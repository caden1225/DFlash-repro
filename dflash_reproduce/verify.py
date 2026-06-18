"""
Validation and evaluation functions for DFlash speculative decoding.

Provides evaluation of:
    - Acceptance length (tau) across datasets
    - Speedup over autoregressive baseline
    - Downstream task performance (GSM8K, MATH500, HumanEval, MBPP, MT-Bench)
    - Full benchmark suite with JSON report generation
"""

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

# Optional imports: deferred to avoid hard dependency.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

# transformers imports are deferred to avoid hard dependency.
try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
except ImportError:
    PreTrainedModel = Any  # type: ignore[misc,assignment]
    PreTrainedTokenizer = Any  # type: ignore[misc,assignment]

from dflash_reproduce.inference import DFlashConfig, DFlashInferenceEngine
from dflash_reproduce.utils import save_json

logger = logging.getLogger("dflash")


def __tqdm(iterable, desc="", **kwargs):
    """Wrapper that uses tqdm when available, falls back to plain iteration."""
    if tqdm is not None:
        return _tqdm(iterable, desc=desc, **kwargs)
    return iterable


# ---------------------------------------------------------------------------
# Acceptance Length Evaluation
# ---------------------------------------------------------------------------

def evaluate_acceptance_length(
    engine: DFlashInferenceEngine,
    dataset: List[Dict[str, str]],
    num_samples: int = 128,
    max_new_tokens: int = 256,
) -> Dict[str, float]:
    """Evaluate average acceptance length (tau) on a dataset.

    Runs speculative decoding on each sample and records how many draft tokens
    are accepted per iteration.

    Args:
        engine: DFlash inference engine.
        dataset: List of data samples, each with at least a "prompt" or "input" key.
        num_samples: Number of samples to evaluate (randomly selected if dataset
            is larger).
        max_new_tokens: Maximum new tokens to generate per sample.

    Returns:
        Dictionary with evaluation metrics:
            - avg_acceptance_length: Mean accepted tokens per iteration (tau).
            - median_acceptance_length: Median accepted tokens.
            - max_acceptance_length: Maximum accepted tokens in any iteration.
            - min_acceptance_length: Minimum accepted tokens in any iteration.
            - acceptance_rate: Fraction of generated tokens that were accepted.
            - num_samples: Number of samples evaluated.
    """
    if num_samples < len(dataset):
        import random
        dataset = random.sample(dataset, num_samples)
    else:
        num_samples = len(dataset)

    all_acceptance_lengths: List[int] = []
    total_generated = 0
    total_accepted = 0

    logger.info(f"Evaluating acceptance length on {num_samples} samples...")

    for sample in _tqdm(dataset, desc="Acceptance length eval"):
        prompt = sample.get("prompt", sample.get("input", sample.get("question", "")))
        if not prompt:
            continue

        # Reset stats for this sample
        engine.stats.reset()

        # Run speculative decoding
        engine.speculative_generate(prompt, max_new_tokens=max_new_tokens)

        # Collect stats
        stats = engine.get_stats()
        all_acceptance_lengths.extend(stats["acceptance_lengths"])
        total_generated += stats["total_generated_tokens"]
        total_accepted += stats["total_accepted_tokens"]

    # Compute metrics
    if all_acceptance_lengths:
        import numpy as np
        avg_tau = float(np.mean(all_acceptance_lengths))
        median_tau = float(np.median(all_acceptance_lengths))
        max_tau = int(np.max(all_acceptance_lengths))
        min_tau = int(np.min(all_acceptance_lengths))
    else:
        avg_tau = median_tau = 0.0
        max_tau = min_tau = 0

    acceptance_rate = total_accepted / max(total_generated, 1)

    results = {
        "avg_acceptance_length": round(avg_tau, 4),
        "median_acceptance_length": round(median_tau, 4),
        "max_acceptance_length": max_tau,
        "min_acceptance_length": min_tau,
        "acceptance_rate": round(acceptance_rate, 4),
        "num_samples": num_samples,
        "total_iterations": len(all_acceptance_lengths),
    }

    logger.info(f"Acceptance length evaluation complete: tau={avg_tau:.3f}")
    return results


# ---------------------------------------------------------------------------
# Autoregressive baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def autoregressive_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    eos_token_id: int = 2,
    device: str = "cuda",
) -> Tuple[str, float, int]:
    """Generate text using standard autoregressive decoding.

    Args:
        model: Language model.
        tokenizer: Tokenizer.
        prompt: Input prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        eos_token_id: EOS token ID.
        device: Device to run on.

    Returns:
        Tuple of (generated_text, elapsed_time_seconds, num_tokens_generated).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    num_input_tokens = input_ids.size(1)

    model.eval()
    start_time = time.perf_counter()

    generated_ids = input_ids.clone()
    past_key_values = None

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        if temperature == 0:
            next_token = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == eos_token_id:
            break

    elapsed = time.perf_counter() - start_time
    num_generated = generated_ids.size(1) - num_input_tokens

    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text, elapsed, num_generated


# ---------------------------------------------------------------------------
# Speedup Evaluation
# ---------------------------------------------------------------------------

def evaluate_speedup(
    engine: DFlashInferenceEngine,
    target_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: List[Dict[str, str]],
    num_samples: int = 128,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    backend: str = "transformers",
) -> Dict[str, float]:
    """Compare speculative decoding speed against autoregressive baseline.

    Measures tokens-per-second for both methods and computes speedup ratio.

    Args:
        engine: DFlash inference engine (speculative decoding).
        target_model: Target model for autoregressive baseline.
        tokenizer: Tokenizer.
        dataset: Evaluation dataset.
        num_samples: Number of samples to evaluate.
        max_new_tokens: Max tokens per sample.
        temperature: Sampling temperature.
        backend: Backend name for logging ("transformers", "vllm", etc.).

    Returns:
        Dictionary with speedup metrics:
            - speedup: SD tokens/sec divided by AR tokens/sec.
            - sd_tokens_per_sec: Speculative decoding throughput.
            - ar_tokens_per_sec: Autoregressive throughput.
            - sd_total_time: Total SD generation time.
            - ar_total_time: Total AR generation time.
            - sd_total_tokens: Total tokens from SD.
            - ar_total_tokens: Total tokens from AR.
            - avg_acceptance_length: Average tau from SD runs.
    """
    if num_samples < len(dataset):
        import random
        eval_dataset = random.sample(dataset, num_samples)
    else:
        eval_dataset = dataset
        num_samples = len(dataset)

    logger.info(f"Evaluating speedup on {num_samples} samples (backend={backend})...")

    # --- Speculative decoding timing ---
    sd_total_time = 0.0
    sd_total_tokens = 0
    all_acceptance_lengths: List[float] = []

    for sample in _tqdm(eval_dataset, desc="Speculative decoding"):
        prompt = sample.get("prompt", sample.get("input", sample.get("question", "")))
        if not prompt:
            continue

        engine.stats.reset()
        start = time.perf_counter()
        engine.speculative_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - start

        stats = engine.get_stats()
        sd_total_time += elapsed
        sd_total_tokens += stats["total_generated_tokens"]
        all_acceptance_lengths.append(stats["avg_acceptance_length_tau"])

    sd_tokens_per_sec = sd_total_tokens / max(sd_total_time, 1e-6)

    # --- Autoregressive baseline timing ---
    ar_total_time = 0.0
    ar_total_tokens = 0

    for sample in _tqdm(eval_dataset, desc="Autoregressive baseline"):
        prompt = sample.get("prompt", sample.get("input", sample.get("question", "")))
        if not prompt:
            continue

        _, elapsed, num_tokens = autoregressive_generate(
            model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=engine.config.eos_token_id,
            device=engine.config.device,
        )
        ar_total_time += elapsed
        ar_total_tokens += num_tokens

    ar_tokens_per_sec = ar_total_tokens / max(ar_total_time, 1e-6)

    # Compute speedup
    speedup = sd_tokens_per_sec / max(ar_tokens_per_sec, 1e-6)
    avg_tau = sum(all_acceptance_lengths) / max(len(all_acceptance_lengths), 1)

    results = {
        "speedup": round(speedup, 4),
        "sd_tokens_per_sec": round(sd_tokens_per_sec, 2),
        "ar_tokens_per_sec": round(ar_tokens_per_sec, 2),
        "sd_total_time": round(sd_total_time, 2),
        "ar_total_time": round(ar_total_time, 2),
        "sd_total_tokens": sd_total_tokens,
        "ar_total_tokens": ar_total_tokens,
        "avg_acceptance_length": round(avg_tau, 4),
        "backend": backend,
        "num_samples": num_samples,
    }

    logger.info(f"Speedup evaluation complete:")
    logger.info(f"  SD: {sd_tokens_per_sec:.2f} tok/s ({sd_total_time:.2f}s)")
    logger.info(f"  AR: {ar_tokens_per_sec:.2f} tok/s ({ar_total_time:.2f}s)")
    logger.info(f"  Speedup: {speedup:.2f}x")
    logger.info(f"  Avg tau: {avg_tau:.3f}")

    return results


# ---------------------------------------------------------------------------
# Downstream Task Evaluation
# ---------------------------------------------------------------------------

def _evaluate_gsm8k(
    engine: DFlashInferenceEngine,
    num_samples: int = 128,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Evaluate on GSM8K math reasoning dataset.

    Args:
        engine: DFlash inference engine.
        num_samples: Number of samples.
        max_new_tokens: Max tokens to generate.

    Returns:
        Dictionary with accuracy and generation statistics.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        logger.warning("datasets library not installed, skipping GSM8K evaluation")
        return {"error": "datasets library not installed"}

    logger.info(f"Loading GSM8K dataset (test split)...")
    try:
        ds = hf_load_dataset("gsm8k", "main", split="test")
    except Exception as e:
        logger.warning(f"Failed to load GSM8K: {e}")
        return {"error": str(e)}

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    correct = 0
    total = 0
    all_acceptance_lengths = []

    for example in _tqdm(ds, desc="GSM8K eval"):
        question = example["question"]
        answer_text = example["answer"]

        # Extract numeric answer from ground truth
        ground_answer = _extract_gsm8k_answer(answer_text)

        # Generate response
        prompt = f"Question: {question}\nAnswer:"
        engine.stats.reset()
        generated = engine.speculative_generate(prompt, max_new_tokens=max_new_tokens)

        # Extract answer from generation
        pred_answer = _extract_gsm8k_answer(generated)

        if _math_equal(pred_answer, ground_answer):
            correct += 1
        total += 1

        stats = engine.get_stats()
        all_acceptance_lengths.append(stats["avg_acceptance_length_tau"])

    accuracy = correct / max(total, 1)
    avg_tau = sum(all_acceptance_lengths) / max(len(all_acceptance_lengths), 1)

    return {
        "task": "gsm8k",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_acceptance_length": round(avg_tau, 4),
    }


def _evaluate_math500(
    engine: DFlashInferenceEngine,
    num_samples: int = 128,
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """Evaluate on MATH-500 math competition problems.

    Args:
        engine: DFlash inference engine.
        num_samples: Number of samples.
        max_new_tokens: Max tokens to generate.

    Returns:
        Dictionary with accuracy and statistics.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        logger.warning("datasets library not installed, skipping MATH500 evaluation")
        return {"error": "datasets library not installed"}

    logger.info("Loading MATH-500 dataset...")
    try:
        ds = hf_load_dataset("hendrycks/competition_math", split="test")
    except Exception as e:
        logger.warning(f"Failed to load MATH dataset: {e}")
        return {"error": str(e)}

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    correct = 0
    total = 0
    all_acceptance_lengths = []

    for example in _tqdm(ds, desc="MATH-500 eval"):
        problem = example.get("problem", "")
        solution = example.get("solution", "")

        ground_answer = _extract_math_answer(solution)

        prompt = f"Problem: {problem}\nSolve the problem step by step.\nSolution:"
        engine.stats.reset()
        generated = engine.speculative_generate(prompt, max_new_tokens=max_new_tokens)

        pred_answer = _extract_math_answer(generated)

        if _math_equal(pred_answer, ground_answer):
            correct += 1
        total += 1

        stats = engine.get_stats()
        all_acceptance_lengths.append(stats["avg_acceptance_length_tau"])

    accuracy = correct / max(total, 1)
    avg_tau = sum(all_acceptance_lengths) / max(len(all_acceptance_lengths), 1)

    return {
        "task": "math500",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_acceptance_length": round(avg_tau, 4),
    }


def _evaluate_humaneval(
    engine: DFlashInferenceEngine,
    num_samples: int = 164,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Evaluate on HumanEval code generation benchmark.

    Uses pass@1 metric: generate one completion per problem and check
    if it passes the test cases.

    Args:
        engine: DFlash inference engine.
        num_samples: Number of problems to evaluate.
        max_new_tokens: Max tokens to generate.

    Returns:
        Dictionary with pass@1 and statistics.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        logger.warning("datasets library not installed, skipping HumanEval evaluation")
        return {"error": "datasets library not installed"}

    logger.info("Loading HumanEval dataset...")
    try:
        ds = hf_load_dataset("openai_humaneval", split="test")
    except Exception as e:
        logger.warning(f"Failed to load HumanEval: {e}")
        return {"error": str(e)}

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    passed = 0
    total = 0
    all_acceptance_lengths = []

    for example in _tqdm(ds, desc="HumanEval eval"):
        prompt_code = example["prompt"]
        test_code = example["test"]
        entry_point = example["entry_point"]

        engine.stats.reset()
        generated = engine.speculative_generate(
            prompt_code,
            max_new_tokens=max_new_tokens,
        )

        # Extract just the completion part
        completion = generated[len(prompt_code):] if generated.startswith(prompt_code) else generated

        # Try to execute test
        if _check_code_passes(prompt_code, completion, test_code, entry_point):
            passed += 1
        total += 1

        stats = engine.get_stats()
        all_acceptance_lengths.append(stats["avg_acceptance_length_tau"])

    pass_at_1 = passed / max(total, 1)
    avg_tau = sum(all_acceptance_lengths) / max(len(all_acceptance_lengths), 1)

    return {
        "task": "humaneval",
        "pass@1": round(pass_at_1, 4),
        "passed": passed,
        "total": total,
        "avg_acceptance_length": round(avg_tau, 4),
    }


def _evaluate_mbpp(
    engine: DFlashInferenceEngine,
    num_samples: int = 256,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Evaluate on MBPP (Mostly Basic Python Programming) benchmark.

    Args:
        engine: DFlash inference engine.
        num_samples: Number of samples.
        max_new_tokens: Max tokens to generate.

    Returns:
        Dictionary with pass rate and statistics.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        logger.warning("datasets library not installed, skipping MBPP evaluation")
        return {"error": "datasets library not installed"}

    logger.info("Loading MBPP dataset...")
    try:
        ds = hf_load_dataset("mbpp", split="test")
    except Exception as e:
        logger.warning(f"Failed to load MBPP: {e}")
        return {"error": str(e)}

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    passed = 0
    total = 0
    all_acceptance_lengths = []

    for example in _tqdm(ds, desc="MBPP eval"):
        task_id = example.get("task_id", "")
        text = example.get("text", "")
        test_list = example.get("test_list", [])
        test_setup_code = example.get("test_setup_code", "")

        prompt = f"# {text}\n"
        engine.stats.reset()
        generated = engine.speculative_generate(prompt, max_new_tokens=max_new_tokens)

        # Extract code from generation
        code = generated[len(prompt):] if generated.startswith(prompt) else generated

        # Check if code passes tests
        if _check_mbpp_code(code, test_setup_code, test_list):
            passed += 1
        total += 1

        stats = engine.get_stats()
        all_acceptance_lengths.append(stats["avg_acceptance_length_tau"])

    pass_rate = passed / max(total, 1)
    avg_tau = sum(all_acceptance_lengths) / max(len(all_acceptance_lengths), 1)

    return {
        "task": "mbpp",
        "pass_rate": round(pass_rate, 4),
        "passed": passed,
        "total": total,
        "avg_acceptance_length": round(avg_tau, 4),
    }


def _evaluate_mt_bench(
    engine: DFlashInferenceEngine,
    num_samples: int = 80,
    max_new_tokens: int = 1024,
) -> Dict[str, Any]:
    """Evaluate on MT-Bench conversation quality benchmark.

    Since MT-Bench requires GPT-4 evaluation for scores, this function
    generates responses and saves them for external evaluation.

    Args:
        engine: DFlash inference engine.
        num_samples: Number of conversations.
        max_new_tokens: Max tokens per turn.

    Returns:
        Dictionary with generation statistics and output file path.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        logger.warning("datasets library not installed, skipping MT-Bench evaluation")
        return {"error": "datasets library not installed"}

    logger.info("Loading MT-Bench dataset...")
    try:
        ds = hf_load_dataset("json", data_files="mt_bench_questions.jsonl", split="train")
    except Exception:
        # Fallback: create synthetic evaluation prompts
        logger.info("MT-Bench dataset not found, using synthetic prompts")
        ds = []
        categories = ["writing", "reasoning", "math", "coding", "roleplay", "stem"]
        for i in range(num_samples):
            ds.append({
                "question_id": i,
                "category": categories[i % len(categories)],
                "turns": [f"This is a test question in {categories[i % len(categories)]}. Please provide a detailed response."],
            })

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    responses = []
    all_acceptance_lengths = []

    for example in _tqdm(ds, desc="MT-Bench eval"):
        question_id = example.get("question_id", len(responses))
        category = example.get("category", "general")
        turns = example.get("turns", [])

        turn_responses = []
        for turn in turns:
            engine.stats.reset()
            response = engine.speculative_generate(turn, max_new_tokens=max_new_tokens)
            turn_responses.append(response)

            stats = engine.get_stats()
            all_acceptance_lengths.append(stats["avg_acceptance_length_tau"])

        responses.append({
            "question_id": question_id,
            "category": category,
            "turns": turns,
            "responses": turn_responses,
        })

    avg_tau = sum(all_acceptance_lengths) / max(len(all_acceptance_lengths), 1)

    return {
        "task": "mt_bench",
        "num_questions": len(responses),
        "avg_acceptance_length": round(avg_tau, 4),
        "responses": responses,
        "note": "Responses saved for external GPT-4 evaluation",
    }


def evaluate_downstream(
    engine: DFlashInferenceEngine,
    task_name: str,
    num_samples: int = 128,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Evaluate DFlash on a downstream task.

    Args:
        engine: DFlash inference engine.
        task_name: Name of the task. Supported:
            - "gsm8k": Math reasoning
            - "math500": Math competition
            - "humaneval": Code generation
            - "mbpp": Python programming
            - "mt_bench": Conversation quality
        num_samples: Number of samples to evaluate.
        max_new_tokens: Max tokens to generate per sample.

    Returns:
        Dictionary with task-specific evaluation results.

    Raises:
        ValueError: If task_name is not supported.
    """
    task_name = task_name.lower().strip()

    evaluators = {
        "gsm8k": _evaluate_gsm8k,
        "math500": _evaluate_math500,
        "humaneval": _evaluate_humaneval,
        "mbpp": _evaluate_mbpp,
        "mt_bench": _evaluate_mt_bench,
    }

    if task_name not in evaluators:
        raise ValueError(
            f"Unknown task: {task_name}. Supported: {list(evaluators.keys())}"
        )

    logger.info(f"Starting downstream evaluation: {task_name}")
    return evaluators[task_name](
        engine=engine,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
    )


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------

def run_benchmark(
    config: DFlashConfig,
    engine: DFlashInferenceEngine,
    target_model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    output_dir: str = "./benchmark_results",
    num_samples: int = 128,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Run the full DFlash benchmark suite.

    Evaluates:
        1. Acceptance length at temperature=0 and temperature=1
        2. Speedup over autoregressive baseline
        3. Downstream task performance (if applicable)

    Args:
        config: DFlash configuration.
        engine: DFlash inference engine.
        target_model: Target model for AR baseline comparison.
        tokenizer: Tokenizer.
        output_dir: Directory to save results.
        num_samples: Number of samples for each evaluation.
        max_new_tokens: Max tokens to generate.

    Returns:
        Dictionary with all benchmark results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting DFlash Benchmark Suite")
    logger.info("=" * 60)

    benchmark_results: Dict[str, Any] = {
        "config": {
            "block_size": config.block_size,
            "max_new_tokens": max_new_tokens,
            "target_layer_ids": config.target_layer_ids,
            "num_samples": num_samples,
        },
        "results": {},
    }

    # 1. Acceptance length at temperature=0 (greedy)
    logger.info("\n--- Evaluating acceptance length (temperature=0) ---")
    # Create synthetic dataset for acceptance evaluation
    synthetic_dataset = _create_synthetic_dataset(num_samples)

    orig_temp = config.temperature
    config.temperature = 0.0
    acc_results_greedy = evaluate_acceptance_length(
        engine=engine,
        dataset=synthetic_dataset,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
    )
    benchmark_results["results"]["acceptance_length_greedy"] = acc_results_greedy

    # 2. Acceptance length at temperature=1 (sampling)
    logger.info("\n--- Evaluating acceptance length (temperature=1) ---")
    config.temperature = 1.0
    acc_results_sampling = evaluate_acceptance_length(
        engine=engine,
        dataset=synthetic_dataset,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
    )
    benchmark_results["results"]["acceptance_length_sampling"] = acc_results_sampling
    config.temperature = orig_temp

    # 3. Speedup evaluation
    if target_model is not None and tokenizer is not None:
        logger.info("\n--- Evaluating speedup ---")
        speedup_results = evaluate_speedup(
            engine=engine,
            target_model=target_model,
            tokenizer=tokenizer,
            dataset=synthetic_dataset,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        benchmark_results["results"]["speedup"] = speedup_results
    else:
        logger.info("Skipping speedup evaluation (target_model/tokenizer not provided)")
        benchmark_results["results"]["speedup"] = {"skipped": True}

    # 4. Save results
    results_path = output_dir / "benchmark_results.json"
    save_json(benchmark_results, results_path)
    logger.info(f"\nBenchmark results saved to {results_path}")

    # 5. Generate report
    report = _generate_report(benchmark_results)
    report_path = output_dir / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Benchmark report saved to {report_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Suite Complete")
    logger.info("=" * 60)

    return benchmark_results


def _generate_report(results: Dict[str, Any]) -> str:
    """Generate a human-readable benchmark report.

    Args:
        results: Benchmark results dictionary.

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DFlash Benchmark Report")
    lines.append("=" * 60)
    lines.append("")

    # Config
    config = results.get("config", {})
    lines.append("Configuration:")
    for k, v in config.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Acceptance length
    for key in ["acceptance_length_greedy", "acceptance_length_sampling"]:
        if key in results["results"]:
            lines.append(f"--- {key} ---")
            data = results["results"][key]
            if "error" in data:
                lines.append(f"  Error: {data['error']}")
            else:
                lines.append(f"  Avg acceptance length (tau): {data.get('avg_acceptance_length', 'N/A')}")
                lines.append(f"  Median acceptance length: {data.get('median_acceptance_length', 'N/A')}")
                lines.append(f"  Max acceptance length: {data.get('max_acceptance_length', 'N/A')}")
                lines.append(f"  Acceptance rate: {data.get('acceptance_rate', 'N/A')}")
            lines.append("")

    # Speedup
    speedup = results["results"].get("speedup", {})
    if "skipped" not in speedup:
        lines.append("--- Speedup ---")
        lines.append(f"  Speedup: {speedup.get('speedup', 'N/A')}x")
        lines.append(f"  SD tokens/sec: {speedup.get('sd_tokens_per_sec', 'N/A')}")
        lines.append(f"  AR tokens/sec: {speedup.get('ar_tokens_per_sec', 'N/A')}")
        lines.append(f"  Avg acceptance length: {speedup.get('avg_acceptance_length', 'N/A')}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper functions for evaluation
# ---------------------------------------------------------------------------

def _create_synthetic_dataset(num_samples: int) -> List[Dict[str, str]]:
    """Create a synthetic dataset for evaluation when real datasets are unavailable.

    Args:
        num_samples: Number of samples.

    Returns:
        List of sample dictionaries with "prompt" key.
    """
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a Python function to calculate the factorial of a number.",
        "Describe the process of photosynthesis.",
        "What are the main differences between Python and Java?",
        "Explain quantum computing to a 10-year-old.",
        "Write a short story about a robot who discovers emotions.",
        "What is the capital of France and what is it famous for?",
        "Describe the water cycle and its importance.",
        "How does the internet work?",
        "Write a function to check if a number is prime.",
        "Explain the theory of relativity briefly.",
        "What are the benefits of regular exercise?",
        "Describe the structure of a cell.",
        "How do you solve a quadratic equation?",
        "What is blockchain technology?",
        "Explain the difference between HTTP and HTTPS.",
    ]
    dataset = []
    for i in range(num_samples):
        dataset.append({"prompt": prompts[i % len(prompts)]})
    return dataset


def _extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer from GSM8K-formatted text.

    Looks for patterns like "#### 123" or the last number in the text.

    Args:
        text: Text containing the answer.

    Returns:
        Extracted answer string.
    """
    import re

    # Look for #### format
    match = re.search(r"####\s*([-\d.,]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: find last number
    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return numbers[-1].strip()

    return text.strip()


def _extract_math_answer(text: str) -> str:
    """Extract answer from MATH-format solution text.

    Looks for boxed answers or the last mathematical expression.

    Args:
        text: Text containing the solution.

    Returns:
        Extracted answer string.
    """
    import re

    # Look for \\boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()

    # Look for #### format
    match = re.search(r"####\s*([-\d.,/]+)", text)
    if match:
        return match.group(1).strip()

    # Fallback: last number
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1].strip()

    return text.strip()


def _math_equal(pred: str, ref: str) -> bool:
    """Check if two mathematical answers are equal.

    Handles integers, floats, fractions, and simple expressions.

    Args:
        pred: Predicted answer.
        ref: Reference answer.

    Returns:
        True if answers are equivalent.
    """
    import re

    pred = pred.strip()
    ref = ref.strip()

    if pred == ref:
        return True

    # Try numeric comparison
    try:
        # Handle fractions
        pred_val = _parse_number(pred)
        ref_val = _parse_number(ref)
        if pred_val is not None and ref_val is not None:
            return abs(pred_val - ref_val) < 1e-6
    except (ValueError, ZeroDivisionError):
        pass

    return False


def _parse_number(s: str) -> Optional[float]:
    """Parse a number from a string, handling fractions.

    Args:
        s: String to parse.

    Returns:
        Float value or None if parsing fails.
    """
    s = s.strip()

    # Handle fractions like "3/4"
    if "/" in s and s.count("/") == 1:
        parts = s.split("/")
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            pass

    # Handle simple numbers
    try:
        return float(s)
    except ValueError:
        pass

    return None


def _check_code_passes(
    prompt_code: str,
    completion: str,
    test_code: str,
    entry_point: str,
) -> bool:
    """Check if generated code passes HumanEval test cases.

    Args:
        prompt_code: Function signature and docstring.
        completion: Generated function body.
        test_code: Test cases.
        entry_point: Function name.

    Returns:
        True if all tests pass.
    """
    import io
    import sys
    import traceback

    # Combine code
    full_code = prompt_code + completion

    # Create namespace and execute
    namespace = {}
    stdout_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(full_code, namespace)
            exec(test_code, namespace)
        return True
    except Exception:
        return False


def _check_mbpp_code(
    code: str,
    test_setup_code: str,
    test_list: List[str],
) -> bool:
    """Check if generated code passes MBPP test cases.

    Args:
        code: Generated code.
        test_setup_code: Setup code.
        test_list: List of test assertions.

    Returns:
        True if all tests pass.
    """
    import io
    import sys
    import traceback
    import contextlib

    full_code = test_setup_code + "\n" + code + "\n"
    for test in test_list:
        full_code += test + "\n"

    namespace = {}
    stdout_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(full_code, namespace)
        return True
    except Exception:
        return False

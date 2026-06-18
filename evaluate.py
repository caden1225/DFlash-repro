#!/usr/bin/env python3
"""
DFlash Evaluation Entry Point

Evaluate a trained DFlash draft model on standard benchmarks.
Usage:
    python evaluate.py --config configs/qwen3_8b.yaml --tasks gsm8k math500 humaneval
    python evaluate.py --draft-model ./checkpoints/dflash --target-model Qwen/Qwen3-8B --tasks all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Ensure project root is on PYTHONPATH
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dflash_reproduce.config import DFlashConfig, load_config
from dflash_reproduce.inference import DFlashInferenceEngine
from dflash_reproduce.model import DFlashDraftModel
from dflash_reproduce.utils import load_target_model, set_seed, setup_logging

logger = logging.getLogger("dflash.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DFlash draft model on standard benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with config file
  python evaluate.py --config configs/qwen3_8b.yaml --tasks gsm8k math500

  # Evaluate all supported tasks
  python evaluate.py --config configs/qwen3_8b.yaml --tasks all

  # Evaluate with explicit model paths
  python evaluate.py --draft-model ./checkpoints/dflash --target-model Qwen/Qwen3-8B --tasks humaneval mbpp

  # Speedup comparison against autoregressive baseline
  python evaluate.py --config configs/qwen3_8b.yaml --speedup --num-samples 128

  # Full benchmark suite
  python evaluate.py --config configs/qwen3_8b.yaml --benchmark
        """.strip(),
    )
    parser.add_argument("--config", "-c", type=str, help="Path to YAML configuration file")
    parser.add_argument("--draft-model", type=str, help="Path or HF name of DFlash draft model")
    parser.add_argument("--target-model", type=str, help="Path or HF name of target model")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Evaluation tasks (gsm8k math500 humaneval mbpp mt_bench). Use 'all' for all tasks.",
    )
    parser.add_argument("--num-samples", type=int, default=128, help="Number of samples per task")
    parser.add_argument("--temperature", type=float, default=0.0, help="Evaluation temperature")
    parser.add_argument("--speedup", action="store_true", help="Run speedup comparison")
    parser.add_argument("--acceptance-length", action="store_true", help="Run acceptance length evaluation")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    return parser.parse_args()


def build_engine(args: argparse.Namespace):
    """Build inference engine from CLI args or config."""
    config: DFlashConfig | None = None
    if args.config:
        config = load_config(args.config)

    draft_model_path = args.draft_model
    target_model_path = args.target_model
    if config is not None:
        draft_model_path = draft_model_path or config.output.checkpoint_dir
        target_model_path = target_model_path or config.model.target_model

    if not draft_model_path or not target_model_path:
        raise ValueError("Must specify --draft-model and --target-model (or use --config)")

    device = args.device or "cuda"
    dtype = getattr(config.model if config else {}, "dtype", "bfloat16")

    logger.info("Loading target model: %s", target_model_path)
    target_model, tokenizer = load_target_model(target_model_path, device=device, dtype=dtype)

    logger.info("Loading draft model: %s", draft_model_path)
    try:
        draft_model = DFlashDraftModel.from_pretrained(draft_model_path)
        draft_model = draft_model.to(device).eval()
    except Exception as e:
        logger.error("Failed to load draft model: %s", e)
        raise

    from types import SimpleNamespace
    ns = SimpleNamespace(
        model=SimpleNamespace(
            target_model=target_model_path,
            block_size=getattr(config.model if config else {}, "block_size", 16),
            target_layer_ids=getattr(config.model if config else {}, "target_layer_ids", None),
            dtype=dtype,
        ),
        inference=SimpleNamespace(
            temperature=args.temperature,
            max_new_tokens=2048,
            device=device,
        ),
    )

    engine = DFlashInferenceEngine(
        config=ns,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
    )
    return engine, tokenizer


def run_speedup_eval(engine, target_model, tokenizer, num_samples: int, temperature: float = 0.0):
    """Run speedup comparison: speculative decoding vs autoregressive baseline."""
    logger.info("=" * 60)
    logger.info("Speedup Evaluation")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("gsm8k", "main", split="test")
        prompts = [item["question"] for item in ds.select(range(min(num_samples, len(ds))))]
    except Exception as e:
        logger.warning("Could not load GSM8K for speedup eval: %s", e)
        prompts = ["What is the capital of France?"] * min(num_samples, 10)

    logger.info("Running speculative decoding on %d prompts ...", len(prompts))
    sd_times = []
    sd_tokens = []
    for prompt in prompts:
        start = time.perf_counter()
        response = engine.speculative_generate(prompt)
        elapsed = time.perf_counter() - start
        gen_tokens = len(tokenizer.encode(response))
        sd_times.append(elapsed)
        sd_tokens.append(gen_tokens)

    logger.info("Running autoregressive baseline on %d prompts ...", len(prompts))
    ar_times = []
    ar_tokens = []
    target_model.eval()
    with torch.no_grad():
        for prompt in prompts:
            start = time.perf_counter()
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            output = target_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
            elapsed = time.perf_counter() - start
            gen_tokens = output.shape[1] - inputs["input_ids"].shape[1]
            ar_times.append(elapsed)
            ar_tokens.append(gen_tokens)

    import numpy as np
    sd_time_total = sum(sd_times)
    ar_time_total = sum(ar_times)
    sd_tokens_total = sum(sd_tokens)
    ar_tokens_total = sum(ar_tokens)

    sd_tok_per_sec = sd_tokens_total / sd_time_total if sd_time_total > 0 else 0
    ar_tok_per_sec = ar_tokens_total / ar_time_total if ar_time_total > 0 else 0
    speedup = sd_tok_per_sec / ar_tok_per_sec if ar_tok_per_sec > 0 else 0

    stats = engine.get_stats() if hasattr(engine, "get_stats") else {}
    results = {
        "speculative_decoding": {
            "total_time": sd_time_total,
            "total_tokens": sd_tokens_total,
            "tokens_per_sec": sd_tok_per_sec,
        },
        "autoregressive": {
            "total_time": ar_time_total,
            "total_tokens": ar_tokens_total,
            "tokens_per_sec": ar_tok_per_sec,
        },
        "speedup": speedup,
        "avg_acceptance_length": stats.get("avg_acceptance_length", 0.0),
        "num_samples": len(prompts),
    }

    logger.info("-" * 40)
    logger.info("Results:")
    logger.info("  Speculative : %.2f tok/s (%d tokens in %.2fs)", sd_tok_per_sec, sd_tokens_total, sd_time_total)
    logger.info("  Autoregressive: %.2f tok/s (%d tokens in %.2fs)", ar_tok_per_sec, ar_tokens_total, ar_time_total)
    logger.info("  Speedup     : %.2fx", speedup)
    logger.info("  Avg tau     : %.2f", results["avg_acceptance_length"])
    logger.info("-" * 40)

    return results


def run_acceptance_length_eval(engine, num_samples: int):
    """Run acceptance length evaluation."""
    logger.info("=" * 60)
    logger.info("Acceptance Length Evaluation")
    logger.info("=" * 60)

    try:
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("gsm8k", "main", split="test")
        prompts = [item["question"] for item in ds.select(range(min(num_samples, len(ds))))]
    except Exception as e:
        logger.warning("Could not load GSM8K: %s", e)
        prompts = ["What is 2+2?"] * min(num_samples, 10)

    acceptance_lengths = []
    for prompt in prompts:
        engine.speculative_generate(prompt)
        stats = engine.get_stats() if hasattr(engine, "get_stats") else {}
        tau = stats.get("avg_acceptance_length", 0.0)
        acceptance_lengths.append(tau)

    import numpy as np
    results = {
        "mean_tau": float(np.mean(acceptance_lengths)),
        "median_tau": float(np.median(acceptance_lengths)),
        "min_tau": float(np.min(acceptance_lengths)),
        "max_tau": float(np.max(acceptance_lengths)),
        "std_tau": float(np.std(acceptance_lengths)),
        "num_samples": len(prompts),
    }

    logger.info("-" * 40)
    logger.info("Acceptance Length (tau):")
    logger.info("  Mean  : %.2f", results["mean_tau"])
    logger.info("  Median: %.2f", results["median_tau"])
    logger.info("  Range : %.2f - %.2f", results["min_tau"], results["max_tau"])
    logger.info("  Std   : %.2f", results["std_tau"])
    logger.info("-" * 40)

    return results


def run_task_eval(engine, tokenizer, task_name: str, num_samples: int, temperature: float = 0.0):
    """Run evaluation on a specific downstream task."""
    logger.info("-" * 40)
    logger.info("Task: %s (n=%d, temp=%.1f)", task_name, num_samples, temperature)

    try:
        from dflash_reproduce.verify import evaluate_downstream
        results = evaluate_downstream(
            engine=engine,
            task_name=task_name,
            num_samples=num_samples,
            temperature=temperature,
        )
    except Exception as e:
        logger.error("Failed to evaluate %s: %s", task_name, e)
        results = {"error": str(e)}

    return results


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(42)

    # Determine tasks to evaluate
    if args.benchmark:
        args.speedup = True
        args.acceptance_length = True
        args.tasks = args.tasks or ["all"]
    elif args.tasks is None and not args.speedup and not args.acceptance_length:
        print("Error: Must specify --tasks, --speedup, --acceptance-length, or --benchmark")
        sys.exit(1)

    if args.tasks and "all" in args.tasks:
        args.tasks = ["gsm8k", "math500", "humaneval", "mbpp", "mt_bench"]

    # Build engine
    logger.info("=" * 70)
    logger.info("DFlash Evaluation")
    logger.info("=" * 70)
    engine, tokenizer = build_engine(args)

    output_dir = Path(args.output_dir or "./eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run acceptance length eval
    if args.acceptance_length:
        all_results["acceptance_length"] = run_acceptance_length_eval(
            engine, args.num_samples
        )

    # Run speedup eval
    if args.speedup:
        all_results["speedup"] = run_speedup_eval(
            engine,
            engine.target_model if hasattr(engine, "target_model") else None,
            tokenizer,
            args.num_samples,
            args.temperature,
        )

    # Run task-specific evals
    if args.tasks:
        task_results = {}
        for task in args.tasks:
            result = run_task_eval(engine, tokenizer, task, args.num_samples, args.temperature)
            task_results[task] = result
        all_results["tasks"] = task_results

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"eval_results_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("Evaluation complete. Results saved to %s", result_file)
    logger.info("=" * 70)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if "acceptance_length" in all_results:
        r = all_results["acceptance_length"]
        print(f"Acceptance Length (tau): {r['mean_tau']:.2f} (mean)")
    if "speedup" in all_results:
        r = all_results["speedup"]
        print(f"Speedup               : {r['speedup']:.2f}x")
        print(f"  Speculative         : {r['speculative_decoding']['tokens_per_sec']:.1f} tok/s")
        print(f"  Autoregressive      : {r['autoregressive']['tokens_per_sec']:.1f} tok/s")
    if "tasks" in all_results:
        for task, result in all_results["tasks"].items():
            if "accuracy" in result:
                print(f"{task:20s}: accuracy={result['accuracy']:.2%}")
            elif "pass_at_1" in result:
                print(f"{task:20s}: pass@1={result['pass_at_1']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()

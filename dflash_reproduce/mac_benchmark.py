#!/usr/bin/env python3
"""
DFlash Mac Benchmark - Compare Official vs Custom Draft Models

Run on MacBook (48GB) to compare:
1. Official DFlash model (z-lab/Qwen3-4B-DFlash-b16) vs
2. Your trained DFlash model

Metrics:
- Acceptance length (tau)
- Generation speed (tok/s)
- Speedup over autoregressive baseline
- GSM8K accuracy (subset)

Usage:
    # Compare official model
    python -m dflash_reproduce.mac_benchmark --official

    # Compare your trained model
    python -m dflash_reproduce.mac_benchmark --custom --draft-path ./mac_checkpoints/dflash_qwen3_4b/final

    # Full comparison
    python -m dflash_reproduce.mac_benchmark --all

    # Quick test (5 samples)
    python -m dflash_reproduce.mac_benchmark --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Setup paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dflash_reproduce.mac_mlx_core import (
    HAS_MLX,
    HAS_MLX_LM,
    load_models,
    stream_generate_ar,
    stream_generate_dflash,
)

logger = logging.getLogger("dflash.mac_benchmark")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_TARGET = "Qwen/Qwen3-4B"
DEFAULT_OFFICIAL_DRAFT = "z-lab/Qwen3-4B-DFlash-b16"
DEFAULT_MAX_TOKENS = 512
DATASET_CACHE = PROJECT_ROOT / "cache" / "gsm8k_test.jsonl"

# --------------------------------------------------------------------------- #
# Dataset Loading
# --------------------------------------------------------------------------- #

def load_gsm8k_samples(num_samples: int = 50, split: str = "test") -> List[Dict[str, str]]:
    """Load GSM8K samples, using local cache if available."""
    samples = []

    # Try local cache first
    if DATASET_CACHE.exists():
        with open(DATASET_CACHE, "r") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                samples.append(json.loads(line))
        if samples:
            return samples

    # Download from HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split=split)
        samples = [{"question": item["question"], "answer": item["answer"]} for item in ds]

        # Cache locally
        DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_CACHE, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        return samples[:num_samples]
    except Exception as e:
        logger.warning("Could not load GSM8K: %s. Using fallback samples.", e)
        return [
            {"question": "What is 15 + 27?", "answer": "42"},
            {"question": "If a train travels at 60 mph for 2.5 hours, how far does it go?", "answer": "150"},
            {"question": "What is the square root of 144?", "answer": "12"},
            {"question": "A rectangle has length 8 and width 5. What is its area?", "answer": "40"},
            {"question": "What is 7 factorial?", "answer": "5040"},
        ] * (num_samples // 5 + 1)


# --------------------------------------------------------------------------- #
# Benchmark Functions
# --------------------------------------------------------------------------- #

def benchmark_speed(
    model,
    draft,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.0,
    block_size: int = 16,
    label: str = "dflash",
) -> Dict[str, Any]:
    """Benchmark generation speed.

    Returns dict with tokens, time, tok/s, and acceptance stats.
    """
    import mlx.core as mx

    total_tokens = 0
    total_time = 0.0
    total_accepted = 0
    total_draft_calls = 0

    logger.info("[%s] Benchmarking %d prompts (max_tokens=%d, temp=%.1f) ...",
                label, len(prompts), max_tokens, temperature)

    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            logger.info("  Progress: %d/%d", i + 1, len(prompts))

        start = time.perf_counter()
        gen_tokens = 0
        last_accepted = 0

        for resp in stream_generate_dflash(
            model, draft, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            block_size=block_size,
        ):
            gen_tokens = resp.generation_tokens
            last_accepted = resp.accepted

        elapsed = time.perf_counter() - start
        total_tokens += gen_tokens
        total_time += elapsed
        total_accepted += last_accepted
        total_draft_calls += 1

        # Clear MLX cache
        mx.clear_cache()

    avg_tau = total_accepted / total_draft_calls if total_draft_calls > 0 else 0
    tok_per_sec = total_tokens / total_time if total_time > 0 else 0

    return {
        "label": label,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_sec": tok_per_sec,
        "avg_acceptance": avg_tau,
        "total_draft_calls": total_draft_calls,
        "num_prompts": len(prompts),
    }


def benchmark_ar_baseline(
    model,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Benchmark autoregressive baseline."""
    import mlx.core as mx

    total_tokens = 0
    total_time = 0.0

    logger.info("[AR] Benchmarking autoregressive baseline on %d prompts ...", len(prompts))

    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            logger.info("  Progress: %d/%d", i + 1, len(prompts))

        start = time.perf_counter()
        gen_tokens = 0

        for resp in stream_generate_ar(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            gen_tokens = resp.generation_tokens

        elapsed = time.perf_counter() - start
        total_tokens += gen_tokens
        total_time += elapsed
        mx.clear_cache()

    tok_per_sec = total_tokens / total_time if total_time > 0 else 0

    return {
        "label": "autoregressive",
        "total_tokens": total_tokens,
        "total_time": total_time,
        "tokens_per_sec": tok_per_sec,
        "avg_acceptance": 1.0,
        "num_prompts": len(prompts),
    }


def extract_final_answer(text: str) -> str:
    """Extract the final numerical answer from GSM8K-style response."""
    import re
    # Look for #### marker
    if "####" in text:
        match = re.search(r"####\s*([-]?[\d,.]+)", text)
        if match:
            return match.group(1).replace(",", "").strip()
    # Look for "The answer is X" pattern
    match = re.search(r"(?:the answer is|answer:\s*)\s*([-]?[\d,.]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()
    # Last number in text
    numbers = re.findall(r"[-]?[\d,.]+", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return ""


def benchmark_gsm8k(
    model,
    draft,
    tokenizer,
    num_samples: int = 50,
    max_tokens: int = 512,
    label: str = "dflash",
) -> Dict[str, Any]:
    """Evaluate on GSM8K subset.

    Returns accuracy and per-sample results.
    """
    import mlx.core as mx

    samples = load_gsm8k_samples(num_samples)
    correct = 0
    results = []

    logger.info("[%s] GSM8K evaluation on %d samples ...", label, len(samples))

    for i, sample in enumerate(samples):
        question = sample["question"]
        true_answer = extract_final_answer(sample["answer"])

        # Generate response
        response_parts = []
        for resp in stream_generate_dflash(
            model, draft, tokenizer, question, max_tokens=max_tokens
        ):
            response_parts.append(resp.text)
        response = "".join(response_parts)

        pred_answer = extract_final_answer(response)
        is_correct = pred_answer == true_answer
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "response": response,
            "true_answer": true_answer,
            "pred_answer": pred_answer,
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0:
            logger.info("  Progress: %d/%d (accuracy so far: %.1f%%)",
                       i + 1, len(samples), 100 * correct / (i + 1))

        mx.clear_cache()

    accuracy = correct / len(samples) if samples else 0

    return {
        "label": label,
        "dataset": "gsm8k",
        "num_samples": len(samples),
        "correct": correct,
        "accuracy": accuracy,
        "per_sample": results,
    }


# --------------------------------------------------------------------------- #
# Report Generation
# --------------------------------------------------------------------------- #

def print_report(results: Dict[str, Any]) -> None:
    """Print formatted benchmark report."""
    print("\n" + "=" * 70)
    print("  DFlash Mac Benchmark Report")
    print("=" * 70)

    # Speed results
    if "speed" in results:
        print("\n  Generation Speed")
        print("  " + "-" * 66)
        speed = results["speed"]
        ar = results.get("ar_baseline")

        print(f"  {'Metric':<30} {'DFlash':>15} {'AR Baseline':>15}")
        print(f"  {'-'*30} {'-'*15} {'-'*15}")
        print(f"  {'Total tokens':<30} {speed['total_tokens']:>15,} {ar['total_tokens'] if ar else 'N/A':>15}")
        print(f"  {'Total time (s)':<30} {speed['total_time']:>15.1f} {ar['total_time'] if ar else 'N/A':>15}")
        print(f"  {'Tokens/sec':<30} {speed['tokens_per_sec']:>15.1f} {ar['tokens_per_sec'] if ar else 'N/A':>15}")
        if ar:
            speedup = speed['tokens_per_sec'] / ar['tokens_per_sec'] if ar['tokens_per_sec'] > 0 else 0
            print(f"  {'Speedup':<30} {speedup:>15.2f}x {'':>15}")
        print(f"  {'Avg acceptance (tau)':<30} {speed['avg_acceptance']:>15.2f} {'':>15}")

    # GSM8K results
    if "gsm8k" in results:
        print("\n  GSM8K Evaluation")
        print("  " + "-" * 66)
        gsm8k = results["gsm8k"]
        print(f"  Model : {gsm8k['label']}")
        print(f"  Samples: {gsm8k['num_samples']}")
        print(f"  Correct: {gsm8k['correct']}/{gsm8k['num_samples']}")
        print(f"  Accuracy: {gsm8k['accuracy']*100:.1f}%")

    # Comparison
    if "comparison" in results:
        print("\n  Model Comparison")
        print("  " + "-" * 66)
        comp = results["comparison"]
        print(f"  {'Model':<25} {'Tok/s':>10} {'Tau':>10} {'Accuracy':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        for name, data in comp.items():
            print(f"  {name:<25} {data.get('tokens_per_sec', 0):>10.1f} "
                  f"{data.get('avg_acceptance', 0):>10.2f} {data.get('accuracy', 0)*100:>9.1f}%")

    print("\n" + "=" * 70)


def save_report(results: Dict[str, Any], output_dir: str = "./mac_eval_results") -> str:
    """Save report to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"mac_benchmark_{timestamp}.json"

    # Remove per-sample data to keep file small
    save_data = {k: v for k, v in results.items()}
    if "gsm8k" in save_data and "per_sample" in save_data["gsm8k"]:
        save_data["gsm8k"] = {k: v for k, v in save_data["gsm8k"].items() if k != "per_sample"}

    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)

    logger.info("Report saved to: %s", path)
    return str(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="DFlash Mac Benchmark - Compare draft models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 samples, official model)
  python -m dflash_reproduce.mac_benchmark --quick

  # Full benchmark (official model, 50 samples)
  python -m dflash_reproduce.mac_benchmark --official --num-samples 50

  # Benchmark your trained model
  python -m dflash_reproduce.mac_benchmark --custom --draft-path ./mac_checkpoints/dflash_qwen3_4b/final

  # Compare both
  python -m dflash_reproduce.mac_benchmark --all --num-samples 50
        """.strip(),
    )
    parser.add_argument("--official", action="store_true", help="Benchmark official DFlash model")
    parser.add_argument("--custom", action="store_true", help="Benchmark custom trained model")
    parser.add_argument("--all", action="store_true", help="Benchmark both and compare")
    parser.add_argument("--quick", action="store_true", help="Quick test with 5 samples")
    parser.add_argument("--draft-path", type=str, default=None, help="Path to custom draft model")
    parser.add_argument("--target-model", type=str, default=DEFAULT_TARGET, help="Target model ID")
    parser.add_argument("--official-draft", type=str, default=DEFAULT_OFFICIAL_DRAFT, help="Official draft ID")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples for eval")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--block-size", type=int, default=16, help="Block size")
    parser.add_argument("--output-dir", type=str, default="./mac_eval_results", help="Output directory")
    parser.add_argument("--skip-ar", action="store_true", help="Skip autoregressive baseline")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate
    if not HAS_MLX:
        print("ERROR: mlx not installed. Run: pip install mlx mlx-lm")
        sys.exit(1)

    # Determine what to run
    if args.quick:
        args.official = True
        args.num_samples = 5
        args.max_tokens = 256

    if args.all:
        args.official = True
        args.custom = True

    if not args.official and not args.custom:
        print("ERROR: Specify --official, --custom, or --all")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Benchmark
    # ------------------------------------------------------------------ #
    all_results = {}
    comparison = {}

    # --- Official Model ---
    if args.official:
        print("\n" + "=" * 70)
        print("  Loading OFFICIAL DFlash model:")
        print(f"    Target: {args.target_model}")
        print(f"    Draft : {args.official_draft}")
        print("=" * 70)

        model, tokenizer, draft = load_models(
            target_id=args.target_model,
            draft_id=args.official_draft,
        )

        # Speed benchmark
        samples = load_gsm8k_samples(args.num_samples)
        prompts = [s["question"] for s in samples]

        speed = benchmark_speed(
            model, draft, tokenizer, prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            block_size=args.block_size,
            label="official",
        )
        all_results["official_speed"] = speed
        comparison["Official DFlash"] = speed

        # AR baseline
        if not args.skip_ar:
            ar = benchmark_ar_baseline(
                model, tokenizer, prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            all_results["ar_baseline"] = ar
            speedup = speed["tokens_per_sec"] / ar["tokens_per_sec"] if ar["tokens_per_sec"] > 0 else 0
            all_results["official_speedup"] = speedup

        # GSM8K
        if not args.quick:
            gsm8k = benchmark_gsm8k(
                model, draft, tokenizer,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                label="official",
            )
            all_results["gsm8k"] = gsm8k
            comparison["Official DFlash"]["accuracy"] = gsm8k["accuracy"]

        print_report({"speed": speed, "ar_baseline": ar if not args.skip_ar else None,
                      "gsm8k": all_results.get("gsm8k")})

    # --- Custom Model ---
    if args.custom:
        if not args.draft_path:
            print("ERROR: --draft-path required for custom model benchmark")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("  Loading CUSTOM DFlash model:")
        print(f"    Target: {args.target_model}")
        print(f"    Draft : {args.draft_path}")
        print("=" * 70)

        # For custom model, we'd need to load from local path
        # This requires the draft model to be saved in MLX-compatible format
        # For now, use the same load path via HF if possible
        model, tokenizer, draft = load_models(
            target_id=args.target_model,
            draft_id=args.draft_path,
        )

        samples = load_gsm8k_samples(args.num_samples)
        prompts = [s["question"] for s in samples]

        speed = benchmark_speed(
            model, draft, tokenizer, prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            block_size=args.block_size,
            label="custom",
        )
        all_results["custom_speed"] = speed
        comparison["Custom DFlash"] = speed

        if not args.skip_ar and "ar_baseline" not in all_results:
            ar = benchmark_ar_baseline(
                model, tokenizer, prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            all_results["ar_baseline"] = ar

        if not args.quick:
            gsm8k = benchmark_gsm8k(
                model, draft, tokenizer,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                label="custom",
            )
            all_results["custom_gsm8k"] = gsm8k
            comparison["Custom DFlash"]["accuracy"] = gsm8k["accuracy"]

        print_report({"speed": speed, "gsm8k": all_results.get("custom_gsm8k")})

    # --- Comparison ---
    if args.all and len(comparison) > 1:
        all_results["comparison"] = comparison
        print("\n" + "=" * 70)
        print("  HEAD-TO-HEAD COMPARISON")
        print("=" * 70)
        print(f"  {'Model':<25} {'Tok/s':>10} {'Tau':>10} {'Accuracy':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        for name, data in comparison.items():
            print(f"  {name:<25} {data.get('tokens_per_sec', 0):>10.1f} "
                  f"{data.get('avg_acceptance', 0):>10.2f} {data.get('accuracy', 0)*100:>9.1f}%")

    # Save report
    report_path = save_report(all_results, args.output_dir)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()

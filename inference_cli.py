#!/usr/bin/env python3
"""
DFlash Inference Entry Point

Run speculative decoding inference with a trained DFlash draft model.
Usage:
    python inference_cli.py --config configs/qwen3_8b.yaml --prompt "What is 2+2?"
    python inference_cli.py --config configs/qwen3_8b.yaml --interactive
    python inference_cli.py --draft-model ./checkpoints/dflash_final --target-model Qwen/Qwen3-8B --prompt "Explain quantum computing"
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

logger = logging.getLogger("dflash.inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DFlash speculative decoding inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt with config file
  python inference_cli.py --config configs/qwen3_8b.yaml --prompt "Explain P=NP"

  # Interactive mode
  python inference_cli.py --draft-model ./checkpoints/dflash --target-model Qwen/Qwen3-8B --interactive

  # Batch inference from JSONL file
  python inference_cli.py --config configs/qwen3_8b.yaml --input prompts.jsonl --output results.jsonl

  # Greedy decoding (temperature=0)
  python inference_cli.py --config configs/qwen3_8b.yaml --prompt "Solve: 2x+5=13" --temperature 0
        """.strip(),
    )
    parser.add_argument("--config", "-c", type=str, help="Path to YAML configuration file")
    parser.add_argument("--draft-model", type=str, help="Path or HF name of trained DFlash draft model")
    parser.add_argument("--target-model", type=str, help="Path or HF name of target (verifier) model")
    parser.add_argument("--prompt", "-p", type=str, help="Single prompt string")
    parser.add_argument("--input", "-i", type=str, help="Path to JSONL file with prompts (one per line)")
    parser.add_argument("--output", "-o", type=str, help="Path to output JSONL file")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum new tokens to generate")
    parser.add_argument("--num-speculative-tokens", type=int, default=None, help="Number of speculative tokens")
    parser.add_argument("--block-size", type=int, default=None, help="Diffusion block size")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default=None, help="Data type (bfloat16/float16/float32)")
    parser.add_argument("--show-stats", action="store_true", default=True, help="Show generation statistics")
    return parser.parse_args()


def load_engine_from_config(args: argparse.Namespace) -> DFlashInferenceEngine:
    """Load inference engine from config file or CLI arguments."""
    config: DFlashConfig | None = None
    if args.config:
        config = load_config(args.config)

    # Determine model paths
    draft_model_path = args.draft_model
    target_model_path = args.target_model
    if config is not None:
        draft_model_path = draft_model_path or config.output.checkpoint_dir
        target_model_path = target_model_path or config.model.target_model

    if not draft_model_path or not target_model_path:
        raise ValueError(
            "Must specify both --draft-model and --target-model (or use --config)"
        )

    # Override inference params from CLI
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    block_size = args.block_size
    device = args.device
    dtype = args.dtype
    if config is not None:
        temperature = temperature if temperature is not None else config.inference.temperature
        max_new_tokens = max_new_tokens or max_new_tokens if max_new_tokens is not None else config.inference.max_new_tokens
        block_size = block_size if block_size is not None else config.model.block_size
        device = device or config.inference.device
        dtype = dtype or config.model.dtype

    temperature = 0.0 if temperature is None else temperature
    max_new_tokens = 2048 if max_new_tokens is None else max_new_tokens
    block_size = 16 if block_size is None else block_size
    device = device or "cuda"

    logger.info("Loading target model: %s", target_model_path)
    target_model, tokenizer = load_target_model(
        target_model_path, device=device, dtype=dtype
    )

    logger.info("Loading DFlash draft model: %s", draft_model_path)
    try:
        draft_model = DFlashDraftModel.from_pretrained(draft_model_path)
        draft_model = draft_model.to(device).eval()
    except Exception as e:
        logger.error("Failed to load draft model: %s", e)
        raise

    # Build a simple config-like namespace for the engine
    engine_config = {
        "model": {
            "target_model": target_model_path,
            "block_size": block_size,
            "target_layer_ids": getattr(config.model if config else {}, "target_layer_ids", None),
            "dtype": dtype or "bfloat16",
        },
        "inference": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "device": device,
        },
    }
    # SimpleNamespace allows dot access
    from types import SimpleNamespace
    ns = SimpleNamespace(
        model=SimpleNamespace(**engine_config["model"]),
        inference=SimpleNamespace(**engine_config["inference"]),
    )

    engine = DFlashInferenceEngine(
        config=ns,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
    )
    return engine


def interactive_mode(engine: DFlashInferenceEngine) -> None:
    """Run an interactive chat session."""
    print("\n" + "=" * 60)
    print("  DFlash Interactive Inference")
    print("  Type 'quit' or 'exit' to leave")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not prompt:
            continue

        start = time.perf_counter()
        response = engine.speculative_generate(prompt)
        elapsed = time.perf_counter() - start

        print(f"\nAssistant> {response}\n")
        if engine.get_stats:
            stats = engine.get_stats()
            gen_tokens = stats.get("generated_tokens", 0)
            print(
                f"  [{gen_tokens} tokens in {elapsed:.2f}s = "
                f"{gen_tokens / elapsed:.1f} tok/s, "
                f"tau={stats.get('avg_acceptance_length', 0):.2f}]\n"
            )


def single_prompt(engine: DFlashInferenceEngine, prompt: str, args) -> None:
    """Run a single prompt and print results."""
    logger.info("Prompt: %s", prompt[:80] + "..." if len(prompt) > 80 else prompt)

    start = time.perf_counter()
    response = engine.speculative_generate(prompt)
    elapsed = time.perf_counter() - start

    print(f"\n{'='*60}")
    print("Response:")
    print(f"{'='*60}")
    print(response)
    print(f"{'='*60}")

    if args.show_stats:
        stats = engine.get_stats() if hasattr(engine, "get_stats") else {}
        gen_tokens = stats.get("generated_tokens", 0)
        accepted = stats.get("accepted_tokens", 0)
        print(f"\nGeneration Stats:")
        print(f"  Tokens generated : {gen_tokens}")
        print(f"  Tokens accepted  : {accepted}")
        print(f"  Acceptance (tau) : {stats.get('avg_acceptance_length', 0):.2f}")
        print(f"  Draft calls      : {stats.get('draft_calls', 0)}")
        print(f"  Target calls     : {stats.get('target_calls', 0)}")
        print(f"  Time             : {elapsed:.2f}s")
        if gen_tokens > 0:
            print(f"  Throughput       : {gen_tokens / elapsed:.1f} tok/s")


def batch_inference(engine: DFlashInferenceEngine, input_path: str, output_path: str) -> None:
    """Run batch inference on a JSONL file."""
    import json

    prompts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj.get("prompt", obj.get("instruction", obj.get("input", ""))))

    logger.info("Running batch inference on %d prompts ...", len(prompts))
    results = []
    for i, prompt in enumerate(prompts):
        response = engine.speculative_generate(prompt)
        results.append({"prompt": prompt, "response": response})
        if (i + 1) % 10 == 0:
            logger.info("Processed %d / %d", i + 1, len(prompts))

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Batch inference complete. Results saved to %s", output_path)


def main() -> None:
    args = parse_args()

    setup_logging()
    set_seed(42)

    # Check required arguments
    if not args.config and (not args.draft_model or not args.target_model):
        print("Error: Must provide --config, or both --draft-model and --target-model")
        sys.exit(1)

    if not args.prompt and not args.input and not args.interactive:
        print("Error: Must provide one of --prompt, --input, or --interactive")
        sys.exit(1)

    # Load engine
    engine = load_engine_from_config(args)

    # Run inference
    if args.interactive:
        interactive_mode(engine)
    elif args.input:
        output_path = args.output or "inference_results.jsonl"
        batch_inference(engine, args.input, output_path)
    elif args.prompt:
        single_prompt(engine, args.prompt, args)


if __name__ == "__main__":
    main()

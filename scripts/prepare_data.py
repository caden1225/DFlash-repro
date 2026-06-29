#!/usr/bin/env python3
"""
DFlash Data Preparation Script

Downloads a dataset, re-generates responses using the target model (for
distribution alignment), tokenizes, and saves the processed data to disk.
This allows data preparation to be done once, independently of training.

Usage:
    # 1. Download and prepare dataset (no response regeneration)
    python scripts/prepare_data.py --config configs/qwen3_8b.yaml

    # 2. Also re-generate responses via vLLM (recommended)
    python scripts/prepare_data.py --config configs/qwen3_8b.yaml --regenerate

    # 3. Also pre-extract and cache hidden states
    python scripts/prepare_data.py --config configs/qwen3_8b.yaml --regenerate --extract-hidden-states

    # 4. Use a different dataset
    python scripts/prepare_data.py --config configs/qwen3_8b.yaml \
        --dataset "HuggingFaceH4/ultrachat_200k" \
        --output-dir ./data/my_training_data

    # 5. Limit samples for quick testing
    python scripts/prepare_data.py --config configs/qwen3_8b.yaml \
        --max-samples 1000 \
        --regenerate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dflash_reproduce.config import DFlashConfig, load_config
from dflash_reproduce.data import (
    load_dataset,
    regenerate_responses,
    tokenize_dataset,
)
from dflash_reproduce.hidden_states import OfflineHiddenStatesExtractor
from dflash_reproduce.utils import load_target_model, set_seed, setup_logging

logger = logging.getLogger("dflash.prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare training data for DFlash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: download and tokenize only
  python scripts/prepare_data.py --config configs/qwen3_8b.yaml

  # Recommended: regenerate responses + tokenize
  python scripts/prepare_data.py --config configs/qwen3_8b.yaml --regenerate

  # Full pipeline: regenerate + tokenize + extract hidden states
  python scripts/prepare_data.py --config configs/qwen3_8b.yaml --regenerate --extract-hidden-states

  # Quick test with limited samples
  python scripts/prepare_data.py --config configs/qwen3_8b.yaml --max-samples 500 --regenerate
        """.strip(),
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-generate responses using the target model (recommended)",
    )
    parser.add_argument(
        "--extract-hidden-states",
        action="store_true",
        help="Pre-extract and cache hidden states from the target model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for prepared data (overrides config)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes for tokenization",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    return parser.parse_args()


def save_dataset_to_jsonl(dataset, output_path: str) -> None:
    """Save a HuggingFace dataset to JSONL format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples to %s", len(dataset), output_path)


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    config = load_config(args.config)

    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.max_samples:
        config.data.max_samples = args.max_samples
    if args.output_dir:
        config.output.checkpoint_dir = args.output_dir

    output_dir = Path(config.output.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DFlash Data Preparation")
    logger.info("=" * 60)
    logger.info("Dataset : %s", config.data.dataset_name)
    logger.info("Target  : %s", config.model.target_model)
    logger.info("Regen   : %s", args.regenerate)
    logger.info("Extract : %s", args.extract_hidden_states)
    logger.info("Output  : %s", output_dir)
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Download / load dataset
    # ------------------------------------------------------------------ #
    logger.info("\n[Step 1/4] Loading dataset...")
    dataset = load_dataset(config.data)

    # Save raw dataset
    raw_path = output_dir / "01_raw.jsonl"
    save_dataset_to_jsonl(dataset, str(raw_path))

    # ------------------------------------------------------------------ #
    # Step 2: Re-generate responses (optional but recommended)
    # ------------------------------------------------------------------ #
    if args.regenerate:
        logger.info("\n[Step 2/4] Re-generating responses via target model...")
        dataset = regenerate_responses(
            dataset=dataset,
            data_config=config.data,
            model_config=config.model,
            cache_path=str(output_dir / "02_regenerated.jsonl"),
        )
    else:
        logger.info("\n[Step 2/4] Skipping response regeneration (use --regenerate to enable)")

    # ------------------------------------------------------------------ #
    # Step 3: Tokenize
    # ------------------------------------------------------------------ #
    logger.info("\n[Step 3/4] Tokenizing...")
    _, tokenizer = load_target_model(config.model)
    tokenized = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        config=config.model,
        num_proc=args.num_proc,
    )

    # Save tokenized dataset
    tokenized_path = output_dir / "03_tokenized"
    tokenized.save_to_disk(str(tokenized_path))
    logger.info("Tokenized dataset saved to %s (%d examples)", tokenized_path, len(tokenized))

    # ------------------------------------------------------------------ #
    # Step 4: Pre-extract hidden states (optional)
    # ------------------------------------------------------------------ #
    if args.extract_hidden_states:
        logger.info("\n[Step 4/4] Extracting and caching hidden states...")

        target_model, _ = load_target_model(config.model)
        layer_ids = config.model.target_layer_ids
        if layer_ids is None:
            from dflash_reproduce.config import build_target_layer_ids

            num_layers = getattr(
                target_model.config, "num_hidden_layers", 28
            )
            layer_ids = build_target_layer_ids(
                num_layers, config.model.draft_num_layers
            )

        extractor = OfflineHiddenStatesExtractor(
            cache_dir=str(output_dir / "04_hidden_states"),
            target_model=config.model.target_model,
            target_layer_ids=layer_ids,
        )

        extractor.extract_and_cache(
            dataset=tokenized,
            batch_size=getattr(config.hidden_states, "batch_size", 4),
        )
        logger.info("Hidden states cached.")
    else:
        logger.info("\n[Step 4/4] Skipping hidden state extraction (use --extract-hidden-states to enable)")

    logger.info("\n" + "=" * 60)
    logger.info("Data preparation complete!")
    logger.info("Output directory: %s", output_dir)
    logger.info("  01_raw.jsonl          - Raw downloaded dataset")
    if args.regenerate:
        logger.info("  02_regenerated.jsonl  - Responses re-generated by target model")
    logger.info("  03_tokenized/         - Tokenized dataset (HuggingFace format)")
    if args.extract_hidden_states:
        logger.info("  04_hidden_states/     - Cached hidden states (.pt files)")
    logger.info("=" * 60)

    # Print next steps
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    if args.extract_hidden_states:
        print(f"  python train.py --config {args.config}")
        print("    (data and hidden states are already prepared)")
    else:
        print(f"  python train.py --config {args.config}")
        print("    (training will extract hidden states online)")
    print("=" * 60)


if __name__ == "__main__":
    main()

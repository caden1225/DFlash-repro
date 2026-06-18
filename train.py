#!/usr/bin/env python3
"""
DFlash Training Entry Point

Train a DFlash draft model for speculative decoding on a target LLM.
Usage:
    python train.py --config configs/qwen3_8b.yaml
    python train.py --config configs/qwen3_8b.yaml --resume checkpoints/epoch_3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Ensure project root is on PYTHONPATH
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dflash_reproduce.config import DFlashConfig, load_config
from dflash_reproduce.data import build_dataloader
from dflash_reproduce.hidden_states import (
    HybridHiddenStatesExtractor,
    OfflineHiddenStatesExtractor,
    OnlineHiddenStatesExtractor,
)
from dflash_reproduce.model import build_draft_model
from dflash_reproduce.trainer import DFlashTrainer
from dflash_reproduce.utils import (
    count_parameters,
    load_target_model,
    set_seed,
    setup_distributed,
    setup_logging,
)

logger = logging.getLogger("dflash.train")


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DFlash draft model for speculative decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with a config file
  python train.py --config configs/qwen3_8b.yaml

  # Resume from a checkpoint
  python train.py --config configs/qwen3_8b.yaml --resume ./checkpoints/epoch_3

  # Override config values via CLI
  python train.py --config configs/qwen3_8b.yaml --training.epochs 10 --training.learning_rate 3e-4

  # Launch with torchrun (distributed training)
  torchrun --standalone --nproc_per_node=4 train.py --config configs/qwen3_8b.yaml
        """.strip(),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Override output directory (overrides config value)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Set up everything but do not start training (validate config)",
    )
    # Allow arbitrary config overrides: --training.epochs 10
    parsed, remaining = parser.parse_known_args()
    overrides = _parse_overrides(remaining)
    parsed.overrides = overrides
    return parsed


def _parse_overrides(args: list[str]) -> dict[str, str]:
    """Parse --section.key value style overrides from remaining CLI args."""
    overrides: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--") and "." in arg and i + 1 < len(args):
            key = arg[2:]  # strip leading --
            value = args[i + 1]
            overrides[key] = value
            i += 2
        else:
            i += 1
    return overrides


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load configuration
    # ------------------------------------------------------------------ #
    config = load_config(args.config, overrides=args.overrides)

    if args.output_dir is not None:
        config.output.checkpoint_dir = args.output_dir

    # Apply CLI overrides
    if args.resume is not None:
        config.output.resume_from = args.resume

    # ------------------------------------------------------------------ #
    # 2. Setup distributed training
    # ------------------------------------------------------------------ #
    rank = setup_distributed(config.distributed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # ------------------------------------------------------------------ #
    # 3. Setup logging & seed
    # ------------------------------------------------------------------ #
    log_dir = Path(config.output.checkpoint_dir) / "logs"
    setup_logging(str(log_dir), rank=rank)
    logger.info("=" * 70)
    logger.info("DFlash Training")
    logger.info("=" * 70)
    logger.info("Config file : %s", args.config)
    logger.info("Rank / world: %d / %d", rank, world_size)
    logger.info("Output dir  : %s", config.output.checkpoint_dir)
    logger.info("Target model: %s", config.model.target_model)
    logger.info("Draft layers: %d", config.model.draft_num_layers)
    logger.info("Block size  : %d", config.model.block_size)
    logger.info("Epochs      : %d", config.training.epochs)
    logger.info("Batch size  : %d", config.training.batch_size)
    logger.info("Learning rate: %.2e", config.training.learning_rate)
    logger.info("=" * 70)

    set_seed(config.training.seed)

    # ------------------------------------------------------------------ #
    # 4. Load target model & tokenizer (to infer architecture)
    # ------------------------------------------------------------------ #
    logger.info("Loading target model configuration ...")
    target_model, tokenizer = load_target_model(config.model)
    target_config = target_model.config

    # Auto-compute target_layer_ids if not specified
    if config.model.target_layer_ids is None:
        from dflash_reproduce.config import build_target_layer_ids
        num_target_layers = getattr(target_config, "num_hidden_layers", 28)
        config.model.target_layer_ids = build_target_layer_ids(
            num_target_layers, config.model.draft_num_layers
        )
        logger.info(
            "Auto-computed target_layer_ids: %s",
            config.model.target_layer_ids,
        )

    # ------------------------------------------------------------------ #
    # 5. Build draft model
    # ------------------------------------------------------------------ #
    logger.info("Building DFlash draft model ...")
    draft_model = build_draft_model(config.model, target_config)
    stats = count_parameters(draft_model)
    logger.info(
        "Draft model parameters: total=%s, trainable=%s",
        f"{stats['total']:,}",
        f"{stats['trainable']:,}",
    )

    # ------------------------------------------------------------------ #
    # 6. Build data loader
    # ------------------------------------------------------------------ #
    logger.info("Building training data loader ...")
    train_dataloader = build_dataloader(config, tokenizer)
    logger.info("Training batches: %d", len(train_dataloader))

    # ------------------------------------------------------------------ #
    # 7. Setup hidden-state extractor
    # ------------------------------------------------------------------ #
    extraction_mode = config.hidden_states.extraction_mode
    logger.info("Hidden-state extraction mode: %s", extraction_mode)

    if extraction_mode == "online":
        extractor = OnlineHiddenStatesExtractor(
            endpoint=config.hidden_states.vllm_endpoint,
            target_layer_ids=config.model.target_layer_ids,
        )
    elif extraction_mode == "offline":
        extractor = OfflineHiddenStatesExtractor(
            cache_dir=config.hidden_states.cache_dir,
            target_model=config.model.target_model,
            target_layer_ids=config.model.target_layer_ids,
        )
    elif extraction_mode == "hybrid":
        extractor = HybridHiddenStatesExtractor(
            cache_dir=config.hidden_states.cache_dir,
            target_model=config.model.target_model,
            target_layer_ids=config.model.target_layer_ids,
            vllm_endpoint=config.hidden_states.vllm_endpoint,
        )
    else:
        raise ValueError(f"Unknown extraction mode: {extraction_mode}")

    # ------------------------------------------------------------------ #
    # 8. Dry-run / training
    # ------------------------------------------------------------------ #
    if args.dry_run:
        logger.info("Dry run completed – configuration is valid.")
        return

    logger.info("Starting training ...")
    trainer = DFlashTrainer(
        config=config,
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        extractor=extractor,
    )

    # Resume if requested
    resume_from = getattr(config.output, "resume_from", None) or args.resume
    if resume_from is not None:
        logger.info("Resuming from checkpoint: %s", resume_from)
        trainer.load_checkpoint(resume_from)

    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()

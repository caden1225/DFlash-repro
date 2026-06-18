"""
DFlash Trainer Module

Implements the complete training loop for the DFlash draft model, including:
  - Weighted cross-entropy loss computation (only on masked positions)
  - Position-dependent loss weighting for block diffusion
  - Distributed training support via PyTorch FSDP
  - Mixed precision training
  - Cosine learning rate scheduler with warmup
  - Checkpoint saving and loading
  - Validation and metric tracking

Usage example:
    >>> from dflash_reproduce.model import DFlashConfig, build_draft_model
    >>> from dflash_reproduce.trainer import DFlashTrainer
    >>> config = DFlashConfig()
    >>> model = build_draft_model(config, target_config)
    >>> trainer = DFlashTrainer(config, model, target_model, tokenizer, train_loader)
    >>> trainer.train()
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast, GradScaler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
try:
    from torch.distributed.fsdp.wrap import (
        lambda_auto_wrap_policy,
    )
    HAS_AUTO_WRAP = True
except ImportError:
    HAS_AUTO_WRAP = False
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dflash_reproduce.model import DFlashConfig, DFlashDraftModel

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Training hyperparameters (defaults)
DEFAULT_LEARNING_RATE = 6e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRADIENT_CLIPPING = 1.0
DEFAULT_WARMUP_RATIO = 0.04
DEFAULT_EPOCHS = 6
DEFAULT_BATCH_SIZE = 4
DEFAULT_LOSS_DECAY_GAMMA = 7.0  # for block_size 16
DEFAULT_NUM_ANCHORS = 512

# FSDP / mixed precision
DEFAULT_USE_FSDP = True
DEFAULT_MIXED_PRECISION = "bf16"  # "bf16", "fp16", or "fp32"


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of steps for linear warmup.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as a ratio of the peak LR.

    Returns:
        A LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# DFlash Trainer
# --------------------------------------------------------------------------- #


class DFlashTrainer:
    """Trainer for the DFlash draft model.

    Handles the full training lifecycle including:
        - Optimizer and scheduler setup
        - Mixed precision training
        - Distributed training via FSDP
        - Weighted loss computation
        - Checkpoint management
        - Logging and metric tracking

    Attributes:
        config: DFlashConfig for the draft model.
        draft_model: The DFlashDraftModel to train.
        target_model: The target (teacher) model for feature extraction.
        tokenizer: Tokenizer for encoding/decoding text.
        optimizer: AdamW optimizer.
        lr_scheduler: Cosine scheduler with warmup.
        train_dataloader: Training data loader.
        val_dataloader: Optional validation data loader.
    """

    def __init__(
        self,
        config: DFlashConfig,
        draft_model: DFlashDraftModel,
        target_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        gradient_clipping: float = DEFAULT_GRADIENT_CLIPPING,
        warmup_ratio: float = DEFAULT_WARMUP_RATIO,
        epochs: int = DEFAULT_EPOCHS,
        loss_decay_gamma: float = DEFAULT_LOSS_DECAY_GAMMA,
        use_fsdp: bool = DEFAULT_USE_FSDP,
        mixed_precision: str = DEFAULT_MIXED_PRECISION,
        output_dir: str = "./checkpoints",
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the DFlashTrainer.

        Args:
            config: DFlashConfig for the draft model.
            draft_model: The draft model to train.
            target_model: The target model for feature extraction. Can be None
                if target features are pre-computed and provided in batches.
            tokenizer: Tokenizer instance.
            train_dataloader: DataLoader for training data.
            val_dataloader: Optional DataLoader for validation data.
            learning_rate: Peak learning rate.
            weight_decay: Weight decay for AdamW.
            gradient_clipping: Max gradient norm for clipping.
            warmup_ratio: Fraction of total steps for warmup.
            epochs: Number of training epochs.
            loss_decay_gamma: Gamma for position-dependent loss weighting.
            use_fsdp: Whether to use FSDP for distributed training.
            mixed_precision: Mixed precision mode ("bf16", "fp16", "fp32").
            output_dir: Directory to save checkpoints.
            logging_steps: Log metrics every N steps.
            save_steps: Save checkpoint every N steps.
            eval_steps: Run validation every N steps.
            device: Device to run training on. Auto-detected if None.
        """
        self.config = config
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        self.warmup_ratio = warmup_ratio
        self.epochs = epochs
        self.loss_decay_gamma = loss_decay_gamma
        self.use_fsdp = use_fsdp
        self.mixed_precision = mixed_precision
        self.output_dir = Path(output_dir)
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps

        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.draft_model.to(self.device)
        if self.target_model is not None:
            self.target_model.to(self.device)
            self.target_model.eval()  # Target model is always in eval mode

        # Move target feature projection to device
        if draft_model.target_feature_proj is not None:
            draft_model.target_feature_proj.to(self.device)

        # Distributed training setup
        self.world_size = 1
        self.local_rank = 0
        self.is_distributed = False

        if dist.is_available() and dist.is_initialized():
            self.is_distributed = True
            self.world_size = dist.get_world_size()
            self.local_rank = dist.get_rank()

        # Mixed precision setup
        self.use_amp = mixed_precision in ("bf16", "fp16")
        self.dtype = torch.float32
        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            self.dtype = torch.float16

        self.scaler = GradScaler() if mixed_precision == "fp16" else None

        # FSDP setup
        if self.use_fsdp and self.is_distributed:
            self.setup_fsdp()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler (initialized later when total steps are known)
        self.lr_scheduler: Optional[LambdaLR] = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []

        # Create output directory
        if self.local_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DFlashTrainer initialized: device=%s, distributed=%s, "
            "world_size=%d, mixed_precision=%s, use_fsdp=%s",
            self.device,
            self.is_distributed,
            self.world_size,
            self.mixed_precision,
            self.use_fsdp,
        )

    def _create_optimizer(self) -> AdamW:
        """Create the AdamW optimizer.

        Returns:
            Configured AdamW optimizer.
        """
        # Collect parameters that require gradients
        params_with_wd = []
        params_no_wd = []

        for name, param in self.draft_model.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay for biases and LayerNorm parameters
            if "bias" in name or "norm" in name or "ln" in name:
                params_no_wd.append(param)
            else:
                params_with_wd.append(param)

        param_groups = [
            {"params": params_with_wd, "weight_decay": self.weight_decay},
            {"params": params_no_wd, "weight_decay": 0.0},
        ]

        optimizer = AdamW(param_groups, lr=self.learning_rate)
        return optimizer

    def _create_scheduler(self, num_training_steps: int) -> LambdaLR:
        """Create the learning rate scheduler.

        Args:
            num_training_steps: Total number of training steps.

        Returns:
            Cosine scheduler with warmup.
        """
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # ------------------------------------------------------------------ #
    # FSDP Setup
    # ------------------------------------------------------------------ #

    def setup_fsdp(self) -> None:
        """Wrap model with FSDP for distributed training."""
        if not self.use_fsdp:
            return

        # Mixed precision config
        mp_dtype = getattr(torch, self.mixed_precision, torch.bfloat16)
        mp_config = MixedPrecision(
            param_dtype=mp_dtype,
            reduce_dtype=mp_dtype,
            buffer_dtype=mp_dtype,
        )

        # Auto wrap policy: wrap each transformer layer
        from dflash_reproduce.model import DFlashTransformerLayer

        auto_wrap_policy = lambda_auto_wrap_policy(
            lambda_fn=lambda module: isinstance(module, DFlashTransformerLayer),
        ) if HAS_AUTO_WRAP else None

        self.draft_model = FSDP(
            self.draft_model,
            mixed_precision=mp_config,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )
        logger.info("FSDP setup complete with mixed_precision=%s", self.mixed_precision)

    # ------------------------------------------------------------------ #
    # Loss Computation
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        loss_weights: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute weighted cross-entropy loss on masked positions.

        Only computes loss at positions where labels != -100 (the ignore index).
        Applies position-dependent weights if provided.

        Formula:
            L = sum(w_i * CE(logits_i, labels_i)) / sum(w_i)

        Args:
            logits: Model output logits [batch, seq_len, vocab_size].
            labels: Target token IDs [batch, seq_len].
                Use -100 for positions to ignore.
            loss_weights: Optional position weights [batch, seq_len] or [seq_len].
            attention_mask: Optional binary mask [batch, seq_len].
                If provided, only computes loss where mask == 1.

        Returns:
            Scalar loss tensor.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for cross-entropy computation
        logits_flat = logits.view(-1, vocab_size)  # [batch * seq_len, vocab_size]
        labels_flat = labels.view(-1)  # [batch * seq_len]

        # Create mask for valid (non-ignored) positions
        valid_mask = labels_flat != -100  # [batch * seq_len]

        if attention_mask is not None:
            attention_mask_flat = attention_mask.view(-1)
            valid_mask = valid_mask & (attention_mask_flat == 1)

        # Filter to valid positions only
        valid_logits = logits_flat[valid_mask]  # [num_valid, vocab_size]
        valid_labels = labels_flat[valid_mask]  # [num_valid]

        if valid_labels.numel() == 0:
            # No valid positions - return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute per-token cross-entropy loss (no reduction)
        ce_loss = F.cross_entropy(
            valid_logits, valid_labels, reduction="none"
        )  # [num_valid]

        # Apply position weights if provided
        if loss_weights is not None:
            if loss_weights.dim() == 1:
                # [seq_len] -> expand to [batch, seq_len] -> flatten
                loss_weights = loss_weights.unsqueeze(0).expand(batch_size, -1)
            loss_weights_flat = loss_weights.view(-1)
            valid_weights = loss_weights_flat[valid_mask]  # [num_valid]

            weighted_loss = (ce_loss * valid_weights).sum() / valid_weights.sum().clamp_min(1e-8)
        else:
            weighted_loss = ce_loss.mean()

        return weighted_loss

    # ------------------------------------------------------------------ #
    # Training Step
    # ------------------------------------------------------------------ #

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dictionary containing:
                - input_ids: [batch, seq_len]
                - target_features: [batch, seq_len, hidden_dim]
                - labels: [batch, seq_len] (target token IDs, -100 for ignore)
                - loss_weights: Optional [batch, seq_len] or [seq_len]
                - attention_mask: Optional sparse attention mask
                - position_ids: Optional [batch, seq_len]

        Returns:
            Dictionary of metrics including 'loss', 'learning_rate', etc.
        """
        self.draft_model.train()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        target_features = batch["target_features"].to(self.device)
        labels = batch["labels"].to(self.device)

        loss_weights = batch.get("loss_weights")
        if loss_weights is not None:
            loss_weights = loss_weights.to(self.device)

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        # Forward pass with optional mixed precision
        with autocast(device_type="cuda", enabled=self.use_amp, dtype=self.dtype):
            logits = self.draft_model(
                input_ids=input_ids,
                target_features=target_features,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            loss = self.compute_loss(
                logits=logits,
                labels=labels,
                loss_weights=loss_weights,
                attention_mask=batch.get("mask_positions"),  # binary mask for masked positions
            )

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler is not None:
            # FP16 training with gradient scaler
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            if self.use_fsdp:
                self.draft_model.clip_grad_norm_(self.gradient_clipping)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.draft_model.parameters(), self.gradient_clipping
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # BF16 or FP32 training
            loss.backward()

            # Gradient clipping
            if self.use_fsdp:
                self.draft_model.clip_grad_norm_(self.gradient_clipping)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.draft_model.parameters(), self.gradient_clipping
                )

            self.optimizer.step()

        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.global_step += 1

        # Collect metrics
        metrics = {
            "loss": loss.detach().item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "epoch": self.current_epoch,
            "step": self.global_step,
        }

        return metrics

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set.

        Returns:
            Dictionary with validation metrics:
                - val_loss: Average validation loss
                - val_perplexity: Perplexity on validation set
                - val_accuracy: Token-level accuracy on masked positions
        """
        if self.val_dataloader is None:
            logger.warning("No validation dataloader provided. Skipping validation.")
            return {}

        self.draft_model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        num_batches = 0

        for batch in self.val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            target_features = batch["target_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            loss_weights = batch.get("loss_weights")
            if loss_weights is not None:
                loss_weights = loss_weights.to(self.device)

            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            position_ids = batch.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(self.device)

            with autocast(device_type="cuda", enabled=self.use_amp, dtype=self.dtype):
                logits = self.draft_model(
                    input_ids=input_ids,
                    target_features=target_features,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

                loss = self.compute_loss(
                    logits=logits,
                    labels=labels,
                    loss_weights=loss_weights,
                    attention_mask=batch.get("mask_positions"),
                )

            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy on masked positions
            predictions = logits.argmax(dim=-1)  # [batch, seq_len]
            valid_mask = labels != -100  # [batch, seq_len]
            correct = ((predictions == labels) & valid_mask).sum().item()
            total_correct += correct
            total_tokens += valid_mask.sum().item()

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")
        accuracy = total_correct / max(total_tokens, 1)

        metrics = {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "val_accuracy": accuracy,
        }

        return metrics

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        """Run the complete training loop.

        Iterates over epochs, processes batches, logs metrics,
        runs validation, and saves checkpoints.
        """
        if self.train_dataloader is None:
            raise ValueError("train_dataloader is required for training.")

        # Calculate total training steps
        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.epochs

        # Initialize scheduler
        if self.lr_scheduler is None:
            self.lr_scheduler = self._create_scheduler(total_steps)

        logger.info(
            "Starting training: epochs=%d, steps_per_epoch=%d, total_steps=%d",
            self.epochs,
            steps_per_epoch,
            total_steps,
        )

        start_time = time.time()

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            self.draft_model.train()

            epoch_start = time.time()
            epoch_losses: List[float] = []

            for step, batch in enumerate(self.train_dataloader):
                # Execute training step
                metrics = self.train_step(batch)
                loss = metrics["loss"]
                epoch_losses.append(loss)
                self.train_loss_history.append(loss)

                # Logging
                if self.global_step % self.logging_steps == 0 and self.local_rank == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / max(elapsed, 1e-8)
                    logger.info(
                        "Epoch [%d/%d] Step [%d/%d] | Loss: %.4f | LR: %.2e | "
                        "Steps/s: %.2f | Time: %.1fs",
                        epoch + 1,
                        self.epochs,
                        step + 1,
                        steps_per_epoch,
                        loss,
                        metrics["learning_rate"],
                        steps_per_sec,
                        elapsed,
                    )

                # Validation
                if self.global_step % self.eval_steps == 0 and self.val_dataloader is not None:
                    val_metrics = self.validate()
                    self.val_loss_history.append(val_metrics.get("val_loss", 0.0))

                    if self.local_rank == 0:
                        logger.info(
                            "Validation at step %d | Loss: %.4f | PPL: %.2f | Acc: %.4f",
                            self.global_step,
                            val_metrics.get("val_loss", 0.0),
                            val_metrics.get("val_perplexity", float("inf")),
                            val_metrics.get("val_accuracy", 0.0),
                        )

                    # Save best model
                    if val_metrics.get("val_loss", float("inf")) < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        if self.local_rank == 0:
                            self.save_checkpoint(
                                epoch=epoch,
                                step=self.global_step,
                                path=str(self.output_dir / "best_model"),
                            )

                    self.draft_model.train()  # Return to train mode

                # Checkpoint saving
                if self.global_step % self.save_steps == 0 and self.local_rank == 0:
                    ckpt_path = self.output_dir / f"checkpoint-{self.global_step}"
                    self.save_checkpoint(
                        epoch=epoch, step=self.global_step, path=str(ckpt_path)
                    )

            # End of epoch logging
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

            if self.local_rank == 0:
                logger.info(
                    "Epoch [%d/%d] completed in %.1fs | Avg Loss: %.4f",
                    epoch + 1,
                    self.epochs,
                    epoch_time,
                    avg_epoch_loss,
                )

            # End-of-epoch checkpoint
            if self.local_rank == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    step=self.global_step,
                    path=str(self.output_dir / f"checkpoint-epoch-{epoch + 1}"),
                )

        # Final checkpoint
        if self.local_rank == 0:
            self.save_checkpoint(
                epoch=self.epochs - 1,
                step=self.global_step,
                path=str(self.output_dir / "final_model"),
            )
            logger.info(
                "Training completed. Total steps: %d, Total time: %.1fs",
                self.global_step,
                time.time() - start_time,
            )

    # ------------------------------------------------------------------ #
    # Checkpoint Management
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, epoch: int, step: int, path: str) -> None:
        """Save a training checkpoint.

        Saves:
            - Model weights (with FSDP unwrapping if needed)
            - Optimizer state
            - Scheduler state
            - Training state (epoch, step, best loss)

        Args:
            epoch: Current epoch number.
            step: Current global step.
            path: Directory path to save the checkpoint.
        """
        ckpt_path = Path(path)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Unwrap FSDP if needed
        model_to_save = self.draft_model
        if isinstance(self.draft_model, FSDP):
            model_to_save = self.draft_model.module

        # Prepare state dicts
        model_state = model_to_save.state_dict()
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = (
            self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        )

        training_state = {
            "epoch": epoch,
            "global_step": step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save
        torch.save(model_state, ckpt_path / "model.pt")
        torch.save(optimizer_state, ckpt_path / "optimizer.pt")
        if scheduler_state is not None:
            torch.save(scheduler_state, ckpt_path / "scheduler.pt")
        torch.save(training_state, ckpt_path / "training_state.pt")

        # Save config
        import json

        config_dict = {
            "num_layers": self.config.num_layers,
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "intermediate_size": self.config.intermediate_size,
            "rms_norm_eps": self.config.rms_norm_eps,
            "rope_theta": self.config.rope_theta,
            "max_position_embeddings": self.config.max_position_embeddings,
            "block_size": self.config.block_size,
            "mask_token_id": self.config.mask_token_id,
        }
        with open(ckpt_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info("Checkpoint saved to %s (epoch=%d, step=%d)", ckpt_path, epoch, step)

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint.

        Restores:
            - Model weights
            - Optimizer state
            - Scheduler state
            - Training state

        Args:
            path: Directory path containing the checkpoint files.

        Raises:
            FileNotFoundError: If checkpoint files are not found.
        """
        ckpt_path = Path(path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

        # Unwrap FSDP if needed
        model_to_load = self.draft_model
        if isinstance(self.draft_model, FSDP):
            model_to_load = self.draft_model.module

        # Load model weights
        model_file = ckpt_path / "model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

        state_dict = torch.load(model_file, map_location=self.device)
        model_to_load.load_state_dict(state_dict)

        # Load optimizer state
        optimizer_file = ckpt_path / "optimizer.pt"
        if optimizer_file.exists():
            optimizer_state = torch.load(optimizer_file, map_location=self.device)
            self.optimizer.load_state_dict(optimizer_state)

        # Load scheduler state
        scheduler_file = ckpt_path / "scheduler.pt"
        if scheduler_file.exists() and self.lr_scheduler is not None:
            scheduler_state = torch.load(scheduler_file)
            self.lr_scheduler.load_state_dict(scheduler_state)

        # Load training state
        training_file = ckpt_path / "training_state.pt"
        if training_file.exists():
            training_state = torch.load(training_file)
            self.current_epoch = training_state.get("epoch", 0)
            self.global_step = training_state.get("global_step", 0)
            self.best_val_loss = training_state.get("best_val_loss", float("inf"))

        logger.info(
            "Checkpoint loaded from %s (epoch=%d, step=%d)",
            ckpt_path,
            self.current_epoch,
            self.global_step,
        )

    # ------------------------------------------------------------------ #
    # Utility Methods
    # ------------------------------------------------------------------ #

    def get_trainable_params(self) -> int:
        """Count the number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.draft_model.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Estimate model size in megabytes.

        Returns:
            Model size in MB.
        """
        param_size = sum(p.numel() * p.element_size() for p in self.draft_model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.draft_model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def print_model_summary(self) -> None:
        """Print a summary of the model architecture and training setup."""
        if self.local_rank != 0:
            return

        print("=" * 60)
        print("DFlash Model Summary")
        print("=" * 60)
        print(f"Trainable parameters: {self.get_trainable_params():,}")
        print(f"Model size: {self.get_model_size_mb():.2f} MB")
        print(f"Device: {self.device}")
        print(f"Distributed: {self.is_distributed} (world_size={self.world_size})")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Use FSDP: {self.use_fsdp}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Gradient clipping: {self.gradient_clipping}")
        print(f"Warmup ratio: {self.warmup_ratio}")
        print(f"Epochs: {self.epochs}")
        print(f"Loss decay gamma: {self.loss_decay_gamma}")
        print("=" * 60)

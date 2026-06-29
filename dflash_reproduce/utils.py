"""
Utility functions for DFlash speculative decoding.

Provides logging, seed setup, device management, distributed training setup,
memory monitoring, timing, model statistics, JSON I/O, model loading, and
text post-processing utilities.
"""

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
# transformers imports are deferred to function-level to avoid
# hard dependency when only using utility functions.


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Optional[str] = None, rank: int = 0) -> logging.Logger:
    """Configure logging with file and console handlers.

    Args:
        log_dir: Directory to save log files. If None, only console logging is used.
        rank: Process rank in distributed training. Only rank 0 writes to files.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("dflash")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [Rank %(rank)d] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (all ranks)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    console_handler.setFormatter(formatter)
    # Inject rank into log records via a filter
    console_handler.addFilter(_RankFilter(rank))
    logger.addHandler(console_handler)

    # File handler (rank 0 only)
    if rank == 0 and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"dflash_rank{rank}.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RankFilter(rank))
        logger.addHandler(file_handler)

    return logger


class _RankFilter(logging.Filter):
    """Filter that injects the current rank into log records."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Detect and return the best available device.

    Preference order: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device: The detected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Distributed training
# ---------------------------------------------------------------------------

class DistributedConfig:
    """Simple configuration container for distributed training."""

    def __init__(
        self,
        backend: str = "nccl",
        world_size: int = 1,
        rank: int = 0,
        local_rank: int = 0,
        init_method: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.init_method = init_method


def setup_distributed(config: DistributedConfig) -> None:
    """Initialize PyTorch distributed training.

    Args:
        config: Distributed configuration.

    Raises:
        RuntimeError: If distributed initialization fails.
    """
    if config.world_size <= 1:
        return  # No distributed training needed

    if not dist.is_initialized():
        init_method = config.init_method or "env://"
        try:
            dist.init_process_group(
                backend=config.backend,
                init_method=init_method,
                world_size=config.world_size,
                rank=config.rank,
            )
            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(config.local_rank)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize distributed training: {e}")


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

def log_memory_usage(tag: str = "", logger: Optional[logging.Logger] = None) -> Dict[str, float]:
    """Log current GPU memory usage.

    Args:
        tag: Optional tag to identify the log entry.
        logger: Logger instance. If None, uses the default dflash logger.

    Returns:
        Dictionary with memory stats (allocated_MB, reserved_MB, max_allocated_MB).
    """
    if logger is None:
        logger = logging.getLogger("dflash")

    stats = {"allocated_MB": 0.0, "reserved_MB": 0.0, "max_allocated_MB": 0.0}

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        stats = {
            "allocated_MB": allocated,
            "reserved_MB": reserved,
            "max_allocated_MB": max_allocated,
        }
        prefix = f"[{tag}] " if tag else ""
        logger.info(
            f"{prefix}GPU Memory: allocated={allocated:.1f}MB, "
            f"reserved={reserved:.1f}MB, max_allocated={max_allocated:.1f}MB"
        )

    return stats


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing code blocks.

    Example:
        >>> with Timer("forward pass"):
        ...     output = model(input)
        forward pass took 0.123s
    """

    def __init__(self, name: str = "block", logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.logger = logger or logging.getLogger("dflash")
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.logger.info(f"{self.name} took {self.elapsed:.4f}s")


# ---------------------------------------------------------------------------
# Model statistics
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to serialize.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from a JSON file.

    Args:
        path: Input file path.

    Returns:
        Deserialized data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class ModelConfig:
    """Configuration for loading a target model."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        use_fast_tokenizer: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device or str(get_device())
        self.dtype = dtype or "auto"
        self.trust_remote_code = trust_remote_code
        self.use_fast_tokenizer = use_fast_tokenizer
        self.cache_dir = cache_dir


def load_target_model(
    config: ModelConfig,
) -> Tuple[Any, Any]:
    """Load a target (verification) model and tokenizer from HuggingFace.

    Args:
        config: Model configuration.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        RuntimeError: If model loading fails.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers library is required to load target models. "
            "Install it with: pip install transformers"
        ) from e

    logger = logging.getLogger("dflash")
    logger.info(f"Loading target model from {config.model_name_or_path}")

    # Resolve dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(config.dtype, config.dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            use_fast=config.use_fast_tokenizer,
            cache_dir=config.cache_dir,
            padding_side="left",
        )
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
            device_map=config.device if config.device != "cpu" else None,
            low_cpu_mem_usage=True,
        )

        if config.device == "cpu" or not torch.cuda.is_available():
            model = model.to(config.device)

        model.eval()

        # Log model info
        param_stats = count_parameters(model)
        logger.info(
            f"Loaded target model: {param_stats['total']:,} params "
            f"({param_stats['trainable']:,} trainable)"
        )

        return model, tokenizer

    except Exception as e:
        raise RuntimeError(f"Failed to load target model from {config.model_name_or_path}: {e}")


# ---------------------------------------------------------------------------
# Text post-processing
# ---------------------------------------------------------------------------

def postprocess_text(text: str) -> str:
    """Clean up generated text by removing special tokens and extra whitespace.

    Args:
        text: Raw generated text.

    Returns:
        Cleaned text.
    """
    # Common special tokens to remove
    special_tokens = [
        "<|endoftext|>", "<|eos|>", "<|pad|>", "<|user|>", "<|assistant|>",
        "<|system|>", "<s>", "</s>", "<pad>", "[PAD]", "[EOS]", "[BOS]",
        "<|im_start|>", "<|im_end|>", "<|eot_id|>", "<|start_header_id|>",
        "<|end_header_id|>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")

    # Remove extra whitespace
    text = " ".join(text.split())

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

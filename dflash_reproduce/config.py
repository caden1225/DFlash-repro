"""
DFlash Configuration Module

Defines configuration classes and loading logic for the DFlash speculative
decoding framework using block diffusion models.

Example:
    >>> config = load_config("config.yaml")
    >>> print(config.model.target_model)
    >>> print(config.training.batch_size)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Build the list of target layer IDs for hidden-state extraction.

    Draft layers are mapped to target-model layers using an evenly-spaced
    scheme that skips the first and last few layers of the target model.
    This follows the DFlash paper's algorithm.

    Args:
        num_target_layers: Number of layers in the target (full-size) model.
        num_draft_layers: Number of layers in the draft (small) model.

    Returns:
        A list of ``num_draft_layers`` target layer indices.

    Raises:
        ValueError: If ``num_draft_layers`` is not positive or exceeds
            ``num_target_layers - 3``.

    Example:
        >>> build_target_layer_ids(28, 5)
        [1, 6, 11, 17, 22]
    """
    if num_draft_layers <= 0:
        raise ValueError(f"num_draft_layers must be positive, got {num_draft_layers}")
    if num_draft_layers > num_target_layers - 3:
        raise ValueError(
            f"num_draft_layers ({num_draft_layers}) cannot exceed "
            f"num_target_layers - 3 ({num_target_layers - 3})"
        )

    if num_draft_layers == 1:
        return [num_target_layers // 2]

    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolve a path string, expanding ``~`` and environment variables."""
    if path is None:
        return None
    return os.path.expandvars(os.path.expanduser(str(path)))


# ---------------------------------------------------------------------------
# Sub-configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the target and draft models.

    Attributes:
        target_model: HuggingFace model identifier or local path.
        target_model_type: Architecture family (``qwen3``, ``llama``, ``gemma``, …).
        draft_num_layers: Number of transformer layers in the draft model.
        draft_vocab_size: Vocabulary size used by the draft model.
        block_size: Diffusion block size (number of masked positions per block).
        target_layer_ids: Explicit list of target layer IDs to extract hidden
            states from. If *None*, computed automatically by
            :func:`build_target_layer_ids`.
        max_seq_len: Maximum sequence length during training / inference.
        dtype: PyTorch dtype string for training (``bfloat16``, ``float16``,
            ``float32``).
    """

    target_model: str = "Qwen/Qwen3-8B"
    target_model_type: str = "qwen3"
    draft_num_layers: int = 5
    draft_vocab_size: int = 8192
    block_size: int = 16
    target_layer_ids: Optional[List[int]] = None
    max_seq_len: int = 3072
    dtype: str = "bfloat16"

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate model configuration parameters.

        Raises:
            ValueError: If any parameter is out of the expected range.
        """
        if not self.target_model:
            raise ValueError("target_model must not be empty")

        valid_types = {"qwen3", "llama", "gemma", "mistral", "phi", "qwen2"}
        if self.target_model_type.lower() not in valid_types:
            logger.warning(
                "target_model_type '%s' not in known set %s; continuing anyway",
                self.target_model_type,
                valid_types,
            )

        if self.draft_num_layers < 1:
            raise ValueError(f"draft_num_layers must be >= 1, got {self.draft_num_layers}")
        if self.draft_vocab_size < 1:
            raise ValueError(f"draft_vocab_size must be >= 1, got {self.draft_vocab_size}")
        if self.block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {self.block_size}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")

        valid_dtypes = {"bfloat16", "float16", "float32", "bf16", "fp16", "fp32"}
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}, got {self.dtype}")

        # Normalise dtype aliases
        dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
        self.dtype = dtype_map.get(self.dtype, self.dtype)


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing.

    Attributes:
        dataset_name: HuggingFace dataset name or local file path.
        dataset_type: Format source – ``huggingface``, ``jsonl``, or ``parquet``.
        text_field: Column / field name containing the conversation data.
        regenerate_responses: Whether to re-generate responses with the target
            model via vLLM.
        regen_temperature: Sampling temperature for response regeneration.
        regen_max_tokens: Maximum new tokens when regenerating.
        train_split: Dataset split to use for training.
        max_samples: Cap on the number of training samples (*None* = unlimited).
    """

    dataset_name: str = "nvidia/Nemotron-Post-Training-Dataset-v2"
    dataset_type: str = "huggingface"
    text_field: str = "conversations"
    regenerate_responses: bool = True
    regen_temperature: float = 0.6
    regen_max_tokens: int = 2048
    train_split: str = "train"
    max_samples: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate data configuration parameters."""
        if not self.dataset_name or not self.dataset_name.strip():
            raise ValueError("dataset_name must not be empty")

        valid_types = {"huggingface", "jsonl", "parquet"}
        if self.dataset_type not in valid_types:
            raise ValueError(f"dataset_type must be one of {valid_types}, got {self.dataset_type}")

        if self.regen_temperature < 0.0:
            raise ValueError(f"regen_temperature must be >= 0, got {self.regen_temperature}")
        if self.regen_max_tokens < 1:
            raise ValueError(f"regen_max_tokens must be >= 1, got {self.regen_max_tokens}")

        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError(f"max_samples must be >= 1 or null, got {self.max_samples}")


@dataclass
class TrainingConfig:
    """Training hyper-parameters.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Per-GPU batch size.
        learning_rate: Peak learning rate.
        weight_decay: Weight-decay coefficient.
        gradient_clipping: Maximum gradient norm.
        warmup_ratio: Fraction of total steps used for linear warmup.
        scheduler: LR scheduler type (``cosine``, ``linear``, ``constant``).
        num_anchors: Number of anchor positions sampled per sequence.
        loss_decay_gamma: Exponential-decay factor for per-position loss weights.
        max_anchors_training: Maximum anchor count for long sequences.
        online_training: Whether to run in online-training mode.
        optimizer: Optimiser name (``adamw``, ``adam``, ``sgd``).
        beta1: Adam beta1.
        beta2: Adam beta2.
        eps: Adam epsilon.
    """

    epochs: int = 6
    batch_size: int = 4
    learning_rate: float = 0.0006
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    warmup_ratio: float = 0.04
    scheduler: str = "cosine"
    num_anchors: int = 512
    loss_decay_gamma: float = 7.0
    max_anchors_training: int = 3072
    online_training: bool = True
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate training configuration parameters."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.gradient_clipping <= 0.0:
            raise ValueError(f"gradient_clipping must be > 0, got {self.gradient_clipping}")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")

        valid_schedulers = {"cosine", "linear", "constant", "cosine_with_restarts"}
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}, got {self.scheduler}")

        if self.num_anchors < 1:
            raise ValueError(f"num_anchors must be >= 1, got {self.num_anchors}")
        if self.loss_decay_gamma <= 0.0:
            raise ValueError(f"loss_decay_gamma must be > 0, got {self.loss_decay_gamma}")
        if self.max_anchors_training < self.num_anchors:
            logger.warning(
                "max_anchors_training (%d) < num_anchors (%d); "
                "this may limit anchor sampling on long sequences",
                self.max_anchors_training,
                self.num_anchors,
            )

        valid_optimizers = {"adamw", "adam", "sgd", "adamw_fused"}
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"optimizer must be one of {valid_optimizers}, got {self.optimizer}"
            )

        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")


@dataclass
class HiddenStatesConfig:
    """Configuration for hidden-state extraction (online or offline).

    Attributes:
        extraction_mode: ``online`` (via vLLM) or ``offline`` (from cache).
        vllm_endpoint: URL of the vLLM OpenAI-compatible API endpoint.
        vllm_port: TCP port on which vLLM listens.
        tensor_parallel_size: vLLM tensor-parallel degree.
        data_parallel_size: vLLM data-parallel degree.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        cache_dir: Local directory for cached hidden-state files.
        num_proc: Number of pre-processing processes.
    """

    extraction_mode: str = "online"
    vllm_endpoint: str = "http://localhost:8000/v1"
    vllm_port: int = 8000
    tensor_parallel_size: int = 1
    data_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9
    cache_dir: str = "./cache/hidden_states"
    num_proc: int = 8

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate hidden-states configuration parameters."""
        valid_modes = {"online", "offline"}
        if self.extraction_mode not in valid_modes:
            raise ValueError(
                f"extraction_mode must be one of {valid_modes}, got {self.extraction_mode}"
            )

        if self.vllm_port < 1 or self.vllm_port > 65535:
            raise ValueError(f"vllm_port must be in [1, 65535], got {self.vllm_port}")

        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )
        if self.data_parallel_size < 1:
            raise ValueError(
                f"data_parallel_size must be >= 1, got {self.data_parallel_size}"
            )

        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )

        if self.num_proc < 1:
            raise ValueError(f"num_proc must be >= 1, got {self.num_proc}")


@dataclass
class InferenceConfig:
    """Configuration for speculative-decoding inference.

    Attributes:
        temperature: Sampling temperature (0.0 = greedy).
        max_new_tokens: Maximum number of tokens to generate.
        num_speculative_tokens: Number of draft tokens attempted per step.
        device: PyTorch device string.
    """

    temperature: float = 0.0
    max_new_tokens: int = 2048
    num_speculative_tokens: int = 15
    device: str = "cuda"

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate inference configuration parameters."""
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {self.num_speculative_tokens}"
            )
        valid_devices = {"cuda", "cpu", "auto"}
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            logger.warning(
                "device '%s' not in known set %s; continuing anyway",
                self.device,
                valid_devices,
            )


@dataclass
class EvaluationConfig:
    """Configuration for downstream evaluation.

    Attributes:
        datasets: List of benchmark names to evaluate on.
        num_samples: Number of evaluation samples per benchmark.
        concurrency: Evaluation concurrency level.
        output_dir: Directory to write evaluation results.
    """

    datasets: List[str] = field(default_factory=lambda: ["gsm8k", "math500", "humaneval"])
    num_samples: int = 128
    concurrency: int = 1
    output_dir: str = "./eval_results"

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate evaluation configuration parameters."""
        if not self.datasets:
            raise ValueError("datasets list must not be empty")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")


@dataclass
class OutputConfig:
    """Configuration for checkpointing and logging.

    Attributes:
        checkpoint_dir: Directory to save model checkpoints.
        save_every: Save a checkpoint every N epochs.
        logging_dir: Directory for TensorBoard / log files.
        log_every: Log metrics every N steps.
    """

    checkpoint_dir: str = "./checkpoints"
    save_every: int = 1
    logging_dir: str = "./logs"
    log_every: int = 10

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate output configuration parameters."""
        if self.save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {self.save_every}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")


@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    Attributes:
        backend: PyTorch distributed backend (``nccl``, ``gloo``, …).
        world_size: Total number of processes (*None* = auto-detect).
        rank: Global rank of this process (*None* = auto-detect).
        local_rank: Local rank on the node (*None* = auto-detect).
        use_fsdp: Whether to wrap the model with FSDP.
    """

    backend: str = "nccl"
    world_size: Optional[int] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    use_fsdp: bool = True

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Validate distributed configuration parameters."""
        valid_backends = {"nccl", "gloo", "mpi", "ucc"}
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got {self.backend}")

        for name, value in [("world_size", self.world_size),
                            ("rank", self.rank),
                            ("local_rank", self.local_rank)]:
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 or null, got {value}")


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

@dataclass
class DFlashConfig:
    """Root configuration container for the DFlash training pipeline.

    Holds nested configuration objects for every subsystem.  Provides
    :meth:`validate` and :meth:`resolve_auto_fields` for post-load
    initialisation and consistency checks.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hidden_states: HiddenStatesConfig = field(default_factory=HiddenStatesConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Run validation on every sub-config and cross-field checks."""
        self.model.validate()
        self.data.validate()
        self.training.validate()
        self.hidden_states.validate()
        self.inference.validate()
        self.evaluation.validate()
        self.output.validate()
        self.distributed.validate()

        # Cross-field consistency checks
        if self.model.block_size == 10 and self.training.loss_decay_gamma != 5.0:
            logger.info(
                "block_size=10 detected; recommended loss_decay_gamma is 5.0 "
                "(current: %f)",
                self.training.loss_decay_gamma,
            )
        elif self.model.block_size == 16 and self.training.loss_decay_gamma != 7.0:
            logger.info(
                "block_size=16 detected; recommended loss_decay_gamma is 7.0 "
                "(current: %f)",
                self.training.loss_decay_gamma,
            )
        elif self.model.block_size == 8 and self.training.loss_decay_gamma != 4.0:
            logger.info(
                "block_size=8 detected; recommended loss_decay_gamma is 4.0 "
                "(current: %f)",
                self.training.loss_decay_gamma,
            )

        if self.model.max_seq_len < self.model.block_size * 2:
            raise ValueError(
                f"max_seq_len ({self.model.max_seq_len}) should be at least "
                f"twice block_size ({self.model.block_size})"
            )

    def resolve_auto_fields(self, num_target_layers: Optional[int] = None) -> None:
        """Resolve fields that are ``None`` or require target-model info.

        Args:
            num_target_layers: Number of layers in the target model.
                Required when ``target_layer_ids`` is ``None``.
        """
        # Resolve target_layer_ids
        if self.model.target_layer_ids is None:
            if num_target_layers is None:
                raise ValueError(
                    "num_target_layers is required to auto-compute target_layer_ids"
                )
            self.model.target_layer_ids = build_target_layer_ids(
                num_target_layers, self.model.draft_num_layers
            )
            logger.info(
                "Auto-computed target_layer_ids: %s", self.model.target_layer_ids
            )

        # Resolve paths
        self.hidden_states.cache_dir = _resolve_path(self.hidden_states.cache_dir)
        self.output.checkpoint_dir = _resolve_path(self.output.checkpoint_dir)
        self.output.logging_dir = _resolve_path(self.output.logging_dir)
        self.evaluation.output_dir = _resolve_path(self.evaluation.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain dictionary.

        Returns:
            Nested dictionary mirroring the YAML structure.
        """
        result: Dict[str, Any] = {}
        for key, sub in self.__dict__.items():
            if hasattr(sub, "__dataclass_fields__"):
                result[key] = {}
                for f_name in sub.__dataclass_fields__:
                    val = getattr(sub, f_name)
                    result[key][f_name] = val
            else:
                result[key] = sub
        return result

    def save(self, path: str) -> None:
        """Save the current configuration to a YAML file.

        Args:
            path: Destination file path.
        """
        path = _resolve_path(path) or path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)
        logger.info("Configuration saved to %s", path)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update *base* with values from *override*."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _flatten_dataclass_defaults(cls: type) -> Dict[str, Any]:
    """Return a dict of field-name → default-value for a dataclass."""
    defaults: Dict[str, Any] = {}
    for f_name, f_def in cls.__dataclass_fields__.items():
        if f_def.default is not MISSING:
            defaults[f_name] = f_def.default
        elif f_def.default_factory is not MISSING:
            defaults[f_name] = f_def.default_factory()
    return defaults


def load_config(
    yaml_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    num_target_layers: Optional[int] = None,
) -> DFlashConfig:
    """Load a :class:`DFlashConfig` from a YAML file and optional overrides.

    Parameters in *overrides* take precedence over the YAML file, which in
    turn takes precedence over the hard-coded defaults.

    Args:
        yaml_path: Path to the YAML configuration file.  If *None*, only
            defaults (and *overrides*) are used.
        overrides: Flat or nested dictionary of config values to override.
        num_target_layers: Number of target-model layers; required when
            ``target_layer_ids`` is not set explicitly.

    Returns:
        A fully validated and resolved :class:`DFlashConfig` instance.

    Raises:
        FileNotFoundError: If *yaml_path* is provided but does not exist.
        ValueError: If validation fails.
    """
    # 1. Start with hard-coded defaults
    config_dict: Dict[str, Any] = {}
    for section_name, section_cls in [
        ("model", ModelConfig),
        ("data", DataConfig),
        ("training", TrainingConfig),
        ("hidden_states", HiddenStatesConfig),
        ("inference", InferenceConfig),
        ("evaluation", EvaluationConfig),
        ("output", OutputConfig),
        ("distributed", DistributedConfig),
    ]:
        config_dict[section_name] = _flatten_dataclass_defaults(section_cls)

    # 2. Merge YAML file
    if yaml_path is not None:
        yaml_path = _resolve_path(yaml_path) or yaml_path
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh)
        if user_cfg:
            _deep_update(config_dict, user_cfg)

    # 3. Merge CLI / programmatic overrides
    if overrides:
        _deep_update(config_dict, overrides)

    # 4. Build dataclass instances
    cfg = DFlashConfig(
        model=ModelConfig(**config_dict.get("model", {})),
        data=DataConfig(**config_dict.get("data", {})),
        training=TrainingConfig(**config_dict.get("training", {})),
        hidden_states=HiddenStatesConfig(**config_dict.get("hidden_states", {})),
        inference=InferenceConfig(**config_dict.get("inference", {})),
        evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
        output=OutputConfig(**config_dict.get("output", {})),
        distributed=DistributedConfig(**config_dict.get("distributed", {})),
    )

    # 5. Resolve auto fields and validate
    cfg.resolve_auto_fields(num_target_layers=num_target_layers)
    cfg.validate()

    logger.info("Configuration loaded successfully from %s", yaml_path or "defaults")
    return cfg

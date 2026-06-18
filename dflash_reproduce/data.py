"""
DFlash Data Preparation Module

Handles dataset loading, response re-generation, tokenization, and the
construction of training batches for the block-diffusion draft model.

The core abstraction is :class:`DFlashTrainingDataset`, which turns a
conversational dataset into sparse-attention training examples where
randomly-sampled *anchor* positions define diffusion blocks.

Example::

    >>> config = load_config("config.yaml")
    >>> tokenizer = AutoTokenizer.from_pretrained(config.model.target_model)
    >>> dataloader = build_dataloader(config, tokenizer)
    >>> for batch in dataloader:
    ...     print(batch["input_ids"].shape)
    ...     print(batch["anchor_positions"].shape)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from dflash_reproduce.config import (
    DFlashConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    build_target_layer_ids,
)

logger = logging.getLogger(__name__)

TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


# ============================================================================
# 1.  Dataset loading
# ============================================================================

def _normalize_conversations(
    examples: Dict[str, List[Any]],
    text_field: str,
) -> Dict[str, List[str]]:
    """Extract ``prompt`` and ``response`` from various conversation formats.

    Supported formats:
    - ``conversations``: list of ``{"from": "human"/"gpt", "value": text}``
    - ``messages``: list of ``{"role": "user"/"assistant", "content": text}``
    - ``instruction`` + ``output``: flat fields
    - ``prompt`` + ``response``: already normalised

    Returns:
        Dictionary with ``prompt`` and ``response`` keys.
    """
    prompts: List[str] = []
    responses: List[str] = []

    for item in examples[text_field]:
        if isinstance(item, str):
            # Plain text – treat the whole thing as a prompt with empty response
            prompts.append(item)
            responses.append("")
            continue

        if not isinstance(item, list):
            item = [item]

        # Try "from"/"value" format (ShareGPT)
        prompt_parts: List[str] = []
        response_parts: List[str] = []
        for turn in item:
            if isinstance(turn, dict):
                role = turn.get("from", turn.get("role", "")).lower()
                content = turn.get("value", turn.get("content", ""))
                if role in ("human", "user"):
                    prompt_parts.append(content)
                elif role in ("gpt", "assistant"):
                    response_parts.append(content)

        prompt = "\n".join(prompt_parts)
        response = "\n".join(response_parts)

        if not prompt and not response:
            # Fallback: use the whole conversation as prompt
            prompt = json.dumps(item, ensure_ascii=False)
            response = ""

        prompts.append(prompt)
        responses.append(response)

    return {"prompt": prompts, "response": responses}


def _load_from_jsonl(path: str) -> HFDataset:
    """Load a dataset from a local JSONL file or directory of JSONL files."""
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isdir(path):
        files = sorted(
            f for f in Path(path).rglob("*.jsonl")
        )
        if not files:
            raise ValueError(f"No .jsonl files found in directory: {path}")
        data_dicts: List[Dict[str, Any]] = []
        for fp in files:
            with open(fp, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        data_dicts.append(json.loads(line))
        return HFDataset.from_list(data_dicts)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data_dicts = [json.loads(line) for line in fh if line.strip()]
        return HFDataset.from_list(data_dicts)


def _load_from_parquet(path: str) -> HFDataset:
    """Load a dataset from a local Parquet file or directory."""
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isdir(path):
        return HFDataset.from_parquet(
            [str(p) for p in Path(path).rglob("*.parquet")]
        )
    return HFDataset.from_parquet(path)


def load_dataset(config: DataConfig) -> HFDataset:
    """Load a dataset according to :class:`DataConfig`.

    Args:
        config: Data configuration.

    Returns:
        A HuggingFace :class:`datasets.Dataset` with at least ``prompt`` and
        ``response`` string columns.

    Raises:
        ValueError: If ``dataset_type`` is unsupported.
        FileNotFoundError: If a local file path does not exist.
    """
    logger.info(
        "Loading dataset: name=%s, type=%s", config.dataset_name, config.dataset_type
    )

    # ------------------------------------------------------------------ #
    # Load raw data
    # ------------------------------------------------------------------ #
    if config.dataset_type == "huggingface":
        ds = hf_load_dataset(config.dataset_name, split=config.train_split)
    elif config.dataset_type == "jsonl":
        ds = _load_from_jsonl(config.dataset_name)
    elif config.dataset_type == "parquet":
        ds = _load_from_parquet(config.dataset_name)
    else:
        raise ValueError(f"Unsupported dataset_type: {config.dataset_type}")

    logger.info("Raw dataset loaded: %d examples", len(ds))

    # ------------------------------------------------------------------ #
    # Normalise to prompt / response
    # ------------------------------------------------------------------ #
    if "prompt" in ds.column_names and "response" in ds.column_names:
        logger.info("Dataset already has prompt/response columns")
    else:
        if config.text_field not in ds.column_names:
            available = ds.column_names
            raise ValueError(
                f"text_field '{config.text_field}' not found in dataset. "
                f"Available columns: {available}"
            )
        ds = ds.map(
            _normalize_conversations,
            fn_kwargs={"text_field": config.text_field},
            batched=True,
            remove_columns=ds.column_names,
            desc="Normalising conversations",
            writer_batch_size=1000,
        )

    # ------------------------------------------------------------------ #
    # Optional sample cap
    # ------------------------------------------------------------------ #
    if config.max_samples is not None and len(ds) > config.max_samples:
        ds = ds.shuffle(seed=42).select(range(config.max_samples))
        logger.info("Down-sampled to %d examples", config.max_samples)

    # Filter out empty prompts
    ds = ds.filter(lambda ex: bool(ex.get("prompt", "").strip()), desc="Filtering empty")
    logger.info("Final dataset: %d examples", len(ds))
    return ds


# ============================================================================
# 2.  Response re-generation via vLLM
# ============================================================================

def regenerate_responses(
    dataset: HFDataset,
    data_config: DataConfig,
    model_config: ModelConfig,
    cache_path: Optional[str] = None,
) -> HFDataset:
    """Re-generate responses using the target model served by vLLM.

    Prompts are batched and sent to the vLLM OpenAI-compatible API.  The
    returned completions replace the original ``response`` column.

    Args:
        dataset: Input dataset with ``prompt`` column.
        data_config: Data configuration (temperature, max_tokens, …).
        model_config: Model configuration (target model name for the API).
        cache_path: If provided, save the re-generated dataset to this path
            and skip regeneration if the file already exists.

    Returns:
        Dataset with updated ``response`` column.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info("Loading cached re-generated responses from %s", cache_path)
        return load_from_disk(cache_path)

    if not data_config.regenerate_responses:
        logger.info("Response regeneration disabled; using original responses")
        return dataset

    try:
        import openai
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for response regeneration. "
            "Install it with: pip install openai"
        )

    # Build system prompt based on model type
    system_prompt = _get_system_prompt(model_config.target_model_type)

    client = openai.OpenAI(
        base_url=f"http://localhost:{data_config.regen_port}/v1"
        if hasattr(data_config, "regen_port")
        else "http://localhost:8000/v1",
        api_key="dummy",
    )

    prompts: List[str] = dataset["prompt"]
    responses: List[str] = []

    batch_size = 32  # API batch size
    total = len(prompts)

    logger.info(
        "Re-generating %d responses with temperature=%.2f, max_tokens=%d",
        total,
        data_config.regen_temperature,
        data_config.regen_max_tokens,
    )

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_prompts = prompts[start_idx:end_idx]

        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ]
            for p in batch_prompts
        ]

        try:
            # vLLM supports batching via the completions API
            completions = client.chat.completions.create(
                model=model_config.target_model,
                messages=messages_batch[0],  # vLLM handles single at a time; loop for safety
                temperature=data_config.regen_temperature,
                max_tokens=data_config.regen_max_tokens,
            )
            batch_responses = [completions.choices[0].message.content or ""]
        except Exception as e:
            logger.error("Error in batch %d-%d: %s", start_idx, end_idx, e)
            batch_responses = [""] * (end_idx - start_idx)

        # Process remaining messages in the batch one by one (safer for vLLM)
        for i in range(1, len(messages_batch)):
            try:
                completion = client.chat.completions.create(
                    model=model_config.target_model,
                    messages=messages_batch[i],
                    temperature=data_config.regen_temperature,
                    max_tokens=data_config.regen_max_tokens,
                )
                batch_responses.append(completion.choices[0].message.content or "")
            except Exception as e:
                logger.error("Error regenerating response %d: %s", start_idx + i, e)
                batch_responses.append("")

        responses.extend(batch_responses)

        if (start_idx // batch_size + 1) % 10 == 0:
            logger.info(
                "Regenerated %d / %d responses", end_idx, total
            )

    # Replace responses
    dataset = dataset.remove_columns("response")
    dataset = dataset.add_column("response", responses)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        dataset.save_to_disk(cache_path)
        logger.info("Saved re-generated dataset to %s", cache_path)

    return dataset


def _get_system_prompt(model_type: str) -> str:
    """Return an appropriate system prompt for the target model family."""
    model_type = model_type.lower()
    if model_type in ("qwen3", "qwen2"):
        return (
            "You are a helpful assistant. Provide clear, accurate, "
            "and concise responses."
        )
    elif model_type == "llama":
        return (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible."
        )
    elif model_type == "gemma":
        return "You are a helpful assistant."
    elif model_type == "mistral":
        return "You are a helpful assistant."
    else:
        return "You are a helpful assistant."


# ============================================================================
# 3.  Tokenization
# ============================================================================

def _apply_chat_template(
    prompt: str,
    response: str,
    tokenizer: TokenizerType,
    model_config: ModelConfig,
) -> Tuple[str, str]:
    """Apply the model's chat template to a prompt-response pair.

    Returns:
        Tuple of (formatted_prompt, formatted_full_text) where the full
        text includes the response.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        try:
            result_full = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # In transformers 4.50+ apply_chat_template may return BatchEncoding
            # when tokenize=True, but with tokenize=False it should return a string.
            # We defensively handle both cases for forward compatibility.
            if hasattr(result_full, "input_ids"):
                full_text = tokenizer.decode(result_full["input_ids"][0], skip_special_tokens=False)
            elif isinstance(result_full, list):
                full_text = tokenizer.decode(result_full, skip_special_tokens=False)
            else:
                full_text = str(result_full)

            # Also format just the prompt (with generation prompt)
            result_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if hasattr(result_prompt, "input_ids"):
                prompt_only = tokenizer.decode(result_prompt["input_ids"][0], skip_special_tokens=False)
            elif isinstance(result_prompt, list):
                prompt_only = tokenizer.decode(result_prompt, skip_special_tokens=False)
            else:
                prompt_only = str(result_prompt)

            return prompt_only, full_text
        except Exception as e:
            logger.warning("Chat template failed: %s; falling back to plain text", e)

    # Fallback: simple concatenation
    full_text = f"User: {prompt}\nAssistant: {response}"
    prompt_only = f"User: {prompt}\nAssistant:"
    return prompt_only, full_text


def _tokenize_example(
    examples: Dict[str, List[str]],
    tokenizer: TokenizerType,
    model_config: ModelConfig,
) -> Dict[str, List[Any]]:
    """Tokenize a batch of prompt-response pairs.

    Returns dict with ``input_ids``, ``attention_mask``, ``labels``,
    and ``response_start_pos``.
    """
    prompts = examples["prompt"]
    responses = examples["response"]

    all_input_ids: List[List[int]] = []
    all_attention_mask: List[List[int]] = []
    all_labels: List[List[int]] = []
    all_response_start: List[int] = []

    max_len = model_config.max_seq_len

    for prompt, response in zip(prompts, responses):
        prompt_text, full_text = _apply_chat_template(
            prompt, response, tokenizer, model_config
        )

        # Tokenize the full conversation
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # Tokenize just the prompt to find response start position
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )
        prompt_len = len(prompt_tokens["input_ids"])

        # Labels: -100 for prompt tokens (don't compute loss), actual ids for response
        labels = [-100] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)
        all_response_start.append(min(prompt_len, len(input_ids) - 1))

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
        "response_start_pos": all_response_start,
    }


def tokenize_dataset(
    dataset: HFDataset,
    tokenizer: TokenizerType,
    config: ModelConfig,
    num_proc: int = 8,
) -> HFDataset:
    """Tokenize a conversational dataset.

    Args:
        dataset: Dataset with ``prompt`` and ``response`` string columns.
        tokenizer: HuggingFace tokenizer for the target model.
        config: Model configuration (``max_seq_len``, …).
        num_proc: Number of processes for batched mapping.

    Returns:
        Tokenized dataset with ``input_ids``, ``attention_mask``,
        ``labels``, and ``response_start_pos`` columns.
    """
    logger.info("Tokenizing dataset with max_seq_len=%d", config.max_seq_len)

    # Set padding side for causal LM
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset.map(
        _tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "model_config": config},
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
        writer_batch_size=1000,
    )

    logger.info("Tokenization complete: %d examples", len(tokenized))
    return tokenized


# ============================================================================
# 4.  Key algorithms: sparse attention mask and loss weights
# ============================================================================

def compute_loss_weights(block_size: int, gamma: float) -> torch.Tensor:
    r"""Compute per-position loss weights inside a diffusion block.

    Earlier masked positions receive higher weight:

    .. math::
        w_i = \exp(-\gamma \cdot i / B)

    where :math:`B` is ``block_size`` and :math:`i \in [0, B-1]`.

    Args:
        block_size: Number of masked positions per block.
        gamma: Decay hyper-parameter (larger = faster decay).

    Returns:
        1-D float tensor of shape ``(block_size,)``.
    """
    positions = torch.arange(block_size, dtype=torch.float32)
    weights = torch.exp(-gamma * positions / block_size)
    return weights


def build_sparse_attention_mask(
    seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    response_start_pos: int,
) -> torch.Tensor:
    """Build a sparse block-attention mask for DFlash training.

    The mask is a ``(seq_len, seq_len)`` boolean tensor where ``True``
    means "allowed to attend".

    Attention rules:

    1. **Prompt** (:math:`<` ``response_start_pos``): causal – each token
       attends only to itself and previous tokens.
    2. **Response block positions** (anchor … anchor + block_size - 1):
       - All block positions can see the **prompt** (causally).
       - Within the *same* block: **bidirectional** attention (all positions
         see each other).
       - Different blocks: **no** cross-attention.
    3. **Non-block response positions**: only see the prompt (causally).

    Args:
        seq_len: Length of the sequence.
        anchor_positions: 1-D integer tensor of anchor indices.
        block_size: Size of each diffusion block.
        response_start_pos: Index where the response starts.

    Returns:
        Boolean tensor of shape ``(seq_len, seq_len)``.
    """
    device = anchor_positions.device
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Build block membership: block_id per position (-1 = not in any block)
    block_ids = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    for block_idx, anchor in enumerate(anchor_positions):
        anchor_int = int(anchor.item())
        end = min(anchor_int + block_size, seq_len)
        block_ids[anchor_int:end] = block_idx

    for i in range(seq_len):
        for j in range(seq_len):
            if i < response_start_pos:
                # Prompt: fully causal
                if j <= i:
                    mask[i, j] = True
            elif block_ids[i] >= 0:
                # Response block position
                if j < response_start_pos:
                    # Can see prompt (causally)
                    mask[i, j] = True
                elif block_ids[j] == block_ids[i]:
                    # Within same block: bidirectional
                    mask[i, j] = True
                # Cannot see other blocks or non-block response positions
            elif j < response_start_pos and j <= i:
                # Non-block response position: only sees prompt (causally)
                mask[i, j] = True

    return mask


def build_sparse_attention_mask_efficient(
    seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    response_start_pos: int,
) -> torch.Tensor:
    """Efficient vectorised version of :func:`build_sparse_attention_mask`.

    Uses batched indexing where possible for better performance on GPU.

    Args:
        seq_len: Length of the sequence.
        anchor_positions: 1-D integer tensor of anchor indices.
        block_size: Size of each diffusion block.
        response_start_pos: Index where the response starts.

    Returns:
        Boolean tensor of shape ``(seq_len, seq_len)``.
    """
    device = anchor_positions.device
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Build block membership
    block_ids = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    for block_idx, anchor in enumerate(anchor_positions):
        anchor_int = int(anchor.item())
        end = min(anchor_int + block_size, seq_len)
        block_ids[anchor_int:end] = block_idx

    # 1. Prompt: fully causal (vectorised)
    if response_start_pos > 0:
        prompt_causal = torch.tril(
            torch.ones(response_start_pos, response_start_pos, dtype=torch.bool, device=device)
        )
        mask[:response_start_pos, :response_start_pos] = prompt_causal

    # 2. Block positions can see prompt (causally)
    resp_region = torch.arange(response_start_pos, seq_len, device=device)
    block_resp_ids = block_ids[resp_region]
    block_mask = block_resp_ids >= 0  # which response positions are in blocks

    if block_mask.any():
        block_positions = resp_region[block_mask]
        # All block positions see all prompt positions
        mask[block_positions.unsqueeze(1), torch.arange(response_start_pos, device=device)] = True

    # 3. Non-block response positions only see prompt (causally)
    non_block_mask = ~block_mask
    if non_block_mask.any():
        non_block_positions = resp_region[non_block_mask]
        for i in non_block_positions:
            mask[i, :min(response_start_pos, int(i.item()) + 1)] = True

    # 4. Within each block: bidirectional attention (vectorised)
    for block_idx in range(len(anchor_positions)):
        pos_in_block = (block_ids == block_idx).nonzero(as_tuple=True)[0]
        if len(pos_in_block) == 0:
            continue
        grid_i = pos_in_block.unsqueeze(1).expand(-1, len(pos_in_block))
        grid_j = pos_in_block.unsqueeze(0).expand(len(pos_in_block), -1)
        mask[grid_i, grid_j] = True

    return mask


# ============================================================================
# 5.  PyTorch Dataset for DFlash training
# ============================================================================

class DFlashTrainingDataset(Dataset):
    """PyTorch Dataset that constructs block-diffusion training examples.

    Each example consists of:

    - ``input_ids``: the full tokenised sequence (prompt + response).
    - ``anchor_positions``: randomly sampled anchor indices in the response.
    - ``block_mask``: integer tensor mapping each position to its block ID
      (or -1 for non-block positions).
    - ``attention_mask``: sparse boolean attention mask implementing the
      block-diffusion attention pattern.
    - ``labels``: target token IDs (``-100`` for non-mask positions).
    - ``loss_weights``: per-position weights inside each block.

    Args:
        tokenized_dataset: HuggingFace dataset with ``input_ids``,
            ``response_start_pos``, and optionally ``labels`` columns.
        model_config: Model configuration.
        training_config: Training configuration.
        loss_weights: Pre-computed loss-weight tensor of shape
            ``(block_size,)``.
    """

    def __init__(
        self,
        tokenized_dataset: HFDataset,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        loss_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.dataset = tokenized_dataset
        self.model_config = model_config
        self.training_config = training_config
        self.block_size = model_config.block_size
        self.max_seq_len = model_config.max_seq_len
        self.num_anchors = training_config.num_anchors
        self.max_anchors_training = training_config.max_anchors_training

        if loss_weights is None:
            self.loss_weights = compute_loss_weights(
                self.block_size, training_config.loss_decay_gamma
            )
        else:
            self.loss_weights = loss_weights

    def __len__(self) -> int:
        return len(self.dataset)

    def _sample_anchors(self, response_start: int, seq_len: int) -> torch.Tensor:
        """Sample anchor positions within the response region.

        Anchors are placed so that each block ``[anchor, anchor+block_size)``
        fits inside the response.  The number of anchors is capped by both
        ``num_anchors`` and the physical space available.

        Args:
            response_start: First index of the response region.
            seq_len: Total sequence length.

        Returns:
            1-D integer tensor of sorted anchor positions.
        """
        # Valid anchor range: response_start to seq_len - block_size
        max_anchor = seq_len - self.block_size
        if max_anchor < response_start:
            # Response too short – return empty
            return torch.tensor([], dtype=torch.long)

        available_positions = list(range(response_start, max_anchor + 1))
        max_possible = len(available_positions)
        num_to_sample = min(self.num_anchors, self.max_anchors_training, max_possible)

        if num_to_sample <= 0:
            return torch.tensor([], dtype=torch.long)

        sampled = random.sample(available_positions, num_to_sample)
        sampled.sort()
        return torch.tensor(sampled, dtype=torch.long)

    def _build_block_mask(
        self,
        seq_len: int,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Build block membership tensor.

        Returns:
            Integer tensor of shape ``(seq_len,)`` where each position
            is mapped to its block ID (or ``-1`` if not in any block).
        """
        block_mask = torch.full((seq_len,), -1, dtype=torch.long)
        for block_idx, anchor in enumerate(anchor_positions):
            anchor_int = int(anchor.item())
            end = min(anchor_int + self.block_size, seq_len)
            block_mask[anchor_int:end] = block_idx
        return block_mask

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Build labels tensor: only mask positions have real labels.

        Non-mask positions (including prompt) are set to ``-100`` so that
        PyTorch's cross-entropy ignores them.

        Args:
            input_ids: Full sequence token IDs.
            anchor_positions: Anchor indices.

        Returns:
            Labels tensor of same shape as ``input_ids``.
        """
        labels = torch.full_like(input_ids, -100)
        seq_len = len(input_ids)
        for anchor in anchor_positions:
            anchor_int = int(anchor.item())
            end = min(anchor_int + self.block_size, seq_len)
            # The "target" for each mask position is the *next* token
            for pos in range(anchor_int, end):
                if pos + 1 < seq_len:
                    labels[pos] = input_ids[pos + 1]
        return labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch a single training example.

        Returns:
            Dictionary with keys:

            - ``input_ids`` (LongTensor, ``(seq_len,)``)
            - ``anchor_positions`` (LongTensor, ``(num_anchors,)``)
            - ``block_mask`` (LongTensor, ``(seq_len,)``)
            - ``attention_mask`` (BoolTensor, ``(seq_len, seq_len)``)
            - ``labels`` (LongTensor, ``(seq_len,)``)
            - ``loss_weights`` (FloatTensor, ``(block_size,)``)
            - ``response_start_pos`` (int)
        """
        example = self.dataset[idx]

        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        response_start = int(example["response_start_pos"])
        seq_len = len(input_ids)

        # Clamp response_start if out of bounds
        response_start = min(response_start, seq_len - 1)

        # Sample anchors
        anchor_positions = self._sample_anchors(response_start, seq_len)

        # Build block mask
        block_mask = self._build_block_mask(seq_len, anchor_positions)

        # Build sparse attention mask (efficient version)
        if len(anchor_positions) > 0:
            attention_mask = build_sparse_attention_mask_efficient(
                seq_len, anchor_positions, self.block_size, response_start
            )
        else:
            # No anchors – simple causal mask
            attention_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool)
            )

        # Build labels
        labels = self._build_labels(input_ids, anchor_positions)

        return {
            "input_ids": input_ids,
            "anchor_positions": anchor_positions,
            "block_mask": block_mask,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": self.loss_weights.clone(),
            "response_start_pos": response_start,
        }


# ============================================================================
# 6.  Collate function
# ============================================================================

def dflash_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""Collate a list of DFlash examples into a batched dictionary.

    Handles variable-length sequences by padding to the maximum length in
    the batch.  Constructs batch-level sparse attention masks by placing
    each example's mask on the block-diagonal.

    Args:
        batch: List of dictionaries returned by
            :meth:`DFlashTrainingDataset.__getitem__`.

    Returns:
        Batched dictionary with keys:

        - ``input_ids`` (LongTensor, ``(batch_size, max_seq_len)``)
        - ``attention_mask`` (BoolTensor,
          ``(batch_size, max_seq_len, max_seq_len)``)
        - ``labels`` (LongTensor, ``(batch_size, max_seq_len)``)
        - ``block_mask`` (LongTensor, ``(batch_size, max_seq_len)``)
        - ``loss_weights`` (FloatTensor, ``(batch_size, block_size)``)
        - ``anchor_positions`` (LongTensor, ``(batch_size, max_num_anchors)``)
        - ``anchor_mask`` (BoolTensor, ``(batch_size, max_num_anchors)``)
        - ``response_start_pos`` (LongTensor, ``(batch_size,)``)
    """
    batch_size = len(batch)

    # Determine max lengths
    max_seq_len = max(len(ex["input_ids"]) for ex in batch)
    max_anchors = max(len(ex["anchor_positions"]) for ex in batch)
    block_size = batch[0]["loss_weights"].shape[0]

    device = batch[0]["input_ids"].device

    # Allocate padded tensors
    input_ids = torch.full((batch_size, max_seq_len), 0, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long, device=device)
    block_mask = torch.full((batch_size, max_seq_len), -1, dtype=torch.long, device=device)
    loss_weights = torch.zeros(batch_size, block_size, dtype=torch.float32, device=device)
    attention_mask = torch.zeros(
        batch_size, max_seq_len, max_seq_len, dtype=torch.bool, device=device
    )
    anchor_positions = torch.zeros(
        batch_size, max_anchors, dtype=torch.long, device=device
    )
    anchor_mask = torch.zeros(batch_size, max_anchors, dtype=torch.bool, device=device)
    response_start_pos = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b_idx, example in enumerate(batch):
        seq_len = len(example["input_ids"])
        num_anchors = len(example["anchor_positions"])

        # Copy sequences
        input_ids[b_idx, :seq_len] = example["input_ids"]
        labels[b_idx, :seq_len] = example["labels"]
        block_mask[b_idx, :seq_len] = example["block_mask"]
        loss_weights[b_idx] = example["loss_weights"]
        response_start_pos[b_idx] = example["response_start_pos"]

        # Copy anchors with padding mask
        if num_anchors > 0:
            anchor_positions[b_idx, :num_anchors] = example["anchor_positions"]
            anchor_mask[b_idx, :num_anchors] = True

        # Place individual attention mask on the diagonal of the batch mask
        attn = example["attention_mask"]
        attention_mask[b_idx, :seq_len, :seq_len] = attn

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "block_mask": block_mask,
        "loss_weights": loss_weights,
        "anchor_positions": anchor_positions,
        "anchor_mask": anchor_mask,
        "response_start_pos": response_start_pos,
    }


# ============================================================================
# 7.  DataLoader builder
# ============================================================================

def build_dataloader(
    config: DFlashConfig,
    tokenizer: TokenizerType,
    dataset: Optional[HFDataset] = None,
    is_train: bool = True,
) -> DataLoader:
    """Build a PyTorch :class:`DataLoader` for DFlash training.

    This is the high-level convenience function that chains together
    dataset loading, optional response regeneration, tokenization, and
    the :class:`DFlashTrainingDataset` wrapper.

    Args:
        config: Full DFlash configuration.
        tokenizer: Tokenizer for the target model.
        dataset: Pre-loaded dataset ( skips loading if provided).
        is_train: Whether this is a training dataloader (affects shuffling).

    Returns:
        A :class:`DataLoader` yielding batched training tensors.
    """
    # Step 1: Load dataset if not provided
    if dataset is None:
        dataset = load_dataset(config.data)

        # Step 2: Optionally regenerate responses
        if config.data.regenerate_responses:
            cache_dir = os.path.expanduser(config.hidden_states.cache_dir)
            cache_path = os.path.join(cache_dir, "regenerated_responses")
            dataset = regenerate_responses(
                dataset,
                config.data,
                config.model,
                cache_path=cache_path,
            )

    # Step 3: Tokenize
    tokenized = tokenize_dataset(
        dataset,
        tokenizer,
        config.model,
        num_proc=config.hidden_states.num_proc,
    )

    # Step 4: Build PyTorch dataset
    loss_weights = compute_loss_weights(
        config.model.block_size, config.training.loss_decay_gamma
    )
    train_dataset = DFlashTrainingDataset(
        tokenized,
        config.model,
        config.training,
        loss_weights=loss_weights,
    )

    # Step 5: Distributed sampler
    sampler: Optional[Sampler] = None
    if is_train:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        if world_size > 1:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=is_train,
            )
            logger.info(
                "Using DistributedSampler: world_size=%d, rank=%d", world_size, rank
            )

    # Step 6: DataLoader
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(is_train and sampler is None),
        sampler=sampler,
        collate_fn=dflash_collate_fn,
        num_workers=0,  # Safe default; >0 can hang with HFDataset
        pin_memory=True,
        drop_last=is_train,
    )

    logger.info(
        "DataLoader built: batch_size=%d, num_batches=%d",
        config.training.batch_size,
        len(dataloader),
    )
    return dataloader


def build_eval_dataloader(
    config: DFlashConfig,
    tokenizer: TokenizerType,
    dataset: Optional[HFDataset] = None,
) -> DataLoader:
    """Build a non-shuffling DataLoader for evaluation.

    Args:
        config: Full DFlash configuration.
        tokenizer: Tokenizer for the target model.
        dataset: Pre-loaded evaluation dataset.

    Returns:
        Evaluation :class:`DataLoader`.
    """
    return build_dataloader(config, tokenizer, dataset=dataset, is_train=False)

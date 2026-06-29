#!/usr/bin/env python3
"""
DFlash Mac Training Script

Train a DFlash draft model on MacBook (48GB) using PyTorch MPS backend.
This script is optimized for limited memory and uses offline hidden state
caching to avoid keeping the target model in memory during training.

Workflow:
    1. Load target model (Qwen3-4B) -> extract hidden states -> cache to disk
    2. Unload target model to free memory
    3. Load cached hidden states + tokenized data
    4. Train draft model on MPS
    5. Save trained draft model

Usage:
    # Full pipeline
    python dflash_reproduce/mac_train.py --config configs/mac_qwen3_4b.yaml

    # Step 1: Extract hidden states only
    python dflash_reproduce/mac_train.py --config configs/mac_qwen3_4b.yaml --extract-only

    # Step 2: Train only (hidden states already cached)
    python dflash_reproduce/mac_train.py --config configs/mac_qwen3_4b.yaml --train-only

    # Dry run (validate config)
    python dflash_reproduce/mac_train.py --config configs/mac_qwen3_4b.yaml --dry-run

Hardware Requirements:
    - MacBook with Apple Silicon (M1/M2/M3/M4)
    - 48GB unified memory (36GB+ may work with smaller settings)
    - macOS 14+ recommended
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Setup paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dflash_reproduce.config import DFlashConfig, load_config
from dflash_reproduce.utils import setup_logging, set_seed

logger = logging.getLogger("dflash.mac_train")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SPECIAL_MASK_TOKEN = "<|diffusion_mask|>"


# --------------------------------------------------------------------------- #
# Step 1: Hidden State Extraction (PyTorch MPS)
# --------------------------------------------------------------------------- #

def extract_hidden_states_mac(
    target_model_name: str,
    tokenized_data: List[Dict[str, Any]],
    layer_ids: List[int],
    cache_dir: str,
    batch_size: int = 2,
    max_samples: Optional[int] = None,
) -> str:
    """Extract hidden states from target model and cache to disk.

    This function:
        1. Loads the target model on MPS (or CPU if MPS unavailable)
        2. Runs forward pass on each sample
        3. Extracts hidden states at specified layers
        4. Concatenates and saves to .pt files
        5. Unloads the model to free memory

    Args:
        target_model_name: HuggingFace model ID.
        tokenized_data: List of dicts with 'input_ids', 'attention_mask', 'labels'.
        layer_ids: Target layer indices to extract.
        cache_dir: Directory to save cached hidden states.
        batch_size: Batch size for extraction.
        max_samples: Maximum samples to process.

    Returns:
        Path to cache directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    expected_files = min(len(tokenized_data), max_samples or len(tokenized_data))
    existing = list(cache_path.glob("hidden_*.pt"))
    if len(existing) >= expected_files:
        logger.info("Hidden states already cached: %d files in %s", len(existing), cache_dir)
        return str(cache_path)

    # Determine device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info("Loading target model '%s' on %s ...", target_model_name, device)

    # Load model in bfloat16 to save memory
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    model.eval()

    num_layers = len(model.model.layers)
    logger.info("Target model loaded: %d layers, extracting from %s", num_layers, layer_ids)

    # Storage for captured hidden states
    captured_states = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured_states[layer_idx] = h.detach().cpu().to(torch.float32)
        return hook

    # Register hooks
    hooks = []
    for lid in layer_ids:
        if lid < 0:
            lid = num_layers + lid
        h = model.model.layers[lid].register_forward_hook(make_hook(lid))
        hooks.append(h)

    # Process samples
    data_subset = tokenized_data[:max_samples] if max_samples else tokenized_data
    logger.info("Extracting hidden states for %d samples ...", len(data_subset))

    for i in range(0, len(data_subset), batch_size):
        batch_end = min(i + batch_size, len(data_subset))
        batch = data_subset[i:batch_end]

        # Prepare batch
        max_len = max(len(s["input_ids"]) for s in batch)
        input_ids = []
        attention_mask = []
        for s in batch:
            ids = s["input_ids"]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [tokenizer.pad_token_id or 0] * pad_len
            input_ids.append(ids)

        input_tensor = torch.tensor(input_ids, device=device)

        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor, output_hidden_states=False)

        # Save hidden states for each sample
        for j, s in enumerate(batch):
            sample_idx = i + j
            seq_len = len(s["input_ids"])

            # Gather hidden states in order
            h_list = []
            for lid in layer_ids:
                if lid < 0:
                    lid = num_layers + lid
                h = captured_states[lid]
                # h shape: [batch, seq, hidden]
                if j < h.shape[0]:
                    h_list.append(h[j, :seq_len])

            if h_list:
                concatenated = torch.cat(h_list, dim=-1)  # [seq, hidden * num_layers]
                save_file = cache_path / f"hidden_{sample_idx:06d}.pt"
                torch.save({
                    "hidden_states": concatenated,
                    "input_ids": torch.tensor(s["input_ids"]),
                    "labels": torch.tensor(s["labels"]) if "labels" in s else None,
                    "prompt_len": s.get("prompt_len", 0),
                }, save_file)

            captured_states.clear()

        if (sample_idx + 1) % 50 == 0:
            logger.info("  Processed %d/%d samples", sample_idx + 1, len(data_subset))
            # Clear memory
            if device.type == "mps":
                torch.mps.empty_cache()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Unload model
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    logger.info("Hidden states cached to: %s", cache_dir)
    return str(cache_path)


# --------------------------------------------------------------------------- #
# Step 2: Training Dataset with Cached Hidden States
# --------------------------------------------------------------------------- #

class CachedHiddenStateDataset(Dataset):
    """Dataset that loads pre-cached hidden states from disk.

    This avoids keeping the target model in memory during training.
    """

    def __init__(
        self,
        cache_dir: str,
        block_size: int = 16,
        num_anchors: int = 128,
        loss_gamma: float = 7.0,
        mask_token_id: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_gamma = loss_gamma
        self.mask_token_id = mask_token_id

        # List all cached files
        self.files = sorted(self.cache_dir.glob("hidden_*.pt"))
        if not self.files:
            raise ValueError(f"No cached hidden states found in {cache_dir}")
        logger.info("Loaded %d cached samples from %s", len(self.files), cache_dir)

        # Pre-compute loss weights
        self.loss_weights = torch.exp(
            -loss_gamma * torch.arange(block_size, dtype=torch.float32) / block_size
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)

        input_ids = data["input_ids"]  # [seq_len]
        hidden_states = data["hidden_states"]  # [seq_len, hidden_dim]
        labels = data["labels"] if data["labels"] is not None else input_ids.clone()
        prompt_len = data.get("prompt_len", 0)

        seq_len = len(input_ids)
        response_start = prompt_len
        response_len = seq_len - response_start

        if response_len <= 0:
            # Fallback: treat entire sequence as response
            response_start = 0
            response_len = seq_len

        # Sample anchor positions within response
        max_anchors = min(self.num_anchors, max(1, response_len // self.block_size))
        if max_anchors > 0:
            anchor_positions = torch.randperm(response_len)[:max_anchors] + response_start
        else:
            anchor_positions = torch.tensor([response_start])

        # Create mask: anchor positions are kept, rest of block is masked
        mask_positions = torch.zeros(seq_len, dtype=torch.bool)
        for anchor in anchor_positions:
            anchor = int(anchor)
            end = min(anchor + self.block_size, seq_len)
            # Mask positions after anchor in the block
            if end > anchor + 1:
                mask_positions[anchor + 1:end] = True

        # Create labels: only compute loss on masked positions
        loss_labels = labels.clone()
        loss_labels[~mask_positions] = -100  # Ignore non-mask positions

        return {
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "labels": loss_labels,
            "mask_positions": mask_positions,
            "loss_weights": self.loss_weights,
        }


def collate_cached_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for cached hidden state dataset."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    hidden_states = []
    labels = []
    mask_positions = []

    for b in batch:
        seq_len = len(b["input_ids"])
        pad_len = max_len - seq_len

        if pad_len > 0:
            ids = torch.cat([b["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)])
            hid = torch.cat([b["hidden_states"], torch.zeros(pad_len, b["hidden_states"].shape[-1])])
            lbl = torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            msk = torch.cat([b["mask_positions"], torch.zeros(pad_len, dtype=torch.bool)])
        else:
            ids = b["input_ids"]
            hid = b["hidden_states"]
            lbl = b["labels"]
            msk = b["mask_positions"]

        input_ids.append(ids)
        hidden_states.append(hid)
        labels.append(lbl)
        mask_positions.append(msk)

    return {
        "input_ids": torch.stack(input_ids),
        "hidden_states": torch.stack(hidden_states),
        "labels": torch.stack(labels),
        "mask_positions": torch.stack(mask_positions),
        "loss_weights": batch[0]["loss_weights"],  # Same for all
    }


# --------------------------------------------------------------------------- #
# Step 3: Draft Model (PyTorch MPS)
# --------------------------------------------------------------------------- #

class TargetFeatureProjection(nn.Module):
    """Project concatenated target hidden states to draft hidden dimension."""

    def __init__(self, input_dim: int, output_dim: int, eps: float = 1e-6):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = nn.RMSNorm(output_dim, eps=eps) if hasattr(nn, "RMSNorm") else nn.LayerNorm(output_dim, eps=eps)

    def forward(self, x):
        return self.norm(self.fc(x))


class KVInjectAttention(nn.Module):
    """Attention with KV injection from target model."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, ctx, attention_mask=None):
        B, S, _ = x.shape
        _, C, _ = ctx.shape

        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K_ctx = self.k_proj(ctx).view(B, C, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_ctx = self.v_proj(ctx).view(B, C, self.num_kv_heads, self.head_dim).transpose(1, 2)
        K_x = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V_x = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        K = torch.cat([K_ctx, K_x], dim=2)
        V = torch.cat([V_ctx, V_x], dim=2)

        scores = (Q * self.scale) @ K.transpose(-2, -1)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class DraftTransformerLayer(nn.Module):
    """Single draft transformer layer."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 intermediate_size: int, eps: float = 1e-6):
        super().__init__()
        self.attn = KVInjectAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        self.input_norm = nn.RMSNorm(hidden_size, eps=eps) if hasattr(nn, "RMSNorm") else nn.LayerNorm(hidden_size, eps=eps)
        self.post_norm = nn.RMSNorm(hidden_size, eps=eps) if hasattr(nn, "RMSNorm") else nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x, ctx, attention_mask=None):
        h = x + self.attn(self.input_norm(x), ctx, attention_mask)
        h = h + self.mlp(self.post_norm(h))
        return h


class DFlashDraftModelTorch(nn.Module):
    """DFlash draft model (PyTorch for MPS training)."""

    def __init__(
        self,
        hidden_size: int = 2048,
        vocab_size: int = 151936,
        draft_vocab_size: int = 8192,
        num_layers: int = 5,
        num_heads: int = 16,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        intermediate_size: int = 8192,
        target_feature_dim: int = 10240,  # 5 layers * 2048 hidden
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size

        # Embeddings: map from full vocab to draft vocab
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.feature_proj = TargetFeatureProjection(target_feature_dim, hidden_size, eps)

        self.layers = nn.ModuleList([
            DraftTransformerLayer(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, eps)
            for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(hidden_size, eps=eps) if hasattr(nn, "RMSNorm") else nn.LayerNorm(hidden_size, eps=eps)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)

    def forward(self, input_ids, hidden_states, labels=None, loss_weights=None):
        x = self.embed_tokens(input_ids)
        ctx = self.feature_proj(hidden_states)

        for layer in self.layers:
            x = layer(x, ctx)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten
            flat_logits = shift_logits.view(-1, self.draft_vocab_size)
            flat_labels = shift_labels.view(-1)

            # Compute weighted cross-entropy
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")

            # Apply position weights
            if loss_weights is not None:
                # Reshape loss to [batch, seq_len]
                batch_size = shift_logits.shape[0]
                seq_len = shift_logits.shape[1]
                loss = loss.view(batch_size, seq_len)

                # Apply weights per position
                weights_expanded = loss_weights[:seq_len].to(loss.device).view(1, -1)
                loss = loss * weights_expanded

            loss = loss.mean()

        return logits, loss


# --------------------------------------------------------------------------- #
# Step 4: Training Loop
# --------------------------------------------------------------------------- #

def train_draft_model_mac(
    config: DFlashConfig,
    cache_dir: str,
    output_dir: str,
) -> str:
    """Train DFlash draft model on Mac using cached hidden states.

    Args:
        config: Training configuration.
        cache_dir: Directory with cached hidden states.
        output_dir: Directory to save checkpoints.

    Returns:
        Path to saved model directory.
    """
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info("Training device: %s", device)

    # Create dataset
    dataset = CachedHiddenStateDataset(
        cache_dir=cache_dir,
        block_size=config.model.block_size,
        num_anchors=config.training.num_anchors,
        loss_gamma=config.training.loss_decay_gamma,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_cached_batch,
        num_workers=0,  # Mac works better with 0 workers
    )

    # Create model
    # Infer dimensions from cached data
    sample = dataset[0]
    hidden_dim = sample["hidden_states"].shape[-1]
    target_feature_dim = hidden_dim  # Already concatenated

    # Get vocab size from tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Build model
    model = DFlashDraftModelTorch(
        hidden_size=2048,  # Qwen3-4B hidden size
        vocab_size=vocab_size,
        draft_vocab_size=config.model.draft_vocab_size,
        num_layers=config.model.draft_num_layers,
        target_feature_dim=target_feature_dim,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Draft model: %s total params, %s trainable",
                f"{total_params:,}", f"{trainable:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.eps,
        weight_decay=config.training.weight_decay,
    )

    # Training loop
    grad_accum = getattr(config.training, "gradient_accumulation", 8)
    num_epochs = config.training.epochs
    global_step = 0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting training: %d epochs, batch_size=%d, grad_accum=%d",
                num_epochs, config.training.batch_size, grad_accum)
    logger.info("=" * 60)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            hidden_states = batch["hidden_states"].to(device)
            labels = batch["labels"].to(device)
            loss_weights = batch["loss_weights"].to(device)

            _, loss = model(input_ids, hidden_states, labels=labels, loss_weights=loss_weights)
            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clipping)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * grad_accum
            num_batches += 1

            if global_step % 10 == 0 and (batch_idx + 1) % grad_accum == 0:
                logger.info("Epoch %d | Step %d | Loss: %.4f",
                           epoch + 1, global_step, epoch_loss / num_batches)

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info("Epoch %d complete | Avg Loss: %.4f", epoch + 1, avg_loss)

        # Save checkpoint
        ckpt_dir = output_path / f"epoch_{epoch + 1}"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        logger.info("Checkpoint saved: %s", ckpt_dir)

    # Save final model
    final_dir = output_path / "final"
    final_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.pt")

    # Save config
    config_dict = {
        "hidden_size": 2048,
        "vocab_size": vocab_size,
        "draft_vocab_size": config.model.draft_vocab_size,
        "num_layers": config.model.draft_num_layers,
        "block_size": config.model.block_size,
        "target_layer_ids": config.model.target_layer_ids,
    }
    with open(final_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Training complete! Final model saved to: %s", final_dir)
    return str(final_dir)


# --------------------------------------------------------------------------- #
# Step 0: Data Preparation
# --------------------------------------------------------------------------- #

def prepare_data_mac(config: DFlashConfig) -> List[Dict[str, Any]]:
    """Prepare training data for Mac.

    Loads dataset, applies chat template, tokenizes.
    Returns list of dicts with input_ids, labels, prompt_len.
    """
    from datasets import load_dataset as hf_load_dataset
    from transformers import AutoTokenizer

    logger.info("Loading dataset: %s", config.data.dataset_name)

    # Load dataset
    if config.data.dataset_type == "huggingface":
        ds = hf_load_dataset(config.data.dataset_name, split=config.data.train_split)
    else:
        raise NotImplementedError(f"Dataset type: {config.data.dataset_type}")

    # Limit samples
    max_samples = config.data.max_samples or config.hidden_states.max_samples
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    logger.info("Dataset loaded: %d samples", len(ds))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model, trust_remote_code=True)

    # Add special mask token
    if SPECIAL_MASK_TOKEN not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_MASK_TOKEN]})

    mask_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_MASK_TOKEN)

    # Process samples
    processed = []
    for item in ds:
        conversations = item.get(config.data.text_field, item.get("conversations", []))

        # Extract prompt and response
        if isinstance(conversations, list) and len(conversations) >= 2:
            prompt = conversations[0].get("value", conversations[0].get("content", ""))
            response = conversations[1].get("value", conversations[1].get("content", ""))
        elif "prompt" in item and "response" in item:
            prompt = item["prompt"]
            response = item["response"]
        elif "instruction" in item and "output" in item:
            prompt = item["instruction"]
            response = item["output"]
        else:
            continue

        # Apply chat template
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        prompt_tokens = tokenizer.encode(
            tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                          tokenize=False, add_generation_prompt=True),
            add_special_tokens=False,
        )

        # Truncate
        max_len = config.model.max_seq_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        # Create labels: -100 for prompt, token_id for response
        labels = [-100] * len(tokens)
        response_start = len(prompt_tokens)
        for i in range(response_start, len(tokens)):
            labels[i] = tokens[i]

        processed.append({
            "input_ids": tokens,
            "labels": labels,
            "prompt_len": response_start,
        })

    logger.info("Processed %d samples", len(processed))
    return processed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash on Mac")
    parser.add_argument("--config", "-c", required=True, help="Config YAML file")
    parser.add_argument("--extract-only", action="store_true", help="Only extract hidden states")
    parser.add_argument("--train-only", action="store_true", help="Only train (HS cached)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    # Load config
    config = load_config(args.config)
    if args.output_dir:
        config.output.checkpoint_dir = args.output_dir

    if args.dry_run:
        logger.info("Config validated successfully")
        return

    set_seed(config.training.seed)

    cache_dir = config.hidden_states.cache_dir
    output_dir = config.output.checkpoint_dir

    # ------------------------------------------------------------------ #
    # Step 0: Prepare data
    # ------------------------------------------------------------------ #
    tokenized_data = None
    if not args.train_only:
        tokenized_data = prepare_data_mac(config)

    # ------------------------------------------------------------------ #
    # Step 1: Extract hidden states
    # ------------------------------------------------------------------ #
    if not args.train_only:
        layer_ids = config.model.target_layer_ids
        if layer_ids is None:
            from dflash_reproduce.config import build_target_layer_ids
            # Qwen3-4B has 36 layers
            layer_ids = build_target_layer_ids(36, config.model.draft_num_layers)
            config.model.target_layer_ids = layer_ids

        extract_hidden_states_mac(
            target_model_name=config.model.target_model,
            tokenized_data=tokenized_data,
            layer_ids=layer_ids,
            cache_dir=cache_dir,
            batch_size=config.hidden_states.batch_size,
            max_samples=config.hidden_states.max_samples,
        )

        if args.extract_only:
            logger.info("Hidden state extraction complete. Exiting.")
            return

    # ------------------------------------------------------------------ #
    # Step 2: Train draft model
    # ------------------------------------------------------------------ #
    if not args.extract_only:
        train_draft_model_mac(config, cache_dir, output_dir)


if __name__ == "__main__":
    main()

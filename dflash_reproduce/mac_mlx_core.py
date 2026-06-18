"""
DFlash MLX Core Inference Module for MacBook

Simplified MLX-based inference for Apple Silicon, adapted from the official
DFlash implementation. Loads a 4-bit quantized target model via mlx-lm and
a DFlash draft model for speculative decoding.

Requirements:
    pip install mlx mlx-lm huggingface_hub

Usage:
    from dflash_reproduce.mac_mlx_core import load_models, stream_generate_dflash
    
    model, tokenizer, draft = load_models(
        target_id="Qwen/Qwen3-4B",
        draft_id="z-lab/Qwen3-4B-DFlash-b16",
    )
    for resp in stream_generate_dflash(model, draft, tokenizer, "What is 2+2?"):
        print(resp.text, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Conditional imports - MLX may not be available in all environments
# --------------------------------------------------------------------------- #
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    logger.warning("mlx not installed. Mac inference will not be available.")

try:
    from huggingface_hub import snapshot_download
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.generate import generation_stream
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from mlx_lm.models.rope_utils import initialize_rope
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper, load_tokenizer
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False
    logger.warning("mlx-lm not installed. Mac inference will not be available.")


# --------------------------------------------------------------------------- #
# DFlash Config (MLX version)
# --------------------------------------------------------------------------- #

@dataclass
class DFlashMLXConfig:
    """Configuration for DFlash draft model (MLX version)."""
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: Tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    layer_types: Tuple[str, ...] = field(default_factory=tuple)
    sliding_window: Optional[int] = None
    final_logit_softcapping: Optional[float] = None

    def _build_rope(self):
        return initialize_rope(
            dims=self.head_dim,
            base=self.rope_theta,
            traditional=False,
            scaling_config=self.rope_scaling,
            max_position_embeddings=self.max_position_embeddings,
        )


# --------------------------------------------------------------------------- #
# DFlash Attention Layer (MLX)
# --------------------------------------------------------------------------- #

class DFlashAttentionMLX(nn.Module):
    """Single DFlash attention layer with KV injection (MLX implementation)."""

    def __init__(self, config: DFlashMLXConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def __call__(
        self,
        x: "mx.array",
        ctx: "mx.array",
        rope,
        cache,
    ) -> "mx.array":
        B, S, _ = x.shape
        _, C, _ = ctx.shape

        queries = self.q_proj(x)
        keys_x = self.k_proj(x)
        values_x = self.v_proj(x)
        keys_ctx = self.k_proj(ctx)
        values_ctx = self.v_proj(ctx)

        # Reshape for GQA
        queries = queries.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys_x = keys_x.reshape(B, S, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        values_x = values_x.reshape(B, S, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        keys_ctx = keys_ctx.reshape(B, C, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        values_ctx = values_ctx.reshape(B, C, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE to draft queries and keys
        queries = rope(queries)
        keys_x = rope(keys_x)

        # Concatenate context KV with draft KV
        keys = mx.concatenate([keys_ctx, keys_x], axis=2)
        values = mx.concatenate([values_ctx, values_x], axis=2)

        # Update cache
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Repeat KV for GQA
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)

        # Attention
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = scores @ values
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)


class DFlashTransformerLayerMLX(nn.Module):
    """Single DFlash transformer layer (MLX)."""

    def __init__(self, config: DFlashMLXConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn = DFlashAttentionMLX(config)
        self.mlp = self._build_mlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _build_mlp(self, config: DFlashMLXConfig):
        # Simple MLP matching Qwen3 architecture
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.silu,
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

    def __call__(self, x, ctx, rope, cache):
        h = x + self.attn(self.input_layernorm(x), ctx, rope, cache)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


# --------------------------------------------------------------------------- #
# DFlash Draft Model (MLX)
# --------------------------------------------------------------------------- #

class DFlashDraftModelMLX(nn.Module):
    """DFlash draft model for speculative decoding (MLX implementation)."""

    def __init__(self, config: DFlashMLXConfig):
        super().__init__()
        self.config = config
        self.layers = [DFlashTransformerLayerMLX(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(
            config.num_target_layers * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = config._build_rope()

        # These will be set from target model
        self.embed_tokens = None
        self.embed_scale = 1.0
        self.lm_head = None

    def steal_weights(self, target_model):
        """Steal embedding and lm_head weights from target model."""
        inner = target_model
        if hasattr(target_model, "model"):
            inner = target_model.model
        elif hasattr(target_model, "language_model"):
            lm = target_model.language_model
            if hasattr(lm, "model"):
                inner = lm.model

        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(self.embed_tokens, "embed_scale",
                                   getattr(inner, "embed_scale", 1.0))

        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (getattr(target_model, "lm_head", None)
                        or getattr(lm, "lm_head", None))
        if self.lm_head is None:
            self.lm_head = self.embed_tokens.as_linear
        return self

    def make_cache(self):
        """Create KV caches for all layers."""
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention" and self.config.sliding_window:
                caches.append(RotatingKVCache(
                    max_size=self.config.sliding_window - 1,
                    keep=0,
                ))
            else:
                caches.append(KVCache())
        return caches

    def __call__(
        self,
        inputs: "mx.array",
        target_hidden: "mx.array",
        cache: list,
        logits_start: int = 0,
    ) -> "mx.array":
        h = self.embed_tokens(inputs) * self.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        if logits_start:
            h = h[:, logits_start:]
        logits = self.lm_head(self.norm(h))
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits


# --------------------------------------------------------------------------- #
# Model Loading
# --------------------------------------------------------------------------- #

def load_models(
    target_id: str = "Qwen/Qwen3-4B",
    draft_id: str = "z-lab/Qwen3-4B-DFlash-b16",
) -> Tuple[Any, Any, DFlashDraftModelMLX]:
    """Load target model (4-bit) and DFlash draft model via MLX.

    Args:
        target_id: HuggingFace model ID for the target model.
        draft_id: HuggingFace model ID for the DFlash draft model.

    Returns:
        Tuple of (target_model, tokenizer, draft_model).
    """
    if not HAS_MLX or not HAS_MLX_LM:
        raise ImportError("mlx and mlx-lm are required. Install: pip install mlx mlx-lm")

    logger.info("Loading target model: %s (4-bit quantized)", target_id)
    model, tokenizer = mlx_lm_load(target_id)

    logger.info("Loading DFlash draft model: %s", draft_id)
    draft = load_draft_mlx(draft_id)
    draft.steal_weights(model)

    # Wrap tokenizer if needed
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    logger.info("Models loaded successfully")
    return model, tokenizer, draft


def load_draft_mlx(draft_id: str) -> DFlashDraftModelMLX:
    """Load a DFlash draft model from HuggingFace (MLX format).

    Args:
        draft_id: HuggingFace model ID (e.g., "z-lab/Qwen3-4B-DFlash-b16").

    Returns:
        Loaded DFlashDraftModelMLX instance.
    """
    if not HAS_MLX_LM:
        raise ImportError("mlx-lm is required")

    path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    cfg = json.loads((path / "config.json").read_text())

    layer_types = tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])

    config = DFlashMLXConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=cfg["block_size"],
        target_layer_ids=tuple(cfg["dflash_config"]["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=cfg["dflash_config"]["mask_token_id"],
        rope_scaling=cfg.get("rope_scaling"),
        layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=cfg.get("final_logit_softcapping"),
    )

    weights = {
        k: v for f in path.glob("*.safetensors")
        for k, v in mx.load(str(f)).items()
    }
    model = DFlashDraftModelMLX(config)
    model.load_weights(list(weights.items()))
    return model


# --------------------------------------------------------------------------- #
# Hidden State Extraction (MLX)
# --------------------------------------------------------------------------- #

class HiddenStateExtractorMLX:
    """Extract hidden states from target model using MLX."""

    def __init__(self, model, layer_ids: List[int]):
        self.model = model
        self.layer_ids = layer_ids
        self._hidden_states = []
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        def hook(_, _inputs, output):
            # output may be a tuple; grab the hidden states tensor
            hidden = output[0] if isinstance(output, tuple) else output
            # Ensure we have the right shape: [batch, seq, hidden]
            if hidden.ndim == 3:
                self._hidden_states[layer_idx] = hidden
            return output
        return hook

    def register_hooks(self):
        """Register forward hooks on target model layers."""
        self._hidden_states = [None] * (max(self.layer_ids) + 2)
        inner = self.model
        if hasattr(self.model, "model"):
            inner = self.model.model
        elif hasattr(self.model, "language_model"):
            lm = self.model.language_model
            if hasattr(lm, "model"):
                inner = lm.model

        layers = inner.layers
        self._hooks = []
        for lid in self.layer_ids:
            # Use a closure to capture the layer index
            hook_fn = self._create_hook(lid)
            layers[lid].register_forward_hook(hook_fn)
            self._hooks.append((lid, hook_fn))

    def _create_hook(self, layer_idx: int):
        import functools
        @functools.wraps(self._hook_fn)
        def wrapper(*args, **kwargs):
            return self._hook_fn(layer_idx, *args, **kwargs)
        return wrapper

    def _hook_fn(self, layer_idx: int, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim == 3:
            self._hidden_states[layer_idx] = hidden

    def remove_hooks(self):
        """Remove all registered hooks."""
        self._hooks.clear()

    @property
    def hidden_states(self) -> list:
        return self._hidden_states

    def extract(self, input_ids: "mx.array") -> "mx.array":
        """Extract and concatenate hidden states from specified layers."""
        self._hidden_states = [None] * (max(self.layer_ids) + 2)
        # Forward pass through target model
        _ = self.model(input_ids[None, :])
        # Gather hidden states
        selected = []
        for lid in self.layer_ids:
            h = self._hidden_states[lid]
            if h is None:
                # Fallback: re-run with hooks
                raise RuntimeError(f"Hidden state for layer {lid} not captured")
            selected.append(h)
        return mx.concatenate(selected, axis=-1)


# --------------------------------------------------------------------------- #
# Speculative Decoding Generation (MLX)
# --------------------------------------------------------------------------- #

@dataclass
class MLXGenerationResponse:
    """Response from MLX speculative decoding generation."""
    text: str
    tokens: List[int]
    accepted: int
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None


def _get_memory_mb() -> float:
    """Get current memory usage in MB."""
    if not HAS_MLX:
        return 0.0
    return mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx.metal, "get_peak_memory") else 0.0


def stream_generate_dflash(
    model,
    draft: DFlashDraftModelMLX,
    tokenizer,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    block_size: Optional[int] = None,
    verbose: bool = False,
) -> Generator[MLXGenerationResponse, None, None]:
    """Generate text using DFlash speculative decoding (MLX version).

    Args:
        model: Target model (loaded via mlx-lm).
        draft: DFlash draft model.
        tokenizer: Tokenizer.
        prompt: Input prompt string.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0=greedy).
        block_size: Diffusion block size (default from draft config).
        verbose: Print debug info.

    Yields:
        MLXGenerationResponse for each generation step.
    """
    if not HAS_MLX or not HAS_MLX_LM:
        raise ImportError("mlx and mlx-lm required")

    if block_size is None:
        block_size = draft.config.block_size
    mask_id = draft.config.mask_token_id
    sampler = make_sampler(temp=temperature if temperature > 0 else None)

    # Tokenize prompt
    if hasattr(tokenizer, "encode"):
        prompt_tokens = tokenizer.encode(prompt)
    else:
        prompt_tokens = tokenizer.tokenizer.encode(prompt, add_special_tokens=False)
    prompt_arr = mx.array(prompt_tokens)

    # Setup caches
    target_cache = None  # Target model manages its own cache
    draft_cache = draft.make_cache()

    # Track hidden states via forward hook
    hidden_states_list = []

    def capture_hidden_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_states_list.append(hidden)
        return output

    # Register hook on target model
    inner_model = model.model if hasattr(model, "model") else model
    target_layers = inner_model.layers
    hooks = []

    # Simple approach: forward pass and capture all hidden states
    # Then select the ones we need
    _hidden_storage = {}

    def make_capture(lid):
        def capture(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            _hidden_storage[lid] = h
            return out
        return capture

    # Clear peak memory
    if hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()

    # Prefill
    tic = time.perf_counter()

    # Forward pass to get first token and capture hidden states
    _hidden_storage.clear()
    handles = []
    for lid in draft.config.target_layer_ids:
        h = target_layers[lid].register_forward_hook(make_capture(lid))
        handles.append(h)

    logits = model(prompt_arr[None, :])
    # Gather hidden states in order
    h_list = [_hidden_storage[lid] for lid in draft.config.target_layer_ids]
    hidden = mx.concatenate(h_list, axis=-1)

    mx.eval(logits, hidden)
    prompt_tps = prompt_arr.size / (time.perf_counter() - tic)

    # Remove hooks
    for h in handles:
        h.remove()

    # Sample first token
    token = sampler(logits[:, -1:])[0, 0].item()
    tokens = [token]
    n = 1

    # Setup detokenizer
    detokenizer = tokenizer.detokenizer if hasattr(tokenizer, "detokenizer") else None

    def _make_response(text_seg, new_tokens, accepted, prompt_sz, ptps, gen_n, start_time, reason=None):
        gen_tps = gen_n / (time.perf_counter() - start_time) if gen_n > 0 else 0
        return MLXGenerationResponse(
            text=text_seg,
            tokens=new_tokens,
            accepted=accepted,
            prompt_tokens=prompt_sz,
            prompt_tps=ptps,
            generation_tokens=gen_n,
            generation_tps=gen_tps,
            peak_memory=_get_memory_mb(),
            finish_reason=reason,
        )

    # Check EOS
    eos_ids = getattr(tokenizer, "eos_token_ids", set())
    if not eos_ids and hasattr(tokenizer, "eos_token_id"):
        eos_ids = {tokenizer.eos_token_id}

    if token in eos_ids:
        text_out = tokenizer.decode([token]) if hasattr(tokenizer, "decode") else ""
        yield _make_response(text_out, [token], 1, prompt_arr.size, prompt_tps, n, tic, "stop")
        return

    text_out = tokenizer.decode([token]) if hasattr(tokenizer, "decode") else ""
    yield _make_response(text_out, [token], 1, prompt_arr.size, prompt_tps, n, tic)

    # Speculative decoding loop
    total_accepted = 0
    total_draft_calls = 0

    while n < max_tokens:
        bs = min(block_size, max_tokens - n + 1)
        if bs <= 1:
            break

        # Draft: generate block
        block_input = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])

        # We need hidden states for the current position
        # Use the last hidden state from prefill/verification
        # For simplicity, re-run target model to get fresh hidden states
        _hidden_storage.clear()
        handles = []
        for lid in draft.config.target_layer_ids:
            h = target_layers[lid].register_forward_hook(make_capture(lid))
            handles.append(h)

        # Forward last token through target to get hidden states
        _ = model(mx.array([[tokens[-1]]]))
        h_list = [_hidden_storage[lid] for lid in draft.config.target_layer_ids]
        current_hidden = mx.concatenate(h_list, axis=-1)
        for h in handles:
            h.remove()

        draft_logits = draft(block_input, current_hidden, draft_cache, logits_start=1)
        draft_tokens = sampler(draft_logits)
        mx.eval(draft_tokens)

        # Verify with target model
        verify_input = mx.array([[tokens[-1]]] + [[t] for t in draft_tokens[0].tolist()])
        verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens], axis=1)

        _hidden_storage.clear()
        handles = []
        for lid in draft.config.target_layer_ids:
            h = target_layers[lid].register_forward_hook(make_capture(lid))
            handles.append(h)

        verify_logits = model(verify_input)
        h_list = [_hidden_storage[lid] for lid in draft.config.target_layer_ids]
        # Update hidden for next iteration
        current_hidden = mx.concatenate(h_list, axis=-1)

        for h in handles:
            h.remove()

        target_tokens = sampler(verify_logits)
        mx.eval(target_tokens, current_hidden)

        d_list = draft_tokens[0].tolist()
        t_list = target_tokens[0].tolist()

        # Find acceptance point
        accepted = next((i for i in range(len(d_list)) if d_list[i] != t_list[i]), len(d_list))
        total_accepted += accepted + 1
        total_draft_calls += 1

        new_tokens = d_list[:accepted] + [t_list[accepted]]
        new_tokens = new_tokens[:max_tokens - n]

        # Check EOS
        eos_idx = next((i for i, t in enumerate(new_tokens) if t in eos_ids), None)
        if eos_idx is not None:
            new_tokens = new_tokens[:eos_idx + 1]

        for t in new_tokens:
            tokens.append(t)
        n += len(new_tokens)

        # Decode
        try:
            new_text = tokenizer.decode(new_tokens)
        except Exception:
            new_text = ""

        is_stop = eos_idx is not None or (n >= max_tokens)
        reason = "stop" if is_stop else None
        yield _make_response(new_text, new_tokens, accepted + 1, prompt_arr.size, prompt_tps, n, tic, reason)

        if is_stop:
            break

        # Clear cache periodically
        if n % 256 == 0:
            mx.clear_cache()

    if verbose:
        avg_acceptance = total_accepted / total_draft_calls if total_draft_calls > 0 else 0
        print(f"\n[Stats] Draft calls: {total_draft_calls}, Avg acceptance: {avg_acceptance:.2f}")


# --------------------------------------------------------------------------- #
# Simple autoregressive baseline (MLX)
# --------------------------------------------------------------------------- #

def stream_generate_ar(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Generator[MLXGenerationResponse, None, None]:
    """Autoregressive generation baseline (MLX)."""
    if not HAS_MLX:
        raise ImportError("mlx required")

    sampler = make_sampler(temp=temperature if temperature > 0 else None)

    if hasattr(tokenizer, "encode"):
        prompt_tokens = tokenizer.encode(prompt)
    else:
        prompt_tokens = tokenizer.tokenizer.encode(prompt, add_special_tokens=False)
    prompt_arr = mx.array(prompt_tokens)

    tic = time.perf_counter()
    logits = model(prompt_arr[None, :])
    mx.eval(logits)
    prompt_tps = prompt_arr.size / (time.perf_counter() - tic)

    token = sampler(logits[:, -1:])[0, 0].item()
    tokens = [token]
    n = 1

    eos_ids = getattr(tokenizer, "eos_token_ids", set())
    if not eos_ids and hasattr(tokenizer, "eos_token_id"):
        eos_ids = {tokenizer.eos_token_id}

    text_out = tokenizer.decode([token]) if hasattr(tokenizer, "decode") else ""
    yield MLXGenerationResponse(text_out, [token], 1, prompt_arr.size, prompt_tps, n,
                                 prompt_arr.size / (time.perf_counter() - tic), _get_memory_mb())

    if token in eos_ids:
        return

    gen_start = time.perf_counter()
    while n < max_tokens:
        logits = model(mx.array([[tokens[-1]]]))
        token = sampler(logits[:, -1:])[0, 0].item()
        tokens.append(token)
        n += 1

        try:
            text_out = tokenizer.decode([token])
        except Exception:
            text_out = ""

        gen_tps = n / (time.perf_counter() - gen_start)
        is_stop = token in eos_ids or n >= max_tokens
        yield MLXGenerationResponse(text_out, [token], 1, prompt_arr.size, prompt_tps, n,
                                     gen_tps, _get_memory_mb(),
                                     "stop" if is_stop else None)
        if is_stop:
            break

        if n % 256 == 0:
            mx.clear_cache()

"""
DFlash Draft Model Architecture

This module implements the complete PyTorch architecture for the DFlash draft model,
which serves as a block diffusion model for speculative decoding.

Key innovations:
  1. Non-causal bidirectional attention for parallel block generation
  2. KV injection: target model hidden states are projected and injected into the
     draft model's KV cache
  3. Anchor mechanism: random anchor points in the sequence, each generating a block

References:
    DFlash paper (block diffusion for speculative decoding)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, PretrainedConfig

# Use PyTorch native RMSNorm (available since PyTorch 2.4)
try:
    from torch.nn import RMSNorm
except ImportError:
    # Fallback for older PyTorch
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Custom RoPE implementation compatible with all transformers versions."""
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[2]  # x shape: [batch, heads, seq, dim]
        t = torch.arange(seq_len, device=x.device, dtype=torch.int64)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Position Embedding to the query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class DFlashConfig:
    """Configuration for the DFlash draft model.

    Attributes:
        num_layers: Number of transformer layers in the draft model.
        hidden_size: Hidden dimension (from target model config).
        vocab_size: Draft vocabulary size (e.g. 8192).
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key-value heads (GQA).
        intermediate_size: MLP intermediate dimension.
        rms_norm_eps: Epsilon for RMSNorm.
        rope_theta: Base frequency for RoPE.
        max_position_embeddings: Maximum sequence length.
        block_size: Block size for diffusion generation.
        mask_token_id: Special mask token ID.
        hidden_dropout_prob: Dropout probability (default 0.0).
        attention_dropout_prob: Attention dropout probability (default 0.0).
        target_layer_ids: List of target-model layer IDs to extract features from.
            Default is [-5, -4, -3, -2, -1] (last 5 layers).
        use_flash_attention: Whether to use Flash Attention when available.
        initializer_range: Standard deviation for weight initialization.
    """

    num_layers: int = 5
    hidden_size: int = 2048
    vocab_size: int = 8192
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    block_size: int = 16
    mask_token_id: int = 0
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    target_layer_ids: Optional[List[int]] = None
    use_flash_attention: bool = True
    initializer_range: float = 0.02
    head_dim: Optional[int] = None

    def __post_init__(self):
        if self.target_layer_ids is None:
            self.target_layer_ids = list(range(-5, 0))  # last 5 layers
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads


# --------------------------------------------------------------------------- #
# Target Feature Extraction
# --------------------------------------------------------------------------- #


def extract_context_feature(
    hidden_states: List[Tensor],
    layer_ids: List[int],
    offset: int = 1,
) -> Tensor:
    """Extract and concatenate hidden states from specified target-model layers.

    This function selects hidden states from the target model at the given layer
    indices, with an offset to accommodate HuggingFace layer numbering.

    Args:
        hidden_states: List of hidden-state tensors from the target model.
            Each tensor has shape [batch, seq_len, hidden_dim].
        layer_ids: Layer indices to extract. Negative values are supported
            (Python-style indexing).
        offset: Offset applied to layer IDs to adapt to HuggingFace numbering.
            Default is 1.

    Returns:
        Concatenated hidden states of shape
        [batch, seq_len, num_layers * hidden_dim].

    Raises:
        ValueError: If a layer ID is out of range.
    """
    selected: List[Tensor] = []
    for lid in layer_ids:
        idx = lid + offset if lid >= 0 else lid + offset
        if idx >= len(hidden_states) or idx < 0:
            raise ValueError(
                f"Layer ID {lid} (adjusted index {idx}) is out of range "
                f"for hidden_states list of length {len(hidden_states)}. "
                f"Valid range with offset={offset}: "
                f"[{-len(hidden_states) + offset}, {len(hidden_states) - 1 + offset}]"
            )
        selected.append(hidden_states[idx])
    return torch.cat(selected, dim=-1)


# --------------------------------------------------------------------------- #
# Target Feature Projection
# --------------------------------------------------------------------------- #


class TargetFeatureProjection(nn.Module):
    """Project concatenated target-model features into the draft hidden dimension.

    Given target features concatenated from multiple layers:
        H_concat = [H^(l1); H^(l2); ...; H^(lk)]
    this module computes:
        H_t = RMSNorm(W_c * H_concat)

    Input shape:  [batch, seq_len, num_target_layers * target_hidden_dim]
    Output shape: [batch, seq_len, draft_hidden_dim]
    """

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.norm = RMSNorm(out_features, eps=eps)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Forward pass.

        Args:
            hidden_states: Concatenated target features
                [batch, seq_len, num_layers * target_hidden_dim].

        Returns:
            Projected and normalized features [batch, seq_len, draft_hidden_dim].
        """
        projected = self.proj(hidden_states)
        normalized = self.norm(projected)
        return normalized


# --------------------------------------------------------------------------- #
# DFlash Attention (KV Injection)
# --------------------------------------------------------------------------- #


class DFlashAttention(nn.Module):
    """DFlash attention layer with KV injection from the target model.

    This is the core attention mechanism of DFlash. For each layer i:
        Q_i = W_i^Q * H_d                          (draft tokens as queries)
        K_i = [W_i^K * H_t ;  W_i^K * H_d]         (target + draft as keys)
        V_i = [W_i^V * H_t ;  W_i^V * H_d]         (target + draft as values)

    The target features H_t are projected once and serve as a persistent prefix
    in the KV cache, enabling the draft model to attend to target-model context.

    The attention supports:
        - Sparse block attention mask (bidirectional inside block, no cross-block)
        - KV cache for autoregressive-like decoding during inference
        - Flash Attention or SDPA for efficient computation
    """

    def __init__(self, config: DFlashConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout_prob

        # Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _repeat_kv(self, hidden_states: Tensor, n_rep: int) -> Tensor:
        """Repeat key/value heads for GQA (grouped query attention).

        Args:
            hidden_states: [batch, num_kv_heads, seq_len, head_dim]
            n_rep: Number of times to repeat each KV head.

        Returns:
            [batch, num_kv_heads * n_rep, seq_len, head_dim]
        """
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: Tensor,
        target_features: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass of DFlash attention with KV injection.

        Args:
            hidden_states: Draft token hidden states [batch, draft_seq_len, hidden_size].
            target_features: Projected target features [batch, target_seq_len, hidden_size].
                These serve as the injected KV prefix.
            attention_mask: Optional sparse attention mask. Shape should be compatible
                with SDPA/Flash Attention. For block attention, use a 2D mask of shape
                [seq_len, seq_len] where 0 = attend, large negative = mask out.
            position_ids: Optional position IDs for RoPE [batch, draft_seq_len].
            past_key_value: Optional cached KV pairs for incremental decoding.
                Tuple of (key_cache, value_cache) each of shape
                [batch, num_kv_heads, cache_len, head_dim].
            use_cache: Whether to return updated KV cache.

        Returns:
            - attention_output: [batch, draft_seq_len, hidden_size]
            - present_key_value: Updated KV cache if use_cache=True, else None.
        """
        bsz, q_len, _ = hidden_states.shape
        target_len = target_features.shape[1]

        # ------------------------------------------------------------------ #
        # 1. Compute Q from draft tokens; K, V from both target + draft
        # ------------------------------------------------------------------ #
        query_states = self.q_proj(hidden_states)

        # For keys and values, we concatenate target features with draft tokens
        # along the sequence dimension, then project
        kv_input = torch.cat([target_features, hidden_states], dim=1)
        key_states = self.k_proj(kv_input)
        value_states = self.v_proj(kv_input)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, target_len + q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, target_len + q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # ------------------------------------------------------------------ #
        # 2. Apply RoPE to Q and K (only to the draft portion of K)
        # ------------------------------------------------------------------ #
        kv_seq_len = key_states.shape[2]

        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)

        # Get cos/sin for the draft positions
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Apply RoPE to query
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, position_ids)

        # For keys: we need position IDs that cover both target + draft positions.
        # Target positions typically come before draft positions in the sequence.
        kv_position_ids = torch.arange(kv_seq_len, device=hidden_states.device).unsqueeze(0)
        # Split key states into target and draft portions
        target_key_states = key_states[:, :, :target_len, :]
        draft_key_states = key_states[:, :, target_len:, :]
        # Apply RoPE only to the draft portion
        _, draft_key_states_rot = apply_rotary_pos_emb(
            draft_key_states, draft_key_states, cos, sin, kv_position_ids[:, target_len:]
        )
        key_states = torch.cat([target_key_states, draft_key_states_rot], dim=2)

        # ------------------------------------------------------------------ #
        # 3. Handle KV cache for incremental decoding
        # ------------------------------------------------------------------ #
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value: Optional[Tuple[Tensor, Tensor]] = None
        if use_cache:
            present_key_value = (key_states, value_states)

        # ------------------------------------------------------------------ #
        # 4. GQA: repeat KV heads to match Q heads
        # ------------------------------------------------------------------ #
        n_rep = self.num_heads // self.num_key_value_heads
        key_states = self._repeat_kv(key_states, n_rep)
        value_states = self._repeat_kv(value_states, n_rep)

        # ------------------------------------------------------------------ #
        # 5. Compute attention using SDPA (which calls Flash Attention when available)
        # ------------------------------------------------------------------ #
        # SDPA expects attention_mask as a float additive mask or boolean mask
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,  # DFlash uses non-causal (bidirectional) attention
        )

        # ------------------------------------------------------------------ #
        # 6. Reshape and project output
        # ------------------------------------------------------------------ #
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


# --------------------------------------------------------------------------- #
# DFlash Transformer Layer
# --------------------------------------------------------------------------- #


class DFlashTransformerLayer(nn.Module):
    """A single DFlash transformer layer with pre-normalization.

    Architecture:
        H' = H_d + Attention(RMSNorm(H_d), target_features)
        H_out = H' + MLP(RMSNorm(H'))

    Supports gradient checkpointing for memory-efficient training.
    """

    def __init__(self, config: DFlashConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Pre-norm layers
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention
        self.self_attn = DFlashAttention(config, layer_idx=layer_idx)

        # MLP (SwiGLU-style as in Qwen/Llama)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        for module in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _mlp(self, hidden_states: Tensor) -> Tensor:
        """SwiGLU MLP forward.

        gate = SiLU(W_gate * x)
        up = W_up * x
        output = W_down * (gate * up)
        """
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)

    def forward(
        self,
        hidden_states: Tensor,
        target_features: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass of a single transformer layer.

        Args:
            hidden_states: Draft hidden states [batch, seq_len, hidden_size].
            target_features: Projected target features [batch, target_seq_len, hidden_size].
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            past_key_value: Optional KV cache.
            use_cache: Whether to return KV cache.

        Returns:
            - updated hidden states [batch, seq_len, hidden_size]
            - updated KV cache if use_cache=True
        """
        residual = hidden_states

        # Attention sub-layer with pre-norm
        hidden_states = self.input_layernorm(hidden_states)

        if self._gradient_checkpointing and self.training:
            # Use gradient checkpointing to trade compute for memory
            attn_output, present_key_value = torch.utils.checkpoint.checkpoint(
                self._attn_forward,
                hidden_states,
                target_features,
                attention_mask,
                position_ids,
                past_key_value,
                use_cache,
                use_reentrant=False,
            )
        else:
            attn_output, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                target_features=target_features,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

        hidden_states = residual + attn_output

        # MLP sub-layer with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self._gradient_checkpointing and self.training:
            mlp_output = torch.utils.checkpoint.checkpoint(
                self._mlp, hidden_states, use_reentrant=False
            )
        else:
            mlp_output = self._mlp(hidden_states)

        hidden_states = residual + mlp_output

        return hidden_states, present_key_value

    def _attn_forward(
        self,
        hidden_states: Tensor,
        target_features: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
        past_key_value: Optional[Tuple[Tensor, Tensor]],
        use_cache: bool,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Wrapper for attention forward used by gradient checkpointing."""
        return self.self_attn(
            hidden_states=hidden_states,
            target_features=target_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )


# --------------------------------------------------------------------------- #
# DFlash Draft Model
# --------------------------------------------------------------------------- #


class DFlashDraftModel(nn.Module):
    """Complete DFlash draft model for block diffusion speculative decoding.

    The draft model takes token IDs and target-model hidden-state features as inputs,
    and outputs logits over the draft vocabulary for each position.

    Architecture overview:
        1. Embed input tokens
        2. Project target features into draft hidden dimension
        3. Pass through N transformer layers (with KV injection attention)
        4. Final RMSNorm + LM head

    Attributes:
        config: DFlashConfig instance.
        embed_tokens: Token embedding layer.
        target_feature_proj: Projection layer for target hidden states.
        layers: List of DFlashTransformerLayer.
        norm: Final RMSNorm.
        lm_head: Language model head mapping to vocab_size.
    """

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Target feature projection (will be initialized lazily if needed)
        # The actual projection is created outside or must match target feature dim
        self.target_feature_proj: Optional[nn.Module] = None

        # Transformer layers
        self.layers = nn.ModuleList(
            [DFlashTransformerLayer(config, layer_idx=i) for i in range(config.num_layers)]
        )

        # Final norm and LM head
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small standard deviation for training stability."""
        std = self.config.initializer_range

        # Token embeddings
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)

        # LM head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

        # Apply default init to all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad:
                    # Skip if already initialized
                    pass

    def set_target_feature_projection(self, projection_module: nn.Module) -> None:
        """Attach the target feature projection module.

        Args:
            projection_module: Typically a TargetFeatureProjection instance.
        """
        self.target_feature_proj = projection_module

    def forward(
        self,
        input_ids: Tensor,
        target_features: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tensor:
        """Forward pass of the DFlash draft model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            target_features: Target model hidden-state features
                [batch_size, target_seq_len, hidden_dim].
                These will be projected if target_feature_proj is attached.
            attention_mask: Optional sparse attention mask for block attention.
                For training, should be a 2D mask [seq_len, seq_len] or
                broadcastable [batch, 1, seq_len, seq_len].
                Use 0 for positions to attend to, and a large negative value
                (e.g. -1e9) for masked positions.
            position_ids: Optional position IDs [batch_size, seq_len].
            past_key_values: Optional list of cached KV pairs for each layer.
                Used during incremental decoding.
            use_cache: Whether to return updated KV cache.

        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size].

        Raises:
            ValueError: If target_feature_proj is not set and target_features
                need projection.
        """
        batch_size, seq_len = input_ids.shape

        # ------------------------------------------------------------------ #
        # 1. Token embeddings
        # ------------------------------------------------------------------ #
        hidden_states = self.embed_tokens(input_ids)

        # ------------------------------------------------------------------ #
        # 2. Project target features if projection module is attached
        # ------------------------------------------------------------------ #
        if self.target_feature_proj is not None:
            projected_target = self.target_feature_proj(target_features)
        else:
            # Assume target_features are already in the correct dimension
            projected_target = target_features

        # Verify dimensions match
        if projected_target.size(-1) != self.config.hidden_size:
            raise ValueError(
                f"Target feature dimension ({projected_target.size(-1)}) does not "
                f"match model hidden_size ({self.config.hidden_size}). "
                f"Set a target_feature_projection module or provide pre-projected features."
            )

        # ------------------------------------------------------------------ #
        # 3. Position IDs
        # ------------------------------------------------------------------ #
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # ------------------------------------------------------------------ #
        # 4. Pass through transformer layers
        # ------------------------------------------------------------------ #
        next_cache: Optional[List[Tuple[Tensor, Tensor]]] = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None

            hidden_states, present_key_value = layer(
                hidden_states=hidden_states,
                target_features=projected_target,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache and next_cache is not None:
                next_cache.append(present_key_value)

        # ------------------------------------------------------------------ #
        # 5. Final norm and LM head
        # ------------------------------------------------------------------ #
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    @torch.no_grad()
    def generate_block(
        self,
        anchor_tokens: Tensor,
        target_features: Tensor,
        num_steps: int = 16,
        temperature: float = 1.0,
        mask_token_id: Optional[int] = None,
    ) -> Tensor:
        """Generate a single block of tokens using iterative diffusion-style denoising.

        This implements a simplified block generation where we start from all-mask
        tokens and iteratively refine predictions. In the full DFlash, this uses
        a diffusion scheduler.

        Args:
            anchor_tokens: Anchor token IDs [batch_size, 1] or [batch_size, anchor_len].
            target_features: Target features [batch_size, target_seq_len, hidden_dim].
            num_steps: Number of denoising steps.
            temperature: Sampling temperature.
            mask_token_id: Token ID used for masked positions. Defaults to config value.

        Returns:
            Generated block token IDs [batch_size, block_size].
        """
        if mask_token_id is None:
            mask_token_id = self.config.mask_token_id

        batch_size = anchor_tokens.shape[0]
        device = anchor_tokens.device

        # Initialize block with mask tokens
        block = torch.full(
            (batch_size, self.config.block_size),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # Concatenate anchor with block for input
        input_ids = torch.cat([anchor_tokens, block], dim=1)

        for step in range(num_steps):
            # Forward pass
            logits = self.forward(input_ids, target_features)

            # Only consider logits for the block positions (after anchor)
            anchor_len = anchor_tokens.shape[1]
            block_logits = logits[:, anchor_len:, :]

            # Sample tokens
            probs = F.softmax(block_logits / temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.config.vocab_size), num_samples=1
            ).view(batch_size, self.config.block_size)

            # Update block (simple parallel decoding: replace all at once)
            # In full DFlash, this follows a diffusion schedule
            block = sampled
            input_ids = torch.cat([anchor_tokens, block], dim=1)

        return block


# --------------------------------------------------------------------------- #
# Model Builder
# --------------------------------------------------------------------------- #


def _resolve_lm_config(target_config: Optional[PretrainedConfig]) -> Optional[PretrainedConfig]:
    """Return the text/LM sub-config for multimodal target models."""
    if target_config is None:
        return None
    text_config = getattr(target_config, "text_config", None)
    if text_config is not None:
        return text_config
    return target_config


def _build_dflash_config_from_training(model_cfg, target_config: PretrainedConfig) -> DFlashConfig:
    """Build draft-model config from training YAML settings and target HF config."""
    lm_cfg = _resolve_lm_config(target_config)
    if lm_cfg is None:
        raise ValueError("target_config is required to build the draft model")

    num_attention_heads = getattr(lm_cfg, "num_attention_heads", 16)
    num_key_value_heads = getattr(
        lm_cfg, "num_key_value_heads", num_attention_heads
    )
    hidden_size = getattr(lm_cfg, "hidden_size", 2048)
    head_dim = getattr(lm_cfg, "head_dim", None)
    intermediate_size = getattr(lm_cfg, "intermediate_size", hidden_size * 4)
    rms_norm_eps = getattr(lm_cfg, "rms_norm_eps", 1e-6)
    rope_theta = getattr(
        lm_cfg,
        "rope_theta",
        getattr(lm_cfg, "rotary_emb_base", 1_000_000.0),
    )
    max_position_embeddings = getattr(lm_cfg, "max_position_embeddings", 32768)

    return DFlashConfig(
        num_layers=model_cfg.draft_num_layers,
        hidden_size=hidden_size,
        vocab_size=model_cfg.draft_vocab_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        block_size=model_cfg.block_size,
        target_layer_ids=model_cfg.target_layer_ids,
        head_dim=head_dim,
    )


def build_draft_model(
    config: DFlashConfig,
    target_config: Optional[PretrainedConfig] = None,
) -> DFlashDraftModel:
    """Build a DFlashDraftModel instance with target feature projection.

    This factory function creates the draft model and automatically sets up
    the target feature projection based on the target model configuration.

    Args:
        config: DFlashConfig for the draft model.
        target_config: Optional HuggingFace config of the target model.
            If provided, the hidden_size is inferred from it, and the
            target feature projection input dimension is computed.

    Returns:
        A DFlashDraftModel instance with target_feature_proj attached.

    Example:
        >>> from transformers import AutoConfig
        >>> target_cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
        >>> draft_cfg = DFlashConfig(hidden_size=target_cfg.hidden_size)
        >>> model = build_draft_model(draft_cfg, target_cfg)
    """
    if hasattr(config, "draft_num_layers") and not hasattr(config, "num_layers"):
        if target_config is None:
            raise ValueError(
                "target_config is required when building from training ModelConfig"
            )
        draft_config = _build_dflash_config_from_training(config, target_config)
    else:
        draft_config = config
        lm_cfg = _resolve_lm_config(target_config)
        if lm_cfg is not None and hasattr(lm_cfg, "hidden_size"):
            draft_config.hidden_size = lm_cfg.hidden_size
            if hasattr(lm_cfg, "num_attention_heads"):
                draft_config.num_attention_heads = lm_cfg.num_attention_heads
            if hasattr(lm_cfg, "num_key_value_heads"):
                draft_config.num_key_value_heads = lm_cfg.num_key_value_heads
            if getattr(lm_cfg, "head_dim", None) is not None:
                draft_config.head_dim = lm_cfg.head_dim
            elif draft_config.head_dim is None:
                draft_config.head_dim = (
                    draft_config.hidden_size // draft_config.num_attention_heads
                )

    # Create the draft model
    model = DFlashDraftModel(draft_config)

    # Compute target feature projection input dimension
    # The target features are concatenated from multiple layers
    num_target_layers = len(draft_config.target_layer_ids)

    lm_cfg = _resolve_lm_config(target_config)
    if lm_cfg is not None and hasattr(lm_cfg, "hidden_size"):
        target_hidden_size = lm_cfg.hidden_size
    else:
        target_hidden_size = draft_config.hidden_size

    projection_in_features = num_target_layers * target_hidden_size
    projection_out_features = draft_config.hidden_size

    # Create and attach the projection module
    target_proj = TargetFeatureProjection(
        in_features=projection_in_features,
        out_features=projection_out_features,
        eps=draft_config.rms_norm_eps,
    )
    model.set_target_feature_projection(target_proj)

    logger.info(
        "Built DFlashDraftModel: layers=%d, hidden_size=%d, vocab_size=%d, "
        "target_proj: %d -> %d (from %d target layers)",
        draft_config.num_layers,
        draft_config.hidden_size,
        draft_config.vocab_size,
        projection_in_features,
        projection_out_features,
        num_target_layers,
    )

    return model


# --------------------------------------------------------------------------- #
# Utility: Create sparse block attention mask
# --------------------------------------------------------------------------- #


def create_block_attention_mask(
    seq_len: int,
    block_size: int,
    num_anchors: int = 1,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create a sparse block attention mask for DFlash training.

    Within each block, tokens attend to each other (bidirectional).
    Between blocks, no attention is allowed.
    Anchors can attend to all positions in their block.

    Args:
        seq_len: Sequence length.
        block_size: Size of each generation block.
        num_anchors: Number of anchor tokens per block.
        device: Target device for the mask.

    Returns:
        Float attention mask of shape [seq_len, seq_len].
        Values are 0.0 for attended positions and -1e9 for masked positions.
    """
    if device is None:
        device = torch.device("cpu")

    mask = torch.full((seq_len, seq_len), -1e9, device=device)

    # For each block, allow bidirectional attention within the block
    num_blocks = seq_len // block_size
    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, seq_len)
        # Bidirectional attention within block
        mask[start:end, start:end] = 0.0

    # Anchors can also attend to their block (already covered above)
    # and the block can attend to anchors

    return mask


def create_loss_weight_mask(
    seq_len: int,
    block_size: int,
    gamma: float = 7.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create position-dependent loss weights for block diffusion training.

    Earlier positions in the block receive higher weights.

    Args:
        seq_len: Sequence length.
        block_size: Size of each block.
        gamma: Decay factor for position weighting.
        device: Target device.

    Returns:
        Weight tensor of shape [seq_len].
    """
    if device is None:
        device = torch.device("cpu")

    weights = torch.ones(seq_len, device=device)
    num_blocks = seq_len // block_size

    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, seq_len)
        for i in range(start, end):
            pos_in_block = i - start
            weights[i] = math.exp(-gamma * pos_in_block / block_size)

    return weights

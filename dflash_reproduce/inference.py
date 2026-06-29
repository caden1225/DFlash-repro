"""
DFlash speculative decoding inference engine.

Implements the core speculative decoding loop using a draft model (block diffusion)
to generate candidate token blocks and a target model to verify them in parallel.

Key features:
    - KV-cache-aware generation for efficiency
    - Target feature extraction for draft model conditioning
    - Greedy and sampling-based decoding
    - Comprehensive generation statistics
    - Batch generation support
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

# transformers imports are deferred to avoid hard dependency.
# These are only needed when using actual model/tokenizer objects.
try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
except ImportError:
    PreTrainedModel = Any  # type: ignore[misc,assignment]
    PreTrainedTokenizer = Any  # type: ignore[misc,assignment]

logger = logging.getLogger("dflash")


# ---------------------------------------------------------------------------
# Configuration placeholder (will be imported from config module when available)
# ---------------------------------------------------------------------------

@dataclass
class DFlashConfig:
    """Configuration for DFlash speculative decoding.

    Attributes:
        block_size: Number of tokens in each speculative block (B).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0 for greedy).
        top_k: Top-k sampling parameter.
        top_p: Nucleus (top-p) sampling parameter.
        target_layer_ids: Which target model layers to extract features from.
        kv_injection_layer: Layer to inject KV cache into the draft model.
        mask_token_id: Token ID used for masked positions in draft input.
        eos_token_id: End-of-sequence token ID.
        device: Device to run inference on.
        dtype: Data type for model weights.
        do_sample: Whether to use sampling or greedy decoding.
        use_cache: Whether to use KV cache.
        pad_token_id: Pad token ID for batch generation.
    """
    block_size: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    target_layer_ids: List[int] = field(default_factory=lambda: [-1, -2, -3])
    kv_injection_layer: int = 0
    mask_token_id: int = 32000
    eos_token_id: int = 2
    device: str = "cuda"
    dtype: str = "fp16"
    do_sample: bool = False
    use_cache: bool = True
    pad_token_id: int = 0

    def __post_init__(self):
        if self.temperature > 0:
            self.do_sample = True


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tensor:
    """Sample token IDs from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: Unnormalized logits tensor of shape [..., vocab_size].
        temperature: Sampling temperature. 0 means greedy (argmax).
        top_k: Number of top tokens to keep (0 = disabled).
        top_p: Nucleus sampling cumulative probability threshold (1.0 = disabled).

    Returns:
        Sampled token IDs of shape [...].
    """
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-8)

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False  # Keep at least one token
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# Target feature extraction
# ---------------------------------------------------------------------------

def extract_context_features(
    hidden_states: Tuple[Tensor, ...],
    layer_ids: List[int],
    offset: int = 1,
) -> Tensor:
    """Extract and concatenate hidden states from specified layers.

    Args:
        hidden_states: Tuple of hidden state tensors from each layer.
            Each tensor has shape [batch, seq_len, hidden_dim].
        layer_ids: List of layer indices to extract (can be negative).
        offset: Offset to skip initial embedding layer (default: 1 for transformers).

    Returns:
        Concatenated features of shape [batch, seq_len, num_layers * hidden_dim].
    """
    selected = []
    for layer_id in layer_ids:
        idx = layer_id + offset if layer_id >= 0 else layer_id + len(hidden_states)
        selected.append(hidden_states[idx])
    return torch.cat(selected, dim=-1)


# ---------------------------------------------------------------------------
# Draft model interface
# ---------------------------------------------------------------------------

class DraftModelInterface:
    """Interface wrapper for draft models used in speculative decoding.

    This handles both DFlashDraftModel instances and generic models.
    """

    def __init__(self, model, config: DFlashConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

    @torch.no_grad()
    def forward(
        self,
        input_ids: Tensor,
        target_features: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run draft model forward pass.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            target_features: Target model hidden states for conditioning
                [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.

        Returns:
            Logits tensor [batch, seq_len, vocab_size].
        """
        # Check if model has the DFlash-specific forward signature
        if hasattr(self.model, "forward_draft"):
            return self.model.forward_draft(
                input_ids=input_ids,
                target_features=target_features,
                attention_mask=attention_mask,
            )
        elif hasattr(self.model, "forward"):
            # Generic fallback: try passing target_features as kwargs
            try:
                return self.model(
                    input_ids=input_ids,
                    target_features=target_features,
                    attention_mask=attention_mask,
                )
            except TypeError:
                # If the model doesn't accept target_features, fall back
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            raise RuntimeError("Draft model has no forward method")


# ---------------------------------------------------------------------------
# Generation statistics
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    """Statistics collected during speculative decoding generation."""

    total_generated_tokens: int = 0
    total_accepted_tokens: int = 0
    draft_forward_calls: int = 0
    target_forward_calls: int = 0
    num_iterations: int = 0
    acceptance_lengths: List[int] = field(default_factory=list)
    total_time_sec: float = 0.0

    @property
    def avg_acceptance_length(self) -> float:
        """Average number of accepted tokens per iteration (tau)."""
        if self.num_iterations == 0:
            return 0.0
        return self.total_accepted_tokens / self.num_iterations

    @property
    def acceptance_rate(self) -> float:
        """Ratio of accepted tokens to total generated tokens."""
        if self.total_generated_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_generated_tokens

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_generated_tokens = 0
        self.total_accepted_tokens = 0
        self.draft_forward_calls = 0
        self.target_forward_calls = 0
        self.num_iterations = 0
        self.acceptance_lengths = []
        self.total_time_sec = 0.0

    def to_dict(self) -> Dict[str, Union[int, float, List[int]]]:
        """Convert stats to dictionary."""
        return {
            "total_generated_tokens": self.total_generated_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "avg_acceptance_length_tau": self.avg_acceptance_length,
            "acceptance_rate": self.acceptance_rate,
            "draft_forward_calls": self.draft_forward_calls,
            "target_forward_calls": self.target_forward_calls,
            "num_iterations": self.num_iterations,
            "total_time_sec": self.total_time_sec,
            "acceptance_lengths": self.acceptance_lengths,
        }


# ---------------------------------------------------------------------------
# DFlash Inference Engine
# ---------------------------------------------------------------------------

class DFlashInferenceEngine:
    """Inference engine for DFlash speculative decoding.

    Uses a small draft model to generate candidate token blocks and a large
    target model to verify them in parallel, achieving speedup over standard
    autoregressive generation.

    Args:
        config: DFlash configuration.
        draft_model: Draft model for token generation.
        target_model: Target (verification) model.
        tokenizer: Tokenizer for encoding/decoding text.
    """

    def __init__(
        self,
        config: DFlashConfig,
        draft_model,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        self.config = config
        self.draft = DraftModelInterface(draft_model, config)
        self.target = target_model
        self.tokenizer = tokenizer
        self.device = torch.device(config.device)
        self.stats = GenerationStats()

        # Ensure models are in eval mode
        self.target.eval()
        if hasattr(draft_model, "eval"):
            draft_model.eval()

        # Move models to device if not already
        if hasattr(self.target, "device_map") and self.target.device_map is not None:
            pass  # Model is on multiple devices
        else:
            self.target = self.target.to(self.device)
        if hasattr(draft_model, "to"):
            draft_model = draft_model.to(self.device)

        logger.info(
            f"DFlashInferenceEngine initialized: block_size={config.block_size}, "
            f"device={config.device}, temperature={config.temperature}"
        )

    # ------------------------------------------------------------------
    # Target model prefill
    # ------------------------------------------------------------------

    @torch.no_grad()
    def target_prefill(
        self,
        input_ids: Tensor,
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], ...], Tensor]:
        """Run target model prefill on the input prompt.

        Args:
            input_ids: Tokenized prompt [batch, prompt_len].

        Returns:
            Tuple of:
                - first_token: First generated token ID [batch].
                - past_key_values: KV cache for subsequent generation.
                - target_features: Extracted hidden states from specified layers.
        """
        attention_mask = torch.ones_like(input_ids)

        outputs = self.target(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get first token (greedy or sampling)
        next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
        first_token = sample_from_logits(
            next_token_logits,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )  # [batch]

        past_key_values = outputs.past_key_values

        # Extract features from specified target layers
        hidden_states = outputs.hidden_states  # tuple of [batch, seq, hidden]
        target_features = extract_context_features(
            hidden_states,
            self.config.target_layer_ids,
        )  # [batch, seq, num_layers * hidden]

        self.stats.target_forward_calls += 1

        return first_token, past_key_values, target_features

    # ------------------------------------------------------------------
    # Block generation (draft model)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_block(
        self,
        input_ids: Tensor,
        target_features: Tensor,
        bonus_token: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate a block of candidate tokens using the draft model.

        Constructs the input sequence as [context_tokens, bonus_token, mask, ..., mask]
        and runs the draft model with non-causal attention to predict all
        masked positions in parallel.

        Args:
            input_ids: Current accepted token sequence [batch, seq_len].
            target_features: Target model hidden states for conditioning
                [batch, seq_len, hidden_dim].
            bonus_token: The last accepted token to start the block [batch].
                If None, uses the last token of input_ids.

        Returns:
            Draft tokens [batch, block_size] where:
                draft_tokens[:, 0] = bonus_token
                draft_tokens[:, 1:] = draft predictions for masked positions.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Use last accepted token as bonus if not provided
        if bonus_token is None:
            bonus_token = input_ids[:, -1]  # [batch]

        # Build draft input: [bonus, mask, mask, ..., mask]
        mask_tokens = torch.full(
            (batch_size, self.config.block_size - 1),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        draft_input = torch.cat([
            bonus_token.unsqueeze(1),  # [batch, 1]
            mask_tokens,               # [batch, block_size - 1]
        ], dim=1)  # [batch, block_size]

        # Build attention mask for non-causal attention:
        # All positions can attend to all positions in the block
        block_attention_mask = torch.ones(
            batch_size,
            self.config.block_size,
            self.config.block_size,
            dtype=torch.bool,
            device=device,
        )

        # Prepare target features for the draft block
        # Use the last position's features as conditioning
        if target_features.size(1) > 0:
            last_features = target_features[:, -1:, :]  # [batch, 1, hidden]
            # Expand to match block size
            block_features = last_features.expand(-1, self.config.block_size, -1)
        else:
            block_features = torch.zeros(
                batch_size, self.config.block_size, target_features.size(-1),
                device=device, dtype=target_features.dtype,
            )

        # Run draft model forward pass
        draft_logits = self.draft.forward(
            input_ids=draft_input,
            target_features=block_features,
            attention_mask=block_attention_mask,
        )  # [batch, block_size, vocab_size]

        # Sample tokens from draft logits
        draft_tokens = sample_from_logits(
            draft_logits,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )  # [batch, block_size]

        # Ensure bonus token is preserved
        draft_tokens[:, 0] = bonus_token

        self.stats.draft_forward_calls += 1

        return draft_tokens

    # ------------------------------------------------------------------
    # Target model block verification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verify_block(
        self,
        input_ids: Tensor,
        draft_tokens: Tensor,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[int, Tensor, Tensor, Optional[Tuple]]:
        """Verify draft tokens using the target model.

        When KV cache (past_key_values) is available, only passes the new
        draft_tokens to the target model. Otherwise, passes the full sequence.
        Finds the first mismatch between draft predictions and target predictions.

        Args:
            input_ids: Current accepted token sequence [batch, seq_len].
            draft_tokens: Draft-generated tokens [batch, block_size].
                draft_tokens[:, 0] is the bonus token (repeated from input_ids[-1]).
            past_key_values: KV cache from previous target model calls.

        Returns:
            Tuple of:
                - accepted_length: Number of accepted tokens (0 to block_size).
                - next_token: Target model's token at first mismatch position [batch].
                - accepted_tokens: All accepted draft tokens [batch, accepted_length].
                - new_past_key_values: Updated KV cache.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        block_size = draft_tokens.size(1)

        # When past_key_values is available, we only need to process new tokens.
        # The draft_tokens include the bonus token at position 0, which the
        # target model needs to see to produce logits for position 1 onwards.
        if past_key_values is not None:
            # Only pass draft tokens; KV cache contains prefix state
            model_input_ids = draft_tokens
            seq_len = input_ids.size(1) + block_size
            attention_mask = torch.ones(
                batch_size, seq_len,
                dtype=torch.long,
                device=device,
            )
        else:
            # No KV cache: pass full sequence
            model_input_ids = torch.cat([input_ids, draft_tokens], dim=1)
            attention_mask = torch.ones(
                batch_size, model_input_ids.size(1),
                dtype=torch.long,
                device=device,
            )

        # Run target model
        outputs = self.target(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

        # Logits for the draft positions.
        # When past_key_values is used, outputs.logits has shape
        # [batch, block_size, vocab_size] (only new tokens).
        # When past_key_values is None, outputs.logits has shape
        # [batch, input_len + block_size, vocab_size].
        if past_key_values is not None:
            # All logits are for draft tokens
            verify_logits = outputs.logits  # [batch, block_size, vocab_size]
        else:
            # Need to extract the last block_size logits
            verify_logits = outputs.logits[:, -block_size:, :]  # [batch, block_size, vocab_size]

        # Get target model predictions for each draft position
        if self.config.temperature == 0:
            target_predictions = verify_logits.argmax(dim=-1)  # [batch, block_size]
        else:
            target_predictions = sample_from_logits(
                verify_logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )  # [batch, block_size]

        # Find first mismatch: compare target predictions with draft tokens
        matches = (target_predictions == draft_tokens).long()  # [batch, block_size]

        # For single-item generation (batch_size=1)
        if batch_size == 1:
            accepted_length = 0
            for i in range(block_size):
                if matches[0, i].item():
                    accepted_length += 1
                else:
                    break

            if accepted_length == block_size:
                # All draft tokens accepted: get bonus token from last position
                next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                next_token = sample_from_logits(
                    next_token_logits,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                )
                accepted_tokens = draft_tokens[0, :accepted_length]
            else:
                # First mismatch: use target prediction as bonus token
                next_token = target_predictions[0, accepted_length:accepted_length + 1]
                accepted_tokens = draft_tokens[0, :accepted_length]

            self.stats.target_forward_calls += 1

            return (
                accepted_length,
                next_token,
                accepted_tokens.unsqueeze(0),
                outputs.past_key_values,
            )

        # Batch verification
        accepted_lengths = []
        all_accepted = []
        all_next_tokens = []

        for b in range(batch_size):
            acc_len = 0
            for i in range(block_size):
                if matches[b, i].item():
                    acc_len += 1
                else:
                    break
            accepted_lengths.append(acc_len)

            if acc_len == block_size:
                next_tok = sample_from_logits(
                    outputs.logits[b, -1, :],
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                )
            else:
                next_tok = target_predictions[b, acc_len]

            all_accepted.append(draft_tokens[b, :acc_len])
            all_next_tokens.append(next_tok)

        min_accepted = min(accepted_lengths)
        next_tokens = torch.stack(all_next_tokens)

        self.stats.target_forward_calls += 1

        return min_accepted, next_tokens, draft_tokens[:, :min_accepted], outputs.past_key_values

    # ------------------------------------------------------------------
    # Speculative decoding generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def speculative_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text using speculative decoding.

        Args:
            prompt: Input prompt string.
            max_new_tokens: Maximum number of new tokens to generate.
                Defaults to config.max_new_tokens.
            temperature: Sampling temperature. Defaults to config.temperature.

        Returns:
            Generated text string (including the prompt).
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        if temperature is not None:
            orig_temp = self.config.temperature
            self.config.temperature = temperature

        self.stats.reset()
        start_time = time.perf_counter()

        try:
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            # [1, prompt_len]

            # Prefill target model
            first_token, past_key_values, target_features = self.target_prefill(input_ids)

            # Initialize accepted tokens with first token
            accepted_ids = torch.cat([
                input_ids,
                first_token.unsqueeze(0).unsqueeze(0),
            ], dim=1)  # [1, prompt_len + 1]

            self.stats.total_generated_tokens += 1
            self.stats.total_accepted_tokens += 1

            # Speculative decoding loop
            while self.stats.total_generated_tokens < max_new_tokens:
                self.stats.num_iterations += 1

                # 1. Extract target features from KV cache / hidden states
                # For simplicity, use the last target_features
                # In a full implementation, we'd re-extract from the KV cache
                current_features = target_features

                # 2. Generate draft block
                bonus_token = accepted_ids[:, -1]  # [1]
                draft_tokens = self.generate_block(
                    input_ids=accepted_ids,
                    target_features=current_features,
                    bonus_token=bonus_token,
                )  # [1, block_size]

                # 3. Verify draft block with target model
                accepted_length, next_token, accepted_tokens, past_key_values = (
                    self.verify_block(
                        input_ids=accepted_ids,
                        draft_tokens=draft_tokens,
                        past_key_values=past_key_values,
                    )
                )

                # 4. Append accepted tokens
                if accepted_length > 0:
                    accepted_ids = torch.cat([
                        accepted_ids,
                        accepted_tokens,
                    ], dim=1)
                    self.stats.total_generated_tokens += accepted_length
                    self.stats.total_accepted_tokens += accepted_length

                # 5. Append bonus token (target model's correction)
                accepted_ids = torch.cat([
                    accepted_ids,
                    next_token.unsqueeze(0).unsqueeze(0),
                ], dim=1)
                self.stats.total_generated_tokens += 1
                self.stats.total_accepted_tokens += 1

                self.stats.acceptance_lengths.append(accepted_length)

                # 6. Check stopping conditions
                if next_token.item() == self.config.eos_token_id:
                    break

                # Check if we've exceeded max_new_tokens
                current_new_tokens = accepted_ids.size(1) - input_ids.size(1)
                if current_new_tokens >= max_new_tokens:
                    break

            # Decode and return
            generated_text = self.tokenizer.decode(
                accepted_ids[0],
                skip_special_tokens=True,
            )

            self.stats.total_time_sec = time.perf_counter() - start_time

            return generated_text

        finally:
            if temperature is not None:
                self.config.temperature = orig_temp

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate text for multiple prompts in batch.

        For simplicity, this currently processes prompts sequentially.
        A full implementation would use padding and batch-consistent
        block sizes for true parallel generation.

        Args:
            prompts: List of input prompt strings.
            max_new_tokens: Maximum number of new tokens per prompt.

        Returns:
            List of generated text strings.
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        results = []
        for prompt in prompts:
            result = self.speculative_generate(prompt, max_new_tokens)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Union[int, float, List[int]]]:
        """Get generation statistics.

        Returns:
            Dictionary with generation statistics:
                - total_generated_tokens: Total tokens generated.
                - total_accepted_tokens: Tokens accepted from draft.
                - avg_acceptance_length (tau): Average accepted tokens per iteration.
                - draft_forward_calls: Number of draft model forward passes.
                - target_forward_calls: Number of target model forward passes.
                - num_iterations: Number of speculative decoding iterations.
                - total_time_sec: Total generation time.
        """
        return self.stats.to_dict()

    def print_stats(self) -> None:
        """Print a formatted summary of generation statistics."""
        stats = self.get_stats()
        logger.info("=" * 50)
        logger.info("Generation Statistics")
        logger.info("=" * 50)
        logger.info(f"Total generated tokens:   {stats['total_generated_tokens']}")
        logger.info(f"Total accepted tokens:    {stats['total_accepted_tokens']}")
        logger.info(f"Avg acceptance length (tau): {stats['avg_acceptance_length_tau']:.3f}")
        logger.info(f"Acceptance rate:          {stats['acceptance_rate']:.3f}")
        logger.info(f"Draft forward calls:      {stats['draft_forward_calls']}")
        logger.info(f"Target forward calls:     {stats['target_forward_calls']}")
        logger.info(f"Num iterations:           {stats['num_iterations']}")
        logger.info(f"Total time:               {stats['total_time_sec']:.3f}s")
        if stats['total_time_sec'] > 0:
            tps = stats['total_generated_tokens'] / stats['total_time_sec']
            logger.info(f"Tokens per second:        {tps:.2f}")
        logger.info("=" * 50)

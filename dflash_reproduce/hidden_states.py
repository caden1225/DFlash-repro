"""
Hidden States Extraction Module

Provides multiple strategies for extracting hidden states from the target model
to be used as KV injection features for the DFlash draft model:

    1. Online extraction via vLLM REST API (for production serving)
    2. Offline extraction with disk caching (for training data preparation)
    3. Hybrid extraction (cache-first with online fallback, for flexible workflows)

All extractors share a unified interface defined by the HiddenStatesExtractor
abstract base class.

Usage example:
    >>> # Offline extraction (recommended for training)
    >>> extractor = OfflineHiddenStatesExtractor(
    ...     cache_dir="./cache/hidden_states",
    ...     target_model_path="Qwen/Qwen2.5-7B",
    ...     target_layer_ids=[-5, -4, -3, -2, -1],
    ... )
    >>> extractor.extract_and_cache(dataset, batch_size=16)
    >>> features = extractor.load_cached(0)  # Load first sample

    >>> # Online extraction (for production serving with vLLM)
    >>> extractor = OnlineHiddenStatesExtractor(
    ...     vllm_endpoint="http://localhost:8000",
    ...     target_layer_ids=[-5, -4, -3, -2, -1],
    ... )
    >>> features = extractor.extract(input_ids)

    >>> # Hybrid extraction (best of both worlds)
    >>> extractor = HybridHiddenStatesExtractor(
    ...     cache_dir="./cache/hidden_states",
    ...     vllm_endpoint="http://localhost:8000",
    ...     target_layer_ids=[-5, -4, -3, -2, -1],
    ... )
    >>> features = extractor.extract(input_ids, index=0)  # Tries cache first
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dflash_reproduce.model import extract_context_feature

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Abstract Base Class
# --------------------------------------------------------------------------- #


class HiddenStatesExtractor(ABC):
    """Abstract base class for hidden states extractors.

    All extractors must implement the `extract` method, which takes input token
    IDs and returns concatenated hidden states from specified target model layers.
    """

    def __init__(self, target_layer_ids: List[int]) -> None:
        """Initialize the extractor.

        Args:
            target_layer_ids: List of target-model layer IDs to extract
                hidden states from. Negative values are supported (Python-style
                indexing, e.g. -1 is the last layer).
        """
        self.target_layer_ids = target_layer_ids
        self.num_layers = len(target_layer_ids)

    @abstractmethod
    def extract(self, input_ids: Tensor) -> Tensor:
        """Extract hidden states from the target model.

        Args:
            input_ids: Input token IDs [batch, seq_len].

        Returns:
            Concatenated hidden states [batch, seq_len, num_layers * hidden_dim].
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output feature dimension per token.

        Returns:
            The dimension of the concatenated hidden states for one token,
            which equals num_layers * target_hidden_dim.
        """
        ...


# --------------------------------------------------------------------------- #
# Online Extractor (vLLM REST API)
# --------------------------------------------------------------------------- #


class OnlineHiddenStatesExtractor(HiddenStatesExtractor):
    """Extract hidden states using vLLM's REST API.

    This extractor sends requests to a running vLLM server and retrieves
    hidden states from specified layers. Suitable for production serving
    where the target model is served via vLLM.

    Requirements:
        - A running vLLM server with the `--return-hidden-states` or equivalent
          flag enabled.
        - The server must expose an endpoint that returns hidden states.

    Note:
        The vLLM API for hidden states extraction is still evolving. This
        implementation assumes a custom endpoint or a compatible API format.
        You may need to adjust the request/response format based on your
        vLLM version and configuration.
    """

    def __init__(
        self,
        vllm_endpoint: str,
        target_layer_ids: List[int],
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        max_batch_size: int = 32,
    ) -> None:
        """Initialize the online extractor.

        Args:
            vllm_endpoint: Base URL of the vLLM server
                (e.g. "http://localhost:8000").
            target_layer_ids: Layer IDs to extract hidden states from.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            max_batch_size: Maximum batch size per API request.
        """
        super().__init__(target_layer_ids)
        self.vllm_endpoint = vllm_endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self._target_hidden_dim: Optional[int] = None

        # Verify connectivity
        self._check_health()

    def _get_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _check_health(self) -> None:
        """Check if the vLLM server is reachable."""
        try:
            response = requests.get(
                f"{self.vllm_endpoint}/health",
                headers=self._get_headers(),
                timeout=10.0,
            )
            if response.status_code == 200:
                logger.info("vLLM server at %s is healthy", self.vllm_endpoint)
            else:
                logger.warning(
                    "vLLM server returned status %d", response.status_code
                )
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Could not connect to vLLM server at %s. "
                "Make sure the server is running.",
                self.vllm_endpoint,
            )
        except Exception as e:
            logger.warning("Health check failed: %s", e)

    def _send_request(self, input_ids: List[List[int]], layer_ids: List[int]) -> Dict[str, Any]:
        """Send a request to the vLLM server to extract hidden states.

        Args:
            input_ids: List of token ID sequences.
            layer_ids: Layer IDs to extract.

        Returns:
            JSON response from the server.

        Raises:
            requests.RequestException: If the request fails.
        """
        # This payload format is vLLM-dependent and may need customization
        payload = {
            "model": "target_model",  # may be required by the API
            "input_ids": input_ids,
            "layers": layer_ids,
            "return_hidden_states": True,
        }

        response = requests.post(
            f"{self.vllm_endpoint}/v1/hidden_states",
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def extract(self, input_ids: Tensor) -> Tensor:
        """Extract hidden states via vLLM API.

        Handles batching automatically if input exceeds max_batch_size.

        Args:
            input_ids: Input token IDs [batch, seq_len].

        Returns:
            Concatenated hidden states [batch, seq_len, num_layers * hidden_dim].

        Raises:
            RuntimeError: If the API request fails.
        """
        batch_size = input_ids.shape[0]

        # Process in chunks if batch is too large
        if batch_size > self.max_batch_size:
            chunks = torch.split(input_ids, self.max_batch_size, dim=0)
            outputs = []
            for chunk in chunks:
                outputs.append(self.extract(chunk))
            return torch.cat(outputs, dim=0)

        # Convert tensor to list for JSON serialization
        input_ids_list = input_ids.cpu().tolist()

        try:
            response = self._send_request(input_ids_list, self.target_layer_ids)

            # Parse response - format depends on the vLLM API
            # Expected: {"hidden_states": list of [batch, seq_len, hidden_dim] per layer}
            hidden_states_data = response.get("hidden_states", [])

            if not hidden_states_data:
                raise RuntimeError("No hidden states returned from vLLM API")

            # Convert to tensors
            hidden_states = [
                torch.tensor(h, dtype=torch.float32, device=input_ids.device)
                for h in hidden_states_data
            ]

            # Concatenate along feature dimension
            result = extract_context_feature(
                hidden_states, self.target_layer_ids, offset=0
            )

            return result.to(input_ids.device)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM API request failed: {e}") from e
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse vLLM response: {e}") from e

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension.

        Note: For the online extractor, this is determined lazily from the
        first API response. If not yet known, returns -1.
        """
        if self._target_hidden_dim is None:
            return -1  # Unknown until first extraction
        return self.num_layers * self._target_hidden_dim

    def set_target_hidden_dim(self, dim: int) -> None:
        """Manually set the target model's hidden dimension.

        Args:
            dim: Hidden dimension of the target model.
        """
        self._target_hidden_dim = dim


# --------------------------------------------------------------------------- #
# Offline Extractor (Pre-compute and Cache)
# --------------------------------------------------------------------------- #


class OfflineHiddenStatesExtractor(HiddenStatesExtractor):
    """Pre-compute and cache hidden states from the target model.

    This extractor loads the target model locally, runs forward passes to extract
    hidden states from specified layers, and saves them to disk for efficient
    loading during training.

    Recommended for training data preparation where the dataset is fixed.

    Cache format:
        Each sample is saved as a ``.pt`` file containing:
            - ``hidden_states``: Tensor [seq_len, num_layers * hidden_dim]
            - ``input_ids``: Tensor [seq_len] (optional, for verification)
            - ``metadata``: Dict with sample info (optional)
    """

    def __init__(
        self,
        cache_dir: str,
        target_model_path: str,
        target_layer_ids: List[int],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        offset: int = 1,
    ) -> None:
        """Initialize the offline extractor.

        Args:
            cache_dir: Directory to save cached hidden states.
            target_model_path: HuggingFace model name or local path
                (e.g. "Qwen/Qwen2.5-7B").
            target_layer_ids: Layer IDs to extract hidden states from.
            device: Device to run the target model on. Auto-detected if None.
            dtype: Data type for the target model.
            trust_remote_code: Whether to trust remote code for model loading.
            offset: Offset for layer ID adaptation (default 1 for HuggingFace).
        """
        super().__init__(target_layer_ids)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.target_model_path = target_model_path
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.offset = offset

        # Load target model config
        self.target_config = AutoConfig.from_pretrained(
            target_model_path, trust_remote_code=trust_remote_code
        )
        self.target_hidden_dim = getattr(self.target_config, "hidden_size", 4096)
        self._output_dim = self.num_layers * self.target_hidden_dim

        # Target model (lazy loading - loaded when needed)
        self._target_model: Optional[nn.Module] = None
        self._tokenizer: Optional[Any] = None

        # Cache index
        self._cache_index_file = self.cache_dir / "index.json"
        self._cache_index: Dict[str, Any] = self._load_cache_index()

        logger.info(
            "OfflineHiddenStatesExtractor: cache_dir=%s, target=%s, "
            "target_hidden_dim=%d, output_dim=%d",
            self.cache_dir,
            target_model_path,
            self.target_hidden_dim,
            self._output_dim,
        )

    def _load_cache_index(self) -> Dict[str, Any]:
        """Load the cache index file if it exists."""
        if self._cache_index_file.exists():
            with open(self._cache_index_file, "r") as f:
                return json.load(f)
        return {"samples": [], "target_model": self.target_model_path, "target_layer_ids": self.target_layer_ids}

    def _save_cache_index(self) -> None:
        """Save the cache index file."""
        with open(self._cache_index_file, "w") as f:
            json.dump(self._cache_index, f, indent=2)

    def _load_target_model(self) -> nn.Module:
        """Lazy-load the target model.

        Returns:
            The loaded target model.
        """
        if self._target_model is None:
            logger.info("Loading target model from %s ...", self.target_model_path)
            self._target_model = AutoModelForCausalLM.from_pretrained(
                self.target_model_path,
                config=self.target_config,
                torch_dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                device_map="auto" if self.device.type == "cuda" else None,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.target_model_path, trust_remote_code=self.trust_remote_code
            )
            self._target_model.eval()
            if self.device.type == "cpu":
                self._target_model.to(self.device)
            logger.info("Target model loaded successfully.")
        return self._target_model

    def _get_cache_file(self, index: int) -> Path:
        """Get the cache file path for a given sample index.

        Args:
            index: Sample index.

        Returns:
            Path to the cache file.
        """
        return self.cache_dir / f"sample_{index:08d}.pt"

    def _is_cached(self, index: int) -> bool:
        """Check if a sample is already cached.

        Args:
            index: Sample index.

        Returns:
            True if the cache file exists.
        """
        return self._get_cache_file(index).exists()

    def extract_single(
        self,
        input_ids: Tensor,
        return_all_layers: bool = False,
    ) -> Union[Tensor, Tuple[List[Tensor], Tensor]]:
        """Extract hidden states for a single sample.

        Args:
            input_ids: Input token IDs [seq_len] or [1, seq_len].
            return_all_layers: If True, return all layer hidden states separately.

        Returns:
            If return_all_layers=False: concatenated hidden states
                [seq_len, num_layers * hidden_dim].
            If return_all_layers=True: tuple of (all_hidden_states, concatenated).
        """
        model = self._load_target_model()

        # Ensure shape is [1, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract hidden states from all layers
        all_hidden_states = outputs.hidden_states  # Tuple of [1, seq_len, hidden_dim]

        # Convert to list of tensors
        hidden_states_list = [h.squeeze(0) for h in all_hidden_states]  # [seq_len, hidden_dim] each

        # Select and concatenate specified layers
        result = extract_context_feature(
            hidden_states_list, self.target_layer_ids, offset=self.offset
        )

        if return_all_layers:
            return hidden_states_list, result

        return result

    def extract_and_cache(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        num_workers: int = 0,
        desc: str = "Extracting hidden states",
    ) -> None:
        """Extract and cache hidden states for an entire dataset.

        Args:
            dataset: PyTorch Dataset yielding samples with an ``input_ids`` field.
                Each sample should be a dict with ``input_ids`` as a tensor
                or list of token IDs.
            batch_size: Batch size for forward passes.
            num_workers: Number of DataLoader workers.
            desc: Description for the progress bar.
        """
        model = self._load_target_model()
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

        total_samples = len(dataset)
        cached_count = 0
        processed_count = 0

        pbar = tqdm(total=total_samples, desc=desc, disable=self._is_main_process is False)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            batch_size_actual = input_ids.shape[0]
            start_index = batch_idx * batch_size

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Extract hidden states from all layers
                all_hidden_states = outputs.hidden_states

                # Process each sample in the batch
                for i in range(batch_size_actual):
                    sample_index = start_index + i

                    # Skip if already cached
                    if self._is_cached(sample_index):
                        cached_count += 1
                        continue

                    # Extract hidden states for this sample
                    sample_hidden_states = [
                        h[i] for h in all_hidden_states
                    ]  # List of [seq_len, hidden_dim]

                    concatenated = extract_context_feature(
                        sample_hidden_states, self.target_layer_ids, offset=self.offset
                    )

                    # Move to CPU for saving
                    concatenated = concatenated.cpu()
                    sample_input_ids = input_ids[i].cpu()

                    # Save to cache
                    cache_file = self._get_cache_file(sample_index)
                    torch.save(
                        {
                            "hidden_states": concatenated,
                            "input_ids": sample_input_ids,
                            "metadata": {"index": sample_index, "layer_ids": self.target_layer_ids},
                        },
                        cache_file,
                    )

                    # Update index
                    self._cache_index["samples"].append(
                        {"index": sample_index, "file": str(cache_file.name)}
                    )

                    processed_count += 1

                pbar.update(batch_size_actual)

        pbar.close()
        self._save_cache_index()

        logger.info(
            "Extraction complete: %d samples processed, %d already cached, "
            "%d total in cache",
            processed_count,
            cached_count,
            len(self._cache_index["samples"]),
        )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """Collate function for DataLoader.

        Handles padding of variable-length sequences.

        Args:
            batch: List of samples.

        Returns:
            Batched tensors.
        """
        input_ids_list = [torch.tensor(item["input_ids"]) for item in batch]

        # Pad sequences
        max_len = max(ids.shape[0] for ids in input_ids_list)
        padded_ids = torch.full(
            (len(batch), max_len),
            fill_value=0,  # pad token id
            dtype=torch.long,
        )

        for i, ids in enumerate(input_ids_list):
            padded_ids[i, : ids.shape[0]] = ids

        return {"input_ids": padded_ids}

    def load_cached(self, index: int) -> Tensor:
        """Load cached hidden states for a given sample index.

        Args:
            index: Sample index.

        Returns:
            Concatenated hidden states [seq_len, num_layers * hidden_dim].

        Raises:
            FileNotFoundError: If the cache file does not exist.
        """
        cache_file = self._get_cache_file(index)

        if not cache_file.exists():
            raise FileNotFoundError(
                f"Cache file not found for index {index}: {cache_file}. "
                f"Run extract_and_cache() first."
            )

        data = torch.load(cache_file, map_location="cpu")
        return data["hidden_states"]

    def load_cached_batch(
        self,
        indices: List[int],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Load cached hidden states for multiple indices.

        Args:
            indices: List of sample indices.
            device: Device to move the tensors to.

        Returns:
            Batched hidden states [len(indices), max_seq_len, feature_dim].
        """
        tensors = [self.load_cached(i) for i in indices]

        # Pad to max length
        max_len = max(t.shape[0] for t in tensors)
        feature_dim = tensors[0].shape[1]

        padded = torch.full(
            (len(indices), max_len, feature_dim),
            fill_value=0.0,
            dtype=tensors[0].dtype,
        )

        for i, t in enumerate(tensors):
            padded[i, : t.shape[0]] = t

        if device is not None:
            padded = padded.to(device)

        return padded

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension per token."""
        return self._output_dim

    @property
    def _is_main_process(self) -> bool:
        """Check if this is the main process."""
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True


# --------------------------------------------------------------------------- #
# Hybrid Extractor (Cache + Online Fallback)
# --------------------------------------------------------------------------- #


class HybridHiddenStatesExtractor(HiddenStatesExtractor):
    """Hybrid extractor that checks cache first, falls back to online extraction.

    This is useful for training workflows where:
        - Most samples have been pre-cached
        - New/augmented samples need on-the-fly extraction
        - You want the flexibility of both modes

    The extractor also supports dynamic caching: extracted samples are
    automatically saved to the cache for future use.
    """

    def __init__(
        self,
        cache_dir: str,
        vllm_endpoint: Optional[str] = None,
        target_model_path: Optional[str] = None,
        target_layer_ids: List[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        auto_cache: bool = True,
        offset: int = 1,
    ) -> None:
        """Initialize the hybrid extractor.

        Args:
            cache_dir: Directory for caching hidden states.
            vllm_endpoint: Optional vLLM endpoint for online extraction.
                Required if not all samples are cached.
            target_model_path: Optional path to target model for offline extraction.
                Required if not all samples are cached and vllm_endpoint is not set.
            target_layer_ids: Layer IDs to extract.
            device: Device for offline extraction.
            dtype: Data type for the target model.
            auto_cache: Whether to automatically cache online-extracted samples.
            offset: Offset for layer ID adaptation.
        """
        super().__init__(target_layer_ids or [-5, -4, -3, -2, -1])
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_cache = auto_cache
        self.offset = offset

        # Initialize offline extractor (always available for loading cached data)
        if target_model_path is not None:
            self.offline_extractor = OfflineHiddenStatesExtractor(
                cache_dir=cache_dir,
                target_model_path=target_model_path,
                target_layer_ids=self.target_layer_ids,
                device=device,
                dtype=dtype,
                offset=offset,
            )
        else:
            self.offline_extractor = None

        # Initialize online extractor (if endpoint provided)
        if vllm_endpoint is not None:
            self.online_extractor = OnlineHiddenStatesExtractor(
                vllm_endpoint=vllm_endpoint,
                target_layer_ids=self.target_layer_ids,
            )
        else:
            self.online_extractor = None

        # Cache index
        self._cache_index_file = self.cache_dir / "index.json"
        self._cache_index: Dict[int, str] = self._build_cache_index()

        # Output dimension (from offline extractor if available)
        if self.offline_extractor is not None:
            self._output_dim = self.offline_extractor.output_dim
        else:
            self._output_dim = -1  # Unknown

        logger.info(
            "HybridHiddenStatesExtractor: cache_dir=%s, cached_samples=%d, "
            "offline=%s, online=%s",
            self.cache_dir,
            len(self._cache_index),
            self.offline_extractor is not None,
            self.online_extractor is not None,
        )

    def _build_cache_index(self) -> Dict[int, str]:
        """Build an index of cached samples.

        Returns:
            Dictionary mapping sample index to cache file path.
        """
        index: Dict[int, str] = {}
        if not self.cache_dir.exists():
            return index

        for file_path in self.cache_dir.glob("sample_*.pt"):
            # Extract index from filename (sample_00000001.pt -> 1)
            try:
                idx = int(file_path.stem.split("_")[1])
                index[idx] = str(file_path)
            except (ValueError, IndexError):
                continue

        return index

    def _is_cached(self, index: int) -> bool:
        """Check if a sample is cached.

        Args:
            index: Sample index.

        Returns:
            True if the sample is in the cache.
        """
        return index in self._cache_index

    def _save_to_cache(self, index: int, hidden_states: Tensor, input_ids: Tensor) -> None:
        """Save extracted hidden states to the cache.

        Args:
            index: Sample index.
            hidden_states: Hidden states tensor [seq_len, feature_dim].
            input_ids: Input token IDs [seq_len].
        """
        cache_file = self.cache_dir / f"sample_{index:08d}.pt"

        torch.save(
            {
                "hidden_states": hidden_states.cpu(),
                "input_ids": input_ids.cpu(),
                "metadata": {"index": index, "layer_ids": self.target_layer_ids},
            },
            cache_file,
        )

        self._cache_index[index] = str(cache_file)
        logger.debug("Saved sample %d to cache: %s", index, cache_file)

    def extract(self, input_ids: Tensor, index: Optional[int] = None) -> Tensor:
        """Extract hidden states with cache-first strategy.

        If ``index`` is provided and the sample is cached, loads from cache.
        Otherwise, falls back to online or offline extraction.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            index: Optional sample index for cache lookup.

        Returns:
            Concatenated hidden states [batch, seq_len, num_layers * hidden_dim].

        Raises:
            RuntimeError: If no extraction method is available.
        """
        # Single sample with index - check cache first
        if index is not None and input_ids.dim() <= 2:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            # Check if all samples in the batch are cached
            batch_size = input_ids.shape[0]
            if batch_size == 1 and self._is_cached(index):
                cached = torch.load(self._cache_index[index])["hidden_states"]
                return cached.unsqueeze(0).to(input_ids.device)

        # Fallback to offline extraction
        if self.offline_extractor is not None:
            result = self.offline_extractor.extract_single(input_ids.squeeze(0) if input_ids.dim() == 2 and input_ids.shape[0] == 1 else input_ids)

            # Auto-cache if index is provided
            if self.auto_cache and index is not None and not self._is_cached(index):
                self._save_to_cache(index, result, input_ids.squeeze(0))

            return result.unsqueeze(0) if result.dim() == 2 and input_ids.dim() == 2 else result

        # Fallback to online extraction
        if self.online_extractor is not None:
            result = self.online_extractor.extract(input_ids)

            # Auto-cache if index is provided
            if self.auto_cache and index is not None and not self._is_cached(index):
                # For online extraction, we get batched results
                for i in range(result.shape[0]):
                    self._save_to_cache(index + i, result[i], input_ids[i])

            return result

        raise RuntimeError(
            "No extraction method available. Provide either a target_model_path "
            "for offline extraction or a vllm_endpoint for online extraction."
        )

    def extract_batch(
        self,
        input_ids: Tensor,
        indices: Optional[List[int]] = None,
    ) -> Tensor:
        """Extract hidden states for a batch, with cache-aware loading.

        Attempts to load cached samples individually, and falls back to
        extraction for missing ones.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            indices: Optional list of sample indices for cache lookup.

        Returns:
            Concatenated hidden states [batch, seq_len, num_layers * hidden_dim].
        """
        batch_size = input_ids.shape[0]

        if indices is None:
            # No indices provided - extract all using offline/online method
            if self.offline_extractor is not None:
                results = []
                for i in range(batch_size):
                    result = self.offline_extractor.extract_single(input_ids[i])
                    results.append(result)
                return self._pad_and_stack(results, input_ids.device)
            elif self.online_extractor is not None:
                return self.online_extractor.extract(input_ids)
            else:
                raise RuntimeError("No extraction method available.")

        # Try to load from cache, extract missing ones
        results = []
        missing_indices = []
        missing_positions = []

        for i, idx in enumerate(indices):
            if self._is_cached(idx):
                cached = torch.load(self._cache_index[idx])["hidden_states"]
                results.append((i, cached.to(input_ids.device)))
            else:
                missing_indices.append(idx)
                missing_positions.append(i)
                results.append((i, None))  # Placeholder

        # Extract missing samples
        if missing_positions:
            missing_input_ids = input_ids[missing_positions]

            if self.offline_extractor is not None:
                for pos, idx in zip(missing_positions, missing_indices):
                    result = self.offline_extractor.extract_single(input_ids[pos])
                    results[pos] = (pos, result.to(input_ids.device))

                    if self.auto_cache:
                        self._save_to_cache(idx, result, input_ids[pos])
            elif self.online_extractor is not None:
                extracted = self.online_extractor.extract(missing_input_ids)
                for j, (pos, idx) in enumerate(zip(missing_positions, missing_indices)):
                    results[pos] = (pos, extracted[j].to(input_ids.device))

                    if self.auto_cache:
                        self._save_to_cache(idx, extracted[j], input_ids[pos])
            else:
                raise RuntimeError(
                    f"Samples at indices {missing_indices} not cached and "
                    f"no extraction method available."
                )

        # Reorder and stack results
        ordered_results = [r for _, r in sorted(results, key=lambda x: x[0])]
        return self._pad_and_stack(ordered_results, input_ids.device)

    def _pad_and_stack(self, tensors: List[Tensor], device: torch.device) -> Tensor:
        """Pad a list of tensors to the same length and stack them.

        Args:
            tensors: List of tensors with shape [seq_len, feature_dim].
            device: Target device.

        Returns:
            Batched tensor [batch, max_seq_len, feature_dim].
        """
        max_len = max(t.shape[0] for t in tensors)
        feature_dim = tensors[0].shape[1]

        padded = torch.full(
            (len(tensors), max_len, feature_dim),
            fill_value=0.0,
            dtype=tensors[0].dtype,
            device=device,
        )

        for i, t in enumerate(tensors):
            padded[i, : t.shape[0]] = t

        return padded

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension per token."""
        return self._output_dim

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        return {
            "cached_samples": len(self._cache_index),
            "cache_dir": str(self.cache_dir),
        }



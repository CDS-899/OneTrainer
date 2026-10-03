from collections.abc import Callable
from typing import Any

import torch

_CACHE_ATTRIBUTE = "_sample_prompt_embedding_cache"


def _map_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_map_tensors(v, fn) for v in value)
    return value


def _cache_enabled(model) -> bool:
    # Only while training with a frozen text encoder: then a prompt always encodes to the same tensors.
    # Standalone sampling (no train config) and text encoder / embedding training never use the cache.
    train_config = getattr(model, "train_config", None)
    return train_config is not None and not train_config.train_text_encoder_or_embedding()


def cached_prompt_encoding(model, key: tuple, train_device: torch.device, encode: Callable[[], Any]) -> Any:
    """
    Returns encode() for this key, reusing the result of earlier sampling rounds when possible.

    encode() is responsible for materializing the text encoder, so on a cache hit the text encoder is not
    moved to the train device at all and the transformer is not evicted just to encode a known prompt.
    Cached tensors are kept in RAM.
    """
    if not _cache_enabled(model):
        return encode()

    cache = model.__dict__.setdefault(_CACHE_ATTRIBUTE, {})
    if key in cache:
        return _map_tensors(cache[key], lambda t: t.to(device=train_device))

    result = encode()
    cache[key] = _map_tensors(result, lambda t: t.detach().to(device="cpu"))
    return result

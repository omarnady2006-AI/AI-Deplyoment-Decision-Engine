from __future__ import annotations
"""Copyright (c) 2026 Omar Nady — Deployment Decision Engine. See LICENSE."""
"""core/model_analysis/cache.py — thread-safe LRU model cache + single-flight gate."""
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

_CACHE_MAX_MODELS    = 128
_MAX_MODEL_INFLIGHT  = 1024

_LOADED_MODEL_CACHE:   "OrderedDict[str, Any]" = OrderedDict()
_VALIDATED_MODEL_PATHS: set[str]               = set()
_MODEL_CACHE_LOCK:     threading.Lock          = threading.Lock()
_MODEL_SIZE_CACHE:     dict[str, int]          = {}
_MODEL_INFLIGHT:       "OrderedDict[str, _InflightEntry]" = OrderedDict()
_MODEL_INFLIGHT_LOCK:  threading.Lock          = threading.Lock()


class _InflightEntry:
    __slots__ = ("event","result","error_type","error_args","ready","last_access")
    def __init__(self) -> None:
        self.event:      threading.Event                       = threading.Event()
        self.result:     "tuple[bool, str|None, Any|None]|None" = None
        self.error_type: "type[BaseException]|None"            = None
        self.error_args: "tuple[Any,...]"                      = ()
        self.ready:      bool                                  = False
        self.last_access: float                                = time.time()


def _get_cached_model(model_path: str) -> Any | None:
    with _MODEL_CACHE_LOCK:
        model = _LOADED_MODEL_CACHE.get(model_path)
        if model is None:
            return None
        _LOADED_MODEL_CACHE.move_to_end(model_path)
        return model


def _cache_model(model_path: str, model: Any, validated: bool = False) -> None:
    with _MODEL_CACHE_LOCK:
        try:
            current_size = Path(model_path).stat().st_size
            if model_path in _MODEL_SIZE_CACHE:
                if _MODEL_SIZE_CACHE[model_path] != current_size:
                    _LOADED_MODEL_CACHE.pop(model_path, None)
                    _VALIDATED_MODEL_PATHS.discard(model_path)
                    logger.warning("stage=cache status=invalidated reason=size_mismatch path=%s", model_path)
            _MODEL_SIZE_CACHE[model_path] = current_size
        except OSError:
            logger.warning("exception_occurred", exc_info=True)
        if model_path in _LOADED_MODEL_CACHE:
            _LOADED_MODEL_CACHE.move_to_end(model_path)
        _LOADED_MODEL_CACHE[model_path] = model
        if validated:
            _VALIDATED_MODEL_PATHS.add(model_path)
        while len(_LOADED_MODEL_CACHE) > _CACHE_MAX_MODELS:
            evicted_path, _ = _LOADED_MODEL_CACHE.popitem(last=False)
            _VALIDATED_MODEL_PATHS.discard(evicted_path)
            _MODEL_SIZE_CACHE.pop(evicted_path, None)


def _clear_model_cache(model_path: str) -> None:
    with _MODEL_CACHE_LOCK:
        _LOADED_MODEL_CACHE.pop(model_path, None)
        _VALIDATED_MODEL_PATHS.discard(model_path)


def get_validated_cached_model(model_path: str) -> Any | None:
    """Return cached model only if it passed validate_onnx_model()."""
    with _MODEL_CACHE_LOCK:
        if model_path in _VALIDATED_MODEL_PATHS:
            return _LOADED_MODEL_CACHE.get(model_path)
    return None

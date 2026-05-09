from __future__ import annotations
"""Copyright (c) 2026 Omar Nady — Deployment Decision Engine. See LICENSE."""
"""core/model_analysis/validator.py — ONNX model validation pipeline."""
import logging
import multiprocessing
import time
import zipfile
from pathlib import Path
from typing import Any

from src.core.model_analysis.constants import (
    _MAX_MODEL_FILE_BYTES, _MAX_UNCOMPRESSED_BYTES, _MAX_PARSE_TIMEOUT_S,
    _MAX_PARSE_MEMORY_MB, _MAX_NODE_COUNT, _MAX_GRAPH_DEPTH,
    _MAX_ZIP_FILES, _MAX_ZIP_RATIO,
)
from src.core.model_analysis.cache import (
    _get_cached_model, _cache_model, _MODEL_CACHE_LOCK, _VALIDATED_MODEL_PATHS,
    _MODEL_INFLIGHT, _MODEL_INFLIGHT_LOCK, _MAX_MODEL_INFLIGHT, _InflightEntry,
)

logger = logging.getLogger(__name__)


def validate_onnx_model(model_path: str) -> tuple[bool, str | None, Any | None]:
    """Validate ONNX model with single-flight caching. Returns (valid, error, model)."""
    cached = _get_cached_model(model_path)
    if cached is not None:
        with _MODEL_CACHE_LOCK:
            if model_path in _VALIDATED_MODEL_PATHS:
                return True, None, cached

    is_owner = False
    entry: _InflightEntry
    with _MODEL_INFLIGHT_LOCK:
        entry = _MODEL_INFLIGHT.get(model_path)  # type: ignore[assignment]
        if entry is None:
            _c2 = _get_cached_model(model_path)
            if _c2 is not None:
                with _MODEL_CACHE_LOCK:
                    if model_path in _VALIDATED_MODEL_PATHS:
                        return True, None, _c2
            entry = _InflightEntry()
            _MODEL_INFLIGHT[model_path] = entry
            is_owner = True
            while len(_MODEL_INFLIGHT) > _MAX_MODEL_INFLIGHT:
                _k, _e = _MODEL_INFLIGHT.popitem(last=False)
                if not _e.ready:
                    _MODEL_INFLIGHT[_k] = _e
                    break
        else:
            entry.last_access = time.time()

    if not is_owner:
        entry.event.wait()
        if entry.error_type is not None:
            raise entry.error_type(*entry.error_args)
        return entry.result  # type: ignore[return-value]

    result = None
    error_type = None
    error_args: tuple = ()
    try:
        result = _do_validate_onnx_model(model_path)
    except Exception as exc:
        result, error_type, error_args = None, type(exc), exc.args
    finally:
        with _MODEL_INFLIGHT_LOCK:
            entry.result     = result
            entry.error_type = error_type
            entry.error_args = error_args
            entry.ready      = True
            entry.last_access = time.time()
            entry.event.set()

    if error_type is not None:
        raise error_type(*error_args)
    return result  # type: ignore[return-value]


def _do_validate_onnx_model(model_path: str) -> tuple[bool, str | None, Any | None]:
    try:
        import onnx
        try:
            file_size = Path(model_path).stat().st_size
        except OSError as exc:
            return False, f"Cannot stat model file: {exc}", None
        if file_size > _MAX_MODEL_FILE_BYTES:
            return False, f"Model file too large: {file_size} bytes", None
        real_fmt = _detect_real_format(model_path)
        if real_fmt == "pytorch_zip":
            safe, reason = _check_zip_bomb(model_path)
            if not safe:
                return False, reason, None
            return False, "PyTorch format not supported; upload an ONNX file", None
        if real_fmt == "unknown":
            return False, "Unrecognised model format; upload a valid ONNX file", None
        safe, reason = _check_zip_bomb(model_path)
        if not safe:
            return False, reason, None
        parse_ok, parse_reason = _run_subprocess_validation(model_path)
        if not parse_ok:
            return False, parse_reason, None
        _t0 = time.perf_counter()
        model = onnx.load(model_path)
        if (time.perf_counter() - _t0) > _MAX_PARSE_TIMEOUT_S:
            return False, "parse_timeout: main process load exceeded limit", None
        for initializer in getattr(getattr(model, "graph", None), "initializer", []) or []:
            if getattr(initializer, "data_location", 0) == 1:
                return False, "Model references external data files; not allowed", None
        _cache_model(model_path, model, validated=True)
        logger.info("stage=validate status=passed path=%s", model_path)
        return True, None, model
    except Exception as e:
        return False, f"Invalid ONNX model: {e}", None


def _run_subprocess_validation(model_path: str) -> tuple[bool, str | None]:
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    p = ctx.Process(target=_subprocess_validate_worker, args=(model_path, result_queue))
    p.start()
    p.join(timeout=_MAX_PARSE_TIMEOUT_S)
    if p.is_alive():
        p.kill(); p.join(); p.close()
        return False, "parse_timeout: validation exceeded time limit"
    p.close()
    try:
        result = result_queue.get_nowait()
    except Exception:
        return False, "parse_failed: subprocess returned no result"
    return (True, None) if result.get("ok") else (False, result.get("reason", "parse_failed"))


def _subprocess_validate_worker(model_path: str, result_queue) -> None:
    try:
        try:
            import resource as _resource
            _mem = _MAX_PARSE_MEMORY_MB * 1024 * 1024
            _resource.setrlimit(_resource.RLIMIT_AS, (_mem, _mem))
        except Exception:
            # setrlimit is unsupported on Windows and some container environments.
            # Failure here is intentionally silent: the subprocess will still validate
            # the model; the only risk is unconstrained memory, which is acceptable
            # for a short-lived validation subprocess.
            pass
        import onnx
        from onnx.checker import check_model
        model = onnx.load(model_path)
        for _init in getattr(getattr(model, "graph", None), "initializer", []) or []:
            if getattr(_init, "data_location", 0) == 1:
                result_queue.put({"ok": False, "reason": "external_data_reference"}); return
        graph = model.graph
        if not graph or not graph.node:
            result_queue.put({"ok": False, "reason": "empty_graph"}); return
        node_count = len(graph.node)
        if node_count > _MAX_NODE_COUNT:
            result_queue.put({"ok": False, "reason": f"graph_too_large: {node_count}"}); return
        check_model(model)
        result_queue.put({"ok": True, "reason": None})
    except Exception as exc:
        result_queue.put({"ok": False, "reason": str(exc)})


def _detect_real_format(path: str) -> str:
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        if header[:2] == b"PK":
            return "pytorch_zip"
        if header[:2] in (b"\x08\x01",b"\x0a\x0d",b"\x08\x06",b"\x08\x07",b"\x0a\x02",b"\x0a\x06"):
            return "onnx_or_tf"
        if header and (0x08 <= header[0] <= 0x7a):
            return "onnx_or_tf"
        return "unknown"
    except Exception:
        return "unknown"


def _check_zip_bomb(path: str) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as z:
            entries = z.infolist()
            if len(entries) > _MAX_ZIP_FILES:
                return False, f"zip_too_many_files: {len(entries)}"
            compressed_size = max(Path(path).stat().st_size, 1)
            total_uncompressed = 0
            for entry in entries:
                if ".." in entry.filename or entry.filename.startswith("/"):
                    return False, f"zip_path_traversal: {entry.filename!r}"
                total_uncompressed += entry.file_size
                if total_uncompressed > _MAX_UNCOMPRESSED_BYTES:
                    return False, f"zip_bomb: uncompressed {total_uncompressed}"
            if total_uncompressed / compressed_size > _MAX_ZIP_RATIO:
                return False, f"zip_bomb_ratio: ratio exceeds {_MAX_ZIP_RATIO}"
    except zipfile.BadZipFile:
        pass
    except Exception as exc:
        return False, f"zip_read_error: {exc}"
    return True, ""

"""
core/runtime/registry.py

Central registry for concrete runtime classes (Phase 1–3).

PHASE 1 — RUNTIME REGISTRY
    RUNTIMES: maps string keys → uninstantiated runtime classes.
    Classes are NOT instantiated here; instantiation happens in execution_loop.py.

PHASE 2 — FORMAT → RUNTIME MAPPING
    get_runtimes_for_model(): returns the ordered list of runtime keys
    applicable to a model path based on its file extension.

PHASE 3 — RUNTIME AVAILABILITY FILTER
    is_runtime_available(): dry-probes a runtime class to confirm the
    required backend (onnxruntime, torch, tflite_runtime, tensorrt…) is
    installed and capable of receiving a load() call.

NO business logic.  NO decision changes.  Import-safe — all heavy
dependencies are inside the runtime classes themselves.
"""
from __future__ import annotations

from src.core.runtime.runtimes import (
    ONNX_CPU,
    ONNX_CUDA,
    TENSORRT,
    TORCH,
    TFLITE,
)
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Phase 1: Registry ─────────────────────────────────────────────────────────
# Stores classes only.  Keys match the runtime name strings used throughout
# execution_loop.py and the output contract.

RUNTIMES: dict[str, type] = {
    "ONNX_CPU":  ONNX_CPU,
    "ONNX_CUDA": ONNX_CUDA,
    "TENSORRT":  TENSORRT,
    "TORCH":     TORCH,
    "TFLITE":    TFLITE,
}


# ── Phase 2: Format → runtime mapping ─────────────────────────────────────────

def get_runtimes_for_model(path: str) -> list[str]:
    """
    Return the ordered list of runtime keys applicable to *path*.

    Order within each format is significant: ONNX_CPU always comes first
    so the availability filter can guarantee at least one viable candidate
    before attempting GPU providers.

    Raises RuntimeError for unrecognised extensions.
    """
    if path.endswith(".onnx"):
        assert "__probe__" not in path, f"Probe path must not enter runtime routing: {path!r}"
        return ["ONNX_CPU", "ONNX_CUDA", "TENSORRT"]
    if path.endswith(".pt") or path.endswith(".pth"):
        assert "__probe__" not in path, f"Probe path must not enter runtime routing: {path!r}"
        return ["TORCH"]
    if path.endswith(".tflite"):
        assert "__probe__" not in path, f"Probe path must not enter runtime routing: {path!r}"
        return ["TFLITE"]
    raise RuntimeError(
        f"Unsupported model format: '{path}'. "
        "Supported extensions: .onnx, .pt, .pth, .tflite"
    )


# ── Phase 3: Availability filter ──────────────────────────────────────────────

def is_runtime_available(name: str) -> bool:
    """
    Confirm backend dependencies for runtime *name* are installed.

    Calls check_available() on the runtime class — a pure import-level
    dependency check.  No file path is used.  No model is executed.

    Returns True  → all required packages are importable.
    Returns False → a hard dependency (ImportError) is missing.
    """
    if name not in RUNTIMES:
        logger.warning(
            "runtime_availability_unknown_key",
            extra={"event": "runtime_availability_unknown_key", "runtime": name},
        )
        return False

    runtime_cls = RUNTIMES[name]

    try:
        runtime = runtime_cls()
        runtime.check_available()
        return True
    except ImportError:
        logger.info(
            "runtime_unavailable",
            extra={"event": "runtime_unavailable", "runtime": name,
                   "reason": "ImportError — dependency not installed"},
        )
        return False
    except Exception:
        # Non-import exception from check_available (e.g. CUDA init) —
        # treat as available since the package is importable.
        return True

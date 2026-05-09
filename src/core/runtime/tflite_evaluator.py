"""
core/runtime/tflite_evaluator.py

Pure TFLite runtime benchmark function.
No IO side-effects beyond the model file. No state. No imports from services or api.
"""
from __future__ import annotations

import time
from typing import Any

from src.core.logging_config import get_logger

logger = get_logger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

# Prefer the lightweight standalone runtime; fall back to full TensorFlow.
try:
    from tflite_runtime import interpreter as _tflite_lib

    _Interpreter = _tflite_lib.Interpreter
    _load_delegate = _tflite_lib.load_delegate
    _TFLITE_SOURCE = "tflite_runtime"
except ImportError:
    try:
        from tensorflow.lite.python import interpreter as _tflite_lib  # type: ignore[no-redef]

        _Interpreter = _tflite_lib.Interpreter
        _load_delegate = _tflite_lib.load_delegate
        _TFLITE_SOURCE = "tensorflow.lite"
    except ImportError:
        _Interpreter = None  # type: ignore[assignment,misc]
        _load_delegate = None  # type: ignore[assignment]
        _TFLITE_SOURCE = None

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


_BENCHMARK_RUNS = 10
_WARMUP_RUNS    = 3
_WALL_CAP_S     = 2.0

_DEFAULT_STRESS: dict[str, Any] = {
    "enabled":          False,
    "runs":             0,
    "latency_avg_ms":   None,
    "latency_p95_ms":   None,
    "peak_memory_mb":   None,
    "memory_growth_mb": None,
    "memory_stability": "STABLE",
}


def _resolve_tflite_shape(raw_shape: Any) -> list[int]:
    """Replace dynamic (-1) dimensions with safe concrete values."""
    if raw_shape is None or len(raw_shape) == 0:
        return [1, 3, 224, 224]
    ndim = len(raw_shape)
    spatial_default = 224 if ndim == 4 else 128
    resolved: list[int] = []
    for i, dim in enumerate(raw_shape):
        dim_int = int(dim)
        if i == 0:
            # Batch axis — always force to 1.
            resolved.append(1)
        elif dim_int > 0:
            resolved.append(dim_int)
        else:
            # Dynamic dim (-1 or 0) — use spatial default.
            resolved.append(spatial_default)
    return resolved


def _build_dummy_inputs(
    input_details: list[dict[str, Any]],
) -> dict[int, Any]:
    """
    Build a {tensor_index: ndarray} map for every model input.

    TFLite ``input_details`` entries carry a numpy dtype directly,
    so no string→dtype translation is required (unlike ONNX).
    """
    rng = np.random.default_rng(0)
    dummy: dict[int, Any] = {}

    for detail in input_details:
        idx   = int(detail["index"])
        shape = _resolve_tflite_shape(detail.get("shape"))
        dtype = detail.get("dtype", np.float32)

        if np.issubdtype(dtype, np.floating):
            dummy[idx] = rng.random(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            dummy[idx] = np.zeros(shape, dtype=dtype)
        elif dtype == np.bool_:
            dummy[idx] = np.zeros(shape, dtype=np.bool_)
        else:
            # Fallback for exotic dtypes (e.g. string tensors in TFLite).
            dummy[idx] = np.zeros(shape, dtype=np.float32)

    return dummy


def evaluate_tflite_runtime(
    model_path: str,
    use_gpu_delegate: bool = False,
    stress_runs: int = 0,
) -> dict[str, Any]:
    """
    Safe runtime benchmark for a .tflite model.

    Returns a dict with keys:
        success, latency_avg_ms, latency_p95_ms, memory_mb, error,
        benchmark_runs, stress_test

    Notes
    -----
    * ``stress_runs`` is accepted for API symmetry but has no effect;
      ``stress_test`` is always returned as ``_DEFAULT_STRESS``.
    * ``use_gpu_delegate=True`` attempts to load the GPU delegate.
      If the delegate is unavailable on the current platform, the
      benchmark falls back to the CPU rather than failing outright.
    * Never raises — all exceptions are caught and returned as
      ``success=False`` payloads.
    """
    # ------------------------------------------------------------------ #
    # Guard: runtime availability                                          #
    # ------------------------------------------------------------------ #
    if np is None or _Interpreter is None:
        missing = []
        if np is None:
            missing.append("numpy")
        if _Interpreter is None:
            missing.append(
                "tflite_runtime  (pip install tflite-runtime)  "
                "or  tensorflow  (pip install tensorflow)"
            )
        return {
            "success":        False,
            "latency_avg_ms": None,
            "latency_p95_ms": None,
            "memory_mb":      None,
            "error":          f"Required packages not available: {', '.join(missing)}",
            "benchmark_runs": 0,
            "stress_test":    _DEFAULT_STRESS,
        }

    # ------------------------------------------------------------------ #
    # Benchmark                                                            #
    # ------------------------------------------------------------------ #
    try:
        # Build interpreter, optionally with GPU delegate. ------ #
        experimental_delegates = []

        if use_gpu_delegate and _load_delegate is not None:
            try:
                gpu_delegate = _load_delegate("libGLESv2.so")
                experimental_delegates = [gpu_delegate]
            except Exception as delegate_exc:
                # GPU unavailable on this host — fall back silently to CPU.
                logger.warning(
                    "tflite_gpu_delegate_unavailable",
                    extra={
                        "event":  "tflite_gpu_delegate_unavailable",
                        "reason": str(delegate_exc),
                        "action": "falling_back_to_cpu",
                    },
                )

        interp = _Interpreter(
            model_path=model_path,
            experimental_delegates=experimental_delegates or None,
        )
        interp.allocate_tensors()

        input_details  = interp.get_input_details()
        dummy_inputs   = _build_dummy_inputs(input_details)

        logger.info(
            "tflite_benchmark_start",
            extra={
                "event":        "tflite_benchmark_start",
                "model_path":   model_path,
                "source":       _TFLITE_SOURCE,
                "use_gpu":      use_gpu_delegate,
                "num_inputs":   len(input_details),
            },
        )

        # Warmup ------------------------------------------------- #
        for _ in range(_WARMUP_RUNS):
            for idx, arr in dummy_inputs.items():
                interp.set_tensor(idx, arr)
            interp.invoke()

        # Timed benchmark ---------------------------------------- #
        rss_before = 0
        if psutil is not None:
            rss_before = psutil.Process().memory_info().rss

        timings_ms: list[float] = []
        wall_start = time.perf_counter()

        for _ in range(_BENCHMARK_RUNS):
            for idx, arr in dummy_inputs.items():
                interp.set_tensor(idx, arr)
            t0 = time.perf_counter()
            interp.invoke()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
            if (time.perf_counter() - wall_start) >= _WALL_CAP_S:
                break

        memory_mb = 0.0
        if psutil is not None:
            rss_after = psutil.Process().memory_info().rss
            memory_mb = float(max(0, rss_after - rss_before)) / (1024.0 * 1024.0)

        latency_avg_ms = float(sum(timings_ms) / len(timings_ms))
        timings_sorted = sorted(timings_ms)
        p95_idx        = max(0, int(len(timings_sorted) * 0.95) - 1)
        latency_p95_ms = float(timings_sorted[p95_idx])

        logger.info(
            "tflite_benchmark_complete",
            extra={
                "event":           "tflite_benchmark_complete",
                "model_path":      model_path,
                "source":          _TFLITE_SOURCE,
                "use_gpu":         use_gpu_delegate,
                "latency_avg_ms":  round(latency_avg_ms, 3),
                "latency_p95_ms":  round(latency_p95_ms, 3),
                "memory_mb":       round(memory_mb, 2),
                "runs_completed":  len(timings_ms),
            },
        )

        return {
            "success":        True,
            "latency_avg_ms": latency_avg_ms,
            "latency_p95_ms": latency_p95_ms,
            "memory_mb":      memory_mb,
            "error":          None,
            "benchmark_runs": len(timings_ms),
            "stress_test":    _DEFAULT_STRESS,
        }

    except Exception as bench_exc:
        logger.error(
            "tflite_benchmark_failed",
            extra={
                "event":      "tflite_benchmark_failed",
                "model_path": model_path,
                "source":     _TFLITE_SOURCE,
                "use_gpu":    use_gpu_delegate,
                "reason":     str(bench_exc),
            },
        )
        return {
            "success":        False,
            "latency_avg_ms": None,
            "latency_p95_ms": None,
            "memory_mb":      None,
            "error":          f"tflite_session_failed: {bench_exc}",
            "benchmark_runs": 0,
            "stress_test":    _DEFAULT_STRESS,
        }

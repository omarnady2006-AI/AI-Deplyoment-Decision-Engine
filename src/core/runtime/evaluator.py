"""
core/runtime/evaluator.py

Pure ONNX runtime benchmark function.
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

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

try:
    import onnx
except ImportError:
    onnx = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


_BENCHMARK_RUNS = 10
_WARMUP_RUNS    = 3
_WALL_CAP_S     = 2.0

_ELEM_BYTES: dict[int, int] = {
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8,
    9: 1, 10: 2, 11: 8, 12: 4, 13: 8,
}

_DEFAULT_STRESS: dict[str, Any] = {
    "enabled": False,
    "runs": 0,
    "latency_avg_ms": None,
    "latency_p95_ms": None,
    "peak_memory_mb": None,
    "memory_growth_mb": None,
    "memory_stability": "STABLE",
}


def _graph_memory_mb(model_path: str) -> float:
    """Sum weight tensor sizes from the ONNX protobuf."""
    if onnx is not None:
        try:
            g = onnx.load(model_path).graph
            total = 0
            for init in g.initializer:
                esz = _ELEM_BYTES.get(int(init.data_type), 4)
                n = 1
                for d in init.dims:
                    n *= max(1, int(d))
                total += n * esz
            return float(total) / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    return 0.0


def _resolve_shape(raw_shape: Any) -> list[int]:
    """Replace dynamic dimensions with safe concrete values."""
    if not raw_shape:
        return [1, 3, 224, 224]
    ndim = len(raw_shape)
    spatial_default = 224 if ndim == 4 else 128
    resolved: list[int] = []
    for i, dim in enumerate(raw_shape):
        if i == 0:
            resolved.append(1)
        elif isinstance(dim, int) and dim > 0:
            resolved.append(int(dim))
        else:
            resolved.append(spatial_default)
    return resolved


def _ort_type_to_numpy(type_str: str):
    if np is not None:
        _MAP = {
            "tensor(float)":   np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)":  np.float64,
            "tensor(int64)":   np.int64,
            "tensor(int32)":   np.int32,
            "tensor(int16)":   np.int16,
            "tensor(int8)":    np.int8,
            "tensor(uint64)":  np.uint64,
            "tensor(uint32)":  np.uint32,
            "tensor(uint16)":  np.uint16,
            "tensor(uint8)":   np.uint8,
            "tensor(bool)":    np.bool_,
            "tensor(string)":  None,
        }
        return _MAP.get(type_str, np.float32)
    return None


def _run_benchmark(
    model_path: str,
    provider: str,
    stress_runs: int = 0,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Inner benchmark implementation. All return paths are validated by the
    public evaluate_onnx_runtime wrapper before the result leaves this module.

    Returns a dict with keys:
        success, latency_avg_ms, latency_p95_ms, memory_mb, error,
        benchmark_runs, stress_test
    """
    unavailable_result = {
        "success": False,
        "latency_avg_ms": None,
        "latency_p95_ms": None,
        "memory_mb": None,
        "error": "numpy or onnxruntime not available",
        "benchmark_runs": 0,
        "stress_test": _DEFAULT_STRESS,
    }

    if np is None or ort is None:
        return unavailable_result

    try:
        available_providers = ort.get_available_providers()
    except Exception:
        available_providers = []

    if provider not in available_providers:
        return {
            "success": False,
            "latency_avg_ms": None,
            "latency_p95_ms": None,
            "memory_mb": None,
            "error": (
                f"Provider '{provider}' is unavailable. "
                f"Available providers: {available_providers}"
            ),
            "benchmark_runs": 0,
            "stress_test": _DEFAULT_STRESS,
        }

    try:
        session = ort.InferenceSession(model_path, providers=[provider])
        inputs  = session.get_inputs()

        dummy_inputs: dict[str, Any] = {}
        rng = np.random.default_rng(0)

        for inp in inputs:
            raw_shape = getattr(inp, "shape", None) or []
            dtype_ort = getattr(inp, "type", "tensor(float)")
            shape     = _resolve_shape(raw_shape)
            dtype_np  = _ort_type_to_numpy(dtype_ort)

            if dtype_np is None:
                dummy_inputs[inp.name] = np.array(
                    [""] * int(np.prod(shape)), dtype=object
                ).reshape(shape)
            elif np.issubdtype(dtype_np, np.floating):
                dummy_inputs[inp.name] = rng.random(shape).astype(dtype_np)
            elif np.issubdtype(dtype_np, np.integer):
                dummy_inputs[inp.name] = np.zeros(shape, dtype=dtype_np)
            elif dtype_np is np.bool_:
                dummy_inputs[inp.name] = np.zeros(shape, dtype=np.bool_)
            else:
                dummy_inputs[inp.name] = np.zeros(shape, dtype=np.float32)

        logger.info(
            "onnx_runtime_benchmark_start",
            extra={"event": "onnx_runtime_benchmark_start", "provider": provider},
        )

        for _ in range(_WARMUP_RUNS):
            session.run(None, dummy_inputs)

        rss_before = 0
        if psutil is not None:
            rss_before = psutil.Process().memory_info().rss

        timings_ms: list[float] = []
        wall_start = time.perf_counter()
        for _ in range(_BENCHMARK_RUNS):
            t0 = time.perf_counter()
            session.run(None, dummy_inputs)
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

        if memory_mb < 1.0:
            memory_mb = max(memory_mb, _graph_memory_mb(model_path))

        logger.info(
            "onnx_runtime_benchmark_complete",
            extra={
                "event": "onnx_runtime_benchmark_complete",
                "provider": provider,
                "latency_avg_ms": round(latency_avg_ms, 3),
                "latency_p95_ms": round(latency_p95_ms, 3),
                "memory_mb": round(memory_mb, 2),
                "runs_completed": len(timings_ms),
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
        # FIXED: execution failed — must report success=False.
        # Returning success=True here was the root-cause bug: it caused the pipeline
        # to treat an unexecuted model (e.g. a .pt file loaded against ONNX runtime)
        # as a successfully benchmarked one, zeroing out all risk signals.
        logger.error(
            "onnx_runtime_benchmark_failed",
            extra={"event": "onnx_runtime_benchmark_failed", "provider": provider,
                   "reason": str(bench_exc)},
        )
        return {
            "success":        False,     # FIXED: was True
            "latency_avg_ms": None,      # no execution → no latency
            "latency_p95_ms": None,
            "memory_mb":      None,      # FIXED: was _graph_memory_mb() — nulled to satisfy
                                         # RuntimeEvaluation contract (None iff success=False)
            "error":          f"onnx_session_failed: {bench_exc}",
            "benchmark_runs": 0,
            "stress_test":    _DEFAULT_STRESS,
        }


# ── Invariant gate ────────────────────────────────────────────────────────────

def _assert_result_invariant(result: dict[str, Any], provider: str) -> None:
    """
    Fail-fast contract enforced at every exit from evaluate_onnx_runtime.

    Rules:
      success=True  → latency_avg_ms must be a real finite number
      success=False → latency_avg_ms must be None

    A violation here means a new code path was added to _run_benchmark without
    honouring the contract — catch it at the source before it corrupts
    response_evaluations, telemetry, or ML training signals.
    """
    success = result.get("success", False)
    latency = result.get("latency_avg_ms")

    if success:
        assert latency is not None, (
            f"[evaluator invariant] success=True but latency_avg_ms is None "
            f"for provider '{provider}' — execution was not actually measured"
        )
    else:
        assert latency is None, (
            f"[evaluator invariant] success=False but latency_avg_ms={latency!r} "
            f"for provider '{provider}' — a failed execution must not carry latency"
        )


def evaluate_onnx_runtime(
    model_path: str,
    provider: str,
    stress_runs: int = 0,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    Safe runtime benchmark for an ONNX model on a given execution provider.

    Public entry point.  Delegates to _run_benchmark and enforces the
    execution_success / latency_avg_ms coherence invariant before returning,
    so every caller is guaranteed a contract-valid result dict.

    Returns a dict with keys:
        success, latency_avg_ms, latency_p95_ms, memory_mb, error,
        benchmark_runs, stress_test
    """
    result = _run_benchmark(model_path, provider, stress_runs, batch_size)
    _assert_result_invariant(result, provider)
    return result

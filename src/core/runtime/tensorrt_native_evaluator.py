"""
core/runtime/tensorrt_native_evaluator.py

Native TensorRT serialized-engine benchmark function.
No IO side-effects beyond the engine file. No state. No imports from services or api.

Same return contract as evaluator.py:
    success, latency_avg_ms, latency_p95_ms, memory_mb, error,
    benchmark_runs, stress_test
"""
from __future__ import annotations

import time
from typing import Any

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — all guarded; absence → graceful failure path.
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:                          # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import tensorrt as trt                   # type: ignore[import]
except ImportError:                          # pragma: no cover
    trt = None  # type: ignore[assignment]

try:
    import pycuda.driver as cuda             # type: ignore[import]
except ImportError:                          # pragma: no cover
    cuda = None  # type: ignore[assignment]

try:
    import pycuda.autoinit  # noqa: F401    # side-effect: initialises CUDA context
except ImportError:                          # pragma: no cover
    pass

try:
    import psutil
except ImportError:                          # pragma: no cover
    psutil = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Benchmark constants — mirror evaluator.py exactly.
# ---------------------------------------------------------------------------
_WARMUP_RUNS    = 3
_BENCHMARK_RUNS = 10
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

# Install hint surfaced to callers when the CUDA/TRT stack is absent.
_INSTALL_HINT = (
    "tensorrt and pycuda are required for native TensorRT benchmarking. "
    "Install with: pip install tensorrt pycuda"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_binding_shape(raw_shape: Any) -> list[int]:
    """
    Coerce a TensorRT binding shape (possibly containing -1 dynamic dims)
    into a fully concrete shape safe for buffer allocation.
    """
    if not raw_shape:
        return [1, 3, 224, 224]
    ndim          = len(raw_shape)
    spatial_default = 224 if ndim == 4 else 128
    resolved: list[int] = []
    for i, dim in enumerate(raw_shape):
        d = int(dim)
        if i == 0:
            resolved.append(1)          # batch dim → 1
        elif d > 0:
            resolved.append(d)
        else:
            resolved.append(spatial_default)
    return resolved


def _volume(shape: list[int]) -> int:
    """Element count for a shape list."""
    result = 1
    for d in shape:
        result *= max(1, d)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_tensorrt_native_runtime(
    model_path: str,
    stress_runs: int = 0,
) -> dict[str, Any]:
    """
    Safe native TensorRT engine benchmark.

    Loads a serialized TensorRT engine from *model_path*, allocates host and
    device buffers for every binding, runs _WARMUP_RUNS warm-up passes followed
    by up to _BENCHMARK_RUNS timed inference passes (wall-capped at _WALL_CAP_S),
    and reports latency and RSS-delta memory.

    Device buffers are always freed in a ``finally`` block regardless of the
    execution path.  This function never raises.

    Args:
        model_path:  Path to a serialized TensorRT engine file (.trt / .engine).
        stress_runs: Reserved for future stress-test integration (unused here;
                     mirrors the evaluator.py signature).

    Returns:
        dict with keys:
            success         – bool
            latency_avg_ms  – float | None
            latency_p95_ms  – float | None
            memory_mb       – float | None  (RSS delta; None iff success=False)
            error           – str  | None
            benchmark_runs  – int
            stress_test     – dict  (always _DEFAULT_STRESS for this evaluator)
    """
    # ------------------------------------------------------------------
    # Guard: require tensorrt + pycuda + numpy.
    # ------------------------------------------------------------------
    if trt is None or cuda is None or np is None:
        missing = [
            lib for lib, mod in (("tensorrt", trt), ("pycuda", cuda), ("numpy", np))
            if mod is None
        ]
        return {
            "success":        False,
            "latency_avg_ms": None,
            "latency_p95_ms": None,
            "memory_mb":      None,
            "error":          f"missing dependencies {missing} — {_INSTALL_HINT}",
            "benchmark_runs": 0,
            "stress_test":    _DEFAULT_STRESS,
        }

    # Buffers allocated on the device; collected here so the finally block
    # can always free them even if allocation partially failed.
    device_buffers: list[Any] = []
    stream         = None

    try:
        # ------------------------------------------------------------------
        # 1. Deserialise the engine.
        # ------------------------------------------------------------------
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f:
            engine_data = f.read()
        engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_data)

        if engine is None:
            return {
                "success":        False,
                "latency_avg_ms": None,
                "latency_p95_ms": None,
                "memory_mb":      None,
                "error":          "trt.Runtime.deserialize_cuda_engine returned None",
                "benchmark_runs": 0,
                "stress_test":    _DEFAULT_STRESS,
            }

        # ------------------------------------------------------------------
        # 2. Create execution context.
        # ------------------------------------------------------------------
        context = engine.create_execution_context()

        # ------------------------------------------------------------------
        # 3. Allocate host + device buffers for all bindings.
        # ------------------------------------------------------------------
        num_bindings  = engine.num_bindings
        host_buffers: list[Any] = []
        bindings:     list[Any] = []

        for i in range(num_bindings):
            raw_shape   = tuple(engine.get_binding_shape(i))
            shape       = _resolve_binding_shape(list(raw_shape))
            n_elements  = _volume(shape)
            # TensorRT engines are typically float32; default to float32.
            host_buf    = np.zeros(n_elements, dtype=np.float32)
            host_buffers.append(host_buf)

            dev_buf = cuda.mem_alloc(host_buf.nbytes)
            device_buffers.append(dev_buf)
            bindings.append(int(dev_buf))

        # ------------------------------------------------------------------
        # 4. Copy input buffers to device (inputs contain zeros by design).
        # ------------------------------------------------------------------
        stream = cuda.Stream()

        def _h2d() -> None:
            for i in range(num_bindings):
                if engine.binding_is_input(i):
                    cuda.memcpy_htod_async(
                        device_buffers[i],
                        host_buffers[i],
                        stream,
                    )

        def _run() -> None:
            _h2d()
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()

        # ------------------------------------------------------------------
        # 5. Log start.
        # ------------------------------------------------------------------
        logger.info(
            "tensorrt_native_benchmark_start",
            extra={
                "event":      "tensorrt_native_benchmark_start",
                "model_path": model_path,
                "bindings":   num_bindings,
            },
        )

        # ------------------------------------------------------------------
        # 6. Warm-up passes.
        # ------------------------------------------------------------------
        for _ in range(_WARMUP_RUNS):
            _run()

        # ------------------------------------------------------------------
        # 7. RSS baseline.
        # ------------------------------------------------------------------
        rss_before = 0
        if psutil is not None:
            rss_before = psutil.Process().memory_info().rss

        # ------------------------------------------------------------------
        # 8. Timed benchmark passes (wall-capped).
        # ------------------------------------------------------------------
        timings_ms: list[float] = []
        wall_start = time.perf_counter()

        for _ in range(_BENCHMARK_RUNS):
            t0 = time.perf_counter()
            _run()
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
            if (time.perf_counter() - wall_start) >= _WALL_CAP_S:
                break

        # ------------------------------------------------------------------
        # 9. RSS delta.
        # ------------------------------------------------------------------
        memory_mb = 0.0
        if psutil is not None:
            rss_after = psutil.Process().memory_info().rss
            memory_mb = float(max(0, rss_after - rss_before)) / (1024.0 * 1024.0)

        # ------------------------------------------------------------------
        # 10. Derive statistics.
        # ------------------------------------------------------------------
        latency_avg_ms = float(sum(timings_ms) / len(timings_ms))
        timings_sorted = sorted(timings_ms)
        p95_idx        = max(0, int(len(timings_sorted) * 0.95) - 1)
        latency_p95_ms = float(timings_sorted[p95_idx])

        logger.info(
            "tensorrt_native_benchmark_complete",
            extra={
                "event":           "tensorrt_native_benchmark_complete",
                "model_path":      model_path,
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
            "tensorrt_native_benchmark_failed",
            extra={
                "event":      "tensorrt_native_benchmark_failed",
                "model_path": model_path,
                "reason":     str(bench_exc),
            },
        )
        return {
            "success":        False,
            "latency_avg_ms": None,
            "latency_p95_ms": None,
            "memory_mb":      None,
            "error":          f"tensorrt_native_session_failed: {bench_exc}",
            "benchmark_runs": 0,
            "stress_test":    _DEFAULT_STRESS,
        }

    finally:
        # ------------------------------------------------------------------
        # Always free every device buffer that was successfully allocated.
        # cuda.DeviceAllocation.free() is idempotent; safe to call even if
        # the inference loop was never reached.
        # ------------------------------------------------------------------
        for dev_buf in device_buffers:
            try:
                dev_buf.free()
            except Exception:          # pragma: no cover
                pass                   # best-effort; never propagate

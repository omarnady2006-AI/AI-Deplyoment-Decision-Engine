"""
core/runtime/openvino_evaluator.py

OpenVINO IR (.xml + .bin) runtime benchmark function.
No IO side-effects beyond the model file.  No state.  No imports from services or api.

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
# Optional imports — all guarded; absence → graceful failure path.
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:                              # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from openvino.runtime import Core as _OVCore  # type: ignore[import]
except ImportError:                              # pragma: no cover
    _OVCore = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:                              # pragma: no cover
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

_INSTALL_HINT = (
    "openvino is required for OpenVINO IR benchmarking. "
    "Install with: pip install openvino"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_ov_shape(partial_shape: Any) -> list[int]:
    """
    Convert an OpenVINO PartialShape to a fully-concrete list[int].

    Rules (mirror evaluator._resolve_shape):
      dim 0  → 1            (batch)
      dynamic dim elsewhere → 224 for 4-D inputs, 128 otherwise
      static dim            → its integer value
    """
    try:
        dims = list(partial_shape)
    except Exception:
        return [1, 3, 224, 224]

    if not dims:
        return [1, 3, 224, 224]

    ndim            = len(dims)
    spatial_default = 224 if ndim == 4 else 128
    resolved: list[int] = []

    for i, dim in enumerate(dims):
        if i == 0:
            resolved.append(1)
            continue
        # OpenVINO Dimension: .is_dynamic is True when the dim is -1 / symbolic.
        try:
            is_dyn = dim.is_dynamic
        except AttributeError:
            # Scalar integer — older API returns bare int-like objects.
            is_dyn = (int(dim) < 1)

        if is_dyn:
            resolved.append(spatial_default)
        else:
            resolved.append(max(1, int(dim)))

    return resolved


def _ov_dtype(element_type: Any) -> Any:
    """
    Map an OpenVINO element type to a numpy dtype.

    Float / bfloat16 types → float32 (safest general-purpose input).
    Integer types          → int32.
    Unknown                → float32 (defensive fallback).
    """
    if np is None:
        return None
    try:
        # element_type.is_real() is True for f16 / f32 / f64 / bf16.
        if element_type.is_real():
            return np.float32
    except AttributeError:
        pass
    return np.int32


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_openvino_runtime(
    model_path:  str,
    device_name: str = "CPU",
    stress_runs: int = 0,
) -> dict[str, Any]:
    """
    Safe OpenVINO IR benchmark.

    Loads an IR model from *model_path* (.xml; the matching .bin is discovered
    automatically by OpenVINO), compiles it on *device_name*, builds zero-filled
    numpy inputs for all network inputs, runs _WARMUP_RUNS passes then up to
    _BENCHMARK_RUNS timed passes (wall-capped at _WALL_CAP_S), and reports
    latency and RSS-delta memory.

    GPU path checks ``Core().available_devices`` first and returns
    ``success=False`` immediately if "GPU" is absent.

    This function never raises.

    Args:
        model_path:  Absolute path to the OpenVINO IR .xml file.
        device_name: OpenVINO device string — "CPU" (default) or "GPU".
        stress_runs: Reserved; mirrors the evaluator.py signature.

    Returns:
        dict with keys:
            success         – bool
            latency_avg_ms  – float | None
            latency_p95_ms  – float | None
            memory_mb       – float | None  (None iff success=False)
            error           – str  | None
            benchmark_runs  – int
            stress_test     – dict  (always _DEFAULT_STRESS for this evaluator)
    """
    # ------------------------------------------------------------------
    # Guard: require openvino + numpy.
    # ------------------------------------------------------------------
    if _OVCore is None or np is None:
        missing = [
            lib for lib, mod in (("openvino", _OVCore), ("numpy", np))
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

    try:
        # ------------------------------------------------------------------
        # 1. Instantiate Core and check device availability.
        # ------------------------------------------------------------------
        core = _OVCore()

        if device_name == "GPU":
            try:
                available_devices: list[str] = list(core.available_devices)
            except Exception:
                available_devices = []

            if "GPU" not in available_devices:
                return {
                    "success":        False,
                    "latency_avg_ms": None,
                    "latency_p95_ms": None,
                    "memory_mb":      None,
                    "error":          (
                        "GPU device not available in OpenVINO. "
                        f"Available devices: {available_devices}"
                    ),
                    "benchmark_runs": 0,
                    "stress_test":    _DEFAULT_STRESS,
                }

        # ------------------------------------------------------------------
        # 2. Load and compile the model.
        #    core.read_model() auto-discovers the .bin from the .xml path.
        # ------------------------------------------------------------------
        model    = core.read_model(model=model_path)
        compiled = core.compile_model(model, device_name)

        # ------------------------------------------------------------------
        # 3. Create infer request.
        # ------------------------------------------------------------------
        infer_request = compiled.create_infer_request()

        # ------------------------------------------------------------------
        # 4. Build dummy inputs for all network inputs.
        # ------------------------------------------------------------------
        dummy_inputs: dict[Any, Any] = {}

        for inp in compiled.inputs:
            try:
                partial_shape = inp.partial_shape
            except AttributeError:
                partial_shape = inp.get_partial_shape()

            try:
                elem_type = inp.element_type
            except AttributeError:
                elem_type = inp.get_element_type()

            shape    = _resolve_ov_shape(partial_shape)
            dtype_np = _ov_dtype(elem_type)
            if dtype_np is None:
                dtype_np = np.float32

            dummy_inputs[inp] = np.zeros(shape, dtype=dtype_np)

        # ------------------------------------------------------------------
        # 5. Log start.
        # ------------------------------------------------------------------
        logger.info(
            "openvino_benchmark_start",
            extra={
                "event":       "openvino_benchmark_start",
                "model_path":  model_path,
                "device_name": device_name,
                "num_inputs":  len(dummy_inputs),
            },
        )

        # ------------------------------------------------------------------
        # 6. Warm-up passes.
        # ------------------------------------------------------------------
        for _ in range(_WARMUP_RUNS):
            infer_request.infer(inputs=dummy_inputs)

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
            infer_request.infer(inputs=dummy_inputs)
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
            "openvino_benchmark_complete",
            extra={
                "event":           "openvino_benchmark_complete",
                "model_path":      model_path,
                "device_name":     device_name,
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
            "openvino_benchmark_failed",
            extra={
                "event":       "openvino_benchmark_failed",
                "model_path":  model_path,
                "device_name": device_name,
                "reason":      str(bench_exc),
            },
        )
        return {
            "success":        False,
            "latency_avg_ms": None,
            "latency_p95_ms": None,
            "memory_mb":      None,
            "error":          f"openvino_session_failed: {bench_exc}",
            "benchmark_runs": 0,
            "stress_test":    _DEFAULT_STRESS,
        }

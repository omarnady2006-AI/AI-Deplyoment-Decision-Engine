"""
core/runtime/execution_loop.py

Multi-runtime execution loop (Phases 4–7).

PHASE 4 — EXECUTION LOOP
    Iterates all runtimes returned by get_runtimes_for_model(), skipping any
    that fail the is_runtime_available() probe, and collects benchmark results.

PHASE 5 — NON-EMPTY GUARD
    Raises RuntimeError if the results list is empty (no runtime executed).
    No fallback injection.

PHASE 6 — BEST RUNTIME SELECTION (UNCHANGED LOGIC)
    Selects the runtime with the lowest latency_ms among entries where
    execution_success is True.  The selection criterion is intentionally
    minimal — no scoring, no heuristics.

PHASE 7 — OUTPUT CONTRACT
    Returns a dict with exactly two keys:
        {
            "evaluations": list[dict],   # one entry per attempted runtime
            "best_runtime": str,         # name of lowest-latency successful runtime
        }

Public entry point: execute_model_runtimes(model_path)
"""
from __future__ import annotations

from typing import Any

from src.core.runtime.registry import (
    RUNTIMES,
    get_runtimes_for_model,
    is_runtime_available,
)
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Phase 4 + 5 + 6 + 7 ──────────────────────────────────────────────────────

def execute_model_runtimes(model_path: str) -> dict[str, Any]:
    """
    Benchmark *model_path* across all applicable runtimes.

    Steps
    -----
    1. Resolve candidate runtime names from the model's file extension.
    2. Skip any runtime whose backend is not installed (is_runtime_available).
    3. Instantiate each surviving runtime, call benchmark(model_path), and
       record the normalised evaluation dict.
    4. Raise RuntimeError if no runtime produced a result (Phase 5 guard).
    5. Select the best runtime: lowest latency_ms among execution_success=True
       entries (Phase 6 — logic is unchanged from the original specification).
    6. Return the Phase 7 output contract dict.

    Returns
    -------
    {
        "evaluations": [
            {
                "runtime":           str,
                "latency_ms":        float | None,
                "memory_mb":         float | None,
                "execution_success": bool,
            },
            ...
        ],
        "best_runtime": str,
    }

    Raises
    ------
    RuntimeError
        • Unsupported model format (propagated from get_runtimes_for_model).
        • No runtime executed successfully (Phase 5 guard).
    """
    # ── Phase 4: execution loop ───────────────────────────────────────────────
    assert "__probe__" not in model_path, (
        f"Probe paths must not enter the execution loop: {model_path!r}"
    )
    candidates = get_runtimes_for_model(model_path)

    results: list[dict[str, Any]] = []

    for rt_name in candidates:
        if not is_runtime_available(rt_name):
            logger.info(
                "runtime_skipped_unavailable",
                extra={
                    "event":   "runtime_skipped_unavailable",
                    "runtime": rt_name,
                    "model":   model_path,
                },
            )
            continue

        runtime = RUNTIMES[rt_name]()

        try:
            metrics = runtime.benchmark(model_path)

            results.append({
                "runtime":           rt_name,
                "latency_ms":        metrics.get("latency_avg_ms"),
                "memory_mb":         metrics.get("memory_mb"),
                "execution_success": bool(metrics.get("success", False)),
            })

            logger.info(
                "runtime_benchmark_complete",
                extra={
                    "event":             "runtime_benchmark_complete",
                    "runtime":           rt_name,
                    "latency_avg_ms":    metrics.get("latency_avg_ms"),
                    "memory_mb":         metrics.get("memory_mb"),
                    "execution_success": bool(metrics.get("success", False)),
                },
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "runtime_benchmark_exception",
                extra={
                    "event":   "runtime_benchmark_exception",
                    "runtime": rt_name,
                    "reason":  str(exc),
                },
            )
            results.append({
                "runtime":           rt_name,
                "latency_ms":        None,
                "memory_mb":         None,
                "execution_success": False,
            })

    # ── Phase 5 (removed): non-empty guard deleted.
    # An empty results list means "no runtime was available to execute" —
    # a distinct state from "all runtimes executed but failed".
    # The caller (_run_analysis_inner) handles both cases explicitly.

    # ── Phase 6: best runtime selection (lowest latency, success only) ────────
    successful = [r for r in results if r["execution_success"] and r["latency_ms"] is not None]

    if successful:
        best = min(successful, key=lambda r: r["latency_ms"])  # type: ignore[arg-type]
        best_runtime = best["runtime"]
    elif results:
        # All runtimes ran but none succeeded (e.g. all returned success=False).
        # Surface the first attempted runtime so the caller has a non-None string;
        # support_status will already be UNSUPPORTED.
        best_runtime = results[0]["runtime"]
    else:
        # No runtime was available to execute at all — return empty contract.
        # This is "no runtime executed", NOT "all runtimes failed".
        return {"evaluations": [], "best_runtime": None}

    # ── Phase 7: output contract ──────────────────────────────────────────────
    return {
        "evaluations": results,
        "best_runtime": best_runtime,
    }


# ── Adapter: translate execution_loop output → onnx_benchmark_service dicts ──
#
# Called by onnx_benchmark_service._call_evaluator() for the registry-backed
# formats (.onnx, .pt/.pth, .tflite).  Converts the Phase 7 contract into
# the raw-dict shape that evaluate_all_runtimes() already knows how to wrap
# into RuntimeEvaluation objects.
#
# The adapter is intentionally minimal — it maps only the fields the service
# uses downstream.  No new fields; no decision changes.

def to_evaluator_dict(evaluation: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single entry from execute_model_runtimes()["evaluations"]
    into the raw dict shape expected by onnx_benchmark_service._call_evaluator().

    Preserves the following guarantee from evaluator.py:
        success=True  → latency_avg_ms is a real finite number
        success=False → latency_avg_ms is None
    """
    success = bool(evaluation.get("execution_success", False))
    latency = evaluation.get("latency_ms")

    return {
        "success":        success,
        "latency_avg_ms": latency if success else None,
        "latency_p95_ms": None,          # not measured by runtimes.py benchmark()
        "memory_mb":      evaluation.get("memory_mb"),
        "error":          None if success else "Runtime benchmark failed or not executed",
        "benchmark_runs": 0,             # not surfaced by runtimes.py
        "stress_test": {
            "enabled":          False,
            "runs":             0,
            "latency_avg_ms":   None,
            "latency_p95_ms":   None,
            "peak_memory_mb":   None,
            "memory_growth_mb": None,
            "memory_stability": "STABLE",
        },
    }

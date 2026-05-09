"""
core/scoring/utility_scorer.py

Pure utility scoring logic.
Accepts RuntimeEvaluation (typed contract) instead of raw dict.

INVARIANTS:
  - Accepts RuntimeEvaluation — no raw dict
  - Reads only canonical field names
  - No IO. No state. No imports from services or api.
"""
from __future__ import annotations

import math

from src.core.contracts.runtime_evaluation import RuntimeEvaluation


def score_runtime_utility(
    evaluation: RuntimeEvaluation,
    scored_latencies: list[float],
    reference_latency_ms: float = 200.0,
) -> float:
    """
    Compute utility score [0.0, 1.0] for a single runtime evaluation.

    Args:
        evaluation:           Typed RuntimeEvaluation.
        scored_latencies:     All latency_avg_ms values from SUPPORTED successful
                              runtimes (used for relative scoring).
        reference_latency_ms: Calibration reference latency (ms).

    Returns:
        utility in [0.0, 1.0].
    """
    # Read canonical fields — no aliases
    if not evaluation.execution_success or evaluation.support_status == "UNSUPPORTED":
        return 0.0

    lat_ms = evaluation.latency_avg_ms
    if lat_ms is None:
        return 0.0

    runtime_latency = float(lat_ms)
    min_lat = min(scored_latencies) if scored_latencies else None
    max_lat = max(scored_latencies) if scored_latencies else None

    if len(scored_latencies) > 1 and min_lat is not None and max_lat is not None:
        if max_lat == min_lat:
            relative_score = 1.0
        else:
            relative_score = (max_lat - runtime_latency) / (max_lat - min_lat)
        absolute_normalized = runtime_latency / reference_latency_ms
        absolute_score      = 1.0 / (1.0 + math.log1p(absolute_normalized))
        utility = (relative_score + absolute_score) / 2.0
    elif len(scored_latencies) == 1:
        normalized = runtime_latency / reference_latency_ms
        utility    = 1.0 / (1.0 + math.log1p(normalized))
    else:
        utility = 0.0

    # Canonical field names for warnings and stress
    if evaluation.support_status == "SUPPORTED_WITH_WARNINGS":
        utility *= 0.8

    if evaluation.stress_memory_stability == "UNSTABLE":
        utility *= 0.5

    return max(0.0, min(1.0, float(utility)))

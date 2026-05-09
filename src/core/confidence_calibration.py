"""
core/confidence_calibration.py

Adjust a raw confidence score using historical calibration statistics.
"""
from __future__ import annotations

from typing import Any


def adjust_confidence_with_history(
    confidence_score: float,
    runtime: Any,
    stats: dict,
) -> float:
    """
    Apply a small empirical correction to confidence_score based on past accuracy.

    Returns the adjusted score in [0.0, 1.0].  When stats contains no useful
    history for the given runtime the original score is returned unchanged.
    """
    if not stats:
        return confidence_score

    runtime_key = runtime.value if hasattr(runtime, "value") else str(runtime)
    runtime_stats = stats.get(runtime_key) or stats.get("all")
    if not runtime_stats:
        return confidence_score

    accuracy = runtime_stats.get("accuracy")
    if accuracy is None:
        return confidence_score

    try:
        accuracy_f = float(accuracy)
    except (TypeError, ValueError):
        return confidence_score

    # Blend: 80 % original + 20 % empirical accuracy
    blended = confidence_score * 0.8 + accuracy_f * 0.2
    return max(0.0, min(1.0, blended))

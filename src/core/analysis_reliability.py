"""
core/analysis_reliability.py

Evaluate the reliability of the analysis pipeline for the current model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AnalysisReliability:
    level: str    # "LOW" | "MEDIUM" | "HIGH"
    score: float  # [0.0, 1.0]


def evaluate_analysis_reliability(
    facts: dict,
    perf_risk: Any,
    execution_behavior: Any,
    validations: dict,
) -> AnalysisReliability:
    """
    Derive a reliability estimate from validation coverage and model characteristics.

    A runtime with success=True counts as a covered data point.
    """
    total = len(validations)
    if total == 0:
        return AnalysisReliability(level="LOW", score=0.0)

    success_count = sum(
        1 for v in validations.values() if getattr(v, "success", False)
    )
    coverage = success_count / total

    # Penalise highly dynamic models — inference behaviour is harder to predict.
    has_dynamic = bool(facts.get("model.has_dynamic_shapes", False))
    penalty = 0.10 if has_dynamic else 0.0

    score = max(0.0, min(1.0, coverage - penalty))

    if score >= 0.75:
        level = "HIGH"
    elif score >= 0.40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return AnalysisReliability(level=level, score=round(score, 6))

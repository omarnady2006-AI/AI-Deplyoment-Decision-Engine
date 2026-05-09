"""
core/model_summary.py

RuntimeAssessment dataclass and summarize_model factory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeAssessment:
    """Per-runtime assessment record used in the second pipeline pass."""
    runtime: Any          # RuntimeName
    decision: str         # "ALLOW" | "ALLOW_WITH_CONDITIONS" | "BLOCK"
    constraints: list
    confidence: Any       # DecisionConfidence
    utility_score: float
    validation: Any       # RuntimeValidationResult


class _ModelSummary:
    """Lightweight model summary passed between pipeline stages."""

    def __init__(self, assessments: list[RuntimeAssessment], facts: dict) -> None:
        self._assessments = assessments
        self._facts = facts
        self.parameter_scale_class: str = str(facts.get("parameter_scale_class", "small"))

    def best_assessment(self) -> RuntimeAssessment | None:
        allowed = [a for a in self._assessments if a.decision not in ("BLOCK", "UNSUPPORTED")]
        if not allowed:
            return None
        return max(allowed, key=lambda a: a.utility_score)


def summarize_model(assessments: list, facts: dict) -> _ModelSummary:
    """Build a _ModelSummary from scored assessments and model facts."""
    return _ModelSummary(assessments, facts)

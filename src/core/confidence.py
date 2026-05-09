"""
core/confidence.py

DecisionConfidence dataclass and compute_confidence factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionConfidence:
    score: float
    level: str          # "LOW" | "MEDIUM" | "HIGH"
    reasons: list[str]  = field(default_factory=list)


def compute_confidence(
    decision: str,
    diagnostics: list,
    constraints: list,
    validation: Any,
    reliability_score: float | None = None,
) -> DecisionConfidence:
    """Compute a DecisionConfidence from per-runtime evaluation data."""
    base = 0.7

    fail_count = sum(1 for d in diagnostics if getattr(d, "severity", None) == "FAIL")
    warn_count  = sum(1 for d in diagnostics if getattr(d, "severity", None) == "WARN")

    score = base - (fail_count * 0.15) - (warn_count * 0.05)

    if reliability_score is not None:
        score = score * 0.6 + reliability_score * 0.4

    if decision in ("BLOCK", "UNSUPPORTED"):
        score = min(score, 0.35)
    elif decision in ("ALLOW_WITH_CONDITIONS", "SUPPORTED_WITH_WARNINGS"):
        score = min(score, 0.70)

    score = max(0.0, min(1.0, score))

    if score >= 0.80:
        level = "HIGH"
    elif score >= 0.50:
        level = "MEDIUM"
    else:
        level = "LOW"

    reasons: list[str] = []
    if fail_count:
        reasons.append(f"{fail_count} FAIL-severity diagnostic(s) detected")
    if warn_count:
        reasons.append(f"{warn_count} WARN-severity diagnostic(s) detected")
    if not reasons:
        reasons.append("no critical issues detected")

    return DecisionConfidence(score=round(score, 6), level=level, reasons=reasons)

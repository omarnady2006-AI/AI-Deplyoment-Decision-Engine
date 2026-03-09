"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TYPE_CHECKING

from src.core.decision import DeploymentConstraint, DeploymentDecision
from src.diagnostics.report import Diagnostic
from src.validation.runtime_validator import RuntimeValidationResult

if TYPE_CHECKING:
    from src.core.latency_measurement import LatencyStatistics


HIGH_OPERATOR_COUNT_CONFIDENCE_PENALTY = 0.05


class ConfidenceLevel(str, Enum):
    """Confidence level enum for gate decisions."""
    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    
    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert a confidence score to a confidence level."""
        if score >= 0.80:
            return cls.HIGH
        if score >= 0.50:
            return cls.MEDIUM
        return cls.LOW


@dataclass(frozen=True)
class DecisionConfidence:
    score: float
    level: Literal["LOW", "MEDIUM", "HIGH"]
    reasons: list[str]


def _clamp_score(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _confidence_level(score: float) -> Literal["LOW", "MEDIUM", "HIGH"]:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"


def compute_confidence(
    decision: DeploymentDecision,
    diagnostics: list[Diagnostic],
    constraints: list[DeploymentConstraint],
    validation: RuntimeValidationResult,
    reliability_score: float | None = None,
    latency_variance: float | None = None,
    latency_measurement: "LatencyStatistics | None" = None,
    sla_passed: bool | None = None,
    memory_within_limit: bool | None = None,
    operator_count: int | None = None,
    parameter_count: int | None = None,
    latency_margin: float | None = None,
) -> DecisionConfidence:
    """
    Compute confidence based on quality metrics, not raw ratios.
    
    Confidence depends on:
    - Latency stability (CV)
    - SLA pass/fail
    - Latency margin (proximity to SLA threshold)
    - Memory stability
    - Model complexity (operator_count)
    """
    if decision == DeploymentDecision.BLOCK:
        return DecisionConfidence(
            score=0.0,
            level="LOW",
            reasons=["decision_blocked"],
        )
    
    if any(diagnostic.severity == "FAIL" for diagnostic in diagnostics):
        return DecisionConfidence(
            score=0.0,
            level="LOW",
            reasons=["blocking FAIL diagnostics present"],
        )
    
    base_confidence = 0.5
    reasons: list[str] = ["base_confidence_0.5"]
    
    if sla_passed is not None:
        if sla_passed:
            base_confidence += 0.20
            reasons.append("sla_passed")
        else:
            base_confidence -= 0.15
            reasons.append("sla_failed")
    
    # Latency margin influence
    if latency_margin is not None:
        if latency_margin < 0.3:
            base_confidence += 0.10
            reasons.append("comfortable_latency_margin")
        elif latency_margin < 0.6:
            base_confidence += 0.05
            reasons.append("adequate_latency_margin")
        elif latency_margin > 0.9:
            base_confidence -= 0.15
            reasons.append("tight_latency_margin")
    
    if memory_within_limit is not None:
        if memory_within_limit:
            base_confidence += 0.10
            reasons.append("memory_within_limit")
        else:
            base_confidence -= 0.10
            reasons.append("memory_exceeds_limit")
    
    if not validation.executed:
        base_confidence -= 0.15
        reasons.append("validation did not execute")
    elif validation.success:
        base_confidence += 0.25
        reasons.append("validation stable")
    else:
        base_confidence -= 0.15
        reasons.append("validation failed")
    
    if latency_variance is not None:
        variance_penalty = min(0.2, latency_variance * 0.1)
        base_confidence -= variance_penalty
        reasons.append(f"latency variance adjustment ({variance_penalty:.2f} penalty)")
    
    if latency_measurement is not None:
        if latency_measurement.mean_ms > 0:
            cv = latency_measurement.std_dev_ms / latency_measurement.mean_ms
            if cv < 0.05:
                base_confidence += 0.1
                reasons.append("very stable latency (CV < 5%)")
            elif cv < 0.1:
                base_confidence += 0.05
                reasons.append("stable latency (CV < 10%)")
            elif cv > 0.3:
                base_confidence -= 0.1
                reasons.append("high latency variance (CV > 30%)")
    
    uncertainty = 0.0
    if not validation.executed:
        uncertainty += 0.1
    if latency_variance is None and latency_measurement is None:
        uncertainty += 0.1
    if reliability_score is None:
        uncertainty += 0.05
    
    if uncertainty > 0:
        base_confidence *= (1.0 - min(uncertainty, 0.3))
        reasons.append(f"uncertainty penalty applied ({uncertainty:.2f})")
    
    is_unstable_domain = "unstable_domain" in (validation.error_message or "").lower()
    if (
        is_unstable_domain
        and DeploymentConstraint.INPUT_SIZE_LIMITED in constraints
    ):
        base_confidence -= 0.25
        reasons.append("unstable domain overlaps input profile")
    
    if DeploymentConstraint.NUMERICAL_RISK in constraints:
        base_confidence -= 0.30
        reasons.append("numerical risk detected")
    
    warn_count = sum(1 for diagnostic in diagnostics if diagnostic.severity == "WARN")
    if warn_count > 0:
        warn_penalty = min(0.25, warn_count * 0.05)
        base_confidence -= warn_penalty
        reasons.append(f"{warn_count} WARN diagnostics present")
    
    score = _clamp_score(base_confidence)
    if reliability_score is not None:
        reliability_multiplier = 0.6 + 0.4 * _clamp_score(float(reliability_score))
        score = _clamp_score(score * reliability_multiplier)
        reasons.append(
            f"analysis reliability adjustment applied ({reliability_multiplier:.2f}x)"
        )
    if decision == DeploymentDecision.SUPPORTED and not constraints and warn_count == 0:
        reasons.append("no risky operators")
    if not reasons:
        reasons.append("confidence computed from validation and diagnostics")
    
    if operator_count is not None and operator_count > 500:
        operator_penalty = HIGH_OPERATOR_COUNT_CONFIDENCE_PENALTY
        score = _clamp_score(score - operator_penalty)
        reasons.append(f"high_operator_count_penalty_{operator_count}")
    
    return DecisionConfidence(
        score=score,
        level=_confidence_level(score),
        reasons=reasons,
    )

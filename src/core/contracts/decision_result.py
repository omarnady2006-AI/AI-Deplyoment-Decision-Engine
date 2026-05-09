"""
core/contracts/decision_result.py

Canonical typed output of the ML deployment decision pipeline.

INVARIANTS (must hold at all times):
  - MLDecisionResult is NEVER constructed outside analysis_pipeline_service.py
  - Fields are FROZEN — no mutation after construction
  - deployment_decision is one of: "edge_int8" | "edge_fp16" | "cloud_cpu" | "cloud_gpu"
  - confidence is a float in (0.0, 1.0]
  - REMOVED: decision, risk_score, risk_level, sla_passed, mapped_status — these are
    rule-based artefacts; the ML engine produces deployment_decision + confidence only.

Layer dependency:
    feature_extractor -> ml_decision_engine -> [MLDecisionResult] -> API routes -> UI
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any


# ── Valid deployment targets — single source of truth ─────────────────────────
VALID_DEPLOYMENT_TARGETS: frozenset[str] = frozenset({
    "edge_int8",
    "edge_fp16",
    "cloud_cpu",
    "cloud_gpu",
})

# ── Confidence threshold constants — kept for UI rendering only ───────────────
CONFIDENCE_HIGH:   float = 0.8
CONFIDENCE_MEDIUM: float = 0.5


def _confidence_level(confidence: float) -> str:
    if confidence >= CONFIDENCE_HIGH:
        return "HIGH"
    if confidence >= CONFIDENCE_MEDIUM:
        return "MEDIUM"
    return "LOW"


@dataclass(frozen=True)
class MLDecisionResult:
    """
    Immutable final deployment decision produced by the ML model.

    Fields:
        deployment_decision   "edge_int8" | "edge_fp16" | "cloud_cpu" | "cloud_gpu"
        confidence            Predicted probability of the chosen target. (0.0, 1.0].
        timestamp             Unix epoch seconds at construction time.
    """
    deployment_decision: str
    confidence: float
    timestamp: float = field(default_factory=_time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_decision": self.deployment_decision,
            "confidence": {
                "score": self.confidence,
                "level": _confidence_level(self.confidence),
            },
            "timestamp": self.timestamp,
        }


def build_ml_decision_result(
    *,
    deployment_decision: str,
    confidence: float,
) -> MLDecisionResult:
    if deployment_decision not in VALID_DEPLOYMENT_TARGETS:
        raise ValueError(
            f"build_ml_decision_result: deployment_decision {deployment_decision!r} "
            f"is not a valid target; must be one of {sorted(VALID_DEPLOYMENT_TARGETS)}"
        )
    if not isinstance(confidence, float):
        confidence = float(confidence)
    if not (0.0 < confidence <= 1.0):
        raise ValueError(
            f"build_ml_decision_result: confidence={confidence!r} out of (0.0, 1.0]."
        )
    return MLDecisionResult(
        deployment_decision=deployment_decision,
        confidence=confidence,
    )


# Backward-compat aliases so unchanged imports keep working
DecisionResult = MLDecisionResult
build_decision_result = build_ml_decision_result

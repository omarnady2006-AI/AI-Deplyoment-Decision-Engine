"""
core/recommendation_policy.py

Apply post-selection policy rules to produce a final runtime recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.runtime import RuntimeName


@dataclass
class RecommendationPolicy:
    final_runtime: RuntimeName | None
    status: str   # "RECOMMEND" | "CONDITIONAL" | "REJECT" | "UNCERTAIN"


def apply_recommendation_policy(
    recommendation: Any,
    confidence: dict,
    reliability: Any,
    perf_risk: Any,
    validations: dict,
) -> RecommendationPolicy:
    """
    Apply policy rules and return the final runtime + status.

    final_runtime is None when the recommended decision is BLOCK.
    status maps to the PipelineOutput.recommendation_status field consumed by
    main._enforce_decision_gate().
    """
    best = getattr(recommendation, "best_runtime", None)
    evaluations = getattr(recommendation, "evaluations", [])

    best_eval = next((e for e in evaluations if e.runtime == best), None)
    decision = getattr(best_eval, "decision", "BLOCK") if best_eval else "BLOCK"

    rel_level = getattr(reliability, "level", "LOW")
    rel_score = getattr(reliability, "score", 0.0)

    conf_obj = confidence.get(best) if best else None
    conf_score = getattr(conf_obj, "score", 0.5) if conf_obj else 0.5

    if decision == "BLOCK":
        return RecommendationPolicy(final_runtime=None, status="REJECT")

    if decision == "ALLOW_WITH_CONDITIONS":
        return RecommendationPolicy(final_runtime=best, status="CONDITIONAL")

    # decision == "ALLOW"
    if rel_level == "LOW" or conf_score < 0.35:
        return RecommendationPolicy(final_runtime=best, status="UNCERTAIN")

    return RecommendationPolicy(final_runtime=best, status="RECOMMEND")

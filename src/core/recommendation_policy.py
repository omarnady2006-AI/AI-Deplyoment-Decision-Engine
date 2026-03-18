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
from typing import Literal

from src.core.confidence import DecisionConfidence
from src.core.decision import DeploymentConstraint, DeploymentDecision
from src.core.runtime import RuntimeName
from src.core.runtime_selector import RuntimeRecommendation
from src.validation.runtime_validator import RuntimeValidationResult


@dataclass(frozen=True)
class RecommendationPolicy:
    final_runtime: RuntimeName | None
    status: Literal["RECOMMEND", "UNCERTAIN", "DO_NOT_DEPLOY"]
    reason: str


def apply_recommendation_policy(
    recommendation: RuntimeRecommendation,
    confidence: dict[RuntimeName, DecisionConfidence],
    reliability,
    perf_risk,
    validations: dict[RuntimeName, RuntimeValidationResult] | list[RuntimeValidationResult],
) -> RecommendationPolicy:
    evaluations = list(recommendation.evaluations)
    if not evaluations:
        return RecommendationPolicy(
            final_runtime=None,
            status="DO_NOT_DEPLOY",
            reason="No runtime evaluations available.",
        )

    by_runtime = {item.runtime: item for item in evaluations}
    best_runtime = recommendation.best_runtime
    best_eval = by_runtime[best_runtime]
    best_conf = confidence.get(best_runtime)
    best_conf_score = best_conf.score if best_conf is not None else 0.0
    best_conf_level = best_conf.level if best_conf is not None else "LOW"
    reliability_level = str(getattr(reliability, "level", "LOW"))

    if isinstance(validations, dict):
        validation_items = list(validations.values())
    else:
        validation_items = list(validations)

    if best_eval.decision == DeploymentDecision.UNSUPPORTED:
        return RecommendationPolicy(
            final_runtime=None,
            status="DO_NOT_DEPLOY",
            reason="Top runtime is unsupported.",
        )

    crash_like = 0
    for item in validation_items:
        message = (item.error_message or "").lower()
        if "crash" in message or "timed out" in message:
            crash_like += 1
    if validation_items and crash_like == len(validation_items):
        return RecommendationPolicy(
            final_runtime=None,
            status="DO_NOT_DEPLOY",
            reason="Validation crashes/timeouts observed across all runtimes.",
        )

    if (
        DeploymentConstraint.NUMERICAL_RISK in best_eval.constraints
        and reliability_level == "LOW"
    ):
        return RecommendationPolicy(
            final_runtime=None,
            status="DO_NOT_DEPLOY",
            reason="Numerical risk with low reliability.",
        )

    if reliability_level == "LOW" and best_conf_score < 0.7:
        return RecommendationPolicy(
            final_runtime=None,
            status="UNCERTAIN",
            reason="Low reliability and insufficient confidence.",
        )

    sorted_by_utility = sorted(
        evaluations, key=lambda item: item.utility_score, reverse=True
    )
    if len(sorted_by_utility) >= 2:
        delta = sorted_by_utility[0].utility_score - sorted_by_utility[1].utility_score
        if delta < 0.05:
            return RecommendationPolicy(
                final_runtime=None,
                status="UNCERTAIN",
                reason="Top runtime utilities are near-equal.",
            )

    if (
        str(getattr(perf_risk, "estimated_latency_class", "")) in {"HIGH", "EXTREME"}
        and best_conf_level == "MEDIUM"
    ):
        return RecommendationPolicy(
            final_runtime=None,
            status="UNCERTAIN",
            reason="High perf risk with only medium confidence.",
        )

    runtime_constraints = {item.runtime: item.constraints for item in evaluations}
    domain_filtered = False
    for item in validation_items:
        message = (item.error_message or "").lower()
        if "unstable_domain" not in message:
            continue
        constraints = runtime_constraints.get(item.runtime, [])
        if DeploymentConstraint.INPUT_SIZE_LIMITED not in constraints:
            domain_filtered = True
            break
    if domain_filtered:
        return RecommendationPolicy(
            final_runtime=None,
            status="UNCERTAIN",
            reason="Domain instability detected but filtered by input profile.",
        )

    return RecommendationPolicy(
        final_runtime=best_runtime,
        status="RECOMMEND",
        reason="Signals are consistent enough for automatic recommendation.",
    )

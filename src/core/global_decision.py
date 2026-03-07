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

from src.core.runtime import RuntimeName
from src.core.runtime_selector import RuntimeRecommendation


class GlobalDeployability(str, Enum):
    SAFE = "SAFE"
    RISKY = "RISKY"
    UNSAFE = "UNSAFE"


@dataclass(frozen=True)
class GlobalDecision:
    status: GlobalDeployability
    best_runtime: RuntimeName | None
    best_utility: float
    reason: str


def compute_global_decision(recommendation: RuntimeRecommendation) -> GlobalDecision:
    if not recommendation.evaluations:
        return GlobalDecision(
            status=GlobalDeployability.UNSAFE,
            best_runtime=None,
            best_utility=0.0,
            reason="No runtime evaluations were produced.",
        )

    best_evaluation = max(
        recommendation.evaluations, key=lambda evaluation: evaluation.utility_score
    )
    best_utility = best_evaluation.utility_score

    if best_utility >= 0.75:
        return GlobalDecision(
            status=GlobalDeployability.SAFE,
            best_runtime=recommendation.best_runtime,
            best_utility=best_utility,
            reason="Utility is high enough for normal deployment.",
        )
    if best_utility >= 0.45:
        return GlobalDecision(
            status=GlobalDeployability.RISKY,
            best_runtime=recommendation.best_runtime,
            best_utility=best_utility,
            reason="Utility is moderate; deployment is possible with caution.",
        )
    return GlobalDecision(
        status=GlobalDeployability.UNSAFE,
        best_runtime=None,
        best_utility=best_utility,
        reason="Utility is too low for safe deployment.",
    )

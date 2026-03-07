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

from src.core.decision import DeploymentConstraint, DeploymentDecision
from src.core.model_summary import ModelCapabilitySummary, RuntimeAssessment


@dataclass(frozen=True)
class DeploymentProfile:
    realtime_gpu: str
    realtime_cpu: str
    mobile: str
    edge_device: str
    memory_pressure: str
    explanation: list[str]


def _best_assessment(assessments: list[RuntimeAssessment]) -> RuntimeAssessment | None:
    if not assessments:
        return None
    return max(assessments, key=lambda item: item.utility_score)


def predict_deployment_profile(
    summary: ModelCapabilitySummary, assessments: list[RuntimeAssessment]
) -> DeploymentProfile:
    best = _best_assessment(assessments)
    has_input_size_limited = any(
        DeploymentConstraint.INPUT_SIZE_LIMITED in assessment.constraints
        for assessment in assessments
    )
    has_batch_limited = any(
        DeploymentConstraint.BATCH_LIMITED in assessment.constraints
        for assessment in assessments
    )

    if (
        best is not None
        and best.decision == DeploymentDecision.SUPPORTED
        and summary.size_class in {"small", "medium"}
        and summary.runtime_stability == "high"
    ):
        realtime_gpu = "yes"
    elif summary.runtime_stability == "moderate" or (
        best is not None and best.decision == DeploymentDecision.SUPPORTED_WITH_WARNINGS
    ):
        realtime_gpu = "possible"
    else:
        realtime_gpu = "no"

    if summary.size_class == "small" and summary.numerical_stability == "stable":
        realtime_cpu = "possible"
    elif summary.size_class == "medium":
        realtime_cpu = "unlikely"
    else:
        realtime_cpu = "no"

    if has_input_size_limited or has_batch_limited or summary.size_class == "large":
        mobile = "not suitable"
    elif summary.size_class == "medium" and summary.numerical_stability == "stable":
        mobile = "limited"
    elif summary.size_class == "small" and summary.numerical_stability == "stable":
        mobile = "suitable"
    else:
        mobile = "limited"

    if summary.size_class == "small":
        edge_device = "good"
    elif summary.size_class == "medium":
        edge_device = "depends"
    else:
        edge_device = "poor"

    if summary.size_class == "large":
        memory_pressure = "high"
    elif summary.size_class == "medium":
        memory_pressure = "moderate"
    else:
        memory_pressure = "low"

    explanation: list[str] = []
    explanation.append(
        f"GPU realtime is {realtime_gpu} based on best runtime decision and runtime stability ({summary.runtime_stability})."
    )
    explanation.append(
        f"CPU realtime is {realtime_cpu} due to model size class {summary.size_class} and numerical stability {summary.numerical_stability}."
    )
    explanation.append(
        f"Mobile suitability is {mobile} considering size limits and batching/input constraints."
    )
    explanation.append(
        f"Edge-device suitability is {edge_device} from size class {summary.size_class}."
    )
    explanation.append(
        f"Expected memory pressure is {memory_pressure} for size class {summary.size_class}."
    )

    return DeploymentProfile(
        realtime_gpu=realtime_gpu,
        realtime_cpu=realtime_cpu,
        mobile=mobile,
        edge_device=edge_device,
        memory_pressure=memory_pressure,
        explanation=explanation,
    )

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

from src.core.confidence import DecisionConfidence
from src.core.decision import DeploymentConstraint, DeploymentDecision
from src.core.runtime import RuntimeName
from src.validation.runtime_validator import RuntimeValidationResult


@dataclass(frozen=True)
class RuntimeAssessment:
    runtime: RuntimeName
    decision: DeploymentDecision
    constraints: list[DeploymentConstraint]
    confidence: DecisionConfidence
    utility_score: float
    validation: RuntimeValidationResult


@dataclass(frozen=True)
class ModelCapabilitySummary:
    model_type: str
    size_class: str
    numerical_stability: str
    batching_support: str
    input_flexibility: str
    runtime_stability: str
    risk_level: str
    explanation: list[str]


def _best_assessment(assessments: list[RuntimeAssessment]) -> RuntimeAssessment | None:
    if not assessments:
        return None
    return max(assessments, key=lambda item: item.utility_score)


def summarize_model(
    assessments: list[RuntimeAssessment], facts: dict
) -> ModelCapabilitySummary:
    has_conv = bool(facts.get("model.has_conv", facts.get("model.has_conv_transpose", False)))
    has_attention = bool(
        facts.get("model.has_attention", facts.get("model.uses_attention", False))
    )
    if has_conv and not has_attention:
        model_type = "CNN"
    elif has_attention and not has_conv:
        model_type = "Transformer"
    elif has_conv and has_attention:
        model_type = "Hybrid"
    else:
        model_type = "Unknown"

    size_value = str(facts.get("parameter_scale_class", "unknown")).lower()
    if size_value in {"small", "medium", "large"}:
        size_class = size_value
    else:
        size_class = "unknown"

    has_numerical_constraint = any(
        DeploymentConstraint.NUMERICAL_RISK in assessment.constraints
        for assessment in assessments
    )
    has_nan_inf = any(
        "nan" in (assessment.validation.error_message or "").lower()
        or "inf" in (assessment.validation.error_message or "").lower()
        for assessment in assessments
    )
    if has_nan_inf:
        numerical_stability = "dangerous"
    elif has_numerical_constraint:
        numerical_stability = "unstable"
    else:
        numerical_stability = "stable"

    has_batch_limited = any(
        DeploymentConstraint.BATCH_LIMITED in assessment.constraints
        for assessment in assessments
    )
    has_shape_instability = any(
        "shape changes across runs"
        in (assessment.validation.error_message or "").lower()
        for assessment in assessments
    )
    if has_batch_limited:
        batching_support = "limited"
    elif has_shape_instability:
        batching_support = "single batch only"
    else:
        batching_support = "batch capable"

    has_input_size_limited = any(
        DeploymentConstraint.INPUT_SIZE_LIMITED in assessment.constraints
        for assessment in assessments
    )
    has_dynamic_shapes = bool(
        facts.get("model.has_dynamic_shapes", facts.get("model.dynamic_shapes", False))
    )
    if has_input_size_limited:
        input_flexibility = "fixed size"
    elif has_dynamic_shapes:
        input_flexibility = "flexible"
    else:
        input_flexibility = "static size"

    best = _best_assessment(assessments)
    if best is None:
        runtime_stability = "low"
        risk_level = "high"
    else:
        if best.confidence.score >= 0.8:
            runtime_stability = "high"
        elif best.confidence.score >= 0.5:
            runtime_stability = "moderate"
        else:
            runtime_stability = "low"

        if best.decision == DeploymentDecision.UNSUPPORTED:
            risk_level = "high"
        elif best.decision == DeploymentDecision.SUPPORTED_WITH_WARNINGS or best.utility_score < 0.75:
            risk_level = "medium"
        else:
            risk_level = "low"

    explanation: list[str] = []
    explanation.append(f"Model type inferred as {model_type}.")
    explanation.append(f"Parameter scale class assessed as {size_class}.")
    explanation.append(
        f"Numerical stability is {numerical_stability} based on constraints and validation signals."
    )
    explanation.append(f"Batching support is {batching_support}.")
    explanation.append(f"Input flexibility is {input_flexibility}.")
    if best is not None:
        explanation.append(
            f"Best runtime is {best.runtime.value} with confidence {best.confidence.score:.2f} and utility {best.utility_score:.2f}."
        )
    explanation.append(f"Overall operational risk is {risk_level}.")

    return ModelCapabilitySummary(
        model_type=model_type,
        size_class=size_class,
        numerical_stability=numerical_stability,
        batching_support=batching_support,
        input_flexibility=input_flexibility,
        runtime_stability=runtime_stability,
        risk_level=risk_level,
        explanation=explanation,
    )

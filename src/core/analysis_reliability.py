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

from src.core.runtime import RuntimeName
from src.validation.runtime_validator import RuntimeValidationResult


@dataclass(frozen=True)
class AnalysisReliability:
    score: float
    level: str
    weak_signals: list[str]
    explanation: list[str]


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _as_validation_list(
    validations: dict[RuntimeName, RuntimeValidationResult] | list[RuntimeValidationResult],
) -> list[RuntimeValidationResult]:
    if isinstance(validations, dict):
        return list(validations.values())
    if isinstance(validations, list):
        return validations
    return []


def evaluate_analysis_reliability(
    facts: dict,
    perf_risk,
    execution_behavior,
    validations: dict[RuntimeName, RuntimeValidationResult] | list[RuntimeValidationResult],
) -> AnalysisReliability:
    score = 0.70
    weak_signals: list[str] = []
    explanation: list[str] = []

    items = _as_validation_list(validations)
    operator_counts_raw = facts.get("operator_counts", {})
    operator_counts = (
        operator_counts_raw if isinstance(operator_counts_raw, dict) else {}
    )
    total_ops = max(1, sum(int(value) for value in operator_counts.values()))
    conv_count = int(operator_counts.get("Conv", 0)) + int(
        operator_counts.get("ConvTranspose", 0)
    )
    matmul_count = int(operator_counts.get("MatMul", 0)) + int(
        operator_counts.get("Gemm", 0)
    )
    attention_present = bool(
        facts.get("model.has_attention", False)
        or int(operator_counts.get("Attention", 0)) > 0
        or int(operator_counts.get("MultiHeadAttention", 0)) > 0
    )
    dynamic_shapes = bool(facts.get("model.has_dynamic_shapes", False))
    parameter_count = int(facts.get("parameter_count", 0) or 0)
    recommended_runtime = str(facts.get("recommended_runtime", ""))

    predicted_supported = any(
        item.predicted_decision.value in {"SUPPORTED", "SUPPORTED_WITH_WARNINGS"}
        for item in items
    )
    all_validation_stable = bool(items) and all(item.success for item in items)
    failed_count = sum(1 for item in items if not item.success)

    if (
        perf_risk.estimated_latency_class in {"HIGH", "EXTREME"}
        and predicted_supported
    ):
        score -= 0.12
        weak_signals.append("high_latency_vs_supported_decision")

    if (
        execution_behavior.bound == "MEMORY_BOUND"
        and recommended_runtime == RuntimeName.TENSORRT.value
    ):
        score -= 0.08
        weak_signals.append("memory_bound_but_gpu_chosen")

    if attention_present and matmul_count <= max(1, conv_count // 2):
        score -= 0.07
        weak_signals.append("attention_without_matmul_dominance")

    if dynamic_shapes and all_validation_stable:
        score -= 0.06
        weak_signals.append("dynamic_shapes_but_validation_stable")

    if parameter_count < 8_000_000 and perf_risk.memory_pressure == "HIGH":
        score -= 0.08
        weak_signals.append("small_model_but_high_memory_estimate")

    if (
        (execution_behavior.bound == "DYNAMIC_UNSTABLE"
         or perf_risk.estimated_latency_class in {"HIGH", "EXTREME"})
        and failed_count > 0
    ):
        score += 0.10
        explanation.append("Validation instability aligns with static risk prediction.")

    conv_ratio = float(conv_count) / float(total_ops)
    if conv_ratio > 0.70:
        score += 0.07
        explanation.append("Strong Conv-dominant operator pattern improves confidence.")

    size_class = str(facts.get("parameter_scale_class", "unknown")).lower()
    if (
        (size_class == "small" and perf_risk.estimated_latency_class in {"LOW", "MEDIUM"})
        or (size_class == "medium" and perf_risk.estimated_latency_class in {"MEDIUM", "HIGH"})
        or (size_class == "large" and perf_risk.estimated_latency_class in {"HIGH", "EXTREME"})
    ):
        score += 0.08
        explanation.append("Parameter scale is consistent with latency estimate.")

    if failed_count >= 2:
        score += 0.08
        explanation.append("Multiple runtime failures provide consistent negative evidence.")

    score = _clamp(score)
    if score >= 0.75:
        level = "HIGH"
    elif score >= 0.45:
        level = "MEDIUM"
    else:
        level = "LOW"

    explanation.append(f"Final reliability score={score:.2f}, level={level}.")
    return AnalysisReliability(
        score=score,
        level=level,
        weak_signals=weak_signals,
        explanation=explanation,
    )

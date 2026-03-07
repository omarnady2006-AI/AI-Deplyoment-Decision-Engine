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

from src.core.model_summary import ModelCapabilitySummary
from src.core.perf_risk import PerfRiskReport


BoundClass = Literal[
    "COMPUTE_BOUND", "MEMORY_BOUND", "LATENCY_BOUND", "DYNAMIC_UNSTABLE"
]
GpuScalingClass = Literal["POOR", "MODERATE", "HIGH"]
BatchScalingClass = Literal["POOR", "GOOD"]


@dataclass(frozen=True)
class ExecutionBehavior:
    bound: BoundClass
    gpu_scaling: GpuScalingClass
    batch_scaling: BatchScalingClass
    explanation: list[str]


def _count(operator_counts: dict[str, int], name: str) -> int:
    return int(operator_counts.get(name, 0))


def classify_execution_behavior(
    facts: dict, perf_risk: PerfRiskReport, model_summary: ModelCapabilitySummary
) -> ExecutionBehavior:
    operator_counts_raw = facts.get("operator_counts", {})
    operator_counts = (
        operator_counts_raw if isinstance(operator_counts_raw, dict) else {}
    )
    parameter_count = int(facts.get("parameter_count", 0) or 0)
    dynamic_shapes = bool(facts.get("model.has_dynamic_shapes", False))
    dynamic_batch = bool(facts.get("model.dynamic_batch", False))
    depth = int(facts.get("sequential_depth_estimate", 0) or 0)

    conv = _count(operator_counts, "Conv") + _count(operator_counts, "ConvTranspose")
    matmul = _count(operator_counts, "MatMul") + _count(operator_counts, "Gemm")
    attention = _count(operator_counts, "Attention") + _count(
        operator_counts, "MultiHeadAttention"
    )
    resize = _count(operator_counts, "Resize")
    norm = _count(operator_counts, "LayerNormalization") + _count(
        operator_counts, "BatchNormalization"
    )
    reduce_ops = sum(
        _count(operator_counts, name)
        for name in ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin")
    )
    elementwise_ops = sum(
        _count(operator_counts, name)
        for name in (
            "Add",
            "Mul",
            "Sub",
            "Div",
            "Relu",
            "Sigmoid",
            "Tanh",
            "Clip",
            "Exp",
            "Sqrt",
        )
    )
    operator_diversity = len(operator_counts)
    total_ops = max(1, sum(operator_counts.values()))

    memory_score = 0
    compute_score = 0
    latency_score = 0
    dynamic_score = 0

    if elementwise_ops >= max(20, conv + matmul):
        memory_score += 2
    if norm >= 4:
        memory_score += 1
    if parameter_count < 8_000_000 and depth >= 80:
        memory_score += 1
    if resize >= 3:
        memory_score += 1
    if perf_risk.memory_pressure == "HIGH":
        memory_score += 1

    if conv + matmul >= max(12, int(0.35 * total_ops)):
        compute_score += 2
    if parameter_count >= 25_000_000:
        compute_score += 2
    elif parameter_count >= 8_000_000:
        compute_score += 1
    if attention > 0:
        compute_score += 1

    if depth <= 12:
        latency_score += 2
    if parameter_count < 2_000_000:
        latency_score += 1
    if operator_diversity <= 6:
        latency_score += 1
    if perf_risk.estimated_latency_class in {"LOW", "MEDIUM"} and total_ops < 30:
        latency_score += 1

    if dynamic_shapes and attention > 0:
        dynamic_score += 2
    if dynamic_shapes and resize > 0:
        dynamic_score += 1
    if dynamic_batch and reduce_ops > 0:
        dynamic_score += 1

    if dynamic_score >= 2:
        bound: BoundClass = "DYNAMIC_UNSTABLE"
    elif compute_score >= memory_score and compute_score >= latency_score:
        bound = "COMPUTE_BOUND"
    elif memory_score >= latency_score:
        bound = "MEMORY_BOUND"
    else:
        bound = "LATENCY_BOUND"

    if bound == "COMPUTE_BOUND":
        gpu_scaling: GpuScalingClass = "HIGH"
    elif bound == "DYNAMIC_UNSTABLE":
        gpu_scaling = "POOR"
    elif bound == "MEMORY_BOUND":
        gpu_scaling = "MODERATE"
    else:
        gpu_scaling = "MODERATE" if parameter_count >= 8_000_000 else "POOR"

    batch_scaling: BatchScalingClass = (
        "GOOD" if model_summary.batching_support == "batch capable" else "POOR"
    )

    explanation: list[str] = []
    explanation.append(
        f"Bound={bound} from compute({compute_score}) memory({memory_score}) latency({latency_score}) dynamic({dynamic_score}) signals."
    )
    explanation.append(
        f"GPU scaling={gpu_scaling} based on operator density, parameter count, and dynamic-shape risk."
    )
    explanation.append(
        f"Batch scaling={batch_scaling} from batching support={model_summary.batching_support}."
    )

    return ExecutionBehavior(
        bound=bound,
        gpu_scaling=gpu_scaling,
        batch_scaling=batch_scaling,
        explanation=explanation,
    )

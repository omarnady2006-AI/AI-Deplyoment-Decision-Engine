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

from src.core.model_summary import ModelCapabilitySummary, RuntimeAssessment


LatencyClass = Literal["LOW", "MEDIUM", "HIGH", "EXTREME"]
MemoryPressureClass = Literal["LOW", "MODERATE", "HIGH"]
ParallelismClass = Literal["POOR", "LIMITED", "GOOD"]


@dataclass(frozen=True)
class PerfRiskReport:
    estimated_latency_class: LatencyClass
    memory_pressure: MemoryPressureClass
    parallelism: ParallelismClass
    realtime_cpu_viable: bool
    realtime_gpu_viable: bool
    explanation: list[str]


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _operator_count(operator_counts: dict[str, int], key: str) -> int:
    return int(operator_counts.get(key, 0))


def estimate_inference_risk(
    facts: dict, model_summary: ModelCapabilitySummary, assessments: list[RuntimeAssessment]
) -> PerfRiskReport:
    operator_counts_raw = facts.get("operator_counts", {})
    operator_counts = (
        operator_counts_raw if isinstance(operator_counts_raw, dict) else {}
    )
    parameter_count = _to_int(facts.get("parameter_count", 0), default=0)
    dynamic_shapes = bool(
        facts.get("model.has_dynamic_shapes", facts.get("model.dynamic_shapes", False))
    )
    has_attention = bool(
        facts.get("model.has_attention", False)
        or _operator_count(operator_counts, "Attention") > 0
        or _operator_count(operator_counts, "MultiHeadAttention") > 0
    )

    resize_count = _operator_count(operator_counts, "Resize")
    convtranspose_count = _operator_count(operator_counts, "ConvTranspose")
    conv_count = _operator_count(operator_counts, "Conv") + convtranspose_count
    matmul_count = _operator_count(operator_counts, "MatMul") + _operator_count(
        operator_counts, "Gemm"
    )
    sequential_depth = _to_int(facts.get("sequential_depth_estimate", 0), default=0)
    if sequential_depth <= 0:
        sequential_depth = (
            conv_count
            + matmul_count
            + _operator_count(operator_counts, "LayerNormalization")
            + _operator_count(operator_counts, "BatchNormalization")
            + _operator_count(operator_counts, "Attention")
            + _operator_count(operator_counts, "MultiHeadAttention")
        )

    conv_matmul_ratio = float(conv_count + 1) / float(matmul_count + 1)
    large_resize_chain = resize_count >= 3

    latency_score = 0
    if parameter_count >= 100_000_000:
        latency_score += 4
    elif parameter_count >= 25_000_000:
        latency_score += 3
    elif parameter_count >= 8_000_000:
        latency_score += 2
    elif parameter_count >= 2_000_000:
        latency_score += 1

    if sequential_depth >= 160:
        latency_score += 3
    elif sequential_depth >= 80:
        latency_score += 2
    elif sequential_depth >= 40:
        latency_score += 1

    if has_attention:
        latency_score += 2
    if large_resize_chain:
        latency_score += 1
    if convtranspose_count > 0:
        latency_score += 1
    if dynamic_shapes:
        latency_score += 1
    if model_summary.batching_support != "batch capable":
        latency_score += 1
    if conv_matmul_ratio >= 6.0 or conv_matmul_ratio <= 0.2:
        latency_score += 1

    if latency_score >= 8:
        latency_class: LatencyClass = "EXTREME"
    elif latency_score >= 5:
        latency_class = "HIGH"
    elif latency_score >= 3:
        latency_class = "MEDIUM"
    else:
        latency_class = "LOW"

    memory_score = 0
    if parameter_count >= 100_000_000:
        memory_score += 3
    elif parameter_count >= 25_000_000:
        memory_score += 2
    elif parameter_count >= 8_000_000:
        memory_score += 1
    if dynamic_shapes:
        memory_score += 1
    if large_resize_chain:
        memory_score += 1
    if convtranspose_count > 0:
        memory_score += 1

    if memory_score >= 4:
        memory_pressure: MemoryPressureClass = "HIGH"
    elif memory_score >= 2:
        memory_pressure = "MODERATE"
    else:
        memory_pressure = "LOW"

    parallel_penalty = 0
    if sequential_depth >= 120:
        parallel_penalty += 1
    if dynamic_shapes:
        parallel_penalty += 1
    if model_summary.batching_support != "batch capable":
        parallel_penalty += 1
    if has_attention:
        parallel_penalty += 1
    if convtranspose_count > 0:
        parallel_penalty += 1

    if parallel_penalty >= 4:
        parallelism: ParallelismClass = "POOR"
    elif parallel_penalty >= 2:
        parallelism = "LIMITED"
    else:
        parallelism = "GOOD"

    has_runtime_signal = len(assessments) > 0
    realtime_cpu_viable = (
        latency_class in {"LOW", "MEDIUM"}
        and memory_pressure != "HIGH"
        and model_summary.numerical_stability != "dangerous"
        and model_summary.size_class in {"small", "medium"}
        and has_runtime_signal
    )
    realtime_gpu_viable = (
        latency_class != "EXTREME"
        and model_summary.numerical_stability != "dangerous"
        and has_runtime_signal
    )

    explanation: list[str] = []
    explanation.append(
        f"Estimated latency class {latency_class} from parameter count {parameter_count}, depth {sequential_depth}, and operator mix."
    )
    explanation.append(
        f"Memory pressure {memory_pressure} based on parameter scale and dynamic/resize/transpose signals."
    )
    explanation.append(
        f"Parallelism rated {parallelism} using batching support ({model_summary.batching_support}), depth, and dynamic-shape behavior."
    )
    explanation.append(
        f"Conv/MatMul ratio is {conv_matmul_ratio:.2f}; attention patterns={'yes' if has_attention else 'no'}."
    )
    explanation.append(
        f"Realtime viability: CPU={'yes' if realtime_cpu_viable else 'no'}, GPU={'yes' if realtime_gpu_viable else 'no'}."
    )

    return PerfRiskReport(
        estimated_latency_class=latency_class,
        memory_pressure=memory_pressure,
        parallelism=parallelism,
        realtime_cpu_viable=realtime_cpu_viable,
        realtime_gpu_viable=realtime_gpu_viable,
        explanation=explanation,
    )

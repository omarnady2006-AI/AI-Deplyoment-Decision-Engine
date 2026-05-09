"""
core/perf_risk.py

PerfRiskReport dataclass and estimate_inference_risk factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerfRiskReport:
    estimated_latency_class: str = "UNKNOWN"   # "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
    memory_pressure: str = "LOW"               # "LOW" | "MEDIUM" | "HIGH"
    parallelism: str = "UNKNOWN"               # "SEQUENTIAL" | "PARALLEL" | "UNKNOWN"
    realtime_cpu_viable: bool = True
    realtime_gpu_viable: bool = False
    explanation: list = field(default_factory=list)


def estimate_inference_risk(
    facts: dict,
    model_summary: Any,
    assessments: list,
) -> PerfRiskReport:
    """Estimate inference performance risk from model facts and runtime assessments."""
    scale = facts.get("parameter_scale_class", "small")
    op_count = int(facts.get("sequential_depth_estimate") or facts.get("operator_count") or 0)

    if scale == "large" or op_count > 500:
        lat_class = "HIGH"
        mem_pressure = "HIGH"
        rt_cpu = False
    elif scale == "medium" or op_count > 100:
        lat_class = "MEDIUM"
        mem_pressure = "MEDIUM"
        rt_cpu = True
    else:
        lat_class = "LOW"
        mem_pressure = "LOW"
        rt_cpu = True

    _GPU_RUNTIMES = {"ONNX_CUDA", "TensorRT", "TF_GPU", "TFLite_GPU", "TensorRT_Native", "OpenVINO_GPU"}
    gpu_viable = any(
        getattr(a.runtime, "value", str(a.runtime)) in _GPU_RUNTIMES
        for a in assessments
        if a.decision not in ("BLOCK", "UNSUPPORTED")
    )

    explanation = [
        f"Parameter scale: {scale}",
        f"Estimated operator depth: {op_count}",
        f"Estimated latency class: {lat_class}",
    ]

    return PerfRiskReport(
        estimated_latency_class=lat_class,
        memory_pressure=mem_pressure,
        parallelism="PARALLEL" if gpu_viable else "SEQUENTIAL",
        realtime_cpu_viable=rt_cpu,
        realtime_gpu_viable=gpu_viable,
        explanation=explanation,
    )

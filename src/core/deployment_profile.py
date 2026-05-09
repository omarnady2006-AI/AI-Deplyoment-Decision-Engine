"""
core/deployment_profile.py

Predict deployment profile characteristics from model summary and assessments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DeploymentProfilePrediction:
    latency_class: str     # "LOW" | "MEDIUM" | "HIGH"
    memory_class: str      # "SMALL" | "MEDIUM" | "LARGE"
    recommended_runtime: str


def predict_deployment_profile(
    summary: Any,
    assessments: list,
) -> DeploymentProfilePrediction:
    """Predict the deployment profile from model summary and runtime assessments."""
    scale = getattr(summary, "parameter_scale_class", "small")

    lat_class_map = {"small": "LOW", "medium": "MEDIUM", "large": "HIGH"}
    lat_class = lat_class_map.get(scale, "MEDIUM")
    mem_class = scale.upper()

    # Best non-blocked assessment
    best = next(
        (a for a in assessments if a.decision not in ("BLOCK", "UNSUPPORTED")),
        None,
    )
    rt_name = best.runtime.value if best else "ONNX_CPU"

    return DeploymentProfilePrediction(
        latency_class=lat_class,
        memory_class=mem_class,
        recommended_runtime=rt_name,
    )

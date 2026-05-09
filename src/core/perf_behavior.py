"""
core/perf_behavior.py

Classify execution bound and GPU scaling from facts + perf risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionBehavior:
    bound: str          # e.g. "compute-bound" | "memory-bound" | "io-bound"
    gpu_scaling: str    # e.g. "LINEAR" | "SUBLINEAR" | "NONE"


def classify_execution_behavior(
    facts: dict,
    perf_risk: Any,
    model_summary: Any,
) -> ExecutionBehavior:
    """Classify the runtime execution behaviour of the model."""
    has_attention = bool(facts.get("model.has_attention", False))
    has_conv = bool(facts.get("model.has_conv", False))
    has_conv_transpose = bool(facts.get("model.has_conv_transpose", False))

    gpu_viable = getattr(perf_risk, "realtime_gpu_viable", False)

    if has_attention:
        bound = "compute-bound"
        gpu_scaling = "LINEAR" if gpu_viable else "NONE"
    elif has_conv or has_conv_transpose:
        bound = "memory-bound"
        gpu_scaling = "SUBLINEAR" if gpu_viable else "NONE"
    else:
        bound = "io-bound"
        gpu_scaling = "NONE"

    return ExecutionBehavior(bound=bound, gpu_scaling=gpu_scaling)

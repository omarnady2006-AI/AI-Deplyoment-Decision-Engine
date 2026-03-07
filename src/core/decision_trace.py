from __future__ import annotations
"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""
src/core/decision_trace.py

Structured decision trace for debugging and UI explainability.
Constructed inside /api/decision/recommend and attached to the response
only when the caller passes ?debug=true.

No external dependencies. No global state. No APP_STATE mutation.
"""


from typing import List, Optional


class DecisionTrace:
    """
    Immutable value object capturing every decision variable produced
    inside /api/decision/recommend in a single request lifecycle.
    """

    __slots__ = (
        "selected_runtime_before_override",
        "selected_runtime_after_override",
        "sla_target_latency",
        "measured_latency",
        "memory_limit",
        "measured_memory",
        "rejection_reasons",
        "conditional_reasons",
        "confidence_before_scaling",
        "confidence_after_scaling",
        "hardware_override_applied",
        "override_reason",
    )

    def __init__(
        self,
        selected_runtime_before_override: str,
        selected_runtime_after_override: str,
        sla_target_latency: Optional[float],
        measured_latency: Optional[float],
        memory_limit: Optional[float],
        measured_memory: Optional[float],
        rejection_reasons: List[str],
        conditional_reasons: List[str],
        confidence_before_scaling: float,
        confidence_after_scaling: float,
        hardware_override_applied: bool,
        override_reason: Optional[str],
    ) -> None:
        self.selected_runtime_before_override = selected_runtime_before_override
        self.selected_runtime_after_override = selected_runtime_after_override
        self.sla_target_latency = sla_target_latency
        self.measured_latency = measured_latency
        self.memory_limit = memory_limit
        self.measured_memory = measured_memory
        self.rejection_reasons = list(rejection_reasons)
        self.conditional_reasons = list(conditional_reasons)
        self.confidence_before_scaling = confidence_before_scaling
        self.confidence_after_scaling = confidence_after_scaling
        self.hardware_override_applied = hardware_override_applied
        self.override_reason = override_reason

    def to_dict(self) -> dict:
        return {
            "selected_runtime_before_override": self.selected_runtime_before_override,
            "selected_runtime_after_override": self.selected_runtime_after_override,
            "sla_target_latency": self.sla_target_latency,
            "measured_latency": self.measured_latency,
            "memory_limit": self.memory_limit,
            "measured_memory": self.measured_memory,
            "rejection_reasons": self.rejection_reasons,
            "conditional_reasons": self.conditional_reasons,
            "confidence_before_scaling": self.confidence_before_scaling,
            "confidence_after_scaling": self.confidence_after_scaling,
            "hardware_override_applied": self.hardware_override_applied,
            "override_reason": self.override_reason,
        }

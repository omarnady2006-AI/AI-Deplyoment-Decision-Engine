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
Explainability Module

Provides explanation tree generation for deployment decisions.
"""


from typing import Any


def build_explanation_tree(trace: Any) -> dict[str, Any]:
    """
    Build an explanation tree from a decision trace.
    
    Args:
        trace: DecisionTrace object containing decision reasoning
        
    Returns:
        Dict with structured explanation of the decision
    """
    return {
        "decision_summary": {
            "selected_runtime": getattr(trace, "selected_runtime_after_override", "unknown"),
            "final_status": "decided",
            "confidence": getattr(trace, "confidence_after_scaling", 0.5),
        },
        "reasoning_chain": [
            {
                "step": "analysis",
                "description": "Model analysis completed",
                "details": {},
            },
            {
                "step": "benchmark",
                "description": "Runtime benchmarks executed",
                "details": {},
            },
            {
                "step": "sla_check",
                "description": "SLA constraints evaluated",
                "details": {
                    "sla_target_latency": getattr(trace, "sla_target_latency", None),
                    "measured_latency": getattr(trace, "measured_latency", None),
                },
            },
            {
                "step": "hardware_override",
                "description": "Hardware-aware selection applied",
                "details": {
                    "override_applied": getattr(trace, "hardware_override_applied", False),
                    "override_reason": getattr(trace, "override_reason", None),
                },
            },
        ],
        "factors": {
            "latency_factor": {
                "weight": 0.4,
                "value": getattr(trace, "measured_latency", 0) or 0,
            },
            "memory_factor": {
                "weight": 0.3,
                "value": getattr(trace, "measured_memory", 0) or 0,
            },
            "sla_factor": {
                "weight": 0.3,
                "value": "passed" if getattr(trace, "rejection_reasons", []) == [] else "failed",
            },
        },
    }

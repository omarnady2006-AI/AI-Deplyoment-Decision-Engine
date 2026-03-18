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
Advisor Report Module

Generates human-readable advisory reports for deployment decisions.
"""


import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AdvisorInput:
    model_path: str
    model_hash: str
    selected_runtime: str
    decision_status: str
    confidence: float
    predicted_latency_ms: float | None = None
    predicted_memory_mb: float | None = None
    sla_target_latency_ms: float | None = None
    sla_memory_limit_mb: float | None = None
    operator_count: int = 0
    parameter_count: int = 0
    has_dynamic_shapes: bool = False
    unsupported_ops: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    conditional_reasons: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "selected_runtime": self.selected_runtime,
            "decision_status": self.decision_status,
            "confidence": self.confidence,
            "predicted_latency_ms": self.predicted_latency_ms,
            "predicted_memory_mb": self.predicted_memory_mb,
            "sla_target_latency_ms": self.sla_target_latency_ms,
            "sla_memory_limit_mb": self.sla_memory_limit_mb,
            "operator_count": self.operator_count,
            "parameter_count": self.parameter_count,
            "has_dynamic_shapes": self.has_dynamic_shapes,
            "unsupported_ops": self.unsupported_ops,
            "warnings": self.warnings,
            "rejection_reasons": self.rejection_reasons,
            "conditional_reasons": self.conditional_reasons,
        }


def generate_advisor_report(input_data: AdvisorInput) -> dict[str, Any]:
    """
    Generate a human-readable advisory report for the deployment decision.
    
    Args:
        input_data: AdvisorInput containing decision details
        
    Returns:
        Dict with structured advisory report
    """
    sections: list[dict[str, Any]] = []
    
    sections.append({
        "title": "Summary",
        "content": _generate_summary(input_data),
    })
    
    if input_data.decision_status == "APPROVED":
        sections.append({
            "title": "Recommendation",
            "content": _generate_approved_recommendation(input_data),
        })
    elif input_data.decision_status == "CONDITIONAL_APPROVAL":
        sections.append({
            "title": "Conditional Approval",
            "content": _generate_conditional_recommendation(input_data),
        })
    elif input_data.decision_status == "REJECTED":
        sections.append({
            "title": "Rejection Analysis",
            "content": _generate_rejection_analysis(input_data),
        })
    
    if input_data.unsupported_ops:
        sections.append({
            "title": "Unsupported Operators",
            "content": _generate_unsupported_ops_section(input_data),
        })
    
    if input_data.has_dynamic_shapes:
        sections.append({
            "title": "Dynamic Shapes Warning",
            "content": "This model has dynamic input shapes. Ensure your inference pipeline handles variable input dimensions correctly.",
        })
    
    sections.append({
        "title": "Performance Estimates",
        "content": _generate_performance_section(input_data),
    })
    
    sections.append({
        "title": "Confidence Analysis",
        "content": _generate_confidence_section(input_data),
    })
    
    return {
        "model_hash": input_data.model_hash,
        "selected_runtime": input_data.selected_runtime,
        "decision_status": input_data.decision_status,
        "confidence": input_data.confidence,
        "sections": sections,
        "generated_at": time.time(),
    }


def _generate_summary(input_data: AdvisorInput) -> str:
    status_emoji = {
        "APPROVED": "✓",
        "CONDITIONAL_APPROVAL": "⚠",
        "REJECTED": "✗",
    }
    
    emoji = status_emoji.get(input_data.decision_status, "?")
    
    summary = (
        f"{emoji} **{input_data.decision_status}** for runtime **{input_data.selected_runtime}**\n\n"
        f"Model: `{input_data.model_path}`\n"
        f"Confidence: {input_data.confidence:.0%}\n"
    )
    
    if input_data.parameter_count > 0:
        summary += f"Parameters: {input_data.parameter_count:,}\n"
    
    if input_data.operator_count > 0:
        summary += f"Operators: {input_data.operator_count}\n"
    
    return summary


def _generate_approved_recommendation(input_data: AdvisorInput) -> str:
    return (
        f"This model is approved for deployment on {input_data.selected_runtime}.\n\n"
        "Recommended actions:\n"
        "1. Deploy the model to your target environment\n"
        "2. Monitor initial inference performance\n"
        "3. Validate output accuracy with test data"
    )


def _generate_conditional_recommendation(input_data: AdvisorInput) -> str:
    reasons = "\n".join(f"- {r}" for r in input_data.conditional_reasons)
    return (
        "This model has conditional approval for deployment.\n\n"
        f"Conditions to address:\n{reasons}\n\n"
        "Recommended actions:\n"
        "1. Review the conditions above\n"
        "2. Test thoroughly in a staging environment\n"
        "3. Consider optimizations to address warnings"
    )


def _generate_rejection_analysis(input_data: AdvisorInput) -> str:
    reasons = "\n".join(f"- {r}" for r in input_data.rejection_reasons)
    return (
        "This model was rejected for deployment.\n\n"
        f"Rejection reasons:\n{reasons}\n\n"
        "Recommended actions:\n"
        "1. Address the issues listed above\n"
        "2. Consider model optimization or alternative runtimes\n"
        "3. Re-run analysis after changes"
    )


def _generate_unsupported_ops_section(input_data: AdvisorInput) -> str:
    ops = ", ".join(input_data.unsupported_ops)
    return (
        "The following operators may not be supported on all backends:\n\n"
        f"{ops}\n\n"
        "Consider verifying operator support for your target runtime."
    )


def _generate_performance_section(input_data: AdvisorInput) -> str:
    lines = ["Estimated performance characteristics:\n"]
    
    if input_data.predicted_latency_ms is not None:
        lines.append(f"- Predicted latency: {input_data.predicted_latency_ms:.2f} ms")
        
        if input_data.sla_target_latency_ms is not None:
            sla_status = "meets" if input_data.predicted_latency_ms <= input_data.sla_target_latency_ms else "exceeds"
            lines.append(f"- SLA target ({input_data.sla_target_latency_ms:.2f} ms): {sla_status}")
    
    if input_data.predicted_memory_mb is not None:
        lines.append(f"- Predicted memory: {input_data.predicted_memory_mb:.2f} MB")
        
        if input_data.sla_memory_limit_mb is not None:
            mem_status = "within" if input_data.predicted_memory_mb <= input_data.sla_memory_limit_mb else "exceeds"
            lines.append(f"- Memory limit ({input_data.sla_memory_limit_mb:.2f} MB): {mem_status}")
    
    return "\n".join(lines)


def _generate_confidence_section(input_data: AdvisorInput) -> str:
    if input_data.confidence >= 0.8:
        level = "HIGH"
        description = "The decision has high confidence based on successful benchmarks and SLA compliance."
    elif input_data.confidence >= 0.5:
        level = "MEDIUM"
        description = "The decision has medium confidence. Some factors may affect reliability."
    else:
        level = "LOW"
        description = "The decision has low confidence. Review warnings and consider additional testing."
    
    return f"Confidence Level: **{level}** ({input_data.confidence:.0%})\n\n{description}"

"""
core/advisor_report.py

AdvisorInput dataclass and generate_advisor_report factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdvisorInput:
    model_path: str
    model_hash: str
    selected_runtime: str
    decision_status: str
    confidence: float
    predicted_latency_ms: float | None
    predicted_memory_mb: float | None
    operator_count: int
    parameter_count: int
    has_dynamic_shapes: bool
    unsupported_ops: list
    warnings: list
    rejection_reasons: list
    conditional_reasons: list


def generate_advisor_report(input: AdvisorInput) -> dict:
    """
    Produce a human-readable advisor report dict from an AdvisorInput.

    The 'report' key contains a formatted string summary; all other keys
    expose individual fields for programmatic consumption.
    """
    lines = [
        f"Model      : {input.model_path}",
        f"Hash       : {input.model_hash}",
        f"Runtime    : {input.selected_runtime}",
        f"Decision   : {input.decision_status}",
        f"Confidence : {input.confidence:.2f}",
        f"Operators  : {input.operator_count}",
        f"Parameters : {input.parameter_count:,}",
        f"Dynamic    : {input.has_dynamic_shapes}",
    ]

    if input.rejection_reasons:
        lines.append("")
        lines.append("FAILURES:")
        for r in input.rejection_reasons:
            lines.append(f"  ✗ {r}")

    if input.conditional_reasons:
        lines.append("")
        lines.append("CONDITIONS:")
        for c in input.conditional_reasons:
            lines.append(f"  ⚠ {c}")

    if input.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in input.warnings:
            lines.append(f"  ! {w}")

    if input.unsupported_ops:
        lines.append("")
        lines.append(f"Unsupported ops: {', '.join(input.unsupported_ops)}")

    return {
        "report":            "\n".join(lines),
        "model_path":        input.model_path,
        "selected_runtime":  input.selected_runtime,
        "decision_status":   input.decision_status,
        "confidence":        input.confidence,
        "operator_count":    input.operator_count,
        "parameter_count":   input.parameter_count,
        "has_dynamic_shapes":input.has_dynamic_shapes,
        "unsupported_ops":   input.unsupported_ops,
        "warnings":          input.warnings,
        "rejection_reasons": input.rejection_reasons,
        "conditional_reasons": input.conditional_reasons,
        "predicted_latency_ms": input.predicted_latency_ms,
        "predicted_memory_mb":  input.predicted_memory_mb,
    }

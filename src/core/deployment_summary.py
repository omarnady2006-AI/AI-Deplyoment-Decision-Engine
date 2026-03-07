"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

from typing import Any


def build_deployment_summary(
    decision: dict[str, Any],
    analysis: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    status = decision.get("status", "REJECTED")
    result = "READY_TO_DEPLOY" if status == "APPROVED" else "NEEDS_FIXES"
    
    recommended_runtime = decision.get("runtime", "ONNX_CPU")
    confidence = float(decision.get("confidence", 0.5))
    
    evaluations = analysis.get("evaluations", [])
    best_eval = {}
    for ev in evaluations:
        if isinstance(ev, dict) and ev.get("runtime") == recommended_runtime:
            best_eval = ev
            break
    
    if not best_eval and evaluations:
        best_eval = evaluations[0] if isinstance(evaluations[0], dict) else {}
    
    estimated_latency_ms = float(best_eval.get("predicted_latency_ms", 0.0) or 0.0)
    estimated_memory_mb = float(best_eval.get("memory_usage_mb", 0.0) or 0.0)
    
    risk_level = decision.get("risk_level", "MEDIUM")
    
    issues: list[str] = decision.get("reasons", [])
    
    suggestions: list[str] = _generate_suggestions(decision, analysis, diagnostics, best_eval)
    
    return {
        "result": result,
        "recommended_runtime": recommended_runtime,
        "estimated_latency_ms": estimated_latency_ms,
        "estimated_memory_mb": estimated_memory_mb,
        "risk_level": risk_level,
        "confidence": confidence,
        "issues": issues,
        "suggestions": suggestions,
    }


def _generate_suggestions(
    decision: dict[str, Any],
    analysis: dict[str, Any],
    diagnostics: dict[str, Any],
    best_eval: dict[str, Any],
) -> list[str]:
    suggestions: list[str] = []
    
    deployment_profile = analysis.get("deployment_profile", {})
    target_latency_ms = deployment_profile.get("target_latency_ms")
    memory_limit_mb = deployment_profile.get("memory_limit_mb")
    
    predicted_latency = best_eval.get("predicted_latency_ms")
    if predicted_latency is not None:
        predicted_latency = float(predicted_latency)
    
    memory_usage = best_eval.get("memory_usage_mb")
    if memory_usage is not None:
        memory_usage = float(memory_usage)
    
    if (
        target_latency_ms is not None
        and predicted_latency is not None
        and predicted_latency > float(target_latency_ms)
    ):
        suggestions.append("Reduce batch size to meet latency SLA")
        suggestions.append("Use TensorRT for optimized inference")
    
    if (
        memory_limit_mb is not None
        and memory_usage is not None
        and memory_usage > float(memory_limit_mb) * 0.8
    ):
        suggestions.append("Lower batch size to reduce memory footprint")
        suggestions.append("Increase memory limit in deployment profile")
    
    raw_analysis = diagnostics.get("raw_analysis", {})
    has_dynamic_shapes = raw_analysis.get("has_dynamic_shapes", False)
    if has_dynamic_shapes:
        suggestions.append("Export static ONNX with fixed input shapes")
    
    stress_test = best_eval.get("stress_test", {})
    if isinstance(stress_test, dict) and stress_test.get("memory_stability") == "UNSTABLE":
        suggestions.append("Run memory optimization pass on model")
    
    return suggestions

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
    deployment_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = decision.get("status", "REJECTED")
    result = "READY_TO_DEPLOY" if status == "APPROVED" else "NEEDS_FIXES"
    
    recommended_runtime = decision.get("selected_runtime", "ONNX_CPU")
    _conf_raw = decision.get("confidence", 0.5)
    confidence = float(_conf_raw["score"] if isinstance(_conf_raw, dict) else _conf_raw)
    
    evaluations = analysis.get("evaluations", [])
    best_eval = {}
    for ev in evaluations:
        if isinstance(ev, dict) and ev.get("runtime") == recommended_runtime:
            best_eval = ev
            break
    
    if not best_eval and evaluations:
        best_eval = evaluations[0] if isinstance(evaluations[0], dict) else {}
    
    estimated_latency_ms = float(best_eval.get("latency_avg_ms", 0.0) or 0.0)
    estimated_memory_mb = float(best_eval.get("memory_mb", 0.0) or 0.0)
    
    risk_level = decision.get("risk_level", "MEDIUM")
    
    issues: list[str] = decision.get("reasons", [])

    # N-02: target_latency_ms and memory_limit_mb live on the deployment profile,
    # not on the decision dict.  Pass them from the correct source object.
    _profile = deployment_profile or {}
    suggestions: list[str] = _generate_suggestions(
        decision, analysis, diagnostics, best_eval,
        target_latency_ms=_profile.get("target_latency_ms"),
        memory_limit_mb=_profile.get("memory_limit_mb"),
    )
    
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
    *,
    target_latency_ms: float | None = None,
    memory_limit_mb: float | None = None,
) -> list[str]:
    suggestions: list[str] = []
    
    predicted_latency = best_eval.get("latency_avg_ms")
    if predicted_latency is not None:
        predicted_latency = float(predicted_latency)
    
    memory_usage = best_eval.get("memory_mb")
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

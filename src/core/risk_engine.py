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
Deployment Risk Engine

Pure deterministic risk evaluation for deployment decisions.

Risk depends ONLY on:
- Model analysis (has_dynamic_shapes, parameter_scale_class, operator_count)
- Runtime metrics (predicted_latency_ms, memory_usage_mb, memory_growth_mb)
- SLA outcome (sla_passed)
- Latency margin (how close to SLA threshold)
- Confidence level

NOT hardware parameters directly. Hardware affects latency scaling which affects SLA.
"""


HIGH_OPERATOR_COUNT_THRESHOLD = 500
MEMORY_GROWTH_THRESHOLD_MB = 5.0
LOW_CONFIDENCE_THRESHOLD = 0.7


def evaluate_deployment_risk(
    analysis: dict,
    deployment_profile: dict,
    predicted_latency_ms: float,
    memory_usage_mb: float,
    sla_target_latency: float,
    confidence_before_scaling: float,
) -> dict:
    """
    Evaluate deployment risk based on analysis and runtime metrics.
    
    Risk depends on:
    - SLA pass/fail (discrete)
    - Latency margin (proximity to SLA threshold)
    - Memory within limit (discrete)
    - Confidence level
    - Model characteristics (parameter_scale_class, operator_count, has_dynamic_shapes)
    
    Returns:
        dict with "risk_score" (int), "risk_level" (str), "sla_passed" (bool), "latency_margin" (float)
    """
    risk_score = 0
    
    has_dynamic_shapes = analysis.get("has_dynamic_shapes") is True
    if has_dynamic_shapes:
        risk_score += 2
    
    memory_growth_mb = analysis.get("memory_growth_mb", 0)
    if memory_growth_mb > MEMORY_GROWTH_THRESHOLD_MB:
        risk_score += 2
    
    if confidence_before_scaling < LOW_CONFIDENCE_THRESHOLD:
        risk_score += 3
    
    parameter_scale_class = analysis.get("parameter_scale_class", "small")
    if parameter_scale_class == "large":
        risk_score += 1
    
    operator_count = analysis.get("operator_count", 0)
    if operator_count > HIGH_OPERATOR_COUNT_THRESHOLD:
        risk_score += 1
    
    sla_passed = True
    latency_margin = 0.0
    sla_target_latency_val = sla_target_latency if sla_target_latency > 0 else None
    predicted_latency_val = predicted_latency_ms if predicted_latency_ms > 0 else None
    
    if sla_target_latency_val is not None and sla_target_latency_val > 0 and predicted_latency_val is not None:
        latency_margin = predicted_latency_val / sla_target_latency_val
        sla_passed = predicted_latency_val <= sla_target_latency_val
        
        # SLA failure penalties (existing logic)
        if not sla_passed:
            if predicted_latency_val > 1.5 * sla_target_latency_val:
                risk_score += 4
            else:
                risk_score += 2
        
        # Latency margin awareness (even when SLA passes)
        if latency_margin > 0.9:
            risk_score += 2
        elif latency_margin > 0.7:
            risk_score += 1
    
    memory_limit_mb = deployment_profile.get("memory_limit_mb") if deployment_profile else None
    memory_within_limit = True
    if memory_limit_mb is not None and memory_limit_mb > 0:
        memory_within_limit = memory_usage_mb <= memory_limit_mb
        if not memory_within_limit:
            if memory_usage_mb > 1.5 * memory_limit_mb:
                risk_score += 4
            else:
                risk_score += 2
    
    if risk_score <= 2:
        risk_level = "LOW_RISK"
    elif risk_score <= 5:
        risk_level = "MEDIUM_RISK"
    else:
        risk_level = "HIGH_RISK"
    
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "sla_passed": sla_passed,
        "memory_within_limit": memory_within_limit,
        "latency_margin": round(latency_margin, 3),
    }

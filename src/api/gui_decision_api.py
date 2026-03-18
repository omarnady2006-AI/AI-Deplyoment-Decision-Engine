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
GUI Decision API

Handles runtime selection, decision pipeline, and advisor reports.
"""

import logging
from pathlib import Path
from typing import Any

from src.core.decision import DeploymentDecision
from src.core.model_analysis import ModelAnalysisResult
from src.core.runtime import RuntimeName
from src.core.runtime_selector import recommend_runtime, RuntimeRecommendation
from src.core.pipeline import PipelineResult, run_pipeline, run_decision_only
from src.diagnostics.report import Diagnostic


logger = logging.getLogger(__name__)


def compute_deployment_decision(
    diagnostics: list[Diagnostic],
    constraints: dict[str, Any] | None = None,
    model_path: str | None = None,
    analysis: ModelAnalysisResult | None = None,
) -> tuple[str, float, str]:
    """
    Deprecated shim: delegate deployment decision to pipeline scoring authority.
    
    Returns:
        Tuple of (decision, confidence, risk_level)
    """
    if analysis is not None:
        decision, confidence = run_decision_only(analysis=analysis, diagnostics=diagnostics)
    elif model_path:
        result = run_pipeline(model_path=model_path, constraints=constraints)
        decision, confidence = result.decision, result.confidence
    else:
        raise ValueError("compute_deployment_decision requires either analysis or model_path")

    if decision == DeploymentDecision.BLOCK.value:
        risk_level = "CRITICAL"
    elif decision == DeploymentDecision.ALLOW_WITH_CONDITIONS.value:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return decision, confidence, risk_level


def select_runtime(
    facts: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    validations: dict[RuntimeName, Any] | None = None,
) -> RuntimeName:
    """
    Select the best runtime for deployment.
    
    Args:
        facts: Model analysis facts
        constraints: Deployment constraints
        validations: Runtime validation results
        
    Returns:
        Selected runtime
    """
    from src.rules.unsupported_operator import UnsupportedOperatorRule
    from src.core.rule import run_rules
    
    rules = [UnsupportedOperatorRule()]
    
    try:
        recommendation = recommend_runtime(
            rules=rules,
            facts=facts,
            validations_by_runtime=validations,
            deployment_profile=constraints,
        )
        return recommendation.best_runtime
    except Exception as e:
        logger.warning(f"runtime_selection_fallback: {e}")
        return RuntimeName.ONNXRUNTIME


def generate_advisor_report(
    model_path: str,
    decision: str,
    confidence: float,
    runtime: RuntimeName,
    diagnostics: list[Diagnostic],
) -> dict[str, Any]:
    """
    Generate an advisor report for deployment.
    
    Returns:
        Advisor report dict
    """
    blocking_reasons = [
        d.message for d in diagnostics if d.severity == "FAIL"
    ]
    
    warnings = [
        d.message for d in diagnostics if d.severity == "WARN"
    ]
    
    recommendations = []
    
    if decision == DeploymentDecision.BLOCK.value:
        recommendations.append("Do not deploy this model in production")
        recommendations.extend(f"Resolve: {r}" for r in blocking_reasons)
    elif decision == DeploymentDecision.ALLOW_WITH_CONDITIONS.value:
        recommendations.append("Deploy with monitoring and guardrails")
        recommendations.extend(f"Monitor: {w}" for w in warnings)
    else:
        recommendations.append("Safe for production deployment")
    
    return {
        "model_path": model_path,
        "decision": decision,
        "confidence": confidence,
        "recommended_runtime": runtime.value,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "recommendations": recommendations,
        "risk_level": "CRITICAL" if decision == DeploymentDecision.BLOCK.value else "MEDIUM" if warnings else "LOW",
    }


def format_decision_summary(result: PipelineResult) -> str:
    """Format a decision summary for display."""
    if not result.success:
        return f"Analysis failed: {result.error}"
    
    lines = [
        f"Model: {Path(result.model_path).name}",
        f"Decision: {result.decision}",
        f"Confidence: {result.confidence:.2%}",
        f"Diagnostics: {len(result.diagnostics)} issues",
    ]
    
    if result.recommended_runtime:
        lines.append(f"Recommended Runtime: {result.recommended_runtime.value}")
    
    return "\n".join(lines)

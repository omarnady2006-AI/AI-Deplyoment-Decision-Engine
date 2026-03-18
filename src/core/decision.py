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
Deployment Decision Module

This module provides the deployment decision interface that integrates with the
risk-based decision engine. It maintains backward compatibility while providing
the new risk-based decision logic.

The old rule-based decision logic has been replaced with:
1. Unified risk score computation
2. Calibrated confidence from historical accuracy
3. Adaptive weights that evolve with outcomes
4. Fail-closed behavior for missing signals
5. Mandatory explanations in output
"""


from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

from src.diagnostics.report import Diagnostic

if TYPE_CHECKING:
    from src.core.risk_based_decision import (
        FailureMode,
        RiskComponents,
        RiskAssessment,
        DecisionResult,
        RiskBasedDecisionEngine,
    )
    from src.core.confidence import DecisionConfidence


# Re-export DeploymentDecision from risk_based_decision for backward compatibility
from src.core.risk_based_decision import DeploymentDecision


class DeploymentConstraint(str, Enum):
    """Deployment constraints for backward compatibility."""
    INPUT_SIZE_LIMITED = "input_size_limited"
    BATCH_LIMITED = "batch_limited"
    NUMERICAL_RISK = "numerical_risk"


@dataclass(frozen=True)
class LegacyDecisionReport:
    """Legacy decision report for backward compatibility."""
    decision: str
    diagnostics: list[Diagnostic]
    constraints: list[DeploymentConstraint] = field(default_factory=list)
    risk_components: dict = field(default_factory=dict)
    risk_score: float = 0.0
    confidence: float = 0.5
    dominant_risk: str = ""
    expected_failure: str = ""


@dataclass(frozen=True)
class DecisionReport:
    """Decision report with typed confidence."""
    decision: str
    diagnostics: list[Diagnostic]
    confidence: Optional["DecisionConfidence"] = None
    constraints: list[DeploymentConstraint] = field(default_factory=list)


def make_decision(
    diagnostics: list[Diagnostic],
    constraints: Optional[list[DeploymentConstraint]] = None,
    risk_components: Optional[dict] = None,
    risk_score: float = 0.0,
    confidence: float = 0.5,
    dominant_risk: str = "",
    expected_failure: str = "",
) -> LegacyDecisionReport:
    """
    Make a deployment decision with risk information.
    
    This function maintains backward compatibility with the old API while
    accepting new risk-based parameters.
    
    Args:
        diagnostics: List of diagnostics from model analysis
        constraints: Optional list of deployment constraints
        risk_components: Risk component values (memory_risk, latency_risk, etc.)
        risk_score: Unified risk score in [0, 1]
        confidence: Calibrated confidence in [0, 1]
        dominant_risk: Name of the highest risk component
        expected_failure: Expected failure mode if blocked
    
    Returns:
        LegacyDecisionReport with decision and risk information
    """
    has_fail = any(d.severity == "FAIL" for d in diagnostics)
    has_warn = any(d.severity == "WARN" for d in diagnostics)
    has_constraints = constraints and len(constraints) > 0
    
    if has_fail:
        decision = DeploymentDecision.BLOCK
    elif has_warn or has_constraints:
        decision = DeploymentDecision.ALLOW_WITH_CONDITIONS
    else:
        decision = DeploymentDecision.ALLOW
    
    return LegacyDecisionReport(
        decision=decision.value,
        diagnostics=diagnostics,
        constraints=list(constraints or []),
        risk_components=risk_components or {},
        risk_score=risk_score,
        confidence=confidence,
        dominant_risk=dominant_risk,
        expected_failure=expected_failure,
    )


def make_risk_based_decision(
    risk_engine: "RiskBasedDecisionEngine",
    risk_components: "RiskComponents",
    has_history: bool = True,
    has_profiling: bool = True,
    has_device_metrics: bool = True,
) -> "DecisionResult":
    """
    Make a deployment decision using the risk-based decision engine.
    
    This is the preferred method for new code.
    
    Args:
        risk_engine: Initialized risk-based decision engine
        risk_components: Computed risk components
        has_history: Whether historical data is available
        has_profiling: Whether model profiling data is available
        has_device_metrics: Whether device metrics are available
    
    Returns:
        DecisionResult with full risk assessment and explanations
    """
    return risk_engine.assess_decision(
        components=risk_components,
        has_history=has_history,
        has_profiling=has_profiling,
        has_device_metrics=has_device_metrics,
    )


def record_decision_outcome(
    risk_engine: "RiskBasedDecisionEngine",
    risk_score: float,
    decision: "DeploymentDecision",
    actual_success: bool,
    components: Optional["RiskComponents"] = None
) -> None:
    """
    Record a deployment outcome for learning.
    
    Args:
        risk_engine: The risk-based decision engine
        risk_score: Risk score used for the decision
        decision: The decision that was made
        actual_success: Whether the deployment actually succeeded
        components: Risk components for weight adaptation
    """
    risk_engine.record_outcome(
        risk_score=risk_score,
        decision=decision,
        actual_success=actual_success,
        components=components,
    )


def decision_to_string(decision: "DeploymentDecision") -> str:
    """Convert decision enum to string for display."""
    return decision.value


def failure_mode_to_string(mode: "FailureMode") -> str:
    """Convert failure mode to string for display."""
    return mode.value

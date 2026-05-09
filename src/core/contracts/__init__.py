"""
core/contracts/

Typed contracts that cross layer boundaries in the ML decision pipeline.

    ModelAnalysis       — immutable parsed model facts (format_router -> pipeline)
    RuntimeEvaluation   — immutable runtime benchmark result (evaluator -> pipeline)
    MLDecisionResult    — final ML deployment decision (pipeline -> API -> UI)
"""
from src.core.contracts.model_analysis import ModelAnalysis
from src.core.contracts.runtime_evaluation import RuntimeEvaluation
from src.core.contracts.decision_result import (
    MLDecisionResult,
    build_ml_decision_result,
    # backward-compat aliases
    DecisionResult,
    build_decision_result,
    VALID_DEPLOYMENT_TARGETS,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
)

__all__ = [
    "ModelAnalysis",
    "RuntimeEvaluation",
    "MLDecisionResult",
    "build_ml_decision_result",
    # compat aliases
    "DecisionResult",
    "build_decision_result",
    "VALID_DEPLOYMENT_TARGETS",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
]

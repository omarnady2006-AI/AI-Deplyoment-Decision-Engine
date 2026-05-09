"""
core/decision.py

Deployment decision vocabulary for the CLI and other non-pipeline callers.
Previously re-exported from risk_based_decision; now defined here directly.
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Any

from src.core.contracts.decision_result import (
    VALID_DEPLOYMENT_TARGETS,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
)


class DeploymentDecision(str, Enum):
    """ML deployment target labels."""
    EDGE_INT8  = "edge_int8"
    EDGE_FP16  = "edge_fp16"
    CLOUD_CPU  = "cloud_cpu"
    CLOUD_GPU  = "cloud_gpu"

    # Legacy aliases kept for CLI compatibility — map to nearest ML target
    SUPPORTED              = "edge_int8"
    SUPPORTED_WITH_WARNINGS = "edge_fp16"
    ALLOW                  = "edge_int8"
    ALLOW_WITH_CONDITIONS  = "edge_fp16"
    BLOCK                  = "cloud_cpu"
    UNSUPPORTED            = "cloud_cpu"


@dataclass
class DecisionReport:
    """Per-runtime decision report (CLI pipeline, not ML decision path)."""
    decision: str
    diagnostics: list
    constraints: list
    confidence: Any = None

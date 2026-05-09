"""
core/global_decision.py

Compute a coarse global decision across all runtime evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GlobalDecision:
    status: str       # "APPROVED" | "CONDITIONAL" | "REJECTED"
    best_runtime: Any  # RuntimeName | None


def compute_global_decision(recommendation: Any) -> GlobalDecision:
    """Derive a global deployment status from the recommendation object."""
    best = getattr(recommendation, "best_runtime", None)
    evaluations = getattr(recommendation, "evaluations", [])

    best_eval = next((e for e in evaluations if e.runtime == best), None)
    decision = getattr(best_eval, "decision", "BLOCK") if best_eval else "BLOCK"

    if decision == "ALLOW":
        status = "APPROVED"
    elif decision == "ALLOW_WITH_CONDITIONS":
        status = "CONDITIONAL"
    else:
        status = "REJECTED"

    return GlobalDecision(status=status, best_runtime=best)

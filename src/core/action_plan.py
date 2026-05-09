"""
core/action_plan.py

Generate an action plan from recommendation, decision reports, and fix suggestions.
"""
from __future__ import annotations

from typing import Any


def generate_action_plan(
    recommendation: Any,
    decision_reports: dict,
    fix_reports: dict,
    deployment_profile: Any,
    model_summary: Any,
) -> dict:
    """
    Aggregate fix suggestions and deployment metadata into an action plan dict.
    """
    best = getattr(recommendation, "best_runtime", None)
    rt_name = best.value if hasattr(best, "value") else str(best)

    report = decision_reports.get(best)
    confidence_score = None
    if report and report.confidence:
        confidence_score = getattr(report.confidence, "score", None)

    all_fixes: list[dict] = []
    for fixes in fix_reports.values():
        all_fixes.extend(fixes)

    return {
        "selected_runtime": rt_name,
        "confidence": confidence_score,
        "fixes": all_fixes,
        "total_fixes": len(all_fixes),
        "deployment_class": getattr(deployment_profile, "latency_class", "UNKNOWN"),
        "recommended_runtime": getattr(deployment_profile, "recommended_runtime", rt_name),
    }

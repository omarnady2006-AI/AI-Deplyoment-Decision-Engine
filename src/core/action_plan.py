"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Union

from src.core.decision import LegacyDecisionReport, DecisionReport, DeploymentConstraint
from src.core.risk_based_decision import DeploymentDecision
from src.core.deployment_profile import DeploymentProfile
from src.core.fix_suggester import FixReport
from src.core.model_summary import ModelCapabilitySummary
from src.core.runtime import RuntimeName
from src.core.runtime_selector import RuntimeRecommendation


@dataclass(frozen=True)
class ActionStep:
    priority: int
    title: str
    reason: str
    actions: list[str]


@dataclass(frozen=True)
class ActionPlan:
    target_runtime: RuntimeName
    steps: list[ActionStep]
    summary: str


def _base_goal_step(decision: DeploymentDecision) -> ActionStep:
    if decision == DeploymentDecision.UNSUPPORTED:
        return ActionStep(
            priority=0,
            title="Make runtime supported",
            reason="Current best runtime is not deployable.",
            actions=[
                "Resolve FAIL diagnostics blocking execution.",
                "Apply operator-level compatibility fixes and re-export the model.",
                "Re-run validation until runtime becomes supported.",
            ],
        )
    if decision == DeploymentDecision.SUPPORTED_WITH_WARNINGS:
        return ActionStep(
            priority=0,
            title="Reduce runtime warnings",
            reason="Model is deployable but has caveats that may affect reliability or speed.",
            actions=[
                "Address warning diagnostics to reduce fallback execution paths.",
                "Benchmark latency and throughput after each model change.",
            ],
        )
    return ActionStep(
        priority=0,
        title="Optimize production performance",
        reason="Model is deployable; next work should improve efficiency and robustness.",
        actions=[
            "Profile bottlenecks in preprocessing, model execution, and postprocessing.",
            "Apply runtime-specific optimizations and verify output parity.",
        ],
    )


DecisionReportType = Union[LegacyDecisionReport, DecisionReport]


def generate_action_plan(
    recommendation: RuntimeRecommendation,
    decision_reports: Mapping[RuntimeName, DecisionReportType],
    fix_reports: Mapping[RuntimeName, FixReport],
    deployment_profile: DeploymentProfile,
    model_summary: ModelCapabilitySummary,
) -> ActionPlan:
    target_runtime = recommendation.best_runtime
    decision_report = decision_reports[target_runtime]
    fix_report = fix_reports[target_runtime]

    decision_enum = DeploymentDecision(decision_report.decision)
    steps: list[ActionStep] = [_base_goal_step(decision_enum)]

    for fix in fix_report.fixes:
        operator_name = fix.operator if fix.operator is not None else "operator compatibility"
        steps.append(
            ActionStep(
                priority=10,
                title=f"Fix {operator_name}",
                reason=fix.explanation,
                actions=list(fix.actions),
            )
        )

    warn_count = sum(1 for diagnostic in decision_report.diagnostics if diagnostic.severity == "WARN")
    if warn_count > 0:
        steps.append(
            ActionStep(
                priority=20,
                title="Resolve warning diagnostics",
                reason="Warnings indicate potential performance or behavior caveats.",
                actions=[
                    "Inspect warning diagnostics for fallback operators and unsupported fast paths.",
                    "Apply graph simplification or operator substitutions for warned nodes.",
                ],
            )
        )

    if deployment_profile.mobile == "not suitable":
        steps.append(
            ActionStep(
                priority=30,
                title="Make model mobile-friendly",
                reason="Current deployment profile indicates mobile targets are not suitable.",
                actions=[
                    "Quantize model to int8.",
                    "Reduce input resolution.",
                    "Remove dynamic shapes.",
                ],
            )
        )

    if deployment_profile.memory_pressure == "high":
        steps.append(
            ActionStep(
                priority=30,
                title="Reduce memory usage",
                reason="Deployment profile indicates high memory pressure.",
                actions=[
                    "Prune channels.",
                    "Apply weight compression.",
                    "Use fp16.",
                ],
            )
        )

    if model_summary.batching_support in {"limited", "single batch only"} or (
        DeploymentConstraint.BATCH_LIMITED in decision_report.constraints
    ):
        steps.append(
            ActionStep(
                priority=30,
                title="Enable batching",
                reason="Batching support is currently limited.",
                actions=[
                    "Fix dynamic batch dimension.",
                    "Avoid shape-changing ops.",
                ],
            )
        )

    if model_summary.numerical_stability == "dangerous":
        steps.append(
            ActionStep(
                priority=30,
                title="Fix numerical instability",
                reason="Validation indicates dangerous numerical behavior.",
                actions=[
                    "Add normalization.",
                    "Clamp activations.",
                ],
            )
        )

    ordered_steps = sorted(steps, key=lambda step: (step.priority, step.title))
    summary = (
        f"Target runtime is {target_runtime.value}. Focus on resolving blockers first, "
        "then apply deployment-fit improvements for memory, batching, and platform constraints."
    )

    return ActionPlan(
        target_runtime=target_runtime,
        steps=ordered_steps,
        summary=summary,
    )

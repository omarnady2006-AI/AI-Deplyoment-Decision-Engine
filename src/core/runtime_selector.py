"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from src.core.confidence import compute_confidence
from src.core.decision import DeploymentConstraint, DeploymentDecision, make_decision
from src.core.rule import Rule, run_rules
from src.core.runtime import RuntimeName
from src.diagnostics.report import Diagnostic
from src.validation.runtime_validator import RuntimeValidationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeEvaluation:
    runtime: RuntimeName
    decision: DeploymentDecision
    diagnostics: list[Diagnostic]
    constraints: list[DeploymentConstraint]
    utility_score: float


@dataclass(frozen=True)
class RuntimeRecommendation:
    best_runtime: RuntimeName
    evaluations: list[RuntimeEvaluation]


def _count_severity(diagnostics: list[Diagnostic], severity: str) -> int:
    return sum(1 for diagnostic in diagnostics if diagnostic.severity == severity)


def recommend_runtime(
    rules: Sequence[Rule],
    facts: dict[str, object],
    constraints_by_runtime: dict[RuntimeName, list[DeploymentConstraint]] | None = None,
    validations_by_runtime: dict[RuntimeName, RuntimeValidationResult] | None = None,
    gpu_scaling: str | None = None,
    reliability_level: str | None = None,
    reliability_score: float | None = None,
    runtime_priors: dict[RuntimeName, float] | dict[str, float] | None = None,
    deployment_profile: dict | None = None,
    benchmark_results: dict[RuntimeName, dict] | dict[str, dict] | None = None,
) -> RuntimeRecommendation:
    runtime_order = list(RuntimeName)

    # Build filtered runtime list from deployment_profile
    active_runtimes = list(runtime_order)
    gpu_available = True
    cuda_available = True
    memory_limit_mb = None
    cpu_cores = None
    target_latency_ms = None

    if deployment_profile is not None:
        gpu_available = bool(deployment_profile.get("gpu_available", True))
        cuda_available = bool(deployment_profile.get("cuda_available", True))
        memory_limit_mb = deployment_profile.get("memory_limit_mb")
        cpu_cores = deployment_profile.get("cpu_cores")
        target_latency_ms = deployment_profile.get("target_latency_ms")

    GPU_RUNTIMES = {RuntimeName.TENSORRT}
    CUDA_RUNTIMES = {RuntimeName.TENSORRT}
    CPU_HEAVY_RUNTIMES = {RuntimeName.ONNXRUNTIME}

    evaluations: list[RuntimeEvaluation] = []
    for runtime in runtime_order:
        diagnostics = run_rules(rules, facts, runtime)
        runtime_constraints = (
            list(constraints_by_runtime.get(runtime, []))
            if constraints_by_runtime is not None
            else []
        )
        decision_report = make_decision(diagnostics, constraints=runtime_constraints)
        decision_enum = DeploymentDecision(decision_report.decision)
        if validations_by_runtime is not None and runtime in validations_by_runtime:
            confidence = compute_confidence(
                decision=decision_enum,
                diagnostics=decision_report.diagnostics,
                constraints=decision_report.constraints,
                validation=validations_by_runtime[runtime],
                reliability_score=reliability_score,
            )
            confidence_score = confidence.score
        else:
            confidence_score = 1.0

        base_score = 0.0
        if decision_enum == DeploymentDecision.SUPPORTED or decision_enum == DeploymentDecision.ALLOW:
            base_score = 1.0
        elif decision_enum in (DeploymentDecision.SUPPORTED_WITH_WARNINGS, DeploymentDecision.ALLOW_WITH_CONDITIONS):
            base_score = 0.6
        utility_score = base_score * confidence_score

        benchmark = benchmark_results.get(runtime) if benchmark_results else None
        measured_latency_ms = None
        measured_memory_mb = None
        latency_variance = None
        
        if benchmark and isinstance(benchmark, dict):
            measured_latency_ms = benchmark.get("latency_avg_ms")
            measured_memory_mb = benchmark.get("memory_mb")
            lat_samples = benchmark.get("latency_samples_ms")
            if lat_samples and isinstance(lat_samples, list) and len(lat_samples) > 1:
                import statistics
                try:
                    latency_variance = statistics.variance(lat_samples)
                except statistics.StatisticsError:
                    latency_variance = None
            logger.debug(
                "benchmark_results_used",
                extra={
                    "runtime": runtime.value,
                    "measured_latency_ms": measured_latency_ms,
                    "measured_memory_mb": measured_memory_mb,
                    "latency_variance": latency_variance,
                }
            )

        if deployment_profile is not None:
            if not gpu_available and runtime in GPU_RUNTIMES:
                utility_score = 0.0
            if not cuda_available and runtime in CUDA_RUNTIMES:
                utility_score = 0.0
            
            if memory_limit_mb is not None:
                if measured_memory_mb is not None:
                    if float(measured_memory_mb) > float(memory_limit_mb):
                        utility_score *= 0.5
                        logger.debug(
                            "memory_penalty_applied",
                            extra={
                                "runtime": runtime.value,
                                "measured_mb": measured_memory_mb,
                                "limit_mb": memory_limit_mb,
                            }
                        )
                else:
                    logger.debug(
                        "memory_check_skipped",
                        extra={"runtime": runtime.value, "reason": "no_measured_memory"},
                    )
            
            if cpu_cores is not None and int(cpu_cores) < 2 and runtime in CPU_HEAVY_RUNTIMES:
                utility_score *= 0.7
            
            if target_latency_ms is not None and target_latency_ms > 0:
                if measured_latency_ms is not None:
                    if float(measured_latency_ms) > float(target_latency_ms):
                        overshoot = (float(measured_latency_ms) - float(target_latency_ms)) / float(target_latency_ms)
                        penalty = min(overshoot * 0.5, 0.5)
                        utility_score = max(0.0, utility_score - penalty)
                        logger.debug(
                            "latency_penalty_applied",
                            extra={
                                "runtime": runtime.value,
                                "measured_ms": measured_latency_ms,
                                "target_ms": target_latency_ms,
                                "penalty": penalty,
                            }
                        )
                else:
                    logger.debug(
                        "latency_check_skipped",
                        extra={"runtime": runtime.value, "reason": "no_measured_latency"},
                    )

        if runtime == RuntimeName.TENSORRT:
            bias_scale = 0.3 if reliability_level == "LOW" else 1.0
            if gpu_scaling == "HIGH":
                utility_score += 0.05 * bias_scale
            elif gpu_scaling == "MODERATE":
                utility_score += 0.02 * bias_scale
            elif gpu_scaling == "POOR":
                utility_score -= 0.03 * bias_scale
        if runtime_priors is not None:
            utility_score *= float(runtime_priors.get(runtime, 0.5))
        if utility_score < 0.0:
            utility_score = 0.0
        if utility_score > 1.0:
            utility_score = 1.0

        evaluations.append(
            RuntimeEvaluation(
                runtime=runtime,
                decision=decision_enum,
                diagnostics=decision_report.diagnostics,
                constraints=decision_report.constraints,
                utility_score=utility_score,
            )
        )

    def _decision_rank(evaluation: RuntimeEvaluation) -> int:
        if evaluation.decision in (DeploymentDecision.SUPPORTED, DeploymentDecision.ALLOW):
            return 2
        if evaluation.decision in (DeploymentDecision.SUPPORTED_WITH_WARNINGS, DeploymentDecision.ALLOW_WITH_CONDITIONS):
            return 1
        return 0

    def rank_key(evaluation: RuntimeEvaluation) -> tuple[float, int, int, int, int]:
        return (
            -evaluation.utility_score,
            -_decision_rank(evaluation),
            _count_severity(evaluation.diagnostics, "FAIL"),
            _count_severity(evaluation.diagnostics, "WARN"),
            runtime_order.index(evaluation.runtime),
        )

    best_evaluation = min(evaluations, key=rank_key)
    return RuntimeRecommendation(
        best_runtime=best_evaluation.runtime,
        evaluations=evaluations,
    )

"""
core/runtime_selector.py

recommend_runtime — select the best runtime from rules + validation results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.runtime import RuntimeName, CANONICAL_RUNTIMES
from src.core.rule import run_rules


# ── Internal evaluation record ────────────────────────────────────────────────

@dataclass
class _RuntimeEvaluation:
    runtime: RuntimeName
    decision: str          # "ALLOW" | "ALLOW_WITH_CONDITIONS" | "BLOCK"
    diagnostics: list      # list[Diagnostic]
    constraints: list      # list[DeploymentConstraint]
    utility_score: float
    latency_avg_ms: float | None = None   # N-01: required by _attach_trace in analysis_routes
    memory_mb: float | None = None        # N-01: required by _attach_trace in analysis_routes
    support_status: str = "SUPPORTED"
    # M-1: precision is absent from _RuntimeEvaluation but present on RuntimeEvaluationResult.
    # Declaring it here (defaulting to None) lets to_analysis_state() propagate it safely
    # through both object types via getattr without AttributeError.
    precision: str | None = None

    def to_dict(self) -> dict:
        return {
            "runtime":           self.runtime.value if hasattr(self.runtime, "value") else str(self.runtime),
            "decision":          self.decision,
            "utility_score":     self.utility_score,
            "support_status":    self.support_status,
            "latency_avg_ms":    self.latency_avg_ms,
            "memory_mb":         self.memory_mb,
            "precision":         self.precision,
            "execution_success": self.decision != "BLOCK",
            "diagnostics":       [str(d) for d in self.diagnostics],
            "confidence_score":  0.0,
        }


@dataclass
class _Recommendation:
    best_runtime: RuntimeName
    evaluations: list      # list[_RuntimeEvaluation]


# ── Decision vocabulary normalisation ─────────────────────────────────────────

_DECISION_MAP: dict[str, str] = {
    "ALLOW":                  "ALLOW",
    "ALLOW_WITH_CONDITIONS":  "ALLOW_WITH_CONDITIONS",
    "BLOCK":                  "BLOCK",
    # Aliases from DeploymentDecision enum
    "SUPPORTED":              "ALLOW",
    "SUPPORTED_WITH_WARNINGS":"ALLOW_WITH_CONDITIONS",
    "UNSUPPORTED":            "BLOCK",
}


def _resolve_decision(validation: Any, diagnostics: list) -> str:
    """Derive decision from validation result + rule diagnostics."""
    has_fail = any(getattr(d, "severity", None) == "FAIL" for d in diagnostics)
    has_warn = any(getattr(d, "severity", None) == "WARN" for d in diagnostics)

    pred = getattr(validation, "predicted_decision", None)
    raw = pred.value if hasattr(pred, "value") else (str(pred) if pred else "ALLOW")
    decision = _DECISION_MAP.get(raw.upper(), "ALLOW")

    # Rule diagnostics can only make things worse, never better.
    if has_fail:
        return "BLOCK"
    if has_warn and decision == "ALLOW":
        return "ALLOW_WITH_CONDITIONS"
    return decision


def _compute_utility(
    decision: str,
    runtime: RuntimeName,
    runtime_priors: dict | None,
    reliability_score: float | None,
    gpu_scaling: str | None,
) -> float:
    """Score utility in [0, 1] for ranking runtimes."""
    if decision == "BLOCK":
        return 0.0

    prior = (runtime_priors or {}).get(runtime.value, 0.5)
    rel = reliability_score if reliability_score is not None else 0.5

    base = 0.8 * prior + 0.2 * rel

    # GPU-capable runtimes get a boost when GPU scaling is favourable.
    _GPU_RUNTIMES = {
        RuntimeName.ONNX_CUDA, RuntimeName.TRT, RuntimeName.TF_GPU,
        RuntimeName.TFLITE_GPU, RuntimeName.TRT_NATIVE, RuntimeName.OV_GPU,
    }
    if gpu_scaling and gpu_scaling not in ("NONE", "") and runtime in _GPU_RUNTIMES:
        base = min(1.0, base * 1.10)

    if decision == "ALLOW_WITH_CONDITIONS":
        base *= 0.70

    return max(0.0, min(1.0, base))


def recommend_runtime(
    rules: list,
    facts: dict,
    constraints_by_runtime: dict,
    validations_by_runtime: dict,
    gpu_scaling: str | None = None,
    reliability_level: str | None = None,
    reliability_score: float | None = None,
    runtime_priors: dict | None = None,
) -> _Recommendation:
    """
    Evaluate all RuntimeName members and return the best-scored recommendation.

    Returns a _Recommendation whose .best_runtime is a RuntimeName and
    .evaluations is a list of _RuntimeEvaluation (one per RuntimeName member).
    """
    evaluations: list[_RuntimeEvaluation] = []

    for runtime in CANONICAL_RUNTIMES:
        diagnostics = run_rules(rules, facts, runtime)
        validation = validations_by_runtime.get(runtime)
        decision = _resolve_decision(validation, diagnostics)
        constraints = constraints_by_runtime.get(runtime, [])
        utility = _compute_utility(
            decision, runtime, runtime_priors, reliability_score, gpu_scaling
        )
        evaluations.append(
            _RuntimeEvaluation(
                runtime=runtime,
                decision=decision,
                diagnostics=diagnostics,
                constraints=constraints,
                utility_score=utility,
            )
        )

    # Prefer any ALLOW over ALLOW_WITH_CONDITIONS over BLOCK.
    # Within the same decision class, prefer highest utility.
    _DECISION_PRIORITY = {"ALLOW": 2, "ALLOW_WITH_CONDITIONS": 1, "BLOCK": 0}
    best = max(
        evaluations,
        key=lambda e: (_DECISION_PRIORITY.get(e.decision, 0), e.utility_score),
    )

    return _Recommendation(best_runtime=best.runtime, evaluations=evaluations)

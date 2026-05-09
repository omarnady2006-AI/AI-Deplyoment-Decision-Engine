"""
services/analysis_pipeline_service.py

Boundary layer between API routes and the core ML pipeline.

SINGLE DECISION PATH (strictly enforced):
    request -> feature_extractor -> ml_decision_engine -> select_action -> deployment_decision

FORBIDDEN:
  - Any import or call to risk_engine, risk_based_decision, risk_input
  - Any if/else logic that determines deployment_decision
  - Any fallback or default decision output
  - Any parallel or hybrid decision path
  - Any reference to risk_score, risk_level, sla_passed

Public callables (consumed by route layer):
  run_analysis_stage(model_path, model_hash, deployment_profile)
  run_full_pipeline(model_path, model_hash, deployment_profile)
  run_decision_stage(analysis_state, deployment_profile)
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

# ── Orchestration: controlled shadow execution executor ───────────────────────
# Bounded pool — shadow jobs never block the main response path.
_shadow_executor: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="shadow_bench",
)

# Canonical deployment target order (mirrors telemetry._DEPLOYMENT_TARGETS)
_DEPLOYMENT_TARGETS: tuple[str, ...] = (
    "edge_int8",
    "edge_fp16",
    "cloud_cpu",
    "cloud_gpu",
)

# Reverse map: deployment target → runtime keys that can satisfy it
_TARGET_TO_RUNTIMES: dict[str, tuple[str, ...]] = {
    "edge_int8": ("ONNX_CPU",),
    "edge_fp16": ("ONNX_CUDA",),
    "cloud_cpu": ("ONNX_CPU",),
    "cloud_gpu": ("ONNX_CUDA", "TENSORRT"),
}

from pydantic import BaseModel
from src.core.telemetry import log_analysis_event


# ── Canonical analysis-state schema ──────────────────────────────────────────

class EvaluationEntry(BaseModel):
    """One runtime's evaluation record as stored in APP_STATE['analysis']."""
    runtime: str
    latency_avg_ms: float | None = None
    memory_mb: float | None = None
    confidence_score: float | None = None
    execution_success: bool = True
    utility_score: float | None = None
    support_status: str | None = None
    decision: str | None = None
    precision: str | None = None
    diagnostics: list[str] = []


class ModelFactsEntry(BaseModel):
    """
    Extended model facts stored in analysis_state.
    Must contain ALL fields required by feature_extractor.extract_features().
    """
    parameter_count: int           = 0
    has_dynamic_shapes: bool       = False
    parameter_scale_class: str     = "small"
    operator_count: int            = 0
    model_size_mb: float           = 0.0
    sequential_depth_estimate: int = 0
    has_conv: bool                 = False
    has_attention: bool            = False
    has_resize: bool               = False
    uses_batch_normalization: bool = False
    uses_layer_normalization: bool = False
    has_non_max_suppression: bool  = False
    has_conv_transpose: bool       = False


class ConfidenceEntry(BaseModel):
    score: float
    level: str


class AnalysisStateSchema(BaseModel):
    """
    Typed contract for APP_STATE['analysis'].

    run_decision_stage validates against this at entry so that callers
    with a malformed dict receive a clear ValidationError rather than
    a silent error propagating into feature extraction.
    """
    best_runtime: str
    confidence: ConfidenceEntry | float
    evaluations: list[EvaluationEntry] = []
    model_facts: ModelFactsEntry | None = None


def _validate_analysis_state(analysis_state: dict[str, Any]) -> AnalysisStateSchema:
    """Validate and coerce analysis_state against AnalysisStateSchema."""
    return AnalysisStateSchema.model_validate(analysis_state)


# ── Internal result objects ───────────────────────────────────────────────────

class _AnalysisResult:
    def __init__(self, *, success, error, analysis, best_runtime,
                 evaluations, overall_confidence, timestamp,
                 response_evaluations: list | None = None):
        self.success = success
        self.error = error
        self.analysis = analysis
        self.best_runtime = best_runtime
        self.evaluations = evaluations
        self.overall_confidence = overall_confidence
        self.timestamp = timestamp
        # Normalized evaluations for API response: keys are latency_ms / memory_mb /
        # execution_success with string runtime names — ready for direct JSON serialization.
        self.response_evaluations: list = response_evaluations if response_evaluations is not None else []

    def to_analysis_state(self) -> dict[str, Any]:
        from src.core.contracts.decision_result import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
        ad = self.analysis.to_dict() if self.analysis else {}
        _score = max(0.0, min(1.0, self.overall_confidence))
        return {
            "best_runtime": self.best_runtime,
            "confidence": {
                "score": _score,
                "level": (
                    "HIGH"   if _score >= CONFIDENCE_HIGH   else
                    "MEDIUM" if _score >= CONFIDENCE_MEDIUM else
                    "LOW"
                ),
            },
            "model_facts": {
                "parameter_count":           int(ad.get("parameter_count", 0)),
                "has_dynamic_shapes":        bool(ad.get("has_dynamic_shapes", False)),
                "parameter_scale_class":     str(ad.get("parameter_scale_class", "small")),
                "operator_count":            int(ad.get("operator_count", 0)),
                "model_size_mb":             float(ad.get("model_size_mb", 0.0)),
                "sequential_depth_estimate": int(ad.get("sequential_depth_estimate", 0)),
                "has_conv":                  bool(ad.get("has_conv", False)),
                "has_attention":             bool(ad.get("has_attention", False)),
                "has_resize":                bool(ad.get("has_resize", False)),
                "uses_batch_normalization":  bool(ad.get("uses_batch_normalization", False)),
                "uses_layer_normalization":  bool(ad.get("uses_layer_normalization", False)),
                "has_non_max_suppression":   bool(ad.get("has_non_max_suppression", False)),
                "has_conv_transpose":        bool(ad.get("has_conv_transpose", False)),
            },
            "evaluations": [
                {
                    "runtime":           normalize_runtime_name(e.runtime),
                    "latency_avg_ms":    e.latency_avg_ms,
                    "memory_mb":         e.memory_mb,
                    "utility_score":     e.utility_score,
                    "support_status":    (getattr(e, "support_status", None)
                                          or getattr(e, "decision", None)),
                    "decision":          getattr(e, "decision", None),
                    "confidence_score":  (getattr(e, "confidence_score", None)
                                          if getattr(e, "confidence_score", None) is not None
                                          else e.utility_score),
                    "precision":         getattr(e, "precision", None),
                    "diagnostics":       e.diagnostics,
                    "execution_success": getattr(e, "decision", "BLOCK") not in (
                        "BLOCK", "UNSUPPORTED"
                    ),
                }
                for e in self.evaluations
            ],
            # Benchmark truth — used by run_decision_stage telemetry (Task 3)
            "response_evaluations": self.response_evaluations,
        }


class _FullPipelineResult(_AnalysisResult):
    def __init__(self, *, ml_decision, hardware_override_applied,
                 hardware_override_info, **kwargs):
        super().__init__(**kwargs)
        self.decision = ml_decision
        self.hardware_override_applied = hardware_override_applied
        self.hardware_override_info = hardware_override_info

    def to_decision_state(self) -> dict[str, Any]:
        d = self.decision.to_dict() if self.decision is not None else {}
        # Propagate best_runtime from the analysis layer so APP_STATE["decision"]
        # carries selected_runtime — required by the frontend hydration path.
        # This is the single source of truth: best_runtime is always sourced
        # from response_evaluations (benchmark truth) and never guessed.
        d["selected_runtime"] = self.best_runtime
        return d

    def best_evaluation(self) -> Any | None:
        if not self.evaluations:
            return None
        rt = getattr(self, "best_runtime", "")
        return next(
            (e for e in self.evaluations
             if (e.runtime.value if hasattr(e.runtime, "value")
                 else str(e.runtime)).upper() == rt.upper()),
            self.evaluations[0],
        )


class _MLDecisionOutput:
    """
    Thin wrapper around MLDecisionResult for the route layer.
    Exposes deployment_decision, confidence, model_label, and is_exploration.
    model_label and is_exploration are used by the telemetry layer ONLY.
    """
    def __init__(self, *, ml_result, model_label: str, is_exploration: bool):
        self._r = ml_result
        self.model_label = model_label
        self.is_exploration = is_exploration
        self.deployment_decision = ml_result.deployment_decision
        self.confidence = ml_result.confidence

    def to_dict(self) -> dict[str, Any]:
        return self._r.to_dict()


# ── SINGLE ML DECISION PATH ───────────────────────────────────────────────────

def _run_ml_decision(
    model_facts: dict[str, Any],
    deployment_profile: dict[str, Any],
) -> "_MLDecisionOutput":
    """
    THE ONLY decision path in the entire runtime system.

    Flow:
        1. Extract feature vector from raw model_facts + deployment_profile
        2. Call ml_decision_engine.predict(features)  -> (model_label, confidence)
        3. Apply epsilon-greedy exploration            -> chosen_action
        4. Wrap result in MLDecisionResult and return

    NO rules. NO thresholds. NO conditionals. NO fallbacks.
    deployment_decision comes EXCLUSIVELY from the ML model + exploration layer.

    Args:
        model_facts:        Dict of raw model analysis facts.
                            Must contain all fields required by feature_extractor.
        deployment_profile: Dict of hardware/SLA configuration.

    Returns:
        _MLDecisionOutput wrapping an MLDecisionResult.

    Raises:
        ValueError:   if feature extraction fails (missing/invalid fields).
        RuntimeError: if the ML model file is missing or prediction fails.
    """
    from src.core.feature_extractor import extract_features
    from src.core.ml_decision_engine import predict, select_action, get_epsilon
    from src.core.contracts.decision_result import build_ml_decision_result

    # Step 1: Extract features — raises ValueError on missing/invalid inputs
    features = extract_features(model_facts, deployment_profile)

    # Step 1 (mandatory): canonical feature-vector log — used for offline audit and
    # cross-run comparison.  The message key ML_FEATURE_VECTOR is stable; do not rename.
    logger.info("ML_FEATURE_VECTOR", extra={"features": features.tolist()})

    # Step 5: input-variation guard — a fully-collapsed vector means deployment_profile
    # fields did not reach the extractor.  Fail fast so the bug surfaces immediately
    # instead of producing a silently wrong confidence score.
    assert len(set(features.tolist())) > 1, (
        "FEATURE VECTOR COLLAPSED — all feature values are identical. "
        "deployment_profile fields are not reaching extract_features(). "
        f"profile keys present: {sorted(deployment_profile.keys())}"
    )

    # Step 2: ML inference — raises RuntimeError if model unavailable
    model_label, confidence = predict(features)

    # Pre-filter model output to enforce physical feasibility BEFORE exploration
    if model_label == "cloud_gpu":
        if not deployment_profile.get("gpu_available", False) \
           and not deployment_profile.get("cuda_available", False):
            model_label = "cloud_cpu"

    # Step 3: Confidence-aware epsilon-greedy exploration
    epsilon = get_epsilon()
    if confidence > 0.85:
        epsilon *= 0.5       # reduce exploration when model is confident
    elif confidence < 0.55:
        epsilon *= 1.3       # increase exploration when model is uncertain
    epsilon = max(0.15, min(0.30, epsilon))   # hard bounds — never allow uncontrolled exploration
    chosen_action = select_action(
        model_label,
        epsilon,
        deployment_profile,
    )

    # Step 4: Soft constraint post-filter (impossible states only — not heuristic tuning)
    if chosen_action == "cloud_gpu":
        if not deployment_profile.get("gpu_available", False) \
           and not deployment_profile.get("cuda_available", False):
            chosen_action = "cloud_cpu"

    # Step 4: Build typed result
    ml_result = build_ml_decision_result(
        deployment_decision=chosen_action,
        confidence=confidence,
    )

    logger.info(
        "ml_decision: model_label=%s chosen_action=%s confidence=%.4f",
        model_label, chosen_action, confidence,
    )

    return _MLDecisionOutput(
        ml_result=ml_result,
        model_label=model_label,
        is_exploration=(chosen_action != model_label),
    )


# ── Runtime name normalizer ───────────────────────────────────────────────────

def normalize_runtime_name(rt: object) -> str:
    """
    Map any runtime identifier (enum, ONNX provider string, or internal key)
    to the canonical UPPER_SNAKE key used in bench_by_name / rt_key lookups.

    Both sides of the benchmark merge (bench_results AND recommendation.evaluations)
    must go through this function so that keys are always comparable.

    Mapping rules (checked in priority order):
      "TENSORRT" substring  → TENSORRT          (before CUDA/GPU — TRT implies GPU
                                                  but is a distinct runtime)
      "CUDA" or "GPU"       → ONNX_CUDA
      "OPENVINO"            → OPENVINO_CPU       (default; GPU variant is handled
                                                  downstream by deployment profile)
      "TFLITE"              → TFLITE_CPU
      "CPU"                 → ONNX_CPU           (catches CPUExecutionProvider,
                                                  ONNX_CPU, TF_CPU, etc.)
      anything else         → rt.upper()         (pass-through for TENSORRT_NATIVE,
                                                  TF_GPU, etc.)
    """
    s = str(rt.value if hasattr(rt, "value") else rt).upper()

    if "TENSORRT" in s:
        result = "TENSORRT"
    elif "CUDA" in s or "GPU" in s:
        result = "ONNX_CUDA"
    elif "OPENVINO" in s:
        result = "OPENVINO_CPU"
    elif "TFLITE" in s:
        result = "TFLITE_CPU"
    elif "CPU" in s:
        result = "ONNX_CPU"
    else:
        result = s

    return result


# ── Shared analysis logic (DRY helper) ───────────────────────────────────────

def _run_analysis(model_path: str, model_hash: str | None,
                  deployment_profile: dict[str, Any]) -> "_AnalysisResult":
    """Shared runtime validation and benchmark logic for both pipeline callables."""
    try:
        return _run_analysis_inner(model_path, model_hash, deployment_profile)
    except Exception:
        import traceback
        print("=== HARD FAILURE ===")
        traceback.print_exc()
        raise


def _run_analysis_inner(model_path: str, model_hash: str | None,
                        deployment_profile: dict[str, Any]) -> "_AnalysisResult":
    """Inner implementation — wrapped by _run_analysis for full traceback logging."""
    print("\n=== ANALYSIS START ===")

    from src.core.model_facts.analyzer import analyze_model_file
    from src.core.runtime import RuntimeName, CANONICAL_RUNTIMES
    from src.core.runtime_selector import recommend_runtime
    from src.rules.unsupported_operator import UnsupportedOperatorRule
    from src.validation.runtime_validator import validate_runtime
    from src.validation.interpret_validation import interpret_validation_result

    analysis = analyze_model_file(model_path, precomputed_hash=model_hash)
    if not analysis.success:
        raise ValueError(
            f"Model analysis failed — cannot run runtime evaluations: {analysis.error}"
        )

    rules = [UnsupportedOperatorRule()]
    constraints_by_runtime: dict = {}
    validations_by_runtime: dict = {}

    for runtime in CANONICAL_RUNTIMES:
        vr = validate_runtime(model_path, runtime, precomputed_analysis=analysis)
        validations_by_runtime[runtime] = vr
        constraints_by_runtime[runtime] = interpret_validation_result(vr)

    facts = analysis.to_dict() if analysis.success else {}
    recommendation = recommend_runtime(
        rules, facts,
        constraints_by_runtime=constraints_by_runtime,
        validations_by_runtime=validations_by_runtime,
    )

    # ── Run benchmarks — sole source of truth for response_evaluations ─────────
    bench_by_name: dict[str, Any] = {}
    try:
        from src.services.onnx_benchmark_service import evaluate_all_runtimes
        from src.services.hardware_profiler import get_server_hw_profile
        server_hw = get_server_hw_profile()
        bench_results = evaluate_all_runtimes(model_path, deployment_profile, server_hw)
        for b in bench_results:
            key = normalize_runtime_name(getattr(b, "provider", None) or b.runtime)
            if key not in bench_by_name:          # first result wins — no silent overwrites
                bench_by_name[key] = b
        print("BENCH KEYS:", list(bench_by_name.keys()))
    except Exception:
        logger.warning(
            "benchmark_failed — no benchmark data available",
            exc_info=True,
        )

    # overall_confidence from recommendation (advisory only — not used for eval truth)
    allowed = sum(
        1 for e in recommendation.evaluations if e.decision != "BLOCK"
    )
    total = len(recommendation.evaluations) or 1
    overall_confidence = max(0.1, min(1.0, allowed / total))

    # ── Build response_evaluations EXCLUSIVELY from benchmark results ─────────
    # recommendation.evaluations is NEVER the source here.
    response_evaluations: list[dict] = []

    # Phase 2: include ALL attempted runtimes — failed ones (latency_avg_ms=None,
    # execution_success=False) are valid results, not missing data.
    # Phase 3: eval_count = len(evaluations), not len(successful evaluations).
    for rt_key, bench in bench_by_name.items():
        if bench is None:
            continue
        response_evaluations.append({
            "runtime":           rt_key,
            # Failed runs have latency_avg_ms=None — preserved, not filtered out.
            "latency_ms":        getattr(bench, "latency_avg_ms", None),
            "memory_mb":         getattr(bench, "memory_mb", None),
            # Always read the real recorded flag — never hardcode True.
            "execution_success": bool(getattr(bench, "execution_success", False)),
        })

    print("EVAL COUNT:", len(response_evaluations))

    # Phase 1 (removed): "Fail fast on empty benchmark results" deleted.
    # An empty list here means no runtime was available — a truthful state,
    # NOT a pipeline failure.  All-failed lists also pass through unchanged.

    # ── Task 1 — Hard invariants (must hold before any downstream use) ────────
    assert len({e["runtime"] for e in response_evaluations}) == len(response_evaluations), \
        "Duplicate runtimes detected"
    for _e in response_evaluations:
        assert "runtime"           in _e, f"evaluation missing 'runtime': {_e}"
        assert "latency_ms"        in _e, f"evaluation missing 'latency_ms': {_e}"
        assert "memory_mb"         in _e, f"evaluation missing 'memory_mb': {_e}"
        assert "execution_success" in _e, f"evaluation missing 'execution_success': {_e}"

    # ── Task 5 — Derive best_runtime via compute_optimal_action ──────────────
    from src.core.telemetry import aggregate_target_outcomes, compute_optimal_action as _coa

    _eval_actions = aggregate_target_outcomes(response_evaluations)
    _constraints = {
        "target_latency_ms": float(deployment_profile.get("target_latency_ms") or 0.0),
        "memory_limit_mb":   float(deployment_profile.get("memory_limit_mb")   or 0.0),
    }
    _optimal_action = _coa(_eval_actions, _constraints)

    # Phase 3/6: best_runtime uses all evaluations; only fall back to
    # latency-min over successful (non-None latency) entries so failed
    # runtimes do not crash min().
    _action_to_runtime: dict[str, str] = {
        ea["action"]: re["runtime"]
        for ea, re in zip(_eval_actions, response_evaluations)
    }
    _successful_evals = [e for e in response_evaluations if e.get("latency_ms") is not None]
    best_runtime = (
        _action_to_runtime.get(_optimal_action)
        or (min(_successful_evals, key=lambda x: x["latency_ms"])["runtime"] if _successful_evals else None)
        or (response_evaluations[0]["runtime"] if response_evaluations else None)
    )

    print("BEST_RUNTIME:", best_runtime)
    print("RUNTIMES:", [e["runtime"] for e in response_evaluations])

    # Guard — best_runtime must resolve to a known measured entry, or None
    # when no runtime was available at all (not a pipeline error).
    if best_runtime is not None:
        assert best_runtime in {_e["runtime"] for _e in response_evaluations}, (
            f"best_runtime {best_runtime!r} not found in response_evaluations"
        )


    return _AnalysisResult(
        success=analysis.success,
        error=analysis.error,
        analysis=analysis,
        best_runtime=best_runtime,
        evaluations=recommendation.evaluations,
        overall_confidence=overall_confidence,
        timestamp=time.time(),
        response_evaluations=response_evaluations,
    )


# ── Telemetry event emitter ───────────────────────────────────────────────────

def _emit_telemetry_event(
    *,
    ml_decision: "_MLDecisionOutput",
    model_facts: dict[str, Any],
    deployment_profile: dict[str, Any],
    raw_evals: list[dict],
    request_id: str | None = None,
    shadow: bool = False,
) -> None:
    """
    Build and write a schema-v3 telemetry event.

    Invariants:
    - Called ONLY after execution; caller guarantees raw_evals is non-empty.
    - compute_optimal_action receives ONLY evaluated_actions + constraints.
      model_label and chosen_action are NEVER passed to it.
    - Raises RuntimeError if feature extraction fails (Fix 4).
    - Silently returns without writing if valid_actions < 2 (Fix 9).
    - shadow=True marks background benchmark events; shadow=False for primary.
    """
    import hashlib
    import uuid
    from datetime import datetime, timezone

    from src.core.telemetry import (
        SCHEMA_VERSION,
        log_analysis_event,
        compute_optimal_action,
        aggregate_target_outcomes,
    )
    from src.core.feature_extractor import extract_features, FEATURE_NAMES

    _TARGET_ORDER: list[str] = ["edge_int8", "edge_fp16", "cloud_cpu", "cloud_gpu"]

    evaluated_actions: list[dict] = aggregate_target_outcomes(raw_evals)

    # Fix 7: Stamp outcome_type on every evaluated_action
    for ea in evaluated_actions:
        ea["outcome_type"] = "measured"

    # Fix 10: Deterministic ordering by canonical target sequence
    _order_map: dict[str, int] = {t: i for i, t in enumerate(_TARGET_ORDER)}
    evaluated_actions.sort(
        key=lambda ea: _order_map.get(ea.get("action", ""), len(_TARGET_ORDER))
    )

    # Fix 9: Minimum coverage guard — only log events with >= 2 fully measured actions
    valid_actions = [
        ea for ea in evaluated_actions
        if ea.get("execution_success", False) and ea.get("latency_ms") is not None
    ]
    if len(valid_actions) < 2:
        return

    constraints: dict = {
        "target_latency_ms": float(deployment_profile.get("target_latency_ms") or 0.0),
        "memory_limit_mb":   float(deployment_profile.get("memory_limit_mb")   or 0.0),
    }

    # optimal_action is derived from real outcomes ONLY — never from model output
    optimal: str = compute_optimal_action(evaluated_actions, constraints)

    # Fix 4: Hard fail on feature extraction — no silent empty-dict fallback
    fv = extract_features(model_facts, deployment_profile)
    logger.info("ML_FEATURE_VECTOR", extra={"features": fv.tolist(), "source": "telemetry"})
    assert len(set(fv.tolist())) > 1, (
        "FEATURE VECTOR COLLAPSED (telemetry path) — "
        f"profile keys present: {sorted(deployment_profile.keys())}"
    )
    features_dict: dict = dict(zip(FEATURE_NAMES, fv.tolist()))

    event_id: str = str(uuid.uuid4())

    # Fix 5: Stable hardware identity digest for cross-session correlation
    cpu_cores: str = str(deployment_profile.get("cpu_cores", ""))
    gpu: str       = str(deployment_profile.get("gpu", ""))
    cuda: str      = str(deployment_profile.get("cuda", ""))
    vram: str      = str(deployment_profile.get("vram", ""))
    hardware_context_id: str = hashlib.sha256(
        f"{cpu_cores}:{gpu}:{cuda}:{vram}".encode()
    ).hexdigest()[:16]

    # Fix 6: Group identity ties this event to its originating request
    group_id: str = request_id or event_id

    # Fix 8: Sanity-check numeric fields before write
    for ea in evaluated_actions:
        latency = ea.get("latency_ms")
        memory  = ea.get("memory_mb")
        assert latency is None or latency >= 0
        assert memory  is None or memory  >= 0

    log_analysis_event({
        "schema_version":      SCHEMA_VERSION,
        "event_id":            event_id,
        "group_id":            group_id,
        "hardware_context_id": hardware_context_id,
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
        "features":            features_dict,
        "model_prediction":    ml_decision.model_label,
        "chosen_action":       ml_decision.deployment_decision,
        "confidence":          ml_decision.confidence,
        "is_exploration":      ml_decision.is_exploration,
        "evaluated_actions":   evaluated_actions,
        "optimal_action":      optimal,
        "label_finalized":     True,
        "shadow":              shadow,
    })


# ── Shadow execution ──────────────────────────────────────────────────────────

def _run_shadow_benchmark(
    *,
    chosen_action: str,
    model_path: str,
    deployment_profile: dict[str, Any],
    model_facts: dict[str, Any],
    ml_decision: "_MLDecisionOutput",
    request_id: str | None,
) -> None:
    """
    Run benchmarks for the deployment targets NOT chosen by the primary decision
    and emit a shadow telemetry event.

    Contract:
    - NEVER called on the main response path — always runs inside _shadow_executor.
    - Errors are logged and swallowed; they MUST NOT propagate to the caller.
    - Emits exactly one telemetry event with shadow=True when >= 2 valid results exist.
    """
    try:
        # Targets the primary decision did not exercise
        remaining: list[str] = [
            t for t in _DEPLOYMENT_TARGETS if t != chosen_action
        ]
        if not remaining:
            return

        # Collect the runtime keys that can satisfy remaining targets
        relevant_runtimes: frozenset[str] = frozenset(
            rt
            for t in remaining
            for rt in _TARGET_TO_RUNTIMES.get(t, ())
        )
        if not relevant_runtimes:
            return

        # Run all runtimes via the benchmark service; filter to relevant only
        from src.services.onnx_benchmark_service import evaluate_all_runtimes
        from src.services.hardware_profiler import get_server_hw_profile

        server_hw = get_server_hw_profile()
        bench_results = evaluate_all_runtimes(model_path, deployment_profile, server_hw)

        raw_evals: list[dict[str, Any]] = []
        for b in bench_results:
            rt_key = normalize_runtime_name(
                getattr(b, "provider", None) or b.runtime
            )
            if rt_key not in relevant_runtimes:
                continue
            raw_evals.append({
                "runtime":           rt_key,
                "latency_ms":        getattr(b, "latency_avg_ms", None),
                "memory_mb":         getattr(b, "memory_mb", None),
                "execution_success": bool(getattr(b, "execution_success", True)),
            })

        if not raw_evals:
            return

        _emit_telemetry_event(
            ml_decision=ml_decision,
            model_facts=model_facts,
            deployment_profile=deployment_profile,
            raw_evals=raw_evals,
            request_id=request_id,
            shadow=True,
        )

    except Exception:
        logger.warning("shadow_benchmark_failed — swallowed", exc_info=True)


# ── Public pipeline callables ─────────────────────────────────────────────────

def run_analysis_stage(
    model_path: str,
    model_hash: str | None,
    deployment_profile: dict[str, Any],
    *,
    request_id: str | None = None,
) -> _AnalysisResult:
    """Run runtime validation and benchmarking for all runtimes. No decision."""
    return _run_analysis(model_path, model_hash, deployment_profile)


def run_full_pipeline(
    model_path: str,
    model_hash: str | None,
    deployment_profile: dict[str, Any],
    *,
    request_id: str | None = None,
) -> _FullPipelineResult:
    """Upload, analyse, and decide in one atomic call via the ML decision engine."""
    result = _run_analysis(model_path, model_hash, deployment_profile)

    ad = result.analysis.to_dict() if result.analysis else {}

    model_facts = {
        "parameter_count":           int(ad.get("parameter_count", 0)),
        "has_dynamic_shapes":        bool(ad.get("has_dynamic_shapes", False)),
        "parameter_scale_class":     str(ad.get("parameter_scale_class", "small")),
        "operator_count":            int(ad.get("operator_count", 0)),
        "model_size_mb":             float(ad.get("model_size_mb", 0.0)),
        "sequential_depth_estimate": int(ad.get("sequential_depth_estimate", 0)),
        "has_conv":                  bool(ad.get("has_conv", False)),
        "has_attention":             bool(ad.get("has_attention", False)),
        "has_resize":                bool(ad.get("has_resize", False)),
        "uses_batch_normalization":  bool(ad.get("uses_batch_normalization", False)),
        "uses_layer_normalization":  bool(ad.get("uses_layer_normalization", False)),
        "has_non_max_suppression":   bool(ad.get("has_non_max_suppression", False)),
        "has_conv_transpose":        bool(ad.get("has_conv_transpose", False)),
    }

    ml_decision = _run_ml_decision(model_facts, deployment_profile)

    # ── Telemetry — fires AFTER execution, ONLY when benchmark results exist ──
    if result.response_evaluations:
        try:
            raw_evals = result.response_evaluations  # benchmark truth only (Task 3)
            _emit_telemetry_event(
                ml_decision=ml_decision,
                model_facts=model_facts,
                deployment_profile=deployment_profile,
                raw_evals=raw_evals,
                request_id=request_id,
                shadow=False,
            )
        except Exception:
            logger.warning("telemetry_event_failed — continuing", exc_info=True)

    # ── Shadow execution — non-blocking, submitted AFTER primary telemetry ────
    _shadow_executor.submit(
        _run_shadow_benchmark,
        chosen_action=ml_decision.deployment_decision,
        model_path=model_path,
        deployment_profile=deployment_profile,
        model_facts=model_facts,
        ml_decision=ml_decision,
        request_id=request_id,
    )

    return _FullPipelineResult(
        success=True,
        error=None,
        analysis=result.analysis,
        best_runtime=result.best_runtime,
        evaluations=result.evaluations,
        overall_confidence=result.overall_confidence,
        timestamp=result.timestamp,
        response_evaluations=result.response_evaluations,  # MUST exist
        ml_decision=ml_decision,
        hardware_override_applied=False,
        hardware_override_info=None,
    )


def run_decision_stage(
    analysis_state: dict[str, Any],
    deployment_profile: dict[str, Any],
    *,
    request_id: str | None = None,
) -> _MLDecisionOutput:
    """
    Run ML decision for a completed analysis.

    Validates analysis_state at entry, extracts model_facts, then calls
    _run_ml_decision — the single ML decision path.

    Emits a telemetry event after the decision when evaluations exist in
    analysis_state.
    """
    _validate_analysis_state(analysis_state)

    raw_mf = analysis_state.get("model_facts") or {}

    model_facts = {
        "parameter_count":           int(raw_mf.get("parameter_count", 0)),
        "has_dynamic_shapes":        bool(raw_mf.get("has_dynamic_shapes", False)),
        "parameter_scale_class":     str(raw_mf.get("parameter_scale_class", "small")),
        "operator_count":            int(raw_mf.get("operator_count", 0)),
        "model_size_mb":             float(raw_mf.get("model_size_mb", 0.0)),
        "sequential_depth_estimate": int(raw_mf.get("sequential_depth_estimate", 0)),
        "has_conv":                  bool(raw_mf.get("has_conv", False)),
        "has_attention":             bool(raw_mf.get("has_attention", False)),
        "has_resize":                bool(raw_mf.get("has_resize", False)),
        "uses_batch_normalization":  bool(raw_mf.get("uses_batch_normalization", False)),
        "uses_layer_normalization":  bool(raw_mf.get("uses_layer_normalization", False)),
        "has_non_max_suppression":   bool(raw_mf.get("has_non_max_suppression", False)),
        "has_conv_transpose":        bool(raw_mf.get("has_conv_transpose", False)),
    }

    ml_decision = _run_ml_decision(model_facts, deployment_profile)

    # ── Telemetry — fires AFTER execution, ONLY when benchmark results exist ──
    # Prefer response_evaluations (benchmark truth) stored by run_analysis_stage.
    state_evals: list[dict] = analysis_state.get("response_evaluations") or []
    if state_evals:
        try:
            raw_evals = state_evals  # already in correct format — no conversion (Task 3)
            _emit_telemetry_event(
                ml_decision=ml_decision,
                model_facts=model_facts,
                deployment_profile=deployment_profile,
                raw_evals=raw_evals,
                request_id=request_id,
                shadow=False,
            )
        except Exception:
            logger.warning("telemetry_event_failed — continuing", exc_info=True)

    return ml_decision

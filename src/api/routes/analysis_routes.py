"""
api/routes/analysis_routes.py

Model analysis and combined analyze-and-decide endpoints.
ZERO logic — each route: parse request -> call pipeline service -> return response.

Output contract for decision-bearing endpoints:
    {
        "deployment_decision": "<edge_int8|edge_fp16|cloud_cpu|cloud_gpu>",
        "ml_confidence": {"score": <float>, "level": "<HIGH|MEDIUM|LOW>"},
        "system_score": {"score": <float>, "level": "<HIGH|MEDIUM|LOW>"}
    }
"""
from __future__ import annotations

import time
import uuid
import json as _json

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.app_state import APP_STATE, APP_STATE_LOCK, get_state_snapshot, _invalid_execution_order
from src.services.upload_service import save_uploaded_model
from src.services.analysis_pipeline_service import run_analysis_stage, run_full_pipeline
from src.core.logging_config import get_logger
from src.core.contracts.decision_result import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

logger = get_logger(__name__)
router = APIRouter()

try:
    import src.core.explainability  # noqa: F401
except Exception:
    logger.warning(
        "src.core.explainability failed to import — explain=True will raise at runtime.",
        exc_info=True,
    )


# ── Request models ────────────────────────────────────────────────────────────

class AnalyzeAndDecideRequest(BaseModel):
    cpu_cores: int          = 1
    ram_gb: float           = 8.0
    gpu_available: bool     = False
    cuda_available: bool    = False
    vram_gb: float | None           = None
    trt_available: bool | None      = None
    stress_test: bool | None        = None
    cpu_arch: str | None            = None
    ram_ddr: str | None             = None
    target_latency_ms: float | None = None
    memory_limit_mb: float | None   = None
    debug: bool   = False
    explain: bool = False


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/api/model/analyze")
async def run_model_analysis():
    """Run runtime benchmarks and score all runtimes. Analysis stage only — no decision."""
    snapshot           = get_state_snapshot()
    model_state        = snapshot.get("model")
    deployment_profile = snapshot.get("deployment_profile")
    diagnostics_state  = snapshot.get("diagnostics")

    if model_state is None:
        return _invalid_execution_order("model")
    if deployment_profile is None:
        return _invalid_execution_order("deployment_profile")
    if diagnostics_state is None:
        return _invalid_execution_order("diagnostics")

    model_path = str(model_state.get("path", ""))
    model_hash = model_state.get("hash")

    if not model_path:
        return JSONResponse(status_code=409, content={
            "success": False, "error": "invalid_execution_order", "required_stage": "model",
        })

    def _normalise_profile(p: dict) -> dict:
        bool_keys = ("trt_available", "stress_test", "gpu_available", "cuda_available")
        return {k: (False if k in bool_keys and v is None else v) for k, v in p.items()}

    if _normalise_profile(diagnostics_state.get("profile_snapshot", {})) \
            != _normalise_profile(deployment_profile):
        return JSONResponse(status_code=409, content={
            "success": False, "error": "invalid_execution_order",
            "required_stage": "diagnostics",
        })

    try:
        request_id = str(uuid.uuid4())
        result = run_analysis_stage(model_path, model_hash, deployment_profile,
                                    request_id=request_id)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        with APP_STATE_LOCK:
            active = APP_STATE.get("model")
            active_hash = active.get("hash") if isinstance(active, dict) else None
            if active_hash != model_hash:
                return JSONResponse(status_code=409, content={
                    "success": False, "error": "invalid_execution_order",
                    "required_stage": "model",
                })
            analysis_payload = result.to_analysis_state()
            APP_STATE["analysis"] = analysis_payload
            APP_STATE["decision"] = None

        response = {
            "success":          True,
            "model_path":       model_path,
            "file_hash":        model_hash,
            "analysis":         result.analysis.to_dict() if result.analysis else {},
            "best_runtime":     result.best_runtime,
            "evaluations":      result.response_evaluations,
            "ml_confidence": {
                "score": result.overall_confidence,
                "level": (
                    "HIGH"   if result.overall_confidence >= CONFIDENCE_HIGH   else
                    "MEDIUM" if result.overall_confidence >= CONFIDENCE_MEDIUM else
                    "LOW"
                ),
            },
            "system_score": {
                "score": result.overall_confidence,
                "level": (
                    "HIGH"   if result.overall_confidence >= CONFIDENCE_HIGH   else
                    "MEDIUM" if result.overall_confidence >= CONFIDENCE_MEDIUM else
                    "LOW"
                ),
            },
            # No decision at analysis-only stage
            "deployment_decision": None,
            "timestamp":           result.timestamp,
        }

        # PHASE 2 — contract enforcement guard
        if "confidence" in response:
            raise RuntimeError("Forbidden legacy field 'confidence' detected")
        assert "confidence" not in response

        required_fields = ["ml_confidence", "system_score", "best_runtime", "evaluations"]
        missing = [f for f in required_fields if f not in response]
        if missing:
            raise HTTPException(status_code=422, detail={
                "error": "pipeline_output_incomplete",
                "missing_fields": missing,
                "pipeline_stage": "analysis",
            })

        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/analyze-and-decide")
async def analyze_and_decide(
    file:              UploadFile | None = File(None),
    cpu_cores:         int           = Form(1),
    ram_gb:            float         = Form(8.0),
    gpu_available:     bool          = Form(False),
    cuda_available:    bool          = Form(False),
    vram_gb:           float  | None = Form(None),
    trt_available:     bool   | None = Form(None),
    stress_test:       bool   | None = Form(None),
    cpu_arch:          str    | None = Form(None),
    ram_ddr:           str    | None = Form(None),
    target_latency_ms: float  | None = Form(None),
    memory_limit_mb:   float  | None = Form(None),
    debug:             bool          = Query(False),
    explain:           bool          = Query(False),
):
    """
    Upload, analyse, and decide in one atomic call.

    Returns:
        {
            "deployment_decision": "<edge_int8|edge_fp16|cloud_cpu|cloud_gpu>",
            "ml_confidence": {"score": <float>, "level": "<HIGH|MEDIUM|LOW>"},
            "system_score": {"score": <float>, "level": "<HIGH|MEDIUM|LOW>"},
            "success": true,
            ...
        }
    """
    try:
        if file is not None and file.filename:
            upload = await save_uploaded_model(file)
            model_path = upload["model_path"]
            if not model_path:
                raise HTTPException(status_code=400,
                                    detail="Upload service returned empty model path")
            model_hash = upload["model_hash"]
        else:
            snapshot = get_state_snapshot()
            model_state = snapshot.get("model")
            if not isinstance(model_state, dict) or not model_state.get("path"):
                raise HTTPException(status_code=400, detail=(
                    "No file provided and no model found in current session. "
                    "Please upload a model file."
                ))
            model_path = str(model_state["path"])
            model_hash = model_state.get("hash")

        deployment_profile = {
            "cpu_cores":         int(cpu_cores),
            "cpu_arch":          cpu_arch,
            "ram_gb":            float(ram_gb),
            "ram_ddr":           ram_ddr,
            "gpu_available":     bool(gpu_available),
            "cuda_available":    bool(cuda_available),
            "trt_available":     bool(trt_available)     if trt_available     is not None else False,
            "stress_test":       bool(stress_test)       if stress_test       is not None else False,
            "vram_gb":           float(vram_gb)           if vram_gb           is not None else None,
            "target_latency_ms": float(target_latency_ms) if target_latency_ms is not None else None,
            "memory_limit_mb":   float(memory_limit_mb)   if memory_limit_mb   is not None else None,
        }

        logger.info("analyze_and_decide_received", extra={
            "event":             "request_received",
            "cpu_cores":         deployment_profile["cpu_cores"],
            "ram_gb":            deployment_profile["ram_gb"],
            "target_latency_ms": deployment_profile["target_latency_ms"],
            "memory_limit_mb":   deployment_profile["memory_limit_mb"],
            "gpu_available":     deployment_profile["gpu_available"],
        })

        request_id = str(uuid.uuid4())
        result = run_full_pipeline(model_path, model_hash, deployment_profile,
                                   request_id=request_id)

        if not result.success:
            raise HTTPException(status_code=400,
                                detail=f"Pipeline failed: {result.error}")

        # Update APP_STATE atomically — decision slot written AFTER response is built
        # (Phase 1: to_decision_state() partial serializer deleted; response dict
        #  is the single source of truth persisted below after Step 9 validation.)
        _facts = result.analysis.to_dict() if result.analysis else {}
        diagnostics_state = {
            "facts":            _facts,
            "raw_analysis":     _facts,
            "profile_snapshot": _json.loads(_json.dumps(deployment_profile)),
            "diagnostics":      (list(_facts.get("diagnostics", []))
                                  if isinstance(_facts.get("diagnostics"), list) else []),
        }

        with APP_STATE_LOCK:
            state = result.to_analysis_state()

            APP_STATE["model"]              = {"path": model_path, "hash": model_hash}
            APP_STATE["deployment_profile"] = deployment_profile
            APP_STATE["diagnostics"]        = diagnostics_state
            APP_STATE["analysis"]           = state
            APP_STATE["decision"]           = None  # overwritten after response is fully built

        dec = result.decision
        trace_id = time.time()

        evals = result.response_evaluations or []

        # Phase 4 (sentinel injection removed): evals is passed through as-is.
        # An empty list means no runtime was available; a list of failed entries
        # means all runtimes executed but none succeeded.  Both are truthful
        # states that the verdict layer (Phase 6) must classify — not hide.

        if evals and not all("latency_ms" in e for e in evals):
            raise RuntimeError("Invalid evaluation schema")

        for e in evals:
            if not isinstance(e.get("runtime"), str):
                raise RuntimeError("Invalid eval: missing runtime string")
            if "latency_ms" not in e:
                raise RuntimeError("Invalid eval: missing latency_ms")

        # FIX 1 — enforce single source of truth invariants at route boundary
        from src.services.analysis_pipeline_service import _TARGET_TO_RUNTIMES as _DECISION_TO_RUNTIME
        _allowed_runtimes = {e["runtime"] for e in evals}
        if result.best_runtime and result.best_runtime not in _allowed_runtimes:
            raise RuntimeError(
                f"best_runtime {result.best_runtime!r} not in evaluations {_allowed_runtimes}"
            )
        _decision_rejected = False
        _rejection_reason  = None
        # STEP 3 — dec.deployment_decision is used only to validate consistency;
        # it does NOT control _selected_runtime.  Runtime is always sourced from
        # measured evaluations (_allowed_runtimes), never from the ML decision target.
        if dec and dec.deployment_decision:
            _rt_candidates = _DECISION_TO_RUNTIME.get(dec.deployment_decision, ())
            if _rt_candidates and not any(r in _allowed_runtimes for r in _rt_candidates):
                _decision_rejected = True
                _rejection_reason  = "runtime_not_measured"
                # Step 1: dec is NOT nulled here.  Removing dec = None preserves
                # the ML decision so deployment_decision is always a real model
                # output.  _decision_rejected records the mismatch for diagnostics
                # without hiding the decision from the UI.

        # PHASE 4 — no default value for decision; fail loudly if ML produced no output
        if dec is None:
            raise RuntimeError("ML engine returned no decision — pipeline failure")
        decision_value: str = dec.deployment_decision

        # STEP 2 — selected_runtime sourced from measured evaluations only.
        # Priority: ML-ranked best_runtime (if in measured set) → deterministic
        # alphabetic fallback over _allowed_runtimes.  Result is always a non-null
        # string — the NO_VALID_RUNTIME path below is unreachable after Step 1.
        _selected_runtime = (
            result.best_runtime
            if result.best_runtime and result.best_runtime in _allowed_runtimes
            else next(
                (e["runtime"] for e in evals if e.get("execution_success")),
                evals[0]["runtime"] if evals else "ONNX_CPU",
            )
        )

        # Phase 6 — verdict layer.  evals may legitimately be empty (no runtime
        # was available to execute) or all-failed.  _selected_runtime is None
        # in both those cases; the verdict maps them to UNSUPPORTED.
        assert _selected_runtime is not None or not evals, \
            "_selected_runtime is None but evals is non-empty (selection logic regression)"

        # ── Step 2 — ops_coverage ────────────────────────────────────────────────
        # Derived from the analysis layer's operator inventory.  When analysis is
        # unavailable the field is still present with score 0.0 — the UI field
        # must never be absent.
        if len(evals) == 1 and not evals[0].get("execution_success"):
            ops_coverage: dict = {
                "covered": 0,
                "total":   0,
                "score":   0.0,
            }
        else:
            success_count = sum(1 for e in evals if e.get("execution_success"))
            total_count   = len(evals)
            ops_coverage = {
                "covered": success_count,
                "total":   total_count,
                "score":   success_count / total_count if total_count else 0.0,
            }

        # ── Step 3 — status ──────────────────────────────────────────────────────
        # Derived from the best eval's execution_success flag.  Always a string.
        _best_eval = next(
            (e for e in evals if e["runtime"] == _selected_runtime), None
        )
        _exec_ok  = bool(_best_eval.get("execution_success")) if _best_eval else False
        if not any(e.get("execution_success") for e in evals):
            exec_status = "NO_RUNTIME"
        elif _exec_ok:
            exec_status = "SUCCESS"
        else:
            exec_status: str = "FALLBACK"

        # ── Phase 1 — ml_confidence: pure ML output, never blended or adjusted ─
        # Sourced exclusively from dec.confidence.  This is the model's own
        # certainty score before any runtime or hardware adjustments are applied.
        _ml_raw: float = max(0.0, min(1.0, float(dec.confidence) if dec else 0.3))
        ml_confidence: dict = {
            "score": _ml_raw,
            "level": (
                "HIGH"   if _ml_raw >= CONFIDENCE_HIGH   else
                "MEDIUM" if _ml_raw >= CONFIDENCE_MEDIUM else
                "LOW"
            ),
        }
        if not isinstance(ml_confidence["score"], (int, float)):
            raise RuntimeError("ml_confidence score must be numeric")

        # ── Phase 1 — system_score: blended runtime + hardware signal ────────
        # Weights: 0.4 ML probability + 0.6 measured runtime performance.
        # Profile penalties apply hardware constraint sensitivity.
        # This is NEVER the model's own confidence — it reflects deployment fit.
        _latency = _best_eval.get("latency_ms") if _best_eval else None
        _success = _best_eval.get("execution_success", False) if _best_eval else False
        _target  = deployment_profile.get("target_latency_ms") or 200.0
        if _latency is not None:
            _latency_ratio = _latency / max(_target, 1e-6)
            _rt_score = max(0.0, min(1.0, 1.0 - _latency_ratio))
        else:
            _rt_score = 0.0
        if not _success:
            _rt_score *= 0.3
        _blended = (0.4 * _ml_raw) + (0.6 * _rt_score)
        _profile_penalty = 0.0
        if not deployment_profile.get("gpu_available"):
            _profile_penalty += 0.05
        if (deployment_profile.get("ram_gb") or 0) < 4:
            _profile_penalty += 0.05
        if (deployment_profile.get("cpu_cores") or 1) <= 2:
            _profile_penalty += 0.05
        _sys_raw: float = max(0.0, min(1.0, _blended - _profile_penalty))
        system_score: dict = {
            "score": _sys_raw,
            "level": (
                "HIGH"   if _sys_raw >= CONFIDENCE_HIGH   else
                "MEDIUM" if _sys_raw >= CONFIDENCE_MEDIUM else
                "LOW"
            ),
        }

        # ── Step 4 — verdict driven by system_score (deployment fit signal) ──
        _conf_score: float = _sys_raw
        if not _exec_ok:
            verdict: str = "UNAVAILABLE"
        elif _conf_score >= 0.7:
            verdict = "OPTIMAL"
        elif _conf_score >= 0.4:
            verdict = "ACCEPTABLE"
        else:
            verdict = "POOR"

        # ── Phase 2 — decision_source: explicit provenance of final decision ─
        # Answers: "did the ML model produce this decision, or was it adjusted?"
        _original_ml = dec.deployment_decision if dec else "unknown"
        _adjusted = (_original_ml != decision_value) or _decision_rejected
        if _decision_rejected:
            _ds_reason: str | None = "runtime_unavailable"
        elif result.hardware_override_applied:
            _ds_reason = "hardware_constraints"
        elif _original_ml != decision_value:
            _ds_reason = "fallback"
        else:
            _ds_reason = None
        decision_source: dict = {
            "type":                "rule_adjusted" if _adjusted else "ml",
            "original_ml_decision": _original_ml,
            "final_decision":       decision_value,
            "reason":               _ds_reason,
        }

        response: dict = {
            # ── Execution provenance ──────────────────────────────────────────
            "trace_id":            trace_id,
            "success":             True,
            # ── ML output contract ────────────────────────────────────────────
            "deployment_decision": decision_value,
            "decision":            decision_value,
            "selected_runtime":    _selected_runtime,
            "best_runtime":        _selected_runtime,
            # ── Phase 1: two distinct confidence signals ──────────────────────
            # ml_confidence  = model probability output (pure ML, never blended)
            # system_score   = blended deployment fit (ML + runtime + hardware)
            # "confidence" key REMOVED — it was ambiguous about which signal it carried
            "ml_confidence":       ml_confidence,
            "system_score":        system_score,
            # ── Phase 2: decision provenance ──────────────────────────────────
            "decision_source":     decision_source,
            # ── Step 6 fields — always populated, never null ──────────────────
            "status":              exec_status,
            "verdict":             verdict,
            "ops_coverage":        ops_coverage,
            # ── Supplementary analysis data ───────────────────────────────────
            "evaluations":         evals,
            "model_path":          model_path,
            "timestamp":           result.timestamp,
            "decision_metadata": {
                "rejected": _decision_rejected,
                "reason":   _rejection_reason,
            },
        }

        if result.hardware_override_applied:
            response["hardware_selector"] = result.hardware_override_info

        if debug or explain:
            _attach_trace(response, result, debug, explain, response["system_score"]["score"])

        # Step 7 — schema completeness: all UI-contract fields must be present.
        required_fields = [
            "deployment_decision",
            "selected_runtime",
            "ml_confidence",
            "system_score",
            "decision_source",
            "decision",
            "ops_coverage",
            "status",
            "verdict",
            "trace_id",
        ]
        missing = [f for f in required_fields if f not in response]
        if missing:
            raise HTTPException(status_code=422, detail={
                "error": "pipeline_output_incomplete",
                "missing_fields": missing,
                "pipeline_stage": "decision",
            })

        # Step 8 — null guard scoped to required UI fields.
        # Optional keys (hardware_selector, decision_trace, explanation_tree) are
        # excluded: they are absent entirely when not requested, which is different
        # from being present-but-null.  decision_metadata.reason is intentionally
        # null when there is no rejection.  The guard targets only the fields the
        # UI contract requires to always carry a real value.
        for _field in required_fields:
            if response.get(_field) is None:
                raise RuntimeError(
                    f"Null value for required UI field '{_field}' — backend contract violated"
                )

        # Step 9 — final type and range invariants.
        assert isinstance(response["selected_runtime"], str)
        assert isinstance(response["best_runtime"], str)
        assert isinstance(response["ml_confidence"]["score"], (int, float))
        assert 0.0 <= response["ml_confidence"]["score"] <= 1.0
        assert isinstance(response["system_score"]["score"], (int, float))
        assert 0.0 <= response["system_score"]["score"] <= 1.0

        # PHASE 2 — contract enforcement guard
        if "confidence" in response:
            raise RuntimeError("Forbidden legacy field 'confidence' detected")
        # PHASE 6 — validation assertion
        assert "confidence" not in response

        # Phase 1 — single source of truth: persist EXACT response dict so
        # APP_STATE["decision"] schema === HTTP response schema (byte-level identical keys).
        # Replaces the prior to_decision_state() partial serializer that dropped
        # status, verdict, ops_coverage, decision, trace_id, and evaluations.
        # PHASE 1 — invariant enforcement before return
        assert "confidence" not in response
        assert "ml_confidence" in response
        assert "system_score" in response
        assert response.get("decision") is not None
        # Phase 5: evaluations is ALWAYS present in the output (may be empty
        # when no runtime was available — truthful, not a failure).
        assert isinstance(response.get("evaluations"), list), \
            "response missing 'evaluations' key"
        _allowed = {e["runtime"] for e in response["evaluations"]}
        # selected_runtime may be None when evaluations is empty.
        assert response.get("selected_runtime") in _allowed or not _allowed

        with APP_STATE_LOCK:
            APP_STATE["decision"] = response
            # PHASE 2 — state consistency invariant
            assert APP_STATE["decision"] == response

        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _attach_trace(response: dict, result, debug: bool, explain: bool, final_score: float) -> None:
    """Attach DecisionTrace and/or explanation tree when requested."""
    try:
        from src.core.decision_trace import DecisionTrace
        from src.core.explainability import build_explanation_tree

        dec = result.decision
        if dec is None:
            if debug:
                response["decision_trace"] = {"error": "no_decision_available"}
            if explain:
                response["explanation_tree"] = {"error": "no_decision_available"}
            return

        best_ev = next(
            (
                e for e in (result.response_evaluations or [])
                if e.get("runtime") == response.get("selected_runtime")
            ),
            None,
        )
        trace = DecisionTrace(
            selected_runtime_before_override=response.get("selected_runtime"),
            selected_runtime_after_override=response.get("selected_runtime"),
            sla_target_latency=None,
            measured_latency=(
                float(best_ev.get("latency_ms"))
                if best_ev and best_ev.get("latency_ms") is not None
                else (
                    float(best_ev.get("latency_avg_ms"))
                    if best_ev and best_ev.get("latency_avg_ms") is not None
                    else None
                )
            ),
            memory_limit=None,
            measured_memory=float(best_ev.get("memory_mb")) if best_ev and best_ev.get("memory_mb") else None,
            rejection_reasons=[],
            conditional_reasons=[],
            confidence_before_scaling=final_score,
            confidence_after_scaling=final_score,
            hardware_override_applied=result.hardware_override_applied,
            override_reason=(
                result.hardware_override_info.get("override_reason")
                if isinstance(result.hardware_override_info, dict) else None
            ),
        )
        if debug:
            response["decision_trace"] = trace.to_dict()
        if explain:
            response["explanation_tree"] = build_explanation_tree(trace)

    except Exception as exc:
        logger.error("decision_trace_attach_failed", exc_info=True)
        if debug:
            response["decision_trace"] = {
                "error": "trace_generation_failed", "detail": str(exc),
            }
        if explain:
            response["explanation_tree"] = {
                "error": "explanation_generation_failed", "detail": str(exc),
            }

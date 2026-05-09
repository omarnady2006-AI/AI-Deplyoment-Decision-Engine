"""
api/routes/decision_routes.py

Decision recommendation endpoint.
ZERO logic — parse request -> call pipeline service -> return response.

Output contract:
    {
        "deployment_decision": "<edge_int8|edge_fp16|cloud_cpu|cloud_gpu>",
        "confidence": {"score": <float>, "level": "<HIGH|MEDIUM|LOW>"}
    }
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.app_state import APP_STATE, APP_STATE_LOCK, get_state_snapshot, _invalid_execution_order
from src.services.analysis_pipeline_service import run_decision_stage
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


@router.post("/api/decision/recommend")
async def recommend_decision(debug: bool = False, explain: bool = False):
    """
    Run ML decision for the current analysis.

    Returns:
        {
            "deployment_decision": "<edge_int8|edge_fp16|cloud_cpu|cloud_gpu>",
            "confidence": {"score": <float>, "level": "<HIGH|MEDIUM|LOW>"},
            "success": true
        }
    """
    snapshot           = get_state_snapshot()
    model              = snapshot.get("model")
    analysis_state     = snapshot.get("analysis")
    deployment_profile = snapshot.get("deployment_profile")

    if model is None:
        return _invalid_execution_order("model")
    if deployment_profile is None:
        return _invalid_execution_order("deployment_profile")
    if analysis_state is None:
        return _invalid_execution_order("analysis")

    try:
        request_id = str(uuid.uuid4())
        ml_output = run_decision_stage(
            analysis_state, deployment_profile, request_id=request_id
        )

        # Write ML decision to APP_STATE
        decision_dict = ml_output.to_dict()
        with APP_STATE_LOCK:
            APP_STATE["decision"] = decision_dict

        logger.info("ml_decision_computed", extra={
            "event":             "ml_decision_computed",
            "deployment_decision": ml_output.deployment_decision,
            "confidence":        ml_output.confidence,
        })

        best_runtime = analysis_state.get("best_runtime")
        selected_runtime = best_runtime

        response: dict = {
            "success":             True,
            "deployment_decision": ml_output.deployment_decision,
            "selected_runtime":    selected_runtime,
            "confidence": {
                "score": ml_output.confidence,
                "level": (
                    "HIGH"   if ml_output.confidence >= CONFIDENCE_HIGH   else
                    "MEDIUM" if ml_output.confidence >= CONFIDENCE_MEDIUM else
                    "LOW"
                ),
            },
            # Include evaluations for frontend rendering
            "evaluations": (
                analysis_state.get("response_evaluations")
                or analysis_state.get("evaluations")
                or []
            ),
        }

        # Include model_path for filename display
        _model = snapshot.get("model") or {}
        response["model_path"] = str(_model.get("path", "")) or None

        if debug or explain:
            _attach_trace(response, analysis_state, ml_output, debug, explain)

        required_fields = ["deployment_decision", "confidence", "selected_runtime"]
        missing = [f for f in required_fields if f not in response]
        if missing:
            raise HTTPException(status_code=422, detail={
                "error": "pipeline_output_incomplete",
                "missing_fields": missing,
                "pipeline_stage": "decision",
            })

        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _attach_trace(response, analysis_state, ml_output, debug, explain):
    """Attach DecisionTrace and/or explanation tree."""
    try:
        from src.core.decision_trace import DecisionTrace
        from src.core.explainability import build_explanation_tree

        evaluations = (
            analysis_state.get("response_evaluations")
            or analysis_state.get("evaluations")
            or []
        )
        best_ev = next(
            (ev for ev in evaluations if isinstance(ev, dict)), {},
        )

        _raw = analysis_state.get("confidence") or 0.5
        trace = DecisionTrace(
            selected_runtime_before_override=analysis_state.get("best_runtime", ""),
            selected_runtime_after_override=analysis_state.get("best_runtime", ""),
            sla_target_latency=None,
            measured_latency=(
                best_ev.get("latency_ms")
                if best_ev.get("latency_ms") is not None
                else best_ev.get("latency_avg_ms")
            ),
            memory_limit=None,
            measured_memory=best_ev.get("memory_mb"),
            rejection_reasons=[],
            conditional_reasons=[],
            confidence_before_scaling=float(
                _raw["score"] if isinstance(_raw, dict) else _raw
            ),
            confidence_after_scaling=ml_output.confidence,
            hardware_override_applied=False,
            override_reason=None,
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

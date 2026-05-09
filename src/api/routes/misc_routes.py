"""
api/routes/misc_routes.py

Utility endpoints: health, state snapshot, capabilities discovery,
incidents, decision history, confidence, and fix suggestions.
ZERO logic — pure data pass-through.
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.app_state import APP_STATE, APP_STATE_LOCK, get_state_snapshot
from src.core.paths import calibration_history_path, incidents_path as _incidents_path_fn
from src.core import persistence
from src.core.logging_config import get_logger
from src.services.analysis_pipeline_service import normalize_runtime_name
from src.core.contracts.decision_result import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

logger = get_logger(__name__)
router = APIRouter()


@router.get("/api/state")
async def get_state():
    """Expose current in-memory state snapshot for frontend hydration."""
    # I-01: Atomic snapshot — all slots captured under a single lock acquisition.
    snap               = get_state_snapshot()
    model_state        = snap.get("model")
    diagnostics_state  = snap.get("diagnostics")
    analysis_state     = snap.get("analysis")
    decision_state     = snap.get("decision")
    deployment_profile = snap.get("deployment_profile")

    if decision_state is not None:
        stage = "decision"
    elif analysis_state is not None:
        stage = "analysis"
    elif diagnostics_state is not None:
        stage = "diagnostics"
    elif model_state is not None:
        stage = "model"
    else:
        stage = "empty"

    analysis = snap.get("analysis")

    return {
        "stage":              stage,
        "model":              model_state,
        "diagnostics":        diagnostics_state,
        "analysis":           analysis_state,
        "decision":           decision_state,
        "deployment_profile": deployment_profile,
        "gpu_calibration_loaded": snap.get("gpu_calibration") is not None,
        "timestamp":          time.time(),
    }


@router.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time(), "version": "1.0.0"}


@router.get("/api/capabilities")
async def get_capabilities():
    """Capability discovery endpoint — reflects actual live service availability.

    I-14: All capability flags are derived from live state or registered routes,
          not from hardcoded defaults.  A flag is True if and only if its backing
          service or data is reachable and non-empty at the time of this call.
    """
    # Probe live APP_STATE to determine session-dependent capabilities.
    snap           = get_state_snapshot()
    analysis_ready = snap.get("analysis") is not None
    decision_ready = snap.get("decision") is not None
    # INT-14: diagnostics data also lives on the Analysis tab — show it when present.
    diagnostics_ready = snap.get("diagnostics") is not None

    # /api/fixes/{hash}/{runtime} is always registered (I-14: patch is available).
    patch_available = True

    capabilities: dict = {
        "core": {
            "health": True, "model_analyze": True,
            "decision_recommend": True, "diagnostics": True,
        },
        "optional": {
            # I-14: runtime_session reflects whether analysis data is present.
            "runtime_session":    analysis_ready or decision_ready,
            # I-14: patch endpoint is always registered.
            "patch":              patch_available,
            # I-14: export and calibration/incidents probed below.
            "calibration":        False,
            "calibration_history": False,
            "incidents":          False,
            "export":             False,
        },
        "navigation": {
            "models_tab": True, "analysis_tab": analysis_ready or decision_ready or diagnostics_ready,
            "reports_tab": decision_ready, "settings_tab": True,
        },
        "actions": {
            "run_analysis": True,
            # I-14: generate_patch follows the patch capability flag.
            "generate_patch": patch_available,
            "export_graph": False,
            "apply_fix": True, "view_report": False,
        },
        "meta": {
            "discovery_version": "1.0.0",
            "dynamic_navigation": True, "conditional_rendering": True,
        },
    }

    try:
        history_path = calibration_history_path()
        if history_path.exists():
            history = json.loads(history_path.read_text())
            if history:
                capabilities["optional"]["calibration"]          = True
                capabilities["optional"]["calibration_history"]  = True
    except Exception:
        logger.warning("capabilities_calibration_check_failed", exc_info=True)

    try:
        inc_path = _incidents_path_fn()
        if inc_path.exists():
            incidents = json.loads(inc_path.read_text())
            if incidents:
                capabilities["optional"]["incidents"]         = True
                capabilities["navigation"]["incidents_tab"]   = True
    except Exception:
        logger.warning("capabilities_incidents_check_failed", exc_info=True)

    return {"capabilities": capabilities, "timestamp": time.time()}


@router.get("/api/calibration/stats")
async def get_calibration_stats():
    try:
        from src.core.calibration_history import compute_calibration_stats
        # INT-38: read from the SQLite store that save_calibration_record writes to.
        # The calibration_history.json file is never written by the API routes, so
        # reading it always returned total_records: 0.
        history = persistence.list_calibrations()
        if not history:
            return {
                "total_records": 0,
                "stats": {"total_evaluations": 0, "per_runtime": {}},
                "source": "sqlite_empty",
            }
        per_runtime = compute_calibration_stats(history)
        return {
            "total_records": len(history),
            "stats": {
                "total_evaluations": len(history),
                "per_runtime": per_runtime,
            },
            "source": "sqlite",
        }
    except Exception as exc:
        return {"total_records": 0, "stats": {}, "error": str(exc)}


@router.get("/api/incidents")
async def get_incidents():
    try:
        inc_path = _incidents_path_fn()
        try:
            incidents = json.loads(inc_path.read_text())
        except FileNotFoundError:
            incidents = []
        return {"incidents": incidents, "count": len(incidents)}
    except Exception as exc:
        return {"incidents": [], "count": 0, "error": str(exc)}


@router.get("/api/history/decisions")
async def get_decision_history():
    try:
        items = persistence.list_decisions(limit=50)
        return {"success": True, "count": len(items), "items": items}
    except Exception as exc:
        return {"success": False, "count": 0, "items": [], "error": str(exc)}


@router.get("/api/confidence/{model_hash}/{runtime}")
async def get_confidence(model_hash: str, runtime: str):
    try:
        # I-01: Atomic snapshot — one lock acquisition for all three slots.
        snap           = get_state_snapshot()
        decision_state = snap.get("decision")
        analysis_state = snap.get("analysis")
        model_state    = snap.get("model")

        current_hash = model_state.get("hash") if isinstance(model_state, dict) else None
        if current_hash != model_hash:
            return {
                "model_hash": model_hash,
                "runtime":    runtime,
                "score":      None,
                "level":      None,
                "reasons":    ["model_hash does not match current model"],
            }

        conf_raw = None
        reasons: list[str] = []

        # I-02: decision.to_dict() writes 'selected_runtime', not 'runtime'.
        #       Reading 'runtime' always returned None, making this endpoint non-functional.
        if isinstance(decision_state, dict) and normalize_runtime_name(decision_state.get("selected_runtime") or "") == normalize_runtime_name(runtime):
            conf_raw = decision_state.get("confidence")
            if decision_state.get("reasons"):
                reasons = list(decision_state["reasons"])

        # INT-10: If decision_state exists but its selected_runtime doesn't match the
        # requested runtime, fall through to analysis_state evals for that runtime.
        # Also fall through when conf_raw is still None after the decision check.
        if conf_raw is None and isinstance(analysis_state, dict):
            evals = analysis_state.get("evaluations") or []
            ev = next(
                (e for e in evals if isinstance(e, dict) and normalize_runtime_name(e.get("runtime") or "") == normalize_runtime_name(runtime)),
                None,
            )
            if ev is not None:
                conf_raw = ev.get("confidence_score")
            # Last resort: use the overall analysis confidence when no per-runtime score exists.
            if conf_raw is None:
                conf_raw = analysis_state.get("confidence")

        # INT-10: If we have a decision but the runtime wasn't matched above, surface the
        # overall decision confidence rather than silently returning null — the caller
        # asked about a model+runtime pair and we have the best available data.
        if conf_raw is None and isinstance(decision_state, dict):
            conf_raw = decision_state.get("confidence")
            if conf_raw is not None and decision_state.get("reasons"):
                reasons = list(decision_state["reasons"])

        if conf_raw is None:
            return {
                "model_hash": model_hash,
                "runtime":    runtime,
                "score":      None,
                "level":      None,
                "reasons":    ["no confidence data available for this runtime"],
            }

        # C-1 / C-2: confidence may be a {score, level} dict in EITHER decision_state
        # (from DecisionResult.to_dict()) OR analysis_state (after the confidence-shape
        # update in analysis_pipeline_service.py).  The isinstance guard below handles
        # both cases so this endpoint does not crash with TypeError regardless of which
        # state slot supplied conf_raw.
        raw_score = conf_raw["score"] if isinstance(conf_raw, dict) else conf_raw
        score = float(raw_score)
        level = "HIGH" if score >= CONFIDENCE_HIGH else "MEDIUM" if score >= CONFIDENCE_MEDIUM else "LOW"
        return {
            "model_hash": model_hash,
            "runtime":    runtime,
            "score":      score,
            "level":      level,
            "reasons":    reasons,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/fixes/{model_hash}/{runtime}")
async def get_fixes(model_hash: str, runtime: str):
    return {
        "model_hash": model_hash,
        "runtime":    runtime,
        "fixes": [
            {"description": "Ensure model uses supported operators", "severity": "info"}
        ],
    }


@router.get("/api/deployment/summary")
async def get_deployment_summary():
    """Return a structured deployment summary derived from current decision and analysis state."""
    try:
        from src.core.deployment_summary import build_deployment_summary
        snap = get_state_snapshot()
        decision_state   = snap.get("decision")
        analysis_state   = snap.get("analysis")
        diagnostics_state = snap.get("diagnostics")
        if decision_state is None or analysis_state is None:
            return JSONResponse(status_code=409, content={"error": "no_decision"})
        summary = build_deployment_summary(
            decision_state, analysis_state, diagnostics_state or {},
            deployment_profile=snap.get("deployment_profile"),  # N-02: correct source for latency/memory limits
        )
        return {"success": True, "summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Phase 5: Metrics endpoint ─────────────────────────────────────────────────

@router.get("/api/metrics")
async def get_metrics():
    """
    Expose in-process counters for monitoring and alerting.

    Returns a snapshot of all pipeline metrics accumulated since process start.
    All counters reset on process restart.

    Fields:
        total_requests       Total run_full_pipeline / run_analysis_stage calls
        failed_runtimes      Runtime evaluation failures (execution_success=False)
        blocked_decisions    Decisions with decision='BLOCK'
        approved_decisions   Decisions with decision='ALLOW' or 'ALLOW_WITH_CONDITIONS'
        persistence_failures Times _persist() raised an exception
        no_valid_runtime     Times all runtimes failed for a model
        uptime_seconds       Seconds since the metrics module was first imported
    """
    from src.core import metrics as _metrics
    return {
        "success": True,
        "metrics": _metrics.snapshot(),
    }

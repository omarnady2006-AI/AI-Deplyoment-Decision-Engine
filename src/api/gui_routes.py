from __future__ import annotations
"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""
GUI Routes

FastAPI route declarations only. No business logic.
All logic delegated to gui_*_api modules.
"""

import json
import logging
import os
import time
import uuid
from math import sqrt
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.api.gui_analysis_api import (
    UPLOAD_DIR,
    UploadResult,
    analyze_uploaded_model,
    cleanup_upload_storage,
    handle_model_upload,
    run_model_pipeline,
    validate_analysis_schema,
)
from src.api.gui_calibration_api import (
    get_benchmark_results,
    get_calibration_history,
    record_benchmark_result,
    run_hardware_calibration,
)
from src.api.gui_state_api import clear_state, get_or_create_session, get_state, set_analysis_result, update_state
from src.core import persistence
from src.core.pipeline import _derive_decision_and_confidence
from src.core.model_analysis import analyze_model, build_adversarial_test_models
from src.diagnostics.report import Diagnostic, DiagnosticSeverity


logger = logging.getLogger(__name__)

# Telemetry: module-level safe import — failure must never affect request handling.
try:
    from src.core.telemetry import log_analysis_event
except Exception:
    log_analysis_event = None

MAX_REQUESTS_PER_MINUTE = 30
MAX_TRACKED_CLIENTS = 10000
RATE_LIMIT_WINDOW_SECONDS = 60
MAX_CALIBRATION_UPLOAD_SIZE = 5 * 1024 * 1024

TRUST_PROXY_HEADERS = os.environ.get("TRUST_PROXY_HEADERS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TRUSTED_PROXY_IPS = {
    ip.strip() for ip in os.environ.get("TRUSTED_PROXY_IPS", "").split(",") if ip.strip()
}

_rate_limit_store: dict[str, deque[float]] = {}
_last_cleanup_time = 0.0
# Maps session_id -> public model_id, populated on upload
_session_model_id_cache: dict[str, str] = {}


def _safe_decision_value(decision: Any) -> str:
    d = str(decision or "").strip().upper()
    if d in {"ALLOW", "ALLOW_WITH_CONDITIONS", "BLOCK"}:
        return d
    return "BLOCK"


class CalibrationRequest(BaseModel):
    model_hash: str
    hardware_profile: dict[str, Any]


class BenchmarkRequest(BaseModel):
    model_hash: str
    latency_ms: float
    memory_mb: float | None = None
    cpu_cores: int = 4
    ram_gb: float = 8.0
    gpu_available: bool = False


class ModelRefRequest(BaseModel):
    model_id: str
    constraints: dict[str, Any] | None = None
    model_config = {"extra": "forbid"}


class PipelineRunRequest(BaseModel):
    model_id: str
    constraints: dict[str, Any] | None = None
    model_config = {"extra": "forbid"}


def _cleanup_rate_limit_store() -> None:
    """Remove old entries and limit store size."""
    global _last_cleanup_time
    now = time.time()

    if now - _last_cleanup_time < 30:
        return

    _last_cleanup_time = now
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    expired_ips = []
    for ip, timestamps in list(_rate_limit_store.items()):
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()
        if not timestamps:
            expired_ips.append(ip)

    for ip in expired_ips:
        del _rate_limit_store[ip]

    if len(_rate_limit_store) > MAX_TRACKED_CLIENTS:
        oldest_ips = sorted(
            _rate_limit_store.keys(),
            key=lambda ip: _rate_limit_store[ip][0] if _rate_limit_store[ip] else now,
        )
        for ip in oldest_ips[: len(_rate_limit_store) - MAX_TRACKED_CLIENTS]:
            del _rate_limit_store[ip]


def _check_rate_limit(identity_key: str) -> bool:
    """Check if identity key is within rate limit."""
    _cleanup_rate_limit_store()

    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    if identity_key not in _rate_limit_store:
        _rate_limit_store[identity_key] = deque(maxlen=MAX_REQUESTS_PER_MINUTE + 1)

    timestamps = _rate_limit_store[identity_key]

    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()

    if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
        return False

    timestamps.append(now)
    return True


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    direct_host = request.client.host if request.client else "unknown"
    if not TRUST_PROXY_HEADERS:
        return direct_host

    if TRUSTED_PROXY_IPS and direct_host not in TRUSTED_PROXY_IPS:
        return direct_host

    forwarded = request.headers.get("X-Forwarded-For")
    if not forwarded:
        return direct_host

    first_hop = forwarded.split(",")[0].strip()
    return first_hop or direct_host


def _get_rate_limit_key(request: Request) -> str:
    return f"ip:{_get_client_ip(request)}"


def _resolve_allowed_model_path(model_id: str) -> str:
    try:
        uuid.UUID(str(model_id))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid model_id") from exc

    model_mapping = persistence.resolve_uploaded_model_mapping(model_id)
    if not model_mapping:
        raise HTTPException(status_code=400, detail="Model not found")

    model_path = str(model_mapping["server_path"])

    resolved_path = Path(model_path).resolve()
    upload_root = Path(UPLOAD_DIR).resolve()
    try:
        resolved_path.relative_to(upload_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc

    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=400, detail="Model file unavailable")

    return str(resolved_path)


def _resolve_allowed_model_info(model_id: str) -> tuple[str, str]:
    try:
        uuid.UUID(str(model_id))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid model_id") from exc

    model_mapping = persistence.resolve_uploaded_model_mapping(model_id)
    if not model_mapping:
        raise HTTPException(status_code=400, detail="Model not found")

    model_path = str(model_mapping["server_path"])
    model_hash = str(model_mapping["model_hash"])

    resolved_path = Path(model_path).resolve()
    upload_root = Path(UPLOAD_DIR).resolve()
    try:
        resolved_path.relative_to(upload_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc

    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=400, detail="Model file unavailable")

    return str(resolved_path), model_hash


def _get_session_id(request: Request) -> str:
    """Extract or create session ID from request."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = request.headers.get("X-Session-ID")
    return get_or_create_session(session_id)


def _derive_default_constraints_from_file(model_path: str) -> dict[str, float]:
    model_size_mb = Path(model_path).stat().st_size / (1024.0 * 1024.0)
    memory_limit = max(512.0, (model_size_mb * 3.0) + 256.0)
    target_latency = max(60.0, (model_size_mb * 0.8) + 80.0)
    return {
        "memory_limit": round(memory_limit, 4),
        "target_latency": round(target_latency, 4),
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    n = float(len(xs))
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    syy = sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    num = (n * sxy) - (sx * sy)
    den = sqrt(max(((n * sxx) - (sx * sx)) * ((n * syy) - (sy * sy)), 0.0))
    if den == 0.0:
        return 0.0
    return round(num / den, 6)


def _build_diagnostics_for_analysis(analysis: Any) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    unsupported_ops = list(getattr(analysis, "unsupported_ops", []) or [])
    if unsupported_ops:
        for op in unsupported_ops:
            diagnostics.append(Diagnostic(
                id="UNSUPPORTED_OPERATOR",
                severity=DiagnosticSeverity.WARN.value,
                title=f"Unsupported Operator: {op}",
                message=f"Unsupported operator: {op}",
                suggestion=f"Replace {op} with a supported alternative",
            ))
    if bool(getattr(analysis, "has_dynamic_shapes", False)):
        diagnostics.append(Diagnostic(
            id="DYNAMIC_SHAPES",
            severity=DiagnosticSeverity.INFO.value,
            title="Dynamic Shapes Detected",
            message="Model has dynamic input shapes",
            suggestion="Consider using static shapes for production deployment",
        ))
    return diagnostics


def _evaluate_model(model_path: str, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    analysis = analyze_model(model_path)
    if not analysis.success:
        return {
            "success": False,
            "model": Path(model_path).name,
            "error": analysis.error,
        }

    # Keep causality validation on the exact engine path used by API requests.
    # This avoids bypassing runtime profiling and preserves single decision authority.
    engine_result = run_model_pipeline(
        model_path,
        constraints=constraints,
    )
    if not engine_result.success:
        return {
            "success": False,
            "model": Path(model_path).name,
            "error": engine_result.error,
        }

    metrics = {}
    try:
        # Recompute decision metrics bundle for reporting only, but using runtime-backed defaults.
        # Decision value itself remains the engine authority.
        _, _, metrics = _derive_decision_and_confidence(
            analysis=analysis,
            model_path=model_path,
            constraints=constraints or _derive_default_constraints_from_file(model_path),
            diagnostics=_build_diagnostics_for_analysis(analysis),
            runtime_metrics=None,
        )
    except Exception:
        metrics = {}

    return {
        "success": True,
        "model": Path(model_path).name,
        "decision": engine_result.decision,
        "confidence": engine_result.confidence,
        "metrics": metrics,
    }


def run_causality_validation() -> dict[str, Any]:
    root = Path(".").resolve()
    generated = build_adversarial_test_models(str(root / "cache" / "uploads"))

    candidate_models = {
        "mnist": str((root / "models" / "mnist-8.onnx").resolve()),
        "resnet": str((root / "models" / "resnet50-v2-7.onnx").resolve()),
        "yolov5": str((root / "models" / "yolov5s.onnx").resolve()),
        "tiny_synthetic": generated.get("tiny_synthetic", ""),
        "empty_graph": generated.get("empty_graph", ""),
    }
    base_models = {k: v for k, v in candidate_models.items() if v and Path(v).exists()}

    baseline_results: dict[str, dict[str, Any]] = {}
    for model_name, model_path in base_models.items():
        baseline_results[model_name] = _evaluate_model(model_path)

    knockout_features = ["parameter_count", "operator_count", "sequential_depth", "compute_cost_score"]
    knockout_rows: list[dict[str, Any]] = []
    for model_name, model_path in base_models.items():
        baseline = baseline_results.get(model_name, {})
        base_decision = baseline.get("decision")
        base_risk = float((baseline.get("metrics") or {}).get("risk_score", 0.0))
        base_confidence = float(baseline.get("confidence", 0.0))
        for feature in knockout_features:
            test_constraints = {
                **_derive_default_constraints_from_file(model_path),
                "__stress_test": {"knockout_features": [feature]},
            }
            result = _evaluate_model(model_path, constraints=test_constraints)
            test_metrics = result.get("metrics", {}) if result.get("success") else {}
            test_decision = result.get("decision")
            test_risk = float(test_metrics.get("risk_score", 0.0))
            test_conf = float(result.get("confidence", 0.0))
            knockout_rows.append({
                "model": model_name,
                "feature_disabled": feature,
                "risk": round(test_risk, 4),
                "decision": test_decision,
                "confidence": round(test_conf, 4),
                "changed": bool(
                    (test_decision != base_decision)
                    or (abs(test_risk - base_risk) > 1e-6)
                    or (abs(test_conf - base_confidence) > 1e-6)
                ),
            })

    sensitivity_tests = [
        ("memory_pressure_x0", {"memory_pressure": 0.0}),
        ("memory_pressure_x10", {"memory_pressure": 10.0}),
        ("latency_pressure_x0", {"latency_pressure": 0.0}),
        ("latency_pressure_x10", {"latency_pressure": 10.0}),
        ("unsupported_pressure_x10", {"unsupported_pressure": 10.0}),
    ]
    sensitivity_rows: list[dict[str, Any]] = []
    for model_name, model_path in base_models.items():
        base = baseline_results.get(model_name, {})
        base_decision = base.get("decision")
        for label, multipliers in sensitivity_tests:
            test_constraints = {
                **_derive_default_constraints_from_file(model_path),
                "__stress_test": {"pressure_multipliers": multipliers},
            }
            result = _evaluate_model(model_path, constraints=test_constraints)
            metrics = result.get("metrics", {}) if result.get("success") else {}
            sensitivity_rows.append({
                "model": model_name,
                "test": label,
                "new_risk": round(float(metrics.get("risk_score", 0.0)), 4),
                "decision": result.get("decision"),
                "decision_change": result.get("decision") != base_decision,
            })

    constraint_rows: list[dict[str, Any]] = []
    for model_name, model_path in base_models.items():
        tests = [
            ("memory_limit_1", {"memory_limit": 1.0}),
            ("memory_limit_100000", {"memory_limit": 100000.0}),
            ("target_latency_1", {"target_latency": 1.0}),
            ("target_latency_100000", {"target_latency": 100000.0}),
        ]
        for label, c in tests:
            result = _evaluate_model(model_path, constraints=c)
            metrics = result.get("metrics", {}) if result.get("success") else {}
            constraint_rows.append({
                "model": model_name,
                "test": label,
                "risk": round(float(metrics.get("risk_score", 0.0)), 4),
                "decision": result.get("decision"),
                "memory_pressure": round(float(metrics.get("memory_pressure", 0.0)), 4),
                "latency_pressure": round(float(metrics.get("latency_pressure", 0.0)), 4),
            })

    metric_realism_rows: list[dict[str, Any]] = []
    for model_name, result in baseline_results.items():
        metrics = result.get("metrics", {}) if result.get("success") else {}
        metric_realism_rows.append({
            "model": model_name,
            "parameters": int(metrics.get("parameter_count", 0)),
            "operators": int(metrics.get("operator_count", 0)),
            "depth": int(metrics.get("node_count", 0) and analyze_model(base_models[model_name]).sequential_depth),
            "memory_mb": round(float(metrics.get("estimated_memory_mb", 0.0)), 4),
            "latency_ms": round(float(metrics.get("estimated_latency_ms", 0.0)), 4),
            "compute_cost": round(float(metrics.get("compute_cost_score", 0.0)), 4),
        })

    distribution_models = dict(base_models)
    for k, v in generated.items():
        if v and Path(v).exists():
            distribution_models[k] = v
    risk_values: list[float] = []
    risk_by_model: list[dict[str, Any]] = []
    for model_name, model_path in distribution_models.items():
        result = _evaluate_model(model_path)
        if not result.get("success"):
            continue
        risk = float((result.get("metrics") or {}).get("risk_score", 0.0))
        risk_values.append(risk)
        risk_by_model.append({"model": model_name, "risk": round(risk, 4)})
    histogram = {"0-2": 0, "2-4": 0, "4-6": 0, "6-8": 0, "8-10": 0}
    for r in risk_values:
        if r < 2:
            histogram["0-2"] += 1
        elif r < 4:
            histogram["2-4"] += 1
        elif r < 6:
            histogram["4-6"] += 1
        elif r < 8:
            histogram["6-8"] += 1
        else:
            histogram["8-10"] += 1

    corr_models = []
    for model_name, model_path in distribution_models.items():
        result = _evaluate_model(model_path)
        if not result.get("success"):
            continue
        metrics = result.get("metrics") or {}
        corr_models.append({
            "model": model_name,
            "parameter_count": float(metrics.get("parameter_count", 0.0)),
            "compute_cost": float(metrics.get("compute_cost_score", 0.0)),
            "latency": float(metrics.get("estimated_latency_ms", 0.0)),
            "memory": float(metrics.get("estimated_memory_mb", 0.0)),
            "risk": float(metrics.get("risk_score", 0.0)),
        })

    correlations = {
        "parameter_count_vs_risk": _pearson(
            [m["parameter_count"] for m in corr_models],
            [m["risk"] for m in corr_models],
        ),
        "compute_cost_vs_latency": _pearson(
            [m["compute_cost"] for m in corr_models],
            [m["latency"] for m in corr_models],
        ),
        "memory_estimate_vs_risk": _pearson(
            [m["memory"] for m in corr_models],
            [m["risk"] for m in corr_models],
        ),
    }

    adversarial_cases = {
        "case_a_huge_params_few_ops": generated.get("case_a_huge_params_few_ops", ""),
        "case_b_many_ops_tiny_params": generated.get("case_b_many_ops_tiny_params", ""),
        "case_c_deep_low_compute": generated.get("case_c_deep_low_compute", ""),
        "case_d_unsupported_only": generated.get("case_d_unsupported_only", ""),
    }
    adversarial_results: list[dict[str, Any]] = []
    for case_name, model_path in adversarial_cases.items():
        if not model_path or not Path(model_path).exists():
            continue
        result = _evaluate_model(model_path)
        metrics = result.get("metrics", {}) if result.get("success") else {}
        adversarial_results.append({
            "case": case_name,
            "risk": round(float(metrics.get("risk_score", 0.0)), 4),
            "decision": result.get("decision"),
            "confidence": round(float(result.get("confidence", 0.0)), 4),
            "parameter_count": int(metrics.get("parameter_count", 0)),
            "operator_count": int(metrics.get("operator_count", 0)),
            "compute_cost": round(float(metrics.get("compute_cost_score", 0.0)), 4),
        })

    determinism_model = base_models.get("mnist") or next(iter(base_models.values()))
    determinism_runs: list[dict[str, Any]] = []
    for i in range(10):
        result = _evaluate_model(determinism_model)
        metrics = result.get("metrics", {}) if result.get("success") else {}
        determinism_runs.append({
            "run": i + 1,
            "risk": round(float(metrics.get("risk_score", 0.0)), 4),
            "decision": result.get("decision"),
            "confidence": round(float(result.get("confidence", 0.0)), 4),
        })
    deterministic = len({(r["risk"], r["decision"], r["confidence"]) for r in determinism_runs}) == 1

    hidden_shortcut_findings: list[dict[str, Any]] = []
    search_terms = ["min(", "max(", "clamp(", "if risk", "decision ="]
    for rel_path in ["src/core/pipeline.py", "src/core/model_analysis.py", "src/api/gui_routes.py"]:
        file_path = root / rel_path
        if not file_path.exists():
            continue
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            lowered = line.lower()
            for term in search_terms:
                if term in lowered:
                    hidden_shortcut_findings.append({
                        "file": rel_path,
                        "line": idx,
                        "match": term,
                        "text": line.strip(),
                    })

    total_knockout = len(knockout_rows)
    changed_knockout = sum(1 for row in knockout_rows if row.get("changed"))
    change_ratio = (changed_knockout / total_knockout) if total_knockout else 0.0
    strong_corr_count = sum(1 for v in correlations.values() if abs(v) >= 0.5)
    boundary_coverage = sum(1 for v in histogram.values() if v > 0)

    if change_ratio >= 0.6 and strong_corr_count >= 2 and boundary_coverage >= 3 and deterministic:
        final_classification = "TRULY MODEL DRIVEN"
    elif change_ratio >= 0.4 and strong_corr_count >= 1 and boundary_coverage >= 2 and deterministic:
        final_classification = "MOSTLY MODEL DRIVEN"
    elif change_ratio >= 0.2:
        final_classification = "WEAKLY MODEL DRIVEN"
    else:
        final_classification = "RULE SYSTEM IN DISGUISE"

    report = {
        "feature_knockout_results": knockout_rows,
        "weight_sensitivity_results": sensitivity_rows,
        "constraint_sensitivity": constraint_rows,
        "metric_ranges": {
            "rows": metric_realism_rows,
            "parameters": {
                "min": min((r["parameters"] for r in metric_realism_rows), default=0),
                "max": max((r["parameters"] for r in metric_realism_rows), default=0),
            },
            "operators": {
                "min": min((r["operators"] for r in metric_realism_rows), default=0),
                "max": max((r["operators"] for r in metric_realism_rows), default=0),
            },
            "memory_mb": {
                "min": round(min((r["memory_mb"] for r in metric_realism_rows), default=0.0), 4),
                "max": round(max((r["memory_mb"] for r in metric_realism_rows), default=0.0), 4),
            },
            "latency_ms": {
                "min": round(min((r["latency_ms"] for r in metric_realism_rows), default=0.0), 4),
                "max": round(max((r["latency_ms"] for r in metric_realism_rows), default=0.0), 4),
            },
            "compute_cost": {
                "min": round(min((r["compute_cost"] for r in metric_realism_rows), default=0.0), 4),
                "max": round(max((r["compute_cost"] for r in metric_realism_rows), default=0.0), 4),
            },
        },
        "risk_distribution": {
            "histogram": histogram,
            "risk_rows": risk_by_model,
        },
        "correlation_findings": correlations,
        "adversarial_tests": adversarial_results,
        "decision_determinism": {
            "model": Path(determinism_model).name if determinism_model else None,
            "runs": determinism_runs,
            "deterministic": deterministic,
        },
        "hidden_shortcut_detection": hidden_shortcut_findings,
        "final_classification": final_classification,
        "classification_evidence": {
            "knockout_change_ratio": round(change_ratio, 4),
            "strong_correlation_count": strong_corr_count,
            "risk_histogram_occupied_bins": boundary_coverage,
            "deterministic": deterministic,
        },
    }
    return report


def register_routes(app: FastAPI) -> None:
    """Register all API routes on the FastAPI app."""

    @app.get("/api/health")
    async def api_health():
        return {"status": "healthy"}

    # ── Deployment profile save / delete ─────────────────────────────────────
    @app.post("/api/deployment/profile")
    async def api_save_profile(request: Request, payload: dict = Body(default={})):
        session_id = _get_session_id(request)
        update_state(session_id, deployment_profile=payload)
        from fastapi.responses import JSONResponse
        resp = JSONResponse(content={"success": True, "profile": payload})
        resp.set_cookie("session_id", session_id, httponly=True, samesite="lax")
        return resp

    @app.delete("/api/deployment/profile")
    async def api_delete_profile(request: Request):
        session_id = _get_session_id(request)
        update_state(session_id, deployment_profile=None)
        return {"success": True}

    # ── Thin wrappers so legacy frontend API calls still resolve ─────────────
    @app.post("/api/model/diagnostics")
    async def api_model_diagnostics(request: Request, payload: ModelRefRequest):
        try:
            model_path, model_hash = _resolve_allowed_model_info(payload.model_id)
            result = await run_in_threadpool(
                run_model_pipeline, model_path, payload.constraints, None, model_hash, None
            )
            return {
                "success": result.success,
                "model_id": payload.model_id,
                "diagnostics": [getattr(d, "message", str(d)) for d in (result.diagnostics or [])],
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"diagnostics_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/model/analyze")
    async def api_model_analyze(request: Request, payload: ModelRefRequest):
        try:
            model_path, model_hash = _resolve_allowed_model_info(payload.model_id)
            result = await run_in_threadpool(
                run_model_pipeline, model_path, payload.constraints, None, model_hash, None
            )
            _dm = getattr(result, "decision_metrics", {}) or {}
            return {
                "success": result.success,
                "model_id": payload.model_id,
                "decision": _safe_decision_value(result.decision),
                "confidence": float(result.confidence or 0.0),
                "recommended_runtime": result.recommended_runtime.value if result.recommended_runtime else None,
                "decision_metrics": _dm,
                "diagnostics": [getattr(d, "message", str(d)) for d in (result.diagnostics or [])],
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"model_analyze_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/decision/recommend")
    async def api_decision_recommend(request: Request, payload: ModelRefRequest):
        try:
            model_path, model_hash = _resolve_allowed_model_info(payload.model_id)
            result = await run_in_threadpool(
                run_model_pipeline, model_path, payload.constraints, None, model_hash, None
            )
            _runtime = result.recommended_runtime.value if result.recommended_runtime else "onnxruntime"
            _dm = getattr(result, "decision_metrics", {}) or {}
            return {
                "success": result.success,
                "best_runtime": _runtime,
                "decision": _safe_decision_value(result.decision),
                "confidence": float(result.confidence or 0.0),
                "evaluations": [{
                    "runtime": _runtime,
                    "decision": _safe_decision_value(result.decision),
                    "predicted_latency_ms": float(_dm.get("estimated_latency_ms") or 0),
                    "utility_score": float(result.confidence or 0.0),
                }],
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"decision_recommend_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/state")
    async def api_get_state(request: Request):
        session_id = _get_session_id(request)
        state = get_state(session_id)
        model_path = state.get("model_path")
        model_hash = state.get("model_hash")
        model_id   = _session_model_id_cache.get(session_id)
        if model_path or model_hash:
            state["model"]    = {"path": model_path, "hash": model_hash, "model_id": model_id}
            state["model_id"] = model_id
            state["stage"]    = "model"
        else:
            state["model"]    = None
            state["model_id"] = None
            state["stage"]    = "empty"
        # Set session cookie so the same session is used across requests
        from fastapi.responses import JSONResponse
        resp = JSONResponse(content=state)
        resp.set_cookie("session_id", session_id, httponly=True, samesite="lax")
        return resp

    @app.post("/api/model/upload")
    async def api_upload_model(request: Request, file: UploadFile = File(...)):
        request_start = time.perf_counter()
        identity_key = _get_rate_limit_key(request)
        if not _check_rate_limit(identity_key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 30 requests per minute.",
            )
        try:
            upload_result: UploadResult = await run_in_threadpool(handle_model_upload, file)
            model_path = upload_result.model_path
            model_hash = upload_result.model_hash

            public_model_id = str(uuid.uuid4())
            persistence.save_uploaded_model_mapping(
                model_id=public_model_id,
                model_hash=model_hash,
                server_path=str(model_path),
                owner_user_id=None,
            )
            persistence.save_model_record(
                file_hash=model_hash,
                operator_count=None,
                parameter_count=None,
                model_path=str(model_path),
            )

            # Update session state so /api/state reflects the uploaded model
            session_id = _get_session_id(request)
            update_state(session_id, model_path=str(model_path), model_hash=model_hash)
            _session_model_id_cache[session_id] = public_model_id

            from fastapi.responses import JSONResponse
            response_body = {
                "success": True,
                "model_id": public_model_id,
                "model_hash": model_hash,
                "performance": {
                    "upload_time_ms": upload_result.upload_time_ms,
                    "validation_time_ms": upload_result.validation_time_ms,
                    "total_request_time_ms": (time.perf_counter() - request_start) * 1000,
                },
            }
            resp = JSONResponse(content=response_body)
            resp.set_cookie("session_id", session_id, httponly=True, samesite="lax")
            return resp
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"upload_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/pipeline/run")
    async def api_run_pipeline(request: Request, payload: PipelineRunRequest):
        request_start = time.perf_counter()
        identity_key = _get_rate_limit_key(request)
        if not _check_rate_limit(identity_key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 30 requests per minute.",
            )
        try:
            model_path, model_hash = _resolve_allowed_model_info(payload.model_id)
            result = await run_in_threadpool(
                run_model_pipeline,
                model_path,
                payload.constraints,
                None,
                model_hash,
                None,
            )

            if not result.success:
                raise HTTPException(status_code=400, detail=result.error or "Pipeline failed")

            return {
                "success": True,
                "model_id": payload.model_id,
                "model_hash": result.model_hash,
                "decision": _safe_decision_value(result.decision),
                "confidence": result.confidence,
                "security_risk": float(getattr(result, "security_risk", 0.0) or 0.0),
                "security_report": getattr(result, "security_report", {}),
                "recommended_runtime": result.recommended_runtime.value if result.recommended_runtime else None,
                "diagnostics_count": len(result.diagnostics),
                "elapsed_ms": result.elapsed_ms,
                "performance": {
                    "validation_time_ms": result.validation_time_ms,
                    "analysis_time_ms": result.analysis_time_ms,
                    "decision_time_ms": result.decision_time_ms,
                    "total_request_time_ms": (time.perf_counter() - request_start) * 1000,
                },
                "request_echo": {
                    "model_id": payload.model_id,
                    "constraints": payload.constraints,
                },
            }
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"pipeline_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/analyze-and-decide")
    async def api_analyze_and_decide(payload: ModelRefRequest, response: Response):
        try:
            print("API ANALYZE HIT")
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            model_path, model_hash = _resolve_allowed_model_info(payload.model_id)
            # Keep API behavior identical to engine authority:
            # run_pipeline() owns default constraint derivation when constraints are absent.
            effective_constraints = payload.constraints
            result = await run_in_threadpool(
                run_model_pipeline,
                model_path,
                effective_constraints,
                None,
                model_hash,
                None,
            )
            # Log decision source for observability; never block on it.
            # The pipeline is the sole active engine.
            _ds = str(getattr(result, "decision_source", "") or "")
            if _ds and _ds != "_derive_decision_and_confidence":
                logger.warning("unexpected_decision_source: %s", _ds)

            server_time = time.time()
            random_marker = uuid.uuid4().hex
            _decision = _safe_decision_value(result.decision)
            _confidence = float(result.confidence or 0.0)
            _dm = getattr(result, "decision_metrics", {}) or {}
            _risk_score = float(_dm.get("risk_score") or _dm.get("risk_value") or 0.0)
            _est_latency = float(_dm.get("estimated_latency_ms") or _dm.get("real_latency_ms") or 0.0)
            _est_memory = float(_dm.get("estimated_memory_mb") or _dm.get("real_memory_mb") or 0.0)
            _recommended_runtime = result.recommended_runtime.value if result.recommended_runtime else "onnxruntime"
            _diagnostics_messages = [
                getattr(d, "message", str(d)) for d in (result.diagnostics or [])
            ]

            # ── Feature 2: Telemetry Logging ──────────────────────────────────
            # log_analysis_event is imported at module level with a None fallback,
            # so this block is fully safe even if the telemetry module is absent.
            if log_analysis_event:
                try:
                    _model_name = Path(model_path).name
                    _detected_framework = str(_dm.get("detected_framework", "onnx"))
                    log_analysis_event({
                        "model": _model_name,
                        "framework": _detected_framework,
                        "risk_score": _risk_score,
                        "decision": _decision,
                        "confidence": _confidence,
                    })
                except Exception as e:
                    logger.warning("telemetry_event_failed", exc_info=True)  # Telemetry failure must not break API
            # ──────────────────────────────────────────────────────────────────

            return {
                # ── Top-level canonical fields (engine authority) ──────────────
                "success": result.success,
                "decision": _decision,
                "confidence": _confidence,
                "risk": _risk_score,
                "risk_score": _risk_score,
                "risk_level": (
                    "CRITICAL" if _risk_score >= 8.0
                    else "HIGH" if _risk_score >= 6.0
                    else "MEDIUM" if _risk_score >= 4.0
                    else "LOW"
                ),
                # ── UI-compatible nested structure ────────────────────────────
                # renderUnifiedResult() reads: result.runtime_details.status
                #                              result.runtime_details.confidence
                #                              result.runtime_details.runtime
                "runtime_details": {
                    "status": _decision,
                    "confidence": _confidence,
                    "runtime": _recommended_runtime,
                },
                # renderUnifiedResult() reads: result.summary.estimated_latency_ms
                #                              result.summary.estimated_memory_mb
                #                              result.summary.recommended_runtime
                #                              result.summary.issues
                "summary": {
                    "estimated_latency_ms": _est_latency if _est_latency > 0 else None,
                    "estimated_memory_mb": _est_memory if _est_memory > 0 else None,
                    "recommended_runtime": _recommended_runtime,
                    "confidence": _confidence,
                    "issues": _diagnostics_messages,
                    "suggestions": [],
                },
                # ── Ancillary fields ──────────────────────────────────────────
                "security_risk": float(getattr(result, "security_risk", 0.0) or 0.0),
                "security_report": getattr(result, "security_report", {}),
                "server_time": server_time,
                "marker": random_marker,
                "performance": {
                    "validation_time_ms": result.validation_time_ms,
                    "analysis_time_ms": result.analysis_time_ms,
                    "decision_time_ms": result.decision_time_ms,
                    "profiling_time_ms": result.profiling_time_ms,
                    "total_ms": result.total_time_ms,
                },
                "phases": result.phases,
                "constraints_applied": effective_constraints,
            }
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"analyze_and_decide_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/calibration/run")
    async def api_run_calibration(request: CalibrationRequest):
        try:
            result = run_hardware_calibration(
                model_hash=request.model_hash,
                hardware_profile=request.hardware_profile,
            )
            return {"success": True, **result}
        except Exception as e:
            logger.exception(f"calibration_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/calibration/stats")
    async def api_get_calibration_stats():
        return get_calibration_history()

    @app.post("/api/calibration/gpu/load")
    async def api_gpu_calibration_load(request: Request, file: UploadFile = File(...)):
        """Load GPU calibration from uploaded file."""
        identity_key = _get_rate_limit_key(request)
        if not _check_rate_limit(identity_key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 30 requests per minute.",
            )
        try:
            total = 0
            chunks: list[bytes] = []
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_CALIBRATION_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="Calibration file too large")
                chunks.append(chunk)

            if total == 0:
                raise HTTPException(status_code=400, detail="Empty calibration file")

            cal_data = json.loads(b"".join(chunks))

            from src.core.gpu_calibration import GPUCalibrationProfile

            profile = GPUCalibrationProfile(cal_data)

            session_id = _get_session_id(request)
            update_state(session_id, gpu_calibration=profile)

            return {
                "success": True,
                "providers": list(cal_data.get("providers", {}).keys())
                if isinstance(cal_data.get("providers"), dict)
                else [],
                "gpu_name": cal_data.get("gpu_name", "unknown"),
            }

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        except Exception as e:
            logger.exception(f"gpu_calibration_load_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/benchmark/{model_hash}")
    async def api_get_benchmark(model_hash: str):
        return get_benchmark_results(model_hash)

    @app.post("/api/benchmark/record")
    async def api_record_benchmark(request: BenchmarkRequest):
        try:
            record_benchmark_result(
                model_hash=request.model_hash,
                cpu_cores=request.cpu_cores,
                ram_gb=request.ram_gb,
                gpu_available=request.gpu_available,
                latency_ms=request.latency_ms,
                memory_mb=request.memory_mb,
            )
            return {"success": True}
        except Exception as e:
            logger.exception(f"benchmark_record_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/causality/validate")
    async def api_causality_validate():
        try:
            return {
                "success": True,
                "report": await run_in_threadpool(run_causality_validation),
            }
        except Exception as e:
            logger.exception(f"causality_validation_failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


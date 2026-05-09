"""
api/routes/deployment_routes.py

Deployment profile management and calibration update endpoints.
ZERO logic — parse request → call service/core → return response.
"""
from __future__ import annotations

import time
import json as _json

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.app_state import APP_STATE, APP_STATE_LOCK
from src.core.scoring.calibration_math import compute_p95_reference
from src.core.logging_config import get_logger
from src.core import persistence

logger = get_logger(__name__)
router = APIRouter()


# ── Request Models ────────────────────────────────────────────────────────────

class DeploymentProfileRequest(BaseModel):
    cpu_cores: int
    ram_gb: float
    gpu_available: bool
    cuda_available: bool
    vram_gb: float | None           = None
    trt_available: bool | None      = None
    stress_test: bool | None        = None
    cpu_arch: str | None            = None
    ram_ddr: str | None             = None
    target_latency_ms: float | None = None
    memory_limit_mb: float | None   = None


class CalibrationUpdateRequest(BaseModel):
    latencies: list[float]
    model_sizes: list[int]
    notes: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/api/deployment/profile")
async def save_deployment_profile(request: DeploymentProfileRequest):
    """Save deployment profile and reset downstream pipeline stages."""
    try:
        cpu_cores = int(request.cpu_cores)
        if cpu_cores < 1:
            raise HTTPException(status_code=422, detail="cpu_cores must be >= 1")
        ram_gb = float(request.ram_gb)
        if ram_gb <= 0:
            raise HTTPException(status_code=422, detail="ram_gb must be > 0")

        profile = {
            "cpu_cores":         cpu_cores,
            "cpu_arch":          request.cpu_arch or None,
            "ram_gb":            ram_gb,
            "ram_ddr":           request.ram_ddr or None,
            "gpu_available":     bool(request.gpu_available),
            "cuda_available":    bool(request.cuda_available),
            "trt_available":     bool(request.trt_available) if request.trt_available is not None else False,
            "stress_test":       bool(request.stress_test) if request.stress_test is not None else False,
            "vram_gb":           float(request.vram_gb) if request.vram_gb is not None else None,
            "target_latency_ms": float(request.target_latency_ms) if request.target_latency_ms is not None else None,
            "memory_limit_mb":   float(request.memory_limit_mb) if request.memory_limit_mb is not None else None,
        }

        with APP_STATE_LOCK:
            if APP_STATE.get("deployment_profile") != profile:
                APP_STATE["diagnostics"] = None
                APP_STATE["analysis"]    = None
                APP_STATE["decision"]    = None
            APP_STATE["deployment_profile"] = _json.loads(_json.dumps(profile))

        return {"success": True, "deployment_profile": profile, "timestamp": time.time()}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/api/deployment/profile")
async def clear_deployment_profile():
    """Clear the deployment profile and reset downstream pipeline."""
    try:
        with APP_STATE_LOCK:
            APP_STATE["deployment_profile"] = None
            APP_STATE["diagnostics"]        = None
            APP_STATE["analysis"]           = None
            APP_STATE["decision"]           = None
        return {"success": True, "message": "Profile cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/calibration/update")
async def update_calibration(request: CalibrationUpdateRequest):
    """Update latency calibration reference from empirical observations.

    # NOTE: This route is not wired to any UI element. Reachable via direct API calls only.
    """
    try:
        if len(request.latencies) != len(request.model_sizes):
            raise HTTPException(status_code=422,
                detail="latencies and model_sizes must have the same length")
        if not request.latencies:
            raise HTTPException(status_code=422, detail="latencies must not be empty")

        latencies = [float(x) for x in request.latencies]
        if any(not __import__("math").isfinite(x) or x <= 0 for x in latencies):
            raise HTTPException(status_code=422,
                detail="latencies must contain positive finite numbers")

        model_sizes = [int(x) for x in request.model_sizes]
        if any(x < 0 for x in model_sizes):
            raise HTTPException(status_code=422,
                detail="model_sizes must contain non-negative integers")

        # Delegate computation to core
        updated_reference = compute_p95_reference(latencies)
        timestamp = time.time()

        with APP_STATE_LOCK:
            cal = APP_STATE.get("calibration")
            if not isinstance(cal, dict):
                cal = {}
                APP_STATE["calibration"] = cal
            cal["latency_reference_ms"] = updated_reference
            cal["updated_at"]           = timestamp

        return {"success": True, "latency_reference_ms": updated_reference, "timestamp": timestamp}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/calibration/gpu/load")
async def load_gpu_calibration(file: UploadFile = File(...)):
    """Load a GPU benchmark calibration JSON into the decision pipeline."""
    import json
    from src.core.gpu_calibration import GPUCalibrationProfile, GPUCalibrationError

    raw_bytes = await file.read()

    try:
        raw_json = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        return JSONResponse(status_code=422,
                            content={"success": False, "error": "invalid_json", "detail": str(exc)})

    try:
        profile = GPUCalibrationProfile(raw_json)
    except GPUCalibrationError as exc:
        return JSONResponse(status_code=422,
                            content={"success": False, "error": "calibration_validation_failed", "detail": str(exc)})
    except Exception as exc:
        logger.exception("load_gpu_calibration: unexpected error: %s", exc)
        return JSONResponse(status_code=500,
                            content={"success": False, "error": "calibration_construction_error", "detail": str(exc)})

    with APP_STATE_LOCK:
        APP_STATE["gpu_calibration"] = profile

    persistence.save_calibration_record(
        provider_list=profile.providers_detected,
        gpu_name=profile.environment.get("gpu_name"),
        raw_json=raw_json,
    )

    summary = profile.summary()
    logger.info("gpu_calibration_loaded", extra={
        "event": "gpu_calibration_loaded",
        "providers_detected": summary["providers_detected"],
        "entry_count": summary["entry_count"],
    })
    return {"success": True, "summary": summary}

"""
api/routes/model_routes.py

Model upload and diagnostics endpoints.
ZERO logic — each route: parse request → call service → return response.
"""
from __future__ import annotations

import asyncio
import functools
import json as _json
import time
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.app_state import APP_STATE, APP_STATE_LOCK, get_state_snapshot, _invalid_execution_order
from src.core.model_facts.analyzer import analyze_model_file
from src.services.upload_service import save_uploaded_model
from src.core import persistence
from src.core.logging_config import get_logger, safe_extra

logger = get_logger(__name__)
router = APIRouter()


@router.post("/api/model/index")
async def index_model(file: UploadFile = File(...)):
    """Accept an uploaded model file, save it, and analyse it."""
    # Phase 6 — input guard
    if not file or not file.filename:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No file provided"}
        )

    try:
        logger.info("step: request received", extra={"upload_filename": file.filename})
        print("STEP 4: backend received file", file.filename)

        filename = file.filename.lower()

        ALLOWED_EXT = (".onnx", ".pt", ".pth", ".tflite")

        if not filename.endswith(ALLOWED_EXT):
            raise RuntimeError(
                f"Unsupported model format: {filename}. "
                "Allowed: .onnx, .pt, .pth, .tflite"
            )

        upload = await save_uploaded_model(file)
        logger.info("step: file saved")
        model_path = upload["model_path"]
        model_hash = upload["model_hash"]
        print("STEP 5: saved path", model_path)

        logger.info("step: computing analysis")
        loop = asyncio.get_event_loop()

        ext = model_path.lower()
        IS_ONNX = ext.endswith(".onnx")

        if IS_ONNX:
            analysis = await loop.run_in_executor(
                None,
                functools.partial(analyze_model_file, model_path, precomputed_hash=model_hash),
            )
        else:
            class AnalysisStub:
                def __init__(self):
                    self.success = True
                    self.operator_count = 0
                    self.parameter_count = 0

                def to_dict(self):
                    return {
                        "success": True,
                        "note": "analysis skipped for non-ONNX format"
                    }

            analysis = AnalysisStub()

        logger.info("step: updating state")

        with APP_STATE_LOCK:
            if analysis.success:
                APP_STATE["model"] = {"path": model_path, "hash": model_hash}
            else:
                APP_STATE["model"] = None
            APP_STATE["diagnostics"] = None
            APP_STATE["analysis"]    = None
            APP_STATE["decision"]    = None

        if analysis.success:
            analysis_dict = analysis.to_dict()
            analysis_dict["file_hash"] = model_hash
            analysis_dict["success"] = True
            await loop.run_in_executor(
                None,
                functools.partial(
                    persistence.save_model_record,
                    file_hash=model_hash,
                    operator_count=analysis.operator_count,
                    parameter_count=analysis.parameter_count,
                ),
            )
            logger.info("model_uploaded", extra={
                "event": "model_uploaded", "model_hash": model_hash,
                "operator_count": analysis.operator_count,
                "parameter_count": analysis.parameter_count,
            })
            print("STEP 6: returning success response")
            return {
                "success": True,
                "data": analysis_dict,
            }
        else:
            logger.warning("model_upload_failed", extra={
                "event": "model_upload_failed", "error": analysis.error,
            })
            return {
                "success": False,
                "error": analysis.error or "Model analysis failed",
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("model_index_failed", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)}
        )


@router.post("/api/model/diagnostics")
async def get_diagnostics():
    """Compute and store diagnostics for the current model.

    # NOTE: This route is not wired to any UI element. Reachable via direct API calls only.

    Idempotent: re-running diagnostics (e.g. with a new deployment profile)
    clears existing diagnostics, analysis, and decision state so the pipeline
    can proceed from a fresh baseline without requiring a full model re-upload.
    """
    # I-01 / I-12: Atomic snapshot at entry — all prerequisite reads from snapshot.
    snapshot           = get_state_snapshot()
    model              = snapshot.get("model")
    deployment_profile = snapshot.get("deployment_profile")

    if model is None:
        return _invalid_execution_order("model")
    if deployment_profile is None:
        return _invalid_execution_order("deployment_profile")

    model_path  = (model.get("path", "") if isinstance(model, dict) else "")
    stored_hash = (model.get("hash")     if isinstance(model, dict) else None)

    if not model_path:
        return JSONResponse(status_code=409, content={
            "success": False, "error": "invalid_execution_order", "required_stage": "model",
        })

    # Compute phase (outside lock) — always re-run; results overwrite existing state
    ext = model_path.lower()
    IS_ONNX = ext.endswith(".onnx")

    if IS_ONNX:
        analysis = analyze_model_file(model_path, precomputed_hash=stored_hash)
    else:
        class AnalysisStub:
            def __init__(self):
                self.success = True
                self.operator_count = 0
                self.parameter_count = 0
                self.parameter_scale_class = "unknown"
                self.has_dynamic_shapes = False

            def to_dict(self):
                return {
                    "success": True,
                    "note": "diagnostics skipped for non-ONNX format"
                }

        analysis = AnalysisStub()

    if not analysis.success:
        return {"success": False, "error": analysis.error}

    facts = analysis.to_dict()

    # Write phase (under lock) — overwrite diagnostics/analysis/decision unconditionally
    with APP_STATE_LOCK:
        cur_model = APP_STATE.get("model")
        cur_hash  = cur_model.get("hash") if isinstance(cur_model, dict) else None
        if cur_hash != stored_hash:
            return JSONResponse(status_code=409, content={
                "success": False, "error": "invalid_execution_order", "required_stage": "model",
            })
        APP_STATE["diagnostics"] = {
            "facts":            facts,
            "raw_analysis":     facts,
            "profile_snapshot": _json.loads(_json.dumps(deployment_profile)),
            "diagnostics":      list(facts.get("diagnostics", [])) if isinstance(facts.get("diagnostics"), list) else [],
        }
        APP_STATE["analysis"] = None
        APP_STATE["decision"] = None

    logger.info("diagnostics_run", extra={
        "event": "diagnostics_run", "model_hash": stored_hash,
        "parameter_scale_class": analysis.parameter_scale_class,
        "has_dynamic_shapes":    analysis.has_dynamic_shapes,
    })
    return {"success": True}

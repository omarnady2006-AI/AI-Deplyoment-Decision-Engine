from __future__ import annotations

from threading import RLock
from typing import Any

from fastapi.responses import JSONResponse


APP_STATE: dict[str, Any] = {
    "model": None,
    "diagnostics": None,
    "analysis": None,
    "decision": None,
    "deployment_profile": None,
    "calibration": {
        "latency_reference_ms": None,
        "updated_at": None,
    },
    "gpu_calibration": None,
}

APP_STATE_LOCK = RLock()


def get_state_snapshot() -> dict[str, Any]:
    """Atomically capture all APP_STATE slots under the lock.

    Every route handler MUST call this once at entry and operate exclusively
    on the returned snapshot for all reads within that request.  Re-reading
    APP_STATE directly after receiving the snapshot re-opens the TOCTOU
    window this function is designed to close.

    Writes to APP_STATE still go through APP_STATE_LOCK with the existing
    CAS-style hash guards.
    """
    with APP_STATE_LOCK:
        return dict(APP_STATE)


def _invalid_execution_order(stage_name: str) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "success": False,
            "error": "invalid_execution_order",
            "required_stage": stage_name,
        },
    )

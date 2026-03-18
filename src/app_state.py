from __future__ import annotations

from functools import wraps
from inspect import iscoroutinefunction
from threading import RLock
from typing import Any, Callable

from fastapi.responses import JSONResponse


APP_STATE = {
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


def _stage_ready(stage_name: str) -> bool:
    with APP_STATE_LOCK:
        return APP_STATE.get(stage_name) is not None


def _invalid_execution_order(stage_name: str) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "error": "invalid_execution_order",
            "required_stage": stage_name,
        },
    )


def require_stage(stage_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _stage_ready(stage_name):
                    return _invalid_execution_order(stage_name)
                return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _stage_ready(stage_name):
                return _invalid_execution_order(stage_name)
            return func(*args, **kwargs)

        return sync_wrapper

    return decorator

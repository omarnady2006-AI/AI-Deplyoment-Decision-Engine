"""
api/gui_app.py

FastAPI application setup.

Responsibilities (ONLY):
    - Create and configure the FastAPI app
    - Register CORS middleware
    - Register all route modules
    - Run startup side-effects (HW profiling, calibration recovery)
    - Mount the static UI

ZERO business logic. ZERO endpoint implementation.
All endpoints live in api/routes/.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.core.logging_config import get_logger
from src.core import persistence

logger = get_logger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _enforce_single_worker()
    _recover_gpu_calibration()
    _warm_server_hardware_profile()
    yield


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Deployment Decision Engine API",
    description="API for deployment decision making and model analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# F-09: Wildcard origins are NOT safe for production — any website can read
# model hashes, decision results, and calibration data via cross-origin requests.
# Read a comma-separated list from ALLOWED_ORIGINS; default to localhost only.
_origins = [o.strip() for o in
            os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,   # was True — invalid with wildcard origin per CORS spec
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Router Registration ────────────────────────────────────────────────────────

from src.api.routes.model_routes      import router as model_router
from src.api.routes.analysis_routes   import router as analysis_router
from src.api.routes.decision_routes   import router as decision_router
from src.api.routes.deployment_routes import router as deployment_router
from src.api.routes.misc_routes       import router as misc_router

app.include_router(model_router)
app.include_router(analysis_router)
app.include_router(decision_router)
app.include_router(deployment_router)
app.include_router(misc_router)

# ── Startup ───────────────────────────────────────────────────────────────────


def _enforce_single_worker() -> None:
    import os
    # F-12 (part 1): lock-file fallback that works even without psutil.
    # Write /tmp/dde.lock with the current PID on startup; if a second worker
    # starts and the PID in the file is still alive, raise RuntimeError.
    _LOCK_FILE = "/tmp/dde.lock"
    current_pid = os.getpid()
    try:
        if os.path.exists(_LOCK_FILE):
            try:
                with open(_LOCK_FILE) as _lf:
                    held_pid = int(_lf.read().strip())
                # Check if the holding process is still alive (signal 0 = existence check)
                try:
                    os.kill(held_pid, 0)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    alive = False
                if alive and held_pid != current_pid:
                    raise RuntimeError(
                        f"Strict APP_STATE authority requires a single worker process; "
                        f"another DDE worker (PID {held_pid}) is already running. "
                        f"Only a single worker is supported — see README."
                    )
            except (ValueError, OSError):
                pass  # corrupt lock file — overwrite it
        with open(_LOCK_FILE, "w") as _lf:
            _lf.write(str(current_pid))
    except RuntimeError:
        raise
    except Exception:
        logger.debug("lock_file_guard_failed", exc_info=True)

    # F-12 (part 2): psutil-based worker-count check (kept for env-var path).
    try:
        import psutil
        workers_env = os.environ.get("UVICORN_WORKERS") or os.environ.get("WEB_CONCURRENCY")
        workers = int(workers_env) if workers_env else None
        if workers is None:
            try:
                parent = psutil.Process().parent()
                cmdline = parent.cmdline() if parent else []
                for i, tok in enumerate(cmdline):
                    if tok == "--workers" and i + 1 < len(cmdline):
                        workers = int(cmdline[i + 1])
                    elif tok.startswith("--workers="):
                        workers = int(tok.split("=", 1)[1])
            except Exception:
                logger.debug("worker_count_detection_failed", exc_info=True)
        if workers is not None and workers > 1:
            raise RuntimeError(
                "Strict APP_STATE authority requires a single worker process; "
                f"configured workers={workers}"
            )
    except ImportError:
        # psutil not installed — lock-file guard above is the fallback.
        pass


def _recover_gpu_calibration() -> None:
    from src.app_state import APP_STATE, APP_STATE_LOCK
    latest_cal = persistence.load_latest_calibration()
    if latest_cal is None:
        return
    try:
        from src.core.gpu_calibration import GPUCalibrationProfile
        profile = GPUCalibrationProfile(latest_cal["raw_json"])
        with APP_STATE_LOCK:
            APP_STATE["gpu_calibration"] = profile
        logger.info("startup_calibration_recovery", extra={
            "event": "startup_calibration_recovery",
            "providers": latest_cal["provider_list"],
        })
    except Exception:
        logger.warning("startup_calibration_recovery_failed", exc_info=True)


def _warm_server_hardware_profile() -> None:
    from src.services.hardware_profiler import profile_server_hardware
    try:
        profile_server_hardware()
        logger.info("server_hardware_profiled",
                    extra={"event": "server_hardware_profiled"})
    except Exception:
        logger.warning("server_hardware_profile_failed", exc_info=True)


# ── Static UI Mount ───────────────────────────────────────────────────────────

from src.core.paths import get_static_dir
_STATIC_DIR = get_static_dir()
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)

"""
paths.py — Phase 7: IO & State Isolation

Single source of truth for all filesystem paths used by the engine.
Every path in the codebase MUST derive from BASE_DIR or from a
run-specific / model-specific sub-directory produced by this module.

No CWD-relative paths. No hardcoded absolute paths (except BASE_DIR itself,
which is anchored to the user's home directory and overridable via env var).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

# ── BASE_DIR ──────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(
    os.environ.get("DEPLOYCHECK_HOME", str(Path.home() / ".deploycheck"))
).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ── RULE 1: INSTANCE ISOLATION ────────────────────────────────────────────────
# Each process gets a stable namespace; override via APP_INSTANCE_ID for
# deterministic test / container deployments.  When the env-var is absent the
# ID is read from / written to BASE_DIR/.instance_id so it survives restarts.
INSTANCE_ID = os.environ.get("APP_INSTANCE_ID")
if not INSTANCE_ID:
    persisted = BASE_DIR / ".instance_id"
    if persisted.exists():
        INSTANCE_ID = persisted.read_text().strip()
    else:
        INSTANCE_ID = str(uuid.uuid4())
        persisted.write_text(INSTANCE_ID)

INSTANCE_DIR: Path = BASE_DIR / "instances" / INSTANCE_ID
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

# ── RULE 13: QUOTA ────────────────────────────────────────────────────────────
MAX_RUN_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB per run (one model file can be several hundred MB)


# ── Sub-directory factories ───────────────────────────────────────────────────

def get_static_dir() -> Path:
    """Absolute path to the GUI static directory, anchored to the project root."""
    return Path(__file__).resolve().parents[2] / "src" / "gui" / "static"


def get_db_dir() -> Path:
    """Instance-scoped database directory."""
    p = INSTANCE_DIR / "db"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_cache_dir() -> Path:
    """Instance-scoped cache directory."""
    p = INSTANCE_DIR / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_model_dir(model_id: str) -> Path:
    """Per-model directory under INSTANCE_DIR."""
    p = INSTANCE_DIR / "models" / model_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── RULE 2: RUN ISOLATION ─────────────────────────────────────────────────────
def get_run_dir(run_id: str) -> Path:
    """Per-run directory under INSTANCE_DIR."""
    p = INSTANCE_DIR / "runs" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_uploads_dir() -> Path:
    """Instance-scoped upload staging directory."""
    p = Path(os.environ.get("UPLOAD_DIR", str(INSTANCE_DIR / "uploads"))).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── RULE 18: LOG ISOLATION ────────────────────────────────────────────────────
def get_log_path(run_id: str) -> Path:
    """Per-run log file under INSTANCE_DIR/logs/."""
    log_dir = INSTANCE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{run_id}.log"


# ── RULE 3: SCOPED DB PATHS ───────────────────────────────────────────────────
def get_db_path(run_id: str | None = None, model_id: str | None = None) -> Path:
    base = get_db_dir()
    base.mkdir(parents=True, exist_ok=True)

    if run_id:
        return base / f"run_{run_id}.db"
    if model_id:
        return base / f"model_{model_id}.db"

    raise RuntimeError("DB must be scoped")


# ── Well-known file paths ─────────────────────────────────────────────────────

def persistence_db_path() -> Path:
    return get_db_dir() / "production_metrics.db"


def ground_truth_db_path() -> Path:
    return get_db_dir() / "deployment_history.db"


def calibration_history_path() -> Path:
    return INSTANCE_DIR / "calibration_history.json"


def benchmark_calibration_path() -> Path:
    return INSTANCE_DIR / "benchmark_calibration.json"


def calibration_cache_path(instance_id: str) -> Path:
    return get_cache_dir() / f"calibration_cache_{instance_id}.pkl"


def incidents_path() -> Path:
    return INSTANCE_DIR / "incidents.json"


# ── RULE 7: STABLE CACHE KEYS ─────────────────────────────────────────────────
def stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 digest — safe across processes and restarts."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()


# ── RULE 6: SAFE READ WITH RETRY ─────────────────────────────────────────────
def safe_read_json(path: Path, retries: int = 3) -> Any:
    for _ in range(retries):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            time.sleep(0.01)
    raise RuntimeError(f"safe_read_json: all {retries} retries failed for {path}")


# ── RULE 8: SYMLINK + PATH ESCAPE GUARD ──────────────────────────────────────
def assert_safe_path(path: Path, root: Path = INSTANCE_DIR) -> Path:
    """Raise if *path* is a symlink or escapes *root*."""
    if path.is_symlink():
        raise RuntimeError(f"Symlink not allowed: {path}")
    resolved = path.resolve(strict=False)
    resolved.relative_to(root)  # raises ValueError on escape
    return resolved


# ── RULE 13: QUOTA ENFORCEMENT ────────────────────────────────────────────────
def enforce_quota(instance_dir: Path = INSTANCE_DIR) -> None:
    """Raise RuntimeError if *instance_dir* exceeds MAX_RUN_BYTES.

    Must be called before EVERY write operation — not only uploads.
    """
    total = sum(
        f.stat().st_size
        for f in instance_dir.rglob("*")
        if f.is_file()
    )
    if total > MAX_RUN_BYTES:
        raise RuntimeError(
            f"Quota exceeded: {total} bytes used in {instance_dir} "
            f"(limit {MAX_RUN_BYTES})"
        )


# ── RULE 14: CLEANUP (TTL) ────────────────────────────────────────────────────
def cleanup_runs(base: Path = INSTANCE_DIR / "runs", ttl: int = 86400) -> None:
    """Remove run directories whose mtime is older than *ttl* seconds."""
    if not base.exists():
        return
    cutoff = time.time() - ttl
    for run_dir in base.iterdir():
        try:
            if run_dir.is_dir() and run_dir.stat().st_mtime < cutoff:
                shutil.rmtree(run_dir, ignore_errors=True)
        except OSError:
            pass


def cleanup_uploads(
    uploads_dir: "Path | None" = None,
    keep_path: "Path | None" = None,
) -> None:
    """Delete all previous upload staging directories, keeping only *keep_path*.

    Called at the start of every new upload so the uploads directory never
    accumulates stale model files from previous sessions.  Without this,
    every POST /api/model/index creates a new uploads/{uuid}/ subdirectory
    that is never removed, causing the instance directory to grow without
    bound and eventually tripping the quota guard in persistence.py.

    Args:
        uploads_dir: Root of the upload staging area.
                     Defaults to INSTANCE_DIR/uploads (same as get_uploads_dir).
        keep_path:   Absolute path to a file whose parent directory should be
                     preserved (the currently-in-progress upload slot).
                     Pass None to remove everything.
    """
    base = uploads_dir or (INSTANCE_DIR / "uploads")
    if not base.exists():
        return

    keep_dir: "Path | None" = keep_path.parent if keep_path else None

    for entry in base.iterdir():
        try:
            if not entry.is_dir():
                continue
            if keep_dir is not None and entry.resolve() == keep_dir.resolve():
                continue
            shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass

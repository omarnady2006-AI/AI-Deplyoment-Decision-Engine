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
GUI Analysis API

Handles model upload, analysis execution, and schema validation.
"""

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
from fastapi import HTTPException, UploadFile

from src.core.model_analysis import analyze_model, ModelAnalysisResult, validate_onnx_model
from src.core.model_hash import compute_model_hash
from src.core.pipeline import run_pipeline, PipelineResult
from src.core import persistence


logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
MIN_UPLOAD_SIZE = 64  # Minimum reasonable size for protobuf header
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/data/uploads")).resolve()
UPLOAD_CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour
MAX_UPLOAD_STORAGE_BYTES = int(
    os.environ.get("UPLOAD_STORAGE_QUOTA_BYTES", str(20 * 1024 * 1024 * 1024))
)

# ── PG5.1: Split semaphores — small vs large uploads ──────────────────────
_LARGE_FILE_THRESHOLD   = 50 * 1024 * 1024   # 50 MB
_MAX_CONCURRENT_SMALL   = int(os.environ.get("MAX_CONCURRENT_SMALL_UPLOADS", "6"))
_MAX_CONCURRENT_LARGE   = int(os.environ.get("MAX_CONCURRENT_LARGE_UPLOADS", "2"))
SMALL_UPLOAD_SEMAPHORE  = threading.Semaphore(_MAX_CONCURRENT_SMALL)
LARGE_UPLOAD_SEMAPHORE  = threading.Semaphore(_MAX_CONCURRENT_LARGE)
# Legacy alias kept for backward compatibility
UPLOAD_SEMAPHORE        = SMALL_UPLOAD_SEMAPHORE

# ── PG11.1: Slow upload kill ───────────────────────────────────────────────
_MIN_UPLOAD_RATE_BYTES_PER_SEC = 1_024   # 1 KB/s minimum sustained rate
_UPLOAD_RATE_CHECK_INTERVAL    = 5.0     # seconds between rate checks

# ── PG11.2: Mid-stream disk check cadence ─────────────────────────────────
_DISK_CHECK_EVERY_BYTES = 50 * 1024 * 1024  # re-check disk every 50 MB

_last_upload_cleanup = 0.0
_upload_storage_bytes: int | None = None
_upload_storage_lock = threading.Lock()

# ── PG11.3: Multi-upload flood protection ─────────────────────────────────
_MAX_ACTIVE_UPLOADS = int(os.environ.get("MAX_ACTIVE_UPLOADS", str(
    _MAX_CONCURRENT_SMALL + _MAX_CONCURRENT_LARGE
)))
_active_uploads: int = 0
_active_uploads_lock = threading.Lock()


@dataclass
class UploadResult:
    model_path: Path
    model_hash: str
    upload_time_ms: float
    validation_time_ms: float


def _ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _current_upload_storage_bytes() -> int:
    if not UPLOAD_DIR.exists():
        return 0
    total = 0
    for file_path in UPLOAD_DIR.glob("*.onnx"):
        try:
            if file_path.is_file():
                total += file_path.stat().st_size
        except OSError:
            continue
    return total


def _get_upload_storage_bytes() -> int:
    global _upload_storage_bytes
    with _upload_storage_lock:
        if _upload_storage_bytes is None:
            _upload_storage_bytes = _current_upload_storage_bytes()
        return _upload_storage_bytes


def _adjust_upload_storage_bytes(delta: int) -> None:
    global _upload_storage_bytes
    with _upload_storage_lock:
        if _upload_storage_bytes is None:
            _upload_storage_bytes = _current_upload_storage_bytes()
        _upload_storage_bytes = max(0, _upload_storage_bytes + delta)


def cleanup_upload_storage(force: bool = False) -> int:
    """Cleanup orphan upload files inside the managed uploads directory."""
    global _last_upload_cleanup
    now = time.time()

    if not force and now - _last_upload_cleanup < UPLOAD_CLEANUP_INTERVAL_SECONDS:
        return 0

    _last_upload_cleanup = now
    _ensure_upload_dir()
    deleted_count = 0

    try:
        referenced_paths = {
            str(Path(p).resolve())
            for p in persistence.list_uploaded_model_paths()
        }

        for model_file in UPLOAD_DIR.glob("*.onnx"):
            try:
                canonical = str(model_file.resolve())
                if canonical not in referenced_paths:
                    file_size = 0
                    try:
                        file_size = model_file.stat().st_size
                    except OSError:
                        file_size = 0
                    model_file.unlink(missing_ok=True)
                    deleted_count += 1
                    if file_size > 0:
                        _adjust_upload_storage_bytes(-file_size)
            except OSError:
                continue
    except Exception as exc:
        logger.warning("Failed to cleanup upload storage: %s", exc)

    if deleted_count > 0:
        logger.info("upload_storage_cleanup_deleted=%d", deleted_count)

    return deleted_count


def handle_model_upload(file: UploadFile) -> UploadResult:
    """
    Handle model file upload.  Routes to small or large semaphore bucket
    based on Content-Length hint (PG5.1/5.2).  Inner logic is isolated.
    """
    global _active_uploads

    # PG11.3 – global flood gate: reject before touching any semaphore
    with _active_uploads_lock:
        if _active_uploads >= _MAX_ACTIVE_UPLOADS:
            logger.warning(
                "stage=upload status=rejected reason=flood_limit active=%d limit=%d",
                _active_uploads, _MAX_ACTIVE_UPLOADS,
            )
            raise HTTPException(status_code=429, detail="Too many active uploads; retry shortly")
        _active_uploads += 1

    try:
        # PG5.1 – pick the right semaphore bucket
        # Prefer Content-Length hint; default to large bucket when unknown.
        _content_len = 0
        try:
            _cl_raw = getattr(getattr(file, "headers", {}), "get", lambda *a: None)("content-length")
            if _cl_raw:
                _content_len = int(_cl_raw)
        except Exception as e:
            logger.warning("exception_occurred", exc_info=True)

        _is_large = (_content_len == 0) or (_content_len >= _LARGE_FILE_THRESHOLD)
        _sem = LARGE_UPLOAD_SEMAPHORE if _is_large else SMALL_UPLOAD_SEMAPHORE

        if not _sem.acquire(blocking=False):
            _bucket = "large" if _is_large else "small"
            logger.warning("stage=upload status=rejected reason=concurrency_limit bucket=%s", _bucket)
            raise HTTPException(status_code=429, detail="Too many concurrent uploads; retry shortly")

        try:
            return _handle_model_upload_inner(file)
        finally:
            _sem.release()
    finally:
        with _active_uploads_lock:
            _active_uploads = max(0, _active_uploads - 1)


def _handle_model_upload_inner(file: UploadFile) -> UploadResult:
    """Inner upload logic — called with the appropriate semaphore already held."""
    logger.info("stage=upload status=start filename=%s", getattr(file, "filename", "unknown"))
    cleanup_upload_storage()
    _ensure_upload_dir()

    # ── PG11.2 / PG3.4: Disk space guard BEFORE writing ───────────────────
    try:
        _disk = psutil.disk_usage(str(UPLOAD_DIR))
        _min_free = MAX_UPLOAD_SIZE + 64 * 1024 * 1024
        if _disk.free < _min_free:
            logger.warning("stage=upload status=rejected reason=insufficient_disk free=%d", _disk.free)
            raise HTTPException(status_code=507, detail="Insufficient server disk space")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("disk_check_failed_best_effort", exc_info=True)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file.filename.endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Only ONNX files are supported")

    upload_token = str(uuid.uuid4())

    # ── PG7.1: Per-request isolated temp directory ─────────────────────────
    request_tmp_dir = (UPLOAD_DIR / f"tmp_{upload_token}").resolve()
    upload_dir_resolved = UPLOAD_DIR.resolve()
    try:
        request_tmp_dir.relative_to(upload_dir_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Invalid upload path") from exc

    temp_path  = request_tmp_dir / "model.onnx.tmp"
    final_path = (UPLOAD_DIR / f"{upload_token}.onnx").resolve()

    current_storage = _get_upload_storage_bytes()
    start_total = time.perf_counter()
    validation_time_ms = 0.0

    try:
        request_tmp_dir.mkdir(parents=True, exist_ok=True)  # PG7.1

        total_size = 0
        file.file.seek(0)

        # Rate-check state (PG11.1)
        _rate_check_start = time.perf_counter()
        _rate_check_bytes = 0
        # Disk re-check state (PG11.2)
        _disk_next_check = _DISK_CHECK_EVERY_BYTES

        # ── PG7.2: Atomic write: stream → temp, then rename ───────────────
        with open(temp_path, "wb") as stream:
            while True:
                chunk = file.file.read(8192)
                if not chunk:
                    break

                total_size        += len(chunk)
                _rate_check_bytes += len(chunk)

                # Size ceiling
                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB",
                    )

                # Quota
                if current_storage + total_size > MAX_UPLOAD_STORAGE_BYTES:
                    raise HTTPException(status_code=413, detail="Upload storage quota exceeded")

                stream.write(chunk)

                # PG11.1 – slow upload kill (checked every interval)
                _now = time.perf_counter()
                _elapsed = _now - _rate_check_start
                if _elapsed >= _UPLOAD_RATE_CHECK_INTERVAL:
                    _rate = _rate_check_bytes / _elapsed
                    if _rate < _MIN_UPLOAD_RATE_BYTES_PER_SEC:
                        logger.warning(
                            "stage=upload status=aborted reason=slow_upload rate=%.0f", _rate
                        )
                        raise HTTPException(
                            status_code=408,
                            detail=f"Upload too slow ({_rate:.0f} B/s); connection aborted",
                        )
                    _rate_check_bytes = 0
                    _rate_check_start = _now

                # PG11.2 – mid-stream disk check
                if total_size >= _disk_next_check:
                    try:
                        _d = psutil.disk_usage(str(UPLOAD_DIR))
                        if _d.free < 64 * 1024 * 1024:
                            raise HTTPException(
                                status_code=507, detail="Server disk space critical during upload"
                            )
                    except HTTPException:
                        raise
                    except Exception as e:
                        logger.warning("exception_occurred", exc_info=True)
                    _disk_next_check += _DISK_CHECK_EVERY_BYTES

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if total_size < MIN_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="Uploaded file is too small")

        # PG7.2 – atomic rename (temp → final)
        os.replace(temp_path, final_path)

        logger.info("stage=upload status=received size_bytes=%d", total_size)

        validation_start = time.perf_counter()
        is_valid, error_msg, _ = validate_onnx_model(str(final_path))
        validation_time_ms = (time.perf_counter() - validation_start) * 1000

        if not is_valid:
            logger.info("stage=validate status=failed reason=%s", error_msg)
            raise HTTPException(status_code=400, detail=error_msg or "Invalid ONNX model")

        model_hash = compute_model_hash(str(final_path))
        _adjust_upload_storage_bytes(total_size)

        logger.info(
            "stage=upload status=complete filename=%s size_bytes=%d hash=%s",
            file.filename,
            total_size,
            model_hash,
        )

        upload_time_ms = (time.perf_counter() - start_total) * 1000
        return UploadResult(
            model_path=final_path,
            model_hash=model_hash,
            upload_time_ms=upload_time_ms,
            validation_time_ms=validation_time_ms,
        )

    except HTTPException:
        # PG7.3 – guaranteed cleanup on failure
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        try:
            request_tmp_dir.rmdir()
        except Exception as e:
            logger.warning("exception_occurred", exc_info=True)
        raise

    except Exception as e:
        # PG7.3 – guaranteed cleanup on any error
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        try:
            request_tmp_dir.rmdir()
        except Exception as e:
            logger.warning("exception_occurred", exc_info=True)
        logger.exception("stage=upload status=error reason=%s", str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def analyze_uploaded_model(model_path: str) -> ModelAnalysisResult:
    logger.info("stage=analyze status=start path=%s", model_path)
    result = analyze_model(model_path)

    if not result.success:
        logger.error("stage=analyze status=failed reason=%s path=%s", result.error, model_path)
    else:
        logger.info("stage=analyze status=complete ops=%d params=%d path=%s",
                    result.operator_count, result.parameter_count, model_path)

    return result


def run_model_pipeline(
    model_path: str,
    constraints: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    model_hash: str | None = None,
    validated_model_obj: Any | None = None,
) -> PipelineResult:
    """Run the full pipeline on a model."""
    logger.info("stage=pipeline status=start path=%s", model_path)
    result = run_pipeline(
        model_path,
        constraints=constraints,
        profile=profile,
        model_hash=model_hash,
        model_obj=validated_model_obj,
    )
    logger.info("stage=pipeline status=complete decision=%s path=%s",
                getattr(result, "decision", "unknown"), model_path)
    return result


def validate_analysis_schema(analysis: ModelAnalysisResult) -> list[str]:
    """Validate that analysis result has all required fields."""
    required_fields = [
        "model_path", "file_hash", "operator_count", "operators",
        "operator_counts", "parameter_count", "has_dynamic_shapes",
        "input_shapes", "unsupported_ops", "sequential_depth",
        "input_count", "output_count", "success"
    ]
    
    missing = []
    for field in required_fields:
        if not hasattr(analysis, field):
            missing.append(field)
    
    return missing

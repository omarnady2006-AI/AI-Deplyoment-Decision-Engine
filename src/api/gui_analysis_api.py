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

_last_upload_cleanup = 0.0
_upload_storage_bytes: int | None = None
_upload_storage_lock = threading.Lock()


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
    Handle model file upload and return path and hash.
    
    Security measures:
    - Enforces maximum upload size (500MB)
    - Enforces minimum upload size
    - Streams file to disk (no memory exhaustion)
    - Stores uploads in dedicated upload directory
    - Performs hourly orphan cleanup and quota checks
    - Validates model structure (non-empty graph)
    - Guarantees partial-file cleanup on failure
    """
    cleanup_upload_storage()
    _ensure_upload_dir()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if ".." in file.filename or "/" in file.filename or "\\" in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not file.filename.endswith('.onnx'):
        raise HTTPException(status_code=400, detail="Only ONNX files are supported")
    
    upload_token = str(uuid.uuid4())
    temp_path = (UPLOAD_DIR / f"{upload_token}.onnx.tmp").resolve()
    final_path = (UPLOAD_DIR / f"{upload_token}.onnx").resolve()
    upload_dir_resolved = UPLOAD_DIR.resolve()

    try:
        temp_path.relative_to(upload_dir_resolved)
        final_path.relative_to(upload_dir_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Invalid upload path") from exc

    current_storage = _get_upload_storage_bytes()

    start_total = time.perf_counter()
    validation_time_ms = 0.0

    try:
        total_size = 0
        file.file.seek(0)

        with open(temp_path, "wb") as stream:
            while True:
                chunk = file.file.read(8192)
                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB",
                    )

                if current_storage + total_size > MAX_UPLOAD_STORAGE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Upload storage quota exceeded",
                    )

                stream.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if total_size < MIN_UPLOAD_SIZE:
            raise HTTPException(status_code=400, detail="Uploaded file is too small")

        os.replace(temp_path, final_path)

        validation_start = time.perf_counter()
        is_valid, error_msg, _ = validate_onnx_model(str(final_path))
        validation_time_ms = (time.perf_counter() - validation_start) * 1000
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg or "Invalid ONNX model")

        model_hash = compute_model_hash(str(final_path))

        _adjust_upload_storage_bytes(total_size)
        
        logger.info("model_uploaded", extra={
            "uploaded_filename": file.filename,
            "size_bytes": total_size,
            "model_hash": model_hash,
            "upload_path": str(final_path),
        })

        upload_time_ms = (time.perf_counter() - start_total) * 1000
        return UploadResult(
            model_path=final_path,
            model_hash=model_hash,
            upload_time_ms=upload_time_ms,
            validation_time_ms=validation_time_ms,
        )
        
    except HTTPException:
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise
        
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        logger.exception(f"upload_failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def analyze_uploaded_model(model_path: str) -> ModelAnalysisResult:
    result = analyze_model(model_path)
    
    if not result.success:
        logger.error(f"analysis_failed: {result.error}")
    
    return result


def run_model_pipeline(
    model_path: str,
    constraints: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    model_hash: str | None = None,
    validated_model_obj: Any | None = None,
) -> PipelineResult:
    """Run the full pipeline on a model."""
    return run_pipeline(
        model_path,
        constraints=constraints,
        profile=profile,
        model_hash=model_hash,
        model_obj=validated_model_obj,
    )


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

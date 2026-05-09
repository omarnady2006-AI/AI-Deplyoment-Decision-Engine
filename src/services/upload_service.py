"""
services/upload_service.py

Orchestrates file upload: streaming save, hash computation, path setup.
Calls core paths/security utilities. Contains ZERO business logic.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from src.core.paths import get_uploads_dir, assert_safe_path, MAX_RUN_BYTES, cleanup_uploads
from src.core.logging_config import get_logger

logger = get_logger(__name__)

_UPLOAD_DIR: Path = get_uploads_dir()
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_uploaded_model(file: UploadFile) -> dict[str, Any]:
    """
    Stream-save an uploaded model file to a per-request directory.

    Returns:
        {model_path, model_hash, safe_name}
    Raises:
        HTTPException 413 if file exceeds quota.
    """
    upload_token = str(uuid.uuid4())
    request_dir  = _UPLOAD_DIR / upload_token
    request_dir.mkdir(parents=True, exist_ok=True)

    # Remove all previous upload staging directories before saving the new file.
    # Each upload used to create a new uploads/{uuid}/ directory that was never
    # cleaned up, causing the instance directory to grow unboundedly and trip
    # the quota guard in persistence.py.  Only the current slot is preserved.
    safe_name = Path(file.filename or "model.onnx").name
    dest      = request_dir / safe_name
    cleanup_uploads(uploads_dir=_UPLOAD_DIR, keep_path=dest)
    assert_safe_path(dest, root=_UPLOAD_DIR)

    tmp_dest = dest.with_suffix(f".{uuid.uuid4().hex}.tmp")
    hasher   = hashlib.sha256()
    bytes_written = 0

    try:
        with open(str(tmp_dest), "wb", opener=lambda p, f: os.open(p, f, 0o600)) as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_RUN_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds quota limit")
                fh.write(chunk)
                hasher.update(chunk)
        tmp_dest.replace(dest)
    finally:
        if tmp_dest.exists():
            tmp_dest.unlink()

    model_path = str(dest)
    try:
        file_hash        = hasher.hexdigest()
        hash_input       = f"{dest.resolve()}:{file_hash}".encode("utf-8")
        model_hash       = hashlib.sha1(hash_input).hexdigest()
    except Exception:
        import uuid as _uuid
        model_hash = _uuid.uuid4().hex

    return {
        "model_path": model_path,
        "model_hash": model_hash,
        "safe_name":  safe_name,
    }

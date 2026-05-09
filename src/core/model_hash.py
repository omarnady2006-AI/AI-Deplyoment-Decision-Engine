"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""
Canonical model hash implementation.

All model hashing in the repository MUST use this module.
Model identity is content-addressable via SHA256.
"""
from pathlib import Path
import hashlib


def compute_model_hash(path: str) -> str:
    """
    Compute SHA256 hash of model file content.
    
    Args:
        path: Path to the model file
        
    Returns:
        64-character hexadecimal SHA256 hash string
        
    Raises:
        RuntimeError: If file does not exist or cannot be read
        
    Notes:
        - Hash depends ONLY on file content (content-addressable)
        - No path, filename, timestamp, or random values are included
        - Same model uploaded twice produces identical hash
    """
    p = Path(path)

    if not p.exists():
        raise RuntimeError(f"Model file does not exist: {path}")

    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to compute model hash for '{path}': {exc}"
        ) from exc

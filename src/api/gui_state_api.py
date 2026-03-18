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
GUI State API

Handles application state, caching, and diagnostics state.

SECURITY: Uses session-scoped state to prevent cross-request contamination.
Each client session has isolated state.
"""

import logging
import secrets
import threading
import time
from copy import deepcopy
from typing import Any


logger = logging.getLogger(__name__)

SESSION_EXPIRY_SECONDS = 3600
MAX_SESSIONS = 1000

_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSION_LAST_ACCESS: dict[str, float] = {}
_SESSIONS_LOCK = threading.Lock()


def _create_empty_state() -> dict[str, Any]:
    return {
        "model_hash": None,
        "model_path": None,
        "analysis": None,
        "decision": None,
        "confidence": None,
        "diagnostics": [],
        "calibration": None,
        "validation_results": {},
        "deployment_profile": None,
        "gpu_calibration": None,
    }


def create_session() -> str:
    """Create a new session and return its ID."""
    session_id = secrets.token_hex(32)
    with _SESSIONS_LOCK:
        _SESSIONS[session_id] = _create_empty_state()
        _SESSION_LAST_ACCESS[session_id] = time.time()
        _cleanup_expired_sessions()
    return session_id


def _cleanup_expired_sessions() -> None:
    """Remove expired sessions. Must be called with lock held."""
    now = time.time()
    expired = [
        sid for sid, last_access in _SESSION_LAST_ACCESS.items()
        if now - last_access > SESSION_EXPIRY_SECONDS
    ]
    for sid in expired:
        _SESSIONS.pop(sid, None)
        _SESSION_LAST_ACCESS.pop(sid, None)
    
    if len(_SESSIONS) > MAX_SESSIONS:
        sorted_sessions = sorted(
            _SESSIONS.keys(),
            key=lambda sid: _SESSION_LAST_ACCESS.get(sid, 0)
        )
        for sid in sorted_sessions[:len(_SESSIONS) - MAX_SESSIONS]:
            _SESSIONS.pop(sid, None)
            _SESSION_LAST_ACCESS.pop(sid, None)


def validate_session(session_id: str) -> bool:
    """Check if a session ID is valid."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            return False
        _SESSION_LAST_ACCESS[session_id] = time.time()
        return True


def get_or_create_session(session_id: str | None) -> str:
    """Get existing session or create new one."""
    if session_id and validate_session(session_id):
        return session_id
    return create_session()


def get_state(session_id: str) -> dict[str, Any]:
    """Get state for a specific session."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = _create_empty_state()
        _SESSION_LAST_ACCESS[session_id] = time.time()
        return deepcopy(_SESSIONS[session_id])


def update_state(session_id: str, **kwargs: Any) -> None:
    """Update state for a specific session."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = _create_empty_state()
        for key, value in kwargs.items():
            if key in _SESSIONS[session_id]:
                _SESSIONS[session_id][key] = value
            else:
                logger.warning(f"Unknown state key: {key}")
        _SESSION_LAST_ACCESS[session_id] = time.time()


def clear_state(session_id: str) -> None:
    """Clear state for a specific session."""
    with _SESSIONS_LOCK:
        if session_id in _SESSIONS:
            _SESSIONS[session_id] = _create_empty_state()
        _SESSION_LAST_ACCESS[session_id] = time.time()


def delete_session(session_id: str) -> None:
    """Delete a session entirely."""
    with _SESSIONS_LOCK:
        _SESSIONS.pop(session_id, None)
        _SESSION_LAST_ACCESS.pop(session_id, None)


def set_analysis_result(
    session_id: str,
    model_path: str,
    model_hash: str,
    analysis: Any,
) -> None:
    """Set analysis result in session state."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = _create_empty_state()
        _SESSIONS[session_id]["model_path"] = model_path
        _SESSIONS[session_id]["model_hash"] = model_hash
        _SESSIONS[session_id]["analysis"] = analysis
        _SESSION_LAST_ACCESS[session_id] = time.time()


def set_decision_result(
    session_id: str,
    decision: str,
    confidence: float,
    diagnostics: list[Any],
) -> None:
    """Set decision result in session state."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = _create_empty_state()
        _SESSIONS[session_id]["decision"] = decision
        _SESSIONS[session_id]["confidence"] = confidence
        _SESSIONS[session_id]["diagnostics"] = diagnostics
        _SESSION_LAST_ACCESS[session_id] = time.time()


def set_validation_result(session_id: str, runtime: str, result: Any) -> None:
    """Set validation result for a specific runtime."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = _create_empty_state()
        _SESSIONS[session_id]["validation_results"][runtime] = result
        _SESSION_LAST_ACCESS[session_id] = time.time()


def get_validation_result(session_id: str, runtime: str) -> Any | None:
    """Get validation result for a specific runtime."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            return None
        return _SESSIONS[session_id].get("validation_results", {}).get(runtime)


def set_calibration(session_id: str, calibration: Any) -> None:
    """Set calibration data in session state."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = _create_empty_state()
        _SESSIONS[session_id]["calibration"] = calibration
        _SESSION_LAST_ACCESS[session_id] = time.time()


def get_calibration(session_id: str) -> Any | None:
    """Get calibration data from session state."""
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            return None
        return _SESSIONS[session_id].get("calibration")


LEGACY_APP_STATE: dict[str, Any] = _create_empty_state()
APP_STATE_LOCK = threading.Lock()

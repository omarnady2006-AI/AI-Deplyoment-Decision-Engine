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
src/core/logging_config.py

Structured JSON logging configuration for the deployment decision engine.

Usage:
    from src.core.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("model_uploaded", extra={"model_hash": "abc123", "path": "/tmp/model.onnx"})

Log level is configurable via the LOG_LEVEL environment variable.
Defaults to INFO if the variable is absent or invalid.

No external dependencies beyond the Python standard library.
No global state mutation beyond standard logging handler registration.
"""


import json
import logging
import os
import time
from typing import Any


_LOG_LEVEL_ENV = "LOG_LEVEL"
_DEFAULT_LEVEL = logging.INFO
_HANDLER_ATTR = "_deploycheck_json_handler_installed"


def _resolve_level() -> int:
    raw = os.environ.get(_LOG_LEVEL_ENV, "").strip().upper()
    level = getattr(logging, raw, None)
    if isinstance(level, int):
        return level
    return _DEFAULT_LEVEL


class _JsonFormatter(logging.Formatter):
    """
    Formats each log record as a single-line JSON object.

    Standard fields always present:
        timestamp   ISO-8601 UTC string
        level       DEBUG / INFO / WARNING / ERROR / CRITICAL
        logger      Logger name
        message     The formatted log message

    Any key/value pairs passed via the `extra` kwarg to the logger call
    are merged into the top-level JSON object.
    """

    _RESERVED = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message",
        "module", "msecs", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": _iso_utc(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Merge caller-supplied extra fields, skipping reserved internals
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _iso_utc(epoch: float) -> str:
    """Convert a Unix timestamp to a compact ISO-8601 UTC string."""
    t = time.gmtime(epoch)
    ms = int((epoch - int(epoch)) * 1000)
    return (
        f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T"
        f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}Z"
    )


def _ensure_root_handler() -> None:
    """
    Install the JSON handler on the root logger exactly once per process.
    Guards against duplicate handler registration on repeated imports.
    """
    root = logging.getLogger()
    if getattr(root, _HANDLER_ATTR, False):
        return

    level = _resolve_level()
    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)

    setattr(root, _HANDLER_ATTR, True)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger backed by the JSON formatter.

    Calling this function multiple times with the same name is safe;
    the root handler is installed only once.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A standard logging.Logger instance.
    """
    _ensure_root_handler()
    return logging.getLogger(name)

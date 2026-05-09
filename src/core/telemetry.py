"""
Telemetry Emission Layer — Schema v3
Writes append-safe JSONL events for build_dataset_from_telemetry.py.
Dual-writes to SQLite via telemetry_db. Rotates JSONL at 50 MB.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from src.core.telemetry_db import init_db, insert_event

_EVENTS_PATH = os.environ.get("TELEMETRY_EVENTS_PATH", "events.jsonl")
_SCHEMA_VERSION = 3
SCHEMA_VERSION = _SCHEMA_VERSION          # public alias required by analysis_pipeline_service
_MIN_SUCCESSFUL_ACTIONS = 3
_MAX_JSONL_BYTES = 50 * 1024 * 1024  # 50 MB

_write_lock = threading.Lock()

init_db()


# ---------------------------------------------------------------------------
# Core computation  (DO NOT MODIFY)
# ---------------------------------------------------------------------------

def compute_optimal_action(
    evaluated_actions: list[dict[str, Any]],
    constraints: dict[str, Any] | None = None,
) -> str | None:
    if not evaluated_actions:
        return None

    valid = [a for a in evaluated_actions if a.get("execution_success")]

    if not valid:
        return None

    # Collect values
    latencies = [a["latency_ms"] for a in valid if a.get("latency_ms") is not None]
    memories  = [a["memory_mb"] for a in valid if a.get("memory_mb") is not None]

    if not latencies:
        return None

    min_lat, max_lat = min(latencies), max(latencies)
    min_mem = min(memories) if memories else 0.0
    max_mem = max(memories) if memories else 1.0

    def normalize(x, lo, hi):
        if hi == lo:
            return 0.0
        return (x - lo) / (hi - lo)

    scored = []

    # Determine weights dynamically based on memory pressure
    mem_values = [a.get("memory_mb", 0.0) for a in valid if a.get("memory_mb") is not None]

    min_mem = min(mem_values) if mem_values else 0.0
    max_mem = max(mem_values) if mem_values else 1.0

    # Spread-based pressure (0 = uniform, 1 = large disparity)
    if max_mem > 0:
        mem_pressure = (max_mem - min_mem) / max_mem
    else:
        mem_pressure = 0.0

    # Adaptive weights: high memory pressure shifts importance away from latency
    w_mem = 0.2 + 0.5 * mem_pressure   # ranges ~0.2 → 0.7
    w_lat = 1.0 - w_mem

    for a in valid:
        if a.get("latency_ms") is None:
            continue

        lat = normalize(a["latency_ms"], min_lat, max_lat)
        mem = normalize(a.get("memory_mb", min_mem), min_mem, max_mem)

        score = w_lat * lat + w_mem * mem

        scored.append((score, a["action"]))

    if not scored:
        return None

    return min(scored, key=lambda x: x[0])[1]


# ---------------------------------------------------------------------------
# Log rotation
# ---------------------------------------------------------------------------

def _rotate_if_needed(path: str) -> None:
    try:
        if os.path.isfile(path) and os.path.getsize(path) >= _MAX_JSONL_BYTES:
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            rotated = f"{path}.{suffix}"
            os.rename(path, rotated)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Event builder
# ---------------------------------------------------------------------------

def _build_event(
    features: dict[str, Any],
    chosen_action: str,
    evaluated_actions: list[dict[str, Any]],
    optimal_action: str,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "chosen_action": chosen_action,
        "evaluated_actions": [
            {
                "action": a["action"],
                "latency_ms": a.get("latency_ms"),
                "memory_mb": a.get("memory_mb"),
                "execution_success": a.get("execution_success"),
            }
            for a in evaluated_actions
        ],
        "optimal_action": optimal_action,
        "label_finalized": True,
    }


# ---------------------------------------------------------------------------
# Append-safe dual writer
# ---------------------------------------------------------------------------

def _append_event(event: dict[str, Any], path: str) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with _write_lock:
        _rotate_if_needed(path)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    insert_event(event)


# ---------------------------------------------------------------------------
# Public API  (DO NOT MODIFY SIGNATURE)
# ---------------------------------------------------------------------------

def _normalize_runtime(rt: str) -> str:
    """
    Task 4 — canonical runtime name normalizer (mirrors analysis_pipeline_service).
    Bans raw .upper() comparisons throughout telemetry.
    """
    s = str(rt).upper()
    if "TENSORRT" in s:
        return "TENSORRT"
    if "CUDA" in s or "GPU" in s:
        return "ONNX_CUDA"
    if "OPENVINO" in s:
        return "OPENVINO_CPU"
    if "TFLITE" in s:
        return "TFLITE_CPU"
    if "CPU" in s:
        return "ONNX_CPU"
    return s


def aggregate_target_outcomes(raw_evals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalise raw evaluation dicts from analysis_pipeline_service into the
    flat action-keyed format consumed by compute_optimal_action and log_analysis_event.

    Input keys (response_evaluations from benchmark):
        runtime, latency_ms, memory_mb, execution_success

    Output keys (telemetry schema v3):
        action, latency_ms, memory_mb, execution_success

    Task 4: uses _normalize_runtime — no raw .upper() comparisons.
    """
    # Canonical map from normalised runtime key → deployment action.
    # Keys match exactly what normalize_runtime_name in analysis_pipeline_service produces.
    _RUNTIME_TO_ACTION: dict[str, str] = {
        "ONNX_CPU":    "cloud_cpu",
        "ONNX_CUDA":   "cloud_gpu",
        "TENSORRT":    "cloud_gpu",
        "OPENVINO_CPU": "edge_fp16",
        "TFLITE_CPU":  "edge_int8",
    }
    result: list[dict[str, Any]] = []
    for e in raw_evals:
        runtime = _normalize_runtime(str(e.get("runtime", "")))
        action  = _RUNTIME_TO_ACTION.get(runtime, runtime.lower())
        result.append({
            "action":            action,
            "latency_ms":        e.get("latency_ms"),
            "memory_mb":         e.get("memory_mb"),
            "execution_success": bool(e.get("execution_success", False)),
        })
    return result


def log_analysis_event(
    features_or_event: "dict[str, Any]",
    chosen_action: "str | None" = None,
    evaluated_actions: "list[dict[str, Any]] | None" = None,
    constraints: "dict[str, Any] | None" = None,
    events_path: str = _EVENTS_PATH,
) -> None:
    """
    Accept two call patterns:

    Pattern A — pre-built event dict (used by analysis_pipeline_service):
        log_analysis_event({...full event dict...})

    Pattern B — legacy positional args (used internally):
        log_analysis_event(features, chosen_action, evaluated_actions, constraints)
    """
    # Pattern A: single pre-built event dict
    if chosen_action is None and evaluated_actions is None:
        event = features_or_event
        _append_event(event, events_path)
        return

    # Pattern B: legacy positional args
    features         = features_or_event
    _evaluated       = evaluated_actions or []
    successful_count = sum(
        1 for a in _evaluated if a.get("execution_success") is True
    )
    if successful_count < _MIN_SUCCESSFUL_ACTIONS:
        return

    optimal_action = compute_optimal_action(_evaluated, constraints)
    if optimal_action is None:
        return

    event = _build_event(features, chosen_action, _evaluated, optimal_action)
    _append_event(event, events_path)

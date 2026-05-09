"""
core/calibration_history.py

Calibration record persistence and statistics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CalibrationRecord:
    model_hash: str
    runtime: Any
    predicted_decision: Any
    actual_outcome: str
    timestamp: float
    bucket: str


def load_history(path: str) -> list:
    """Load calibration history from a JSON file. Returns [] on missing/corrupt file."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def record_calibration(record: CalibrationRecord, path: str) -> None:
    """Append a CalibrationRecord to the JSON history file."""
    history = load_history(path)
    entry = {
        "model_hash": record.model_hash,
        "runtime": record.runtime.value if hasattr(record.runtime, "value") else str(record.runtime),
        "predicted_decision": (
            record.predicted_decision.value
            if hasattr(record.predicted_decision, "value")
            else str(record.predicted_decision)
        ),
        "actual_outcome": record.actual_outcome,
        "timestamp": record.timestamp,
        "bucket": record.bucket,
    }
    history.append(entry)
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")


def compute_calibration_stats(history: list) -> dict:
    """
    Compute per-runtime accuracy stats from calibration history.

    Returns a dict keyed by runtime string with 'total', 'correct', 'accuracy'.
    """
    if not history:
        return {}

    stats: dict[str, dict] = {}
    for entry in history:
        rt = entry.get("runtime", "unknown")
        pred = str(entry.get("predicted_decision", "")).upper()
        outcome = str(entry.get("actual_outcome", "")).upper()

        if rt not in stats:
            stats[rt] = {"total": 0, "correct": 0}

        stats[rt]["total"] += 1

        # Correct if prediction aligned with outcome
        pred_allow = "ALLOW" in pred or "SUPPORTED" in pred
        pred_block = "BLOCK" in pred or "UNSUPPORTED" in pred
        outcome_ok = outcome in ("SUCCESS", "PASS")
        outcome_fail = outcome in ("FAIL", "SLOW")

        if (pred_allow and outcome_ok) or (pred_block and outcome_fail):
            stats[rt]["correct"] += 1

    for s in stats.values():
        s["accuracy"] = s["correct"] / s["total"] if s["total"] else 0.5

    return stats


def compute_runtime_priors(history: list) -> dict:
    """
    Compute per-runtime success priors from calibration history.

    Returns a dict mapping runtime string -> prior probability [0, 1].
    """
    if not history:
        return {}

    counts: dict[str, int] = {}
    for entry in history:
        rt = entry.get("runtime", "unknown")
        if str(entry.get("actual_outcome", "")).upper() == "SUCCESS":
            counts[rt] = counts.get(rt, 0) + 1

    total = sum(counts.values()) or 1
    return {rt: c / total for rt, c in counts.items()}


def compute_model_bucket(summary: Any, facts: dict) -> str:
    """Return the coarse parameter-scale bucket for a model."""
    scale = facts.get("parameter_scale_class") or getattr(
        summary, "parameter_scale_class", "small"
    )
    return str(scale)

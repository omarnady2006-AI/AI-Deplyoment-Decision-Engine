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
Calibration History Module

Tracks historical calibration data for runtime decision improvement.
"""


import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast


_HISTORY_FILE = Path(".deploycheck_calibration_history.json")
logger = logging.getLogger(__name__)


@dataclass
class CalibrationRecord:
    model_hash: str
    runtime: str
    predicted_decision: str
    actual_outcome: Literal["SUCCESS", "FAIL", "SLOW"]
    timestamp: float
    bucket: str = "default"
    latency_ms: float | None = None
    memory_mb: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_hash": self.model_hash,
            "runtime": self.runtime,
            "predicted_decision": self.predicted_decision,
            "actual_outcome": self.actual_outcome,
            "timestamp": self.timestamp,
            "bucket": self.bucket,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
        }


def load_history(path: str | Path = _HISTORY_FILE) -> list[CalibrationRecord]:
    """
    Load calibration history from file.
    """
    p = Path(path)
    try:
        with open(p, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
        
        # Normalize data format
        if isinstance(data, list):
            data = {"records": data}
        elif not isinstance(data, dict):
            logger.warning("Calibration history has invalid format, resetting")
            return []
        
        if "records" not in data:
            data["records"] = []
        
        records = []
        for item in data.get("records", []):
            if not isinstance(item, dict):
                continue
            outcome = cast(Literal["SUCCESS", "FAIL", "SLOW"], item.get("actual_outcome", "FAIL"))
            records.append(CalibrationRecord(
                model_hash=item.get("model_hash", ""),
                runtime=item.get("runtime", ""),
                predicted_decision=item.get("predicted_decision", ""),
                actual_outcome=outcome,
                timestamp=item.get("timestamp", 0.0),
                bucket=item.get("bucket", "default"),
                latency_ms=item.get("latency_ms"),
                memory_mb=item.get("memory_mb"),
            ))
        return records
    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.error(f"Failed to load calibration history from {p}: {e}")
        return []


def record_calibration(record: CalibrationRecord, path: str | Path = _HISTORY_FILE) -> None:
    """
    Record a calibration result to history.
    """
    p = Path(path)
    data: dict[str, Any] = {"records": [], "stats": {}}
    
    try:
        with open(p, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        pass  # file does not exist yet — starting with empty history (expected on first run)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read existing calibration history: %s", e)
    
    if "records" not in data:
        data["records"] = []
    
    data["records"].append(record.to_dict())
    data["last_updated"] = time.time()
    
    try:
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.error(f"Failed to write calibration history: {e}")
        raise


def compute_model_bucket(summary: Any, facts: dict[str, Any]) -> str:
    """
    Compute model bucket for calibration grouping.
    """
    param_count = facts.get("parameter_count", 0)
    
    if param_count >= 25_000_000:
        return "large"
    elif param_count >= 8_000_000:
        return "medium"
    else:
        return "small"


def compute_calibration_stats(records: list[CalibrationRecord]) -> dict[str, Any]:
    """
    Compute statistics from calibration records.
    """
    if not records:
        return {
            "total_records": 0,
            "success_rate": 0.0,
            "by_runtime": {},
        }
    
    total = len(records)
    successes = sum(1 for r in records if r.actual_outcome == "SUCCESS")
    
    by_runtime: dict[str, dict[str, int]] = {}
    for r in records:
        if r.runtime not in by_runtime:
            by_runtime[r.runtime] = {"total": 0, "success": 0}
        by_runtime[r.runtime]["total"] += 1
        if r.actual_outcome == "SUCCESS":
            by_runtime[r.runtime]["success"] += 1
    
    return {
        "total_records": total,
        "success_rate": successes / total if total > 0 else 0.0,
        "by_runtime": by_runtime,
    }


def compute_runtime_priors(records: list[CalibrationRecord]) -> dict[str, float]:
    """
    Compute runtime prior probabilities from calibration history.
    """
    if not records:
        return {}
    
    runtime_counts: dict[str, int] = {}
    runtime_successes: dict[str, int] = {}
    
    for r in records:
        runtime_counts[r.runtime] = runtime_counts.get(r.runtime, 0) + 1
        if r.actual_outcome == "SUCCESS":
            runtime_successes[r.runtime] = runtime_successes.get(r.runtime, 0) + 1
    
    priors: dict[str, float] = {}
    for runtime, count in runtime_counts.items():
        successes = runtime_successes.get(runtime, 0)
        priors[runtime] = successes / count if count > 0 else 0.0
    
    return priors

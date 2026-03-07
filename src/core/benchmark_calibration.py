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
Benchmark Calibration Module

Provides latency calibration and prediction based on historical benchmark data.
"""


import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_CALIBRATION_FILE = Path(".deploycheck_benchmark_calibration.json")
logger = logging.getLogger(__name__)


@dataclass
class CalibrationStats:
    total_models: int = 0
    total_benchmarks: int = 0
    avg_latency_ms: float = 0.0
    last_updated: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_models": self.total_models,
            "total_benchmarks": self.total_benchmarks,
            "avg_latency_ms": self.avg_latency_ms,
            "last_updated": self.last_updated,
        }


def get_calibrated_latency_prediction(
    cpu_cores: int,
    ram_gb: float,
    gpu_available: bool,
    model_hash: str,
    heuristic_latency: float,
) -> dict[str, Any]:
    """
    Get calibrated latency prediction, falling back to heuristic if no calibration exists.
    
    Args:
        cpu_cores: Number of CPU cores
        ram_gb: RAM in GB
        gpu_available: Whether GPU is available
        model_hash: Hash of the model
        heuristic_latency: Heuristic latency estimate
        
    Returns:
        Dict with latency_ms, source, and optionally p95_latency_ms
    """
    if _CALIBRATION_FILE.exists():
        try:
            with open(_CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            
            model_calibrations = data.get("models", {}).get(model_hash, {})
            hardware_key = f"cpu_{cpu_cores}_ram_{ram_gb}_gpu_{gpu_available}"
            
            if hardware_key in model_calibrations:
                cal = model_calibrations[hardware_key]
                return {
                    "latency_ms": cal.get("avg_latency_ms", heuristic_latency),
                    "source": "calibration",
                    "p95_latency_ms": cal.get("p95_latency_ms"),
                }
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Failed to read calibration file: {e}")
    
    return {
        "latency_ms": heuristic_latency,
        "source": "heuristic",
    }


def record_benchmark(
    model_hash: str,
    cpu_cores: int,
    ram_gb: float,
    gpu_available: bool,
    latency_ms: float,
    memory_mb: float | None = None,
) -> None:
    """
    Record a benchmark result for future calibration.
    """
    data: dict[str, Any] = {"models": {}, "stats": {}}
    
    if _CALIBRATION_FILE.exists():
        try:
            with open(_CALIBRATION_FILE, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read calibration file for recording: {e}")
    
    if "models" not in data:
        data["models"] = {}
    
    if model_hash not in data["models"]:
        data["models"][model_hash] = {}
    
    hardware_key = f"cpu_{cpu_cores}_ram_{ram_gb}_gpu_{gpu_available}"
    
    existing = data["models"][model_hash].get(hardware_key, {})
    latencies = existing.get("latencies", [])
    latencies.append(latency_ms)
    
    data["models"][model_hash][hardware_key] = {
        "latencies": latencies[-100:],
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else None,
        "memory_mb": memory_mb,
        "last_updated": time.time(),
    }
    
    with open(_CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_calibration_stats() -> CalibrationStats:
    """
    Get overall calibration statistics.
    """
    if not _CALIBRATION_FILE.exists():
        return CalibrationStats()
    
    try:
        with open(_CALIBRATION_FILE, "r") as f:
            data = json.load(f)
        
        models = data.get("models", {})
        total_models = len(models)
        total_benchmarks = sum(len(m) for m in models.values())
        
        all_latencies = []
        for model_data in models.values():
            for hw_data in model_data.values():
                if isinstance(hw_data, dict):
                    avg = hw_data.get("avg_latency_ms")
                    if avg:
                        all_latencies.append(avg)
        
        return CalibrationStats(
            total_models=total_models,
            total_benchmarks=total_benchmarks,
            avg_latency_ms=sum(all_latencies) / len(all_latencies) if all_latencies else 0.0,
            last_updated=data.get("last_updated", 0.0),
        )
    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.warning(f"Failed to read calibration stats: {e}")
        return CalibrationStats()

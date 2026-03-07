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
GUI Calibration API

Handles hardware calibration, benchmark runs, and calibration history.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.core.benchmark_calibration import (
    get_calibrated_latency_prediction,
    record_benchmark,
    get_calibration_stats,
    CalibrationStats,
)


logger = logging.getLogger(__name__)

_BASELINE_LIMITS: dict[str, Any] = {
    "P50_target_ms": 25.0,
    "P95_target_ms": 120.0,
    "Hard_timeout_ms": 250.0,
    "Process_baseline_mb": 300.0,
    "Soft_limit_mb": 1024.0,
    "Hard_limit_mb": 1536.0,
    "Max_model_size_mb": 500.0,
    "Recommended_model_size_mb": 100.0,
    "Optimal_concurrency": "1-10",
    "Acceptable_concurrency": "10-20",
    "Rate_limit_threshold": "20+",
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _derive_safe_limits(hardware_profile: dict[str, Any]) -> dict[str, Any]:
    """Derive safe limits from baseline plus available runtime/hardware signals."""
    limits = dict(_BASELINE_LIMITS)

    cpu_cores_raw = hardware_profile.get("cpu_cores")
    ram_gb_raw = hardware_profile.get("ram_gb")
    disk_throughput_raw = hardware_profile.get("disk_throughput_mb_s")

    measured_latency_raw = (
        hardware_profile.get("measured_pipeline_latency_ms")
        or hardware_profile.get("pipeline_latency_ms")
    )

    if measured_latency_raw is None:
        stats = get_calibration_stats()
        if stats.avg_latency_ms and stats.avg_latency_ms > 0:
            measured_latency_raw = stats.avg_latency_ms

    try:
        cpu_cores = int(cpu_cores_raw) if cpu_cores_raw is not None else None
    except (TypeError, ValueError):
        cpu_cores = None

    try:
        ram_gb = float(ram_gb_raw) if ram_gb_raw is not None else None
    except (TypeError, ValueError):
        ram_gb = None

    try:
        disk_throughput_mb_s = (
            float(disk_throughput_raw) if disk_throughput_raw is not None else None
        )
    except (TypeError, ValueError):
        disk_throughput_mb_s = None

    try:
        measured_latency_ms = (
            float(measured_latency_raw) if measured_latency_raw is not None else None
        )
    except (TypeError, ValueError):
        measured_latency_ms = None

    if cpu_cores is None and ram_gb is None and disk_throughput_mb_s is None and measured_latency_ms is None:
        return limits

    latency_scale = 1.0
    if cpu_cores is not None and cpu_cores > 0:
        latency_scale *= _clamp(8.0 / float(cpu_cores), 0.6, 2.0)
    if disk_throughput_mb_s is not None and disk_throughput_mb_s > 0:
        latency_scale *= _clamp(300.0 / disk_throughput_mb_s, 0.7, 1.8)
    if measured_latency_ms is not None and measured_latency_ms > 0:
        latency_scale *= _clamp(measured_latency_ms / 25.0, 0.8, 4.0)

    limits["P50_target_ms"] = round(25.0 * latency_scale, 2)
    limits["P95_target_ms"] = round(max(120.0, limits["P50_target_ms"] * 4.8), 2)
    limits["Hard_timeout_ms"] = round(max(250.0, limits["P95_target_ms"] * 2.0), 2)

    if ram_gb is not None and ram_gb > 0:
        soft_limit = _clamp(ram_gb * 128.0, 1024.0, 4096.0)
        hard_limit = _clamp(soft_limit * 1.5, 1536.0, 6144.0)
        limits["Soft_limit_mb"] = round(soft_limit, 2)
        limits["Hard_limit_mb"] = round(hard_limit, 2)

    if cpu_cores is not None and cpu_cores > 0:
        optimal_high = max(1, min(10, cpu_cores))
        acceptable_high = max(10, min(20, cpu_cores * 2))
        limits["Optimal_concurrency"] = f"1-{optimal_high}"
        limits["Acceptable_concurrency"] = f"{optimal_high}-{acceptable_high}"
        limits["Rate_limit_threshold"] = f"{max(20, acceptable_high)}+"

    return limits


def run_hardware_calibration(
    model_hash: str,
    hardware_profile: dict[str, Any],
    runtime: str = "onnxruntime",
) -> dict[str, Any]:
    """
    Run hardware calibration for a model.
    
    Args:
        model_hash: Hash of the model
        hardware_profile: Hardware configuration (cpu_cores, ram_gb, gpu_available)
        runtime: Target runtime
        
    Returns:
        Calibration result dict
    """
    safe_limits = _derive_safe_limits(hardware_profile)

    cpu_cores = hardware_profile.get("cpu_cores", 4)
    ram_gb = hardware_profile.get("ram_gb", 8.0)
    gpu_available = hardware_profile.get("gpu_available", False)
    
    estimated = get_calibrated_latency_prediction(
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        model_hash=model_hash,
        heuristic_latency=float(safe_limits.get("P50_target_ms", 25.0)),
    )
    
    logger.info("hardware_calibration", extra={
        "model_hash": model_hash,
        "runtime": runtime,
        "estimated_latency_ms": estimated.get("latency_ms"),
        "source": estimated.get("source"),
        "safe_limits": safe_limits,
    })
    
    return {
        "model_hash": model_hash,
        "runtime": runtime,
        "estimated_latency_ms": estimated.get("latency_ms"),
        "source": estimated.get("source"),
        "hardware_profile": hardware_profile,
        "safe_limits": safe_limits,
    }


def get_benchmark_results(model_hash: str) -> dict[str, Any]:
    """
    Get benchmark results for a model.
    
    Args:
        model_hash: Hash of the model
        
    Returns:
        Benchmark results dict
    """
    stats = get_calibration_stats()
    
    return {
        "model_hash": model_hash,
        "total_models": stats.total_models,
        "total_benchmarks": stats.total_benchmarks,
        "avg_latency_ms": stats.avg_latency_ms,
        "last_updated": stats.last_updated,
    }


def record_benchmark_result(
    model_hash: str,
    cpu_cores: int,
    ram_gb: float,
    gpu_available: bool,
    latency_ms: float,
    memory_mb: float | None = None,
) -> None:
    """Record a benchmark result."""
    record_benchmark(
        model_hash=model_hash,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        latency_ms=latency_ms,
        memory_mb=memory_mb,
    )
    
    logger.info("benchmark_recorded", extra={
        "model_hash": model_hash,
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
    })


def get_calibration_history() -> dict[str, Any]:
    """Get calibration history and statistics."""
    stats = get_calibration_stats()
    
    return {
        "total_models": stats.total_models,
        "total_benchmarks": stats.total_benchmarks,
        "avg_latency_ms": stats.avg_latency_ms,
        "last_updated": stats.last_updated,
    }

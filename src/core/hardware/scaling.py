"""
core/hardware/scaling.py

Pure functions for hardware scoring and latency/memory scaling.
No IO. No state. No imports from services or api.
All functions take server_hw_profile as an explicit argument.
"""
from __future__ import annotations

import math
from typing import Any

from src.core.hardware.tables import DDR_BANDWIDTH_GBPS


def estimate_target_hw_score(
    cpu_cores: int,
    ram_gb: float,
    ram_ddr: str,
    gpu_available: bool,
    server_hw_profile: dict[str, Any],
) -> float:
    """
    Return a normalised compute score for the user's target hardware
    relative to the server. Score > 1 means target is faster than server.
    """
    srv_cores = float(server_hw_profile.get("cpu_cores") or 4)
    srv_bw    = float(server_hw_profile.get("memory_bandwidth_gbps") or 10.0)

    target_core_score = math.log1p(float(cpu_cores))
    server_core_score = math.log1p(srv_cores)
    core_ratio = target_core_score / max(server_core_score, 1e-9)

    ddr_key = str(ram_ddr or "").lower()
    target_bw = DDR_BANDWIDTH_GBPS.get(ddr_key, srv_bw)
    effective_target_bw = target_bw * min(1.0, float(ram_gb) / 4.0)
    bw_ratio = effective_target_bw / max(srv_bw, 1e-9)

    hw_score = 0.60 * core_ratio + 0.40 * bw_ratio
    return max(0.05, hw_score)


def scale_latency_for_target_hw(
    measured_ms: float,
    cpu_cores: int,
    ram_gb: float,
    ram_ddr: str,
    gpu_available: bool,
    server_hw_profile: dict[str, Any],
) -> float:
    """Scale measured server latency to estimated latency on target hardware."""
    hw_score = estimate_target_hw_score(cpu_cores, ram_gb, ram_ddr, gpu_available, server_hw_profile)
    return round(measured_ms / max(hw_score, 0.05), 2)


def scale_memory_for_target_hw(measured_mb: float, ram_gb: float) -> float:
    """
    Memory usage is mostly constant per model, but low-RAM systems
    may have extra OS overhead. Return a slightly adjusted value.
    """
    if ram_gb < 2.0:
        return measured_mb * 1.3
    if ram_gb < 4.0:
        return measured_mb * 1.1
    return measured_mb

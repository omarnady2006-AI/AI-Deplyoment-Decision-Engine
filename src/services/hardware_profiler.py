"""
services/hardware_profiler.py

Server hardware profile singleton.

Manages the lifecycle of the server-side hardware profile:
- Calls core profiling logic at startup
- Thread-safe read/write via Lock
- No business logic — pure orchestration of core.hardware
"""
from __future__ import annotations

import threading
import time
from typing import Any

from src.core.hardware.tables import SERVER_HW_FALLBACK
from src.core.logging_config import get_logger

logger = get_logger(__name__)

_SERVER_HW_PROFILE: dict[str, Any] = {}
_SERVER_HW_PROFILE_LOCK: threading.Lock = threading.Lock()


def get_server_hw_profile() -> dict[str, Any]:
    """Thread-safe read of the server hardware profile."""
    with _SERVER_HW_PROFILE_LOCK:
        return dict(_SERVER_HW_PROFILE)


def profile_server_hardware() -> None:
    """Benchmark server hardware once to enable latency scaling."""
    new_profile = _measure_server_hardware()
    with _SERVER_HW_PROFILE_LOCK:
        _SERVER_HW_PROFILE.update(new_profile)


def _measure_server_hardware() -> dict[str, Any]:
    """Pure hardware measurement — returns fallback on any failure."""
    try:
        import psutil
        import numpy as np

        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        ram_gb    = psutil.virtual_memory().total / (1024 ** 3)

        size = 512
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        for _ in range(3):
            np.dot(A, B)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            np.dot(A, B)
            times.append(time.perf_counter() - t0)
        avg_s        = sum(times) / len(times)
        matmul_gflops = 2 * (size ** 3) / (avg_s * 1e9)

        arr = np.random.randn(50 * 1024 * 1024 // 8).astype(np.float32)
        for _ in range(3):
            arr.copy()
        bw_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            arr.copy()
            bw_times.append(time.perf_counter() - t0)
        avg_bw_s = sum(bw_times) / len(bw_times)
        memory_bandwidth_gbps = (50 / 1024) / avg_bw_s

        return {
            "cpu_cores":             cpu_cores,
            "ram_gb":                ram_gb,
            "matmul_gflops":         matmul_gflops,
            "memory_bandwidth_gbps": memory_bandwidth_gbps,
        }
    except Exception:
        logger.warning("server_hw_measurement_failed, using fallback", exc_info=True)
        return dict(SERVER_HW_FALLBACK)

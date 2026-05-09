"""
core/hardware/tables.py

Hardware constant tables.
Pure data — no IO, no state, no imports from services or api.
"""
from __future__ import annotations

# DDR bandwidth lookup (GB/s theoretical peak per channel)
DDR_BANDWIDTH_GBPS: dict[str, float] = {
    "ddr3":   25.6,
    "lpddr3": 17.0,
    "ddr4":   51.2,
    "lpddr4": 34.1,
    "ddr5":  102.4,
    "lpddr5": 68.3,
}

# Fallback server hardware profile used when psutil/numpy are unavailable
SERVER_HW_FALLBACK: dict[str, float] = {
    "cpu_cores": 4,
    "ram_gb": 8.0,
    "matmul_gflops": 10.0,
    "memory_bandwidth_gbps": 10.0,
}

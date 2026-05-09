"""
core/scoring/calibration_math.py

Pure statistical functions for calibration reference computation.
No IO. No state. No imports from services or api.
"""
from __future__ import annotations

import math


def compute_p95_reference(latencies: list[float]) -> float:
    """
    Compute max(median, p95) as the latency calibration reference.

    Args:
        latencies: Non-empty list of positive finite latency values (ms).

    Returns:
        A single reference latency (ms) = max(median, p95).
    """
    s = sorted(latencies)
    n = len(s)

    # Median
    mid    = n // 2
    median = float(s[mid]) if n % 2 == 1 else float((s[mid - 1] + s[mid]) / 2.0)

    # 95th percentile (linear interpolation)
    if n == 1:
        p95 = float(s[0])
    else:
        rank = 0.95 * (n - 1)
        lo   = int(math.floor(rank))
        hi   = int(math.ceil(rank))
        p95  = float(s[lo]) if lo == hi else float(s[lo] + (s[hi] - s[lo]) * (rank - lo))

    return float(max(median, p95))

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
Latency Measurement Module

Provides latency measurement data structures and utilities.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LatencyMeasurement:
    """Single latency measurement result."""
    latency_ms: float
    batch_size: int = 1
    runtime: str = "unknown"
    model_hash: str = ""
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStatistics:
    """Aggregated latency statistics."""
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_dev_ms: float = 0.0
    sample_count: int = 0


def compute_latency_statistics(measurements: list[LatencyMeasurement]) -> LatencyStatistics:
    """Compute statistics from latency measurements."""
    if not measurements:
        return LatencyStatistics()
    
    latencies = [m.latency_ms for m in measurements]
    latencies_sorted = sorted(latencies)
    
    n = len(latencies)
    mean = sum(latencies) / n
    
    if n % 2 == 0:
        median = (latencies_sorted[n // 2 - 1] + latencies_sorted[n // 2]) / 2
    else:
        median = latencies_sorted[n // 2]
    
    variance = sum((x - mean) ** 2 for x in latencies) / n if n > 0 else 0.0
    std_dev = variance ** 0.5
    
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)
    
    return LatencyStatistics(
        min_ms=min(latencies),
        max_ms=max(latencies),
        mean_ms=mean,
        median_ms=median,
        p95_ms=latencies_sorted[min(p95_idx, n - 1)] if n > 0 else 0.0,
        p99_ms=latencies_sorted[min(p99_idx, n - 1)] if n > 0 else 0.0,
        std_dev_ms=std_dev,
        sample_count=n,
    )

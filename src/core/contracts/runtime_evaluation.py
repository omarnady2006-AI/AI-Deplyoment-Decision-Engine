"""
core/contracts/runtime_evaluation.py

Canonical typed contract for a single runtime evaluation result.

INVARIANTS (must hold at all times):
  - RuntimeEvaluation is ONLY constructed by services/onnx_benchmark_service.py
    and updated (re-constructed) by the scoring stage in analysis_pipeline_service.py
  - Fields are FROZEN — no mutation; scoring produces NEW instances
  - support_status is one of: "SUPPORTED" | "SUPPORTED_WITH_WARNINGS" | "UNSUPPORTED"
  - No dict aliases (predicted_latency_ms, memory_usage_mb, execution_success) exist
  - latency_avg_ms and memory_mb are None iff execution_success is False

Layer dependency:
    onnx_benchmark_service  →  [RuntimeEvaluation]  →  scoring  →  selection  →  risk
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Valid states for support_status
VALID_SUPPORT_STATUS: frozenset[str] = frozenset({
    "SUPPORTED",
    "SUPPORTED_WITH_WARNINGS",
    "UNSUPPORTED",
})

# Valid values for confidence_level
VALID_CONFIDENCE_LEVELS: frozenset[str] = frozenset({"LOW", "MEDIUM", "HIGH"})


@dataclass(frozen=True)
class RuntimeEvaluation:
    """
    Immutable result of benchmarking one execution provider against a model.

    Fields:
        runtime             Provider name: "ONNX_CPU" | "ONNX_CUDA" | "TensorRT".
        precision           Numeric precision: "FP32" | "FP16/FP32".
        support_status      "SUPPORTED" | "SUPPORTED_WITH_WARNINGS" | "UNSUPPORTED".
        execution_success   True iff the provider could actually run the model.

        latency_avg_ms      Average inference latency in milliseconds. None if not run.
        latency_p95_ms      95th-percentile latency in milliseconds. None if not run.
        memory_mb           Peak RSS memory usage in megabytes. None if not run.

        stress_memory_growth_mb   Memory growth across stress runs, in megabytes.
        stress_memory_stability   "STABLE" | "UNSTABLE".

        confidence_score    [0.0, 1.0] — populated by scoring stage.
        confidence_level    "LOW" | "MEDIUM" | "HIGH" — derived from confidence_score.
        utility_score       [0.0, 1.0] — populated by scoring stage.

        diagnostics         Human-readable issue messages.
    """
    runtime: str
    precision: str
    support_status: str
    execution_success: bool

    # Performance metrics (None = not measured)
    latency_avg_ms: "float | None"          = None   # milliseconds
    latency_p95_ms: "float | None"          = None   # milliseconds
    memory_mb: "float | None"               = None   # megabytes

    # Stress test signals
    stress_memory_growth_mb: float          = 0.0    # megabytes
    stress_memory_stability: str            = "STABLE"

    # Scoring (default 0.0 until scoring stage runs)
    confidence_score: float                 = 0.0
    confidence_level: str                   = "LOW"
    utility_score: float                    = 0.0

    # Diagnostics
    diagnostics: "tuple[str, ...]"          = ()

    def with_scores(
        self,
        confidence_score: float,
        confidence_level: str,
        utility_score: float,
    ) -> "RuntimeEvaluation":
        """Return a new RuntimeEvaluation with scoring fields set.

        This is the ONLY way to populate scores — the original is unchanged.
        """
        import dataclasses
        return dataclasses.replace(
            self,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            utility_score=utility_score,
        )

    def to_dict(self) -> "dict[str, Any]":
        """Canonical serialisation for API responses and APP_STATE storage.

        Uses canonical field names only — no aliases.
        """
        return {
            "runtime":                  self.runtime,
            "precision":                self.precision,
            "support_status":           self.support_status,
            "execution_success":        self.execution_success,
            "latency_avg_ms":           self.latency_avg_ms,
            "latency_p95_ms":           self.latency_p95_ms,
            "memory_mb":                self.memory_mb,
            "stress_memory_growth_mb":  self.stress_memory_growth_mb,
            "stress_memory_stability":  self.stress_memory_stability,
            "confidence_score":         self.confidence_score,
            "confidence_level":         self.confidence_level,
            "utility_score":            self.utility_score,
            "diagnostics":              list(self.diagnostics),
        }

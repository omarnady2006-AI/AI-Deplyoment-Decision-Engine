"""
core/scoring/confidence_scorer.py

Pure per-runtime confidence scoring logic.
Accepts RuntimeEvaluation (typed contract) instead of raw dict.

INVARIANTS:
  - Accepts RuntimeEvaluation — no raw dict
  - Reads only canonical field names
  - No IO. No state. No imports from services or api.
"""
from __future__ import annotations

from src.core.contracts.runtime_evaluation import RuntimeEvaluation

# DDR penalty table — lower DDR = less confidence for memory-heavy models
_DDR_PENALTY: dict[str, float] = {
    "ddr3":   0.82,
    "lpddr3": 0.82,
    "ddr4":   0.95,
    "lpddr4": 0.93,
    "ddr5":   1.0,
    "lpddr5": 0.98,
}


def score_runtime_confidence(
    evaluation: RuntimeEvaluation,
    deployment_profile: dict,
    reference_latency_ms: float = 200.0,
) -> float:
    """
    Compute confidence score [0.10, 1.0] for a single runtime evaluation.

    Args:
        evaluation:           Typed RuntimeEvaluation from benchmark stage.
        deployment_profile:   Deployment hardware/SLA constraints dict.
        reference_latency_ms: Calibration reference latency (ms).

    Returns:
        confidence_score in [0.10, 1.0].
    """
    if not evaluation.execution_success:
        return 0.3

    confidence = 0.90

    cpu_cores_val     = int(deployment_profile.get("cpu_cores") or 1)
    ram_gb_val        = float(deployment_profile.get("ram_gb") or 1.0)
    ram_ddr_val       = str(deployment_profile.get("ram_ddr") or "").lower()
    gpu_avail_val     = bool(deployment_profile.get("gpu_available", False))
    target_latency_ms = deployment_profile.get("target_latency_ms")
    memory_limit_mb   = deployment_profile.get("memory_limit_mb")

    # Read canonical field names directly — no aliases
    lat_ms = evaluation.latency_avg_ms
    mem_mb = evaluation.memory_mb

    # ── Hardware-tier penalty ──────────────────────────────────────────────
    if cpu_cores_val <= 1:
        confidence *= 0.72
    elif cpu_cores_val <= 2:
        confidence *= 0.82
    elif cpu_cores_val <= 4:
        confidence *= 0.92

    if ram_gb_val < 1.0:
        confidence *= 0.55
    elif ram_gb_val < 2.0:
        confidence *= 0.68
    elif ram_gb_val < 4.0:
        confidence *= 0.80
    elif ram_gb_val < 8.0:
        confidence *= 0.92

    if ram_ddr_val in _DDR_PENALTY:
        confidence *= _DDR_PENALTY[ram_ddr_val]

    _GPU_RUNTIMES = {
        "ONNX_CUDA",
        "TensorRT",
        "TFLite_GPU",
        "TF_GPU",
        "TensorRT_Native",
        "OpenVINO_GPU",
    }
    if evaluation.runtime in _GPU_RUNTIMES and not gpu_avail_val:
        confidence *= 0.30

    # ── Latency vs target ──────────────────────────────────────────────────
    if target_latency_ms is not None and lat_ms is not None:
        tgt   = float(target_latency_ms)
        ratio = float(lat_ms) / tgt if tgt >= 0 else 1.0
        if ratio > 2.0:
            confidence *= 0.50
        elif ratio > 1.5:
            confidence *= 0.65
        elif ratio > 1.0:
            confidence *= 0.80
        else:
            confidence = min(1.0, confidence * (1 + (1.0 - ratio) * 0.10))
    elif lat_ms is not None:
        ratio = float(lat_ms) / reference_latency_ms
        if ratio > 5.0:
            confidence *= 0.60
        elif ratio > 2.0:
            confidence *= 0.80

    # ── Memory vs limit ────────────────────────────────────────────────────
    if memory_limit_mb is not None and mem_mb is not None:
        mem_ratio = float(mem_mb) / float(memory_limit_mb)
        if mem_ratio > 1.0:
            confidence *= 0.60
        elif mem_ratio > 0.85:
            confidence *= 0.85

    # ── Warnings & instability ─────────────────────────────────────────────
    if evaluation.support_status == "SUPPORTED_WITH_WARNINGS":
        confidence *= 0.80

    if evaluation.stress_memory_stability == "UNSTABLE":
        confidence *= 0.65

    # ── Bonus: all within comfortable margins ─────────────────────────────
    lat_ok = target_latency_ms is None or (
        lat_ms is not None and float(lat_ms) <= float(target_latency_ms) * 0.75
    )
    mem_ok = memory_limit_mb is None or (
        mem_mb is not None and float(mem_mb) <= float(memory_limit_mb) * 0.75
    )
    if lat_ok and mem_ok and cpu_cores_val >= 4 and ram_gb_val >= 4.0:
        confidence = min(1.0, confidence * 1.05)

    return max(0.10, min(1.0, confidence))

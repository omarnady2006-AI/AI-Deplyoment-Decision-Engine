"""
core/runtime/postprocess.py

Post-processing of a single raw benchmark result into a typed RuntimeEvaluation.

Contains EXACTLY the logic that was previously embedded in
services/onnx_benchmark_service.py — relocated verbatim, no behaviour change:

    • GPU flag gating   (gpu_flags / _flag_values)
    • ONNX provider availability gate
    • Raw bench dict unpacking
    • CPU hardware latency scaling
    • Memory scaling
    • SLA classification (SUPPORTED / SUPPORTED_WITH_WARNINGS / UNSUPPORTED)
    • RuntimeEvaluation construction

Public API
----------
    postprocess_runtime_result(spec, bench, deployment_profile,
                               server_hw_profile, onnx_providers)
        → RuntimeEvaluation

    query_available_providers()
        → list[str]
"""
from __future__ import annotations

from typing import Any

from src.core.hardware.scaling import scale_latency_for_target_hw, scale_memory_for_target_hw
from src.core.contracts.runtime_evaluation import RuntimeEvaluation


# ---------------------------------------------------------------------------
# Provider query (was _query_available_providers in the service)
# ---------------------------------------------------------------------------

def query_available_providers() -> list[str]:
    """Return onnxruntime available execution providers, or [] on import failure."""
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# SLA classification (was _apply_sla_classification in the service)
# ---------------------------------------------------------------------------

def _apply_sla_classification(
    support_status: str,
    diagnostics: list[str],
    latency_avg_ms: "float | None",
    memory_mb: "float | None",
    memory_limit: "float | None",
    target_latency: "float | None",
) -> str:
    """
    Apply memory and latency SLA checks to an already-SUPPORTED runtime.

    Mutates *diagnostics* in place and returns the (possibly demoted)
    support_status string.  Does nothing when support_status is already
    UNSUPPORTED.
    """
    if support_status == "UNSUPPORTED":
        return support_status

    if memory_limit is not None and memory_mb is not None:
        if float(memory_mb) > float(memory_limit):
            support_status = "SUPPORTED_WITH_WARNINGS"
            diagnostics.append("Measured memory exceeds deployment profile limit")

    if target_latency is not None and latency_avg_ms is not None:
        if float(latency_avg_ms) > float(target_latency):
            if support_status == "SUPPORTED":
                support_status = "SUPPORTED_WITH_WARNINGS"
            diagnostics.append("Measured latency exceeds deployment profile target")

    return support_status


# ---------------------------------------------------------------------------
# Main post-processing entry point
# ---------------------------------------------------------------------------

def postprocess_runtime_result(
    spec: dict[str, Any],
    bench: dict[str, Any],
    deployment_profile: dict[str, Any],
    server_hw_profile: dict[str, Any],
    onnx_providers: list[str],
) -> RuntimeEvaluation:
    """
    Convert a raw benchmark dict + spec into a fully-classified RuntimeEvaluation.

    Logic is relocated verbatim from evaluate_all_runtimes() in
    onnx_benchmark_service.py — no behaviour change.

    Steps
    -----
    1. Resolve GPU flags from deployment_profile; gate spec against them.
    2. For kind="onnx", gate against onnxruntime provider availability.
    3. Return early UNSUPPORTED RuntimeEvaluation if either gate fails.
    4. Unpack bench dict fields.
    5. Mark UNSUPPORTED on benchmark failure.
    6. Scale latency for CPU runtimes against target hardware.
    7. Scale memory for target RAM.
    8. Apply SLA classification.
    9. Construct and return RuntimeEvaluation.
    """
    # ── Spec fields ───────────────────────────────────────────────────────────
    name      = spec["name"]
    precision = spec["precision"]
    kind      = spec["kind"]
    is_cpu    = spec.get("is_cpu", False)

    # ── Deployment profile scalars ─────────────────────────────────────────────
    gpu_available  = bool(deployment_profile.get("gpu_available", False))
    cuda_available = bool(deployment_profile.get("cuda_available", False))
    trt_available  = bool(deployment_profile.get("trt_available", False))
    memory_limit   = deployment_profile.get("memory_limit_mb")
    target_latency = deployment_profile.get("target_latency_ms")

    tgt_cpu_cores = int(deployment_profile.get("cpu_cores") or 1)
    tgt_ram_gb    = float(deployment_profile.get("ram_gb") or 1.0)
    tgt_ram_ddr   = str(deployment_profile.get("ram_ddr") or "")
    tgt_gpu       = gpu_available

    _flag_values: dict[str, bool] = {
        "gpu_available":  gpu_available,
        "cuda_available": cuda_available,
        "trt_available":  trt_available,
    }

    diagnostics: list[str] = []
    support_status = "SUPPORTED"

    # ── GPU flag gate (verbatim from service) ──────────────────────────────────
    for flag in spec.get("gpu_flags", []):
        if not _flag_values.get(flag, False):
            support_status = "UNSUPPORTED"
            label = {
                "gpu_available":  "GPU disabled in deployment profile",
                "cuda_available": "CUDA disabled in deployment profile",
                "trt_available":  "TensorRT disabled in deployment profile",
            }.get(flag, f"{flag} disabled in deployment profile")
            diagnostics.append(label)

    # ── ONNX provider availability gate (onnx / pytorch paths only) ───────────
    if kind == "onnx" and support_status == "SUPPORTED":
        provider = spec.get("provider", "")
        if provider not in onnx_providers:
            support_status = "UNSUPPORTED"
            diagnostics.append(f"Provider unavailable: {provider}")

    # ── Early return for UNSUPPORTED (bench result is not used) ───────────────
    if support_status == "UNSUPPORTED":
        return RuntimeEvaluation(
            runtime=name,
            precision=precision,
            support_status="UNSUPPORTED",
            execution_success=False,
            diagnostics=tuple(diagnostics),
        )

    # ── Unpack bench dict ──────────────────────────────────────────────────────
    success        = bool(bench.get("success"))
    latency_avg_ms = bench.get("latency_avg_ms")
    latency_p95_ms = bench.get("latency_p95_ms")
    memory_mb      = bench.get("memory_mb")
    error          = bench.get("error")

    stress_raw              = bench.get("stress_test") or {}
    stress_growth_mb: float = float(stress_raw.get("memory_growth_mb") or 0.0)
    stress_stability: str   = str(stress_raw.get("memory_stability") or "STABLE")

    if not success:
        support_status = "UNSUPPORTED"
        diagnostics.append(str(error) if error else "Benchmark failed")

    # ── Hardware scaling (CPU runtimes only) ───────────────────────────────────
    if success and latency_avg_ms is not None and is_cpu:
        latency_avg_ms = scale_latency_for_target_hw(
            latency_avg_ms, tgt_cpu_cores, tgt_ram_gb, tgt_ram_ddr,
            tgt_gpu, server_hw_profile,
        )
        if latency_p95_ms is not None:
            latency_p95_ms = scale_latency_for_target_hw(
                latency_p95_ms, tgt_cpu_cores, tgt_ram_gb, tgt_ram_ddr,
                tgt_gpu, server_hw_profile,
            )

    if success and memory_mb is not None:
        memory_mb = scale_memory_for_target_hw(memory_mb, tgt_ram_gb)

    # ── SLA classification ─────────────────────────────────────────────────────
    support_status = _apply_sla_classification(
        support_status, diagnostics,
        latency_avg_ms, memory_mb,
        memory_limit, target_latency,
    )

    return RuntimeEvaluation(
        runtime=name,
        precision=precision,
        support_status=support_status,
        execution_success=success,
        latency_avg_ms=float(latency_avg_ms) if latency_avg_ms is not None else None,
        latency_p95_ms=float(latency_p95_ms) if latency_p95_ms is not None else None,
        memory_mb=float(memory_mb) if memory_mb is not None else None,
        stress_memory_growth_mb=stress_growth_mb,
        stress_memory_stability=stress_stability,
        diagnostics=tuple(diagnostics),
    )

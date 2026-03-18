"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.core.gpu_calibration import GPUCalibrationProfile

logger = logging.getLogger(__name__)

_PROVIDER_FALLBACK_ORDER: Tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)

_GPU_PROVIDERS: Tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
)

_MODEL_SIZE_BATCH_MAP: dict = {
    "small": 1,
    "medium": 8,
    "large": 32,
}


class HardwareSelectorDecision:
    __slots__ = (
        "selected_provider",
        "override_applied",
        "override_reason",
        "calibration_latency_ms",
        "calibration_p95_ms",
        "calibration_gpu_memory_mb",
        "rejected_providers",
    )

    def __init__(
        self,
        selected_provider: Optional[str],
        override_applied: bool,
        override_reason: str,
        calibration_latency_ms: Optional[float] = None,
        calibration_p95_ms: Optional[float] = None,
        calibration_gpu_memory_mb: Optional[float] = None,
        rejected_providers: Optional[list] = None,
    ) -> None:
        self.selected_provider = selected_provider
        self.override_applied = override_applied
        self.override_reason = override_reason
        self.calibration_latency_ms = calibration_latency_ms
        self.calibration_p95_ms = calibration_p95_ms
        self.calibration_gpu_memory_mb = calibration_gpu_memory_mb
        self.rejected_providers = rejected_providers or []

    def to_dict(self) -> dict:
        return {
            "selected_provider": self.selected_provider,
            "override_applied": self.override_applied,
            "override_reason": self.override_reason,
            "calibration_latency_ms": self.calibration_latency_ms,
            "calibration_p95_ms": self.calibration_p95_ms,
            "calibration_gpu_memory_mb": self.calibration_gpu_memory_mb,
            "rejected_providers": self.rejected_providers,
        }


def _extract_gpu_availability(deployment_profile: dict) -> bool:
    """
    Extract GPU + CUDA availability from deployment_profile.

    Supports two schemas:
      NESTED (original spec):  profile["hardware"]["gpu_available"]
      FLAT   (gui_app actual): profile["gpu_available"]          ← BUG 1 FIX

    Returns True only when BOTH gpu_available AND cuda_available are truthy.
    """
    if not isinstance(deployment_profile, dict):
        return False
    # NESTED schema: profile["hardware"]["gpu_available"]
    hardware = deployment_profile.get("hardware")
    if isinstance(hardware, dict):
        gpu_flag = hardware.get("gpu_available", False)
        cuda_flag = hardware.get("cuda_available", False)
        return bool(gpu_flag) and bool(cuda_flag)
    # FLAT schema: gui_app stores gpu_available / cuda_available at top level
    gpu_flag = deployment_profile.get("gpu_available", False)
    cuda_flag = deployment_profile.get("cuda_available", False)
    return bool(gpu_flag) and bool(cuda_flag)


def _extract_memory_limit(deployment_profile: dict) -> Optional[float]:
    """
    Extract GPU memory limit from deployment_profile.

    Supports two schemas:
      NESTED: profile["hardware"]["gpu_memory_limit_mb"]
      FLAT:   profile["memory_limit_mb"]                        ← BUG 1 FIX
    """
    if not isinstance(deployment_profile, dict):
        return None
    # NESTED schema
    hardware = deployment_profile.get("hardware")
    if isinstance(hardware, dict):
        val = hardware.get("gpu_memory_limit_mb")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    # FLAT schema: gui_app uses memory_limit_mb at top level
    val = deployment_profile.get("memory_limit_mb")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_target_latency(deployment_profile: dict) -> Optional[float]:
    """
    Extract SLA target latency from deployment_profile.

    Supports two schemas:
      NESTED: profile["sla"]["target_latency_ms"]
      FLAT:   profile["target_latency_ms"]                      ← BUG 1 FIX
    """
    if not isinstance(deployment_profile, dict):
        return None
    # NESTED schema
    sla = deployment_profile.get("sla")
    if isinstance(sla, dict):
        val = sla.get("target_latency_ms")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    # FLAT schema: gui_app stores target_latency_ms at top level
    val = deployment_profile.get("target_latency_ms")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _resolve_batch_size(batch_size: int, model_size: str) -> int:
    if batch_size > 0:
        return batch_size
    return _MODEL_SIZE_BATCH_MAP.get(model_size, 1)


def _nearest_supported_batch(batch_size: int) -> int:
    supported = [1, 8, 32]
    if batch_size in supported:
        return batch_size
    distances = [(abs(batch_size - s), s) for s in supported]
    distances.sort()
    return distances[0][1]


def _collect_rejection_reasons(
    calibration: GPUCalibrationProfile,
    model_size: str,
    resolved_batch: int,
    max_gpu_memory_mb: Optional[float],
    target_latency_ms: Optional[float],
) -> list:
    rejected = []
    for provider in calibration.providers_detected:
        try:
            ref = calibration.get_reference(model_size, resolved_batch, provider)
        except Exception as e:
            logger.debug(f"No calibration reference for provider {provider}: {e}")
            continue

        reasons = []
        if (
            max_gpu_memory_mb is not None
            and ref.get("gpu_peak_memory_mb") is not None
            and ref["gpu_peak_memory_mb"] > max_gpu_memory_mb
        ):
            reasons.append(
                f"gpu_peak_memory_mb={ref['gpu_peak_memory_mb']} "
                f"exceeds limit={max_gpu_memory_mb}"
            )

        if (
            target_latency_ms is not None
            and ref["latency_avg_ms"] > target_latency_ms
        ):
            reasons.append(
                f"latency_avg_ms={ref['latency_avg_ms']} "
                f"exceeds target={target_latency_ms}"
            )

        if reasons:
            rejected.append({"provider": provider, "reasons": reasons})

    return rejected


def select_runtime_with_hardware_awareness(
    deployment_profile: dict,
    model_size: str,
    batch_size: int,
    calibration: Optional[GPUCalibrationProfile],
) -> HardwareSelectorDecision:
    if calibration is None:
        logger.debug(
            "hardware_selector: no calibration loaded — selector returning None."
        )
        return HardwareSelectorDecision(
            selected_provider=None,
            override_applied=False,
            override_reason="no_calibration_loaded",
        )

    if not _extract_gpu_availability(deployment_profile):
        logger.debug(
            "hardware_selector: GPU or CUDA not flagged available in "
            "deployment_profile — selector returning None."
        )
        return HardwareSelectorDecision(
            selected_provider=None,
            override_applied=False,
            override_reason="gpu_not_available_in_deployment_profile",
        )

    max_gpu_memory_mb = _extract_memory_limit(deployment_profile)
    target_latency_ms = _extract_target_latency(deployment_profile)

    resolved_batch = _resolve_batch_size(batch_size, model_size)
    calibration_batch = _nearest_supported_batch(resolved_batch)

    if model_size not in calibration.models_detected:
        logger.warning(
            "hardware_selector: model_size='%s' not in calibration — "
            "returning None.",
            model_size,
        )
        return HardwareSelectorDecision(
            selected_provider=None,
            override_applied=False,
            override_reason=f"model_size_not_in_calibration:{model_size}",
        )

    rejected_providers = _collect_rejection_reasons(
        calibration=calibration,
        model_size=model_size,
        resolved_batch=calibration_batch,
        max_gpu_memory_mb=max_gpu_memory_mb,
        target_latency_ms=target_latency_ms,
    )

    best_provider = calibration.get_best_provider(
        model_size=model_size,
        batch_size=calibration_batch,
        max_gpu_memory_mb=max_gpu_memory_mb,
        target_latency_ms=target_latency_ms,
    )

    if best_provider is None:
        logger.warning(
            "hardware_selector: no provider passed all constraints for "
            "model_size='%s', batch=%d — returning None.",
            model_size,
            calibration_batch,
        )
        return HardwareSelectorDecision(
            selected_provider=None,
            override_applied=False,
            override_reason="no_provider_passed_constraints",
            rejected_providers=rejected_providers,
        )

    try:
        ref = calibration.get_reference(model_size, calibration_batch, best_provider)
    except Exception as exc:
        logger.error(
            "hardware_selector: failed to read calibration reference for "
            "provider='%s': %s",
            best_provider,
            exc,
        )
        return HardwareSelectorDecision(
            selected_provider=None,
            override_applied=False,
            override_reason=f"calibration_reference_read_error:{exc}",
            rejected_providers=rejected_providers,
        )

    override_reason = (
        f"calibration_best_provider:model={model_size},"
        f"batch={calibration_batch},"
        f"latency_avg_ms={ref['latency_avg_ms']},"
        f"gpu_peak_memory_mb={ref.get('gpu_peak_memory_mb')}"
    )

    logger.info(
        "hardware_selector: overriding runtime to '%s' — %s",
        best_provider,
        override_reason,
    )

    return HardwareSelectorDecision(
        selected_provider=best_provider,
        override_applied=True,
        override_reason=override_reason,
        calibration_latency_ms=ref["latency_avg_ms"],
        calibration_p95_ms=ref["latency_p95_ms"],
        calibration_gpu_memory_mb=ref.get("gpu_peak_memory_mb"),
        rejected_providers=rejected_providers,
    )

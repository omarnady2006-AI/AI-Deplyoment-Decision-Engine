"""
core/decision_engine/hw_override.py

Pure hardware-aware runtime override.
Accepts calibration and deployment_profile as explicit arguments.
No IO. No state. No imports from services, api, or app_state.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.core.hardware_selector import (
    HardwareSelectorDecision,
    select_runtime_with_hardware_awareness,
)

logger = logging.getLogger(__name__)


def apply_hardware_aware_override(
    best_runtime: str,
    model_size: str,
    batch_size: int,
    deployment_profile: dict[str, Any],
    calibration: Any | None = None,
    evaluations: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Apply hardware-aware runtime override (pure function).

    Args:
        best_runtime:       Currently selected runtime name.
        model_size:         'small' | 'medium' | 'large'
        batch_size:         Inference batch size.
        deployment_profile: Deployment hardware constraints dict.
        calibration:        Optional GPUCalibrationProfile instance.
        evaluations:        Optional list of evaluation dicts for GPU guard.

    Returns:
        (resolved_runtime, selector_decision_dict)
    """
    gpu_available = bool(deployment_profile.get("gpu_available", False))

    decision: HardwareSelectorDecision = select_runtime_with_hardware_awareness(
        deployment_profile=deployment_profile,
        model_size=model_size,
        batch_size=batch_size,
        calibration=calibration,
    )

    if not (decision.override_applied and decision.selected_provider is not None):
        logger.debug("hw_override: no override | reason: %s", decision.override_reason)
        return best_runtime, decision.to_dict()

    candidate = decision.selected_provider

    # Guard: GPU-targeted runtimes require GPU to actually be available
    if candidate in ("ONNX_CUDA", "TensorRT"):
        if not gpu_available:
            logger.warning(
                "hw_override suppressed: '%s' requires GPU but gpu_available=False; "
                "keeping '%s'.", candidate, best_runtime,
            )
            d = decision.to_dict()
            d["override_suppressed"] = True
            d["suppression_reason"]  = "gpu_not_available_in_profile"
            return best_runtime, d

        eval_map = {
            ev["runtime"]: ev
            for ev in (evaluations or [])
            if isinstance(ev, dict) and "runtime" in ev
        }
        cuda_eval = eval_map.get(candidate)
        if cuda_eval is None:
            logger.warning(
                "hw_override suppressed: no evaluation for '%s'; keeping '%s'.",
                candidate, best_runtime,
            )
            d = decision.to_dict()
            d["override_suppressed"] = True
            d["suppression_reason"]  = "cuda_evaluation_missing"
            return best_runtime, d

        if not cuda_eval.get("execution_success") or \
                str(cuda_eval.get("status", "")).upper() in ("UNSUPPORTED", "FAILED"):
            logger.warning(
                "hw_override suppressed: '%s' evaluation failed; keeping '%s'.",
                candidate, best_runtime,
            )
            d = decision.to_dict()
            d["override_suppressed"] = True
            d["suppression_reason"]  = "cuda_evaluation_failed"
            return best_runtime, d

    logger.info(
        "hw_override: '%s' → '%s' | reason: %s",
        best_runtime, candidate, decision.override_reason,
    )
    return candidate, decision.to_dict()

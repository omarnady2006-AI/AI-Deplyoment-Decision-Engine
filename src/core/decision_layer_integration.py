from __future__ import annotations

"""
decision_layer_integration.py

This module is a REFERENCE INTEGRATION PATCH — not a standalone module.
Paste the relevant blocks into your existing decision endpoint.

It shows exactly:
  1. Where to call select_runtime_with_hardware_awareness()
  2. How to override best_runtime before SLA status assignment
  3. How to log the override without touching SLA logic
"""

import logging
from typing import Any, Optional

from src.app_state import APP_STATE, APP_STATE_LOCK
from src.core.gpu_calibration import GPUCalibrationProfile
from src.core.hardware_selector import (
    HardwareSelectorDecision,
    select_runtime_with_hardware_awareness,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: read current calibration safely
# ---------------------------------------------------------------------------

def _get_gpu_calibration() -> Optional[GPUCalibrationProfile]:
    with APP_STATE_LOCK:
        return APP_STATE.get("gpu_calibration")


def _get_deployment_profile() -> dict:
    with APP_STATE_LOCK:
        profile = APP_STATE.get("deployment_profile") or {}
    return profile


# ---------------------------------------------------------------------------
# Integration patch — call this BEFORE your SLA status assignment block
# ---------------------------------------------------------------------------

def apply_hardware_aware_override(
    best_runtime: str,
    model_size: str,
    batch_size: int,
    evaluations: Optional[list] = None,
    gpu_available: bool = False,
) -> tuple[str, Optional[dict]]:
    """
    Returns (resolved_runtime, selector_decision_dict).

    resolved_runtime: the runtime to use (original or overridden).
    selector_decision_dict: the HardwareSelectorDecision.to_dict() for audit
                            logging and response enrichment, or None if no
                            calibration was applied.

    FIX-ISSUE-3: The previous implementation unconditionally redirected large
    models to ONNX_CUDA regardless of whether:
      (a) the target hardware has a GPU  (gpu_available flag)
      (b) the CUDA evaluation passed     (evaluations list)
    This caused CPU-only deployments to be approved for a runtime that was
    never benchmarked and cannot actually run.

    The override is now gated by two hard pre-conditions:
      1. gpu_available=True  — the deployment profile declares GPU presence.
      2. A corresponding evaluation entry exists in `evaluations` whose
         execution_success is True and status is not UNSUPPORTED / FAILED.
    If either condition fails the override is suppressed and the caller's
    existing best_runtime is returned unchanged.
    """
    calibration = _get_gpu_calibration()
    deployment_profile = _get_deployment_profile()

    decision: HardwareSelectorDecision = select_runtime_with_hardware_awareness(
        deployment_profile=deployment_profile,
        model_size=model_size,
        batch_size=batch_size,
        calibration=calibration,
    )

    if decision.override_applied and decision.selected_provider is not None:
        candidate = decision.selected_provider

        # --- FIX-ISSUE-3: validate GPU availability and evaluation outcome ---
        if candidate in ("ONNX_CUDA", "TensorRT"):
            if not gpu_available:
                logger.warning(
                    "hardware_selector override suppressed: candidate '%s' requires GPU "
                    "but deployment profile declares gpu_available=False. "
                    "Keeping existing runtime '%s'.",
                    candidate,
                    best_runtime,
                )
                decision_dict = decision.to_dict()
                decision_dict["override_suppressed"] = True
                decision_dict["suppression_reason"] = "gpu_not_available_in_profile"
                return best_runtime, decision_dict

            # Confirm the evaluation benchmark for this runtime passed
            eval_map: dict[str, dict] = {}
            for ev in (evaluations or []):
                if isinstance(ev, dict) and "runtime" in ev:
                    eval_map[ev["runtime"]] = ev

            cuda_eval = eval_map.get(candidate)
            if cuda_eval is None:
                logger.warning(
                    "hardware_selector override suppressed: no evaluation entry found "
                    "for runtime '%s'. Keeping existing runtime '%s'.",
                    candidate,
                    best_runtime,
                )
                decision_dict = decision.to_dict()
                decision_dict["override_suppressed"] = True
                decision_dict["suppression_reason"] = "cuda_evaluation_missing"
                return best_runtime, decision_dict

            exec_ok = cuda_eval.get("execution_success", False)
            status = str(cuda_eval.get("status", "")).upper()
            if not exec_ok or status in ("UNSUPPORTED", "FAILED"):
                logger.warning(
                    "hardware_selector override suppressed: '%s' evaluation "
                    "execution_success=%s status=%s. Keeping existing runtime '%s'.",
                    candidate,
                    exec_ok,
                    status,
                    best_runtime,
                )
                decision_dict = decision.to_dict()
                decision_dict["override_suppressed"] = True
                decision_dict["suppression_reason"] = (
                    f"cuda_evaluation_failed: execution_success={exec_ok} status={status}"
                )
                return best_runtime, decision_dict

        logger.info(
            "hardware_selector override: best_runtime '%s' → '%s' | reason: %s",
            best_runtime,
            candidate,
            decision.override_reason,
        )
        return candidate, decision.to_dict()

    logger.debug(
        "hardware_selector: no override applied | reason: %s",
        decision.override_reason,
    )
    return best_runtime, decision.to_dict()


# ---------------------------------------------------------------------------
# Example: how your existing decision endpoint should look after patching
# ---------------------------------------------------------------------------

def example_decision_endpoint_pseudocode() -> dict:
    """
    Illustrative pseudocode — matches the structure of your real endpoint.
    Copy only the marked blocks into your existing endpoint.
    DO NOT replace your entire endpoint with this.
    """

    # --- (existing) collect inputs ---
    analysis: dict = APP_STATE["analysis"]                          # existing
    deployment_profile: dict = APP_STATE["deployment_profile"]     # existing
    model_size: str = analysis.get("model_size", "small")          # existing
    batch_size: int = analysis.get("batch_size", 1)                # existing

    # --- (existing) your normal runtime selection logic ---
    best_runtime: str = "CPUExecutionProvider"                     # existing placeholder

    # =========================================================
    # PATCH — insert this block before SLA status assignment
    # =========================================================
    best_runtime, hw_selector_decision = apply_hardware_aware_override(
        best_runtime=best_runtime,
        model_size=model_size,
        batch_size=batch_size,
        evaluations=evaluations,   # list of evaluation dicts from benchmark step
        gpu_available=bool(deployment_profile.get("gpu_available", False)),
    )
    # =========================================================

    # --- (existing) SLA status assignment — DO NOT MODIFY ---
    latency_ms: float = analysis.get("latency_ms", 0.0)           # existing
    sla_target_ms: float = deployment_profile.get(                 # existing
        "sla", {}
    ).get("target_latency_ms", float("inf"))
    sla_status: str = "met" if latency_ms <= sla_target_ms else "violated"  # existing

    # --- (existing) build final decision dict ---
    decision: dict[str, Any] = {
        "best_runtime": best_runtime,
        "sla_status": sla_status,
        # add hw_selector_decision for audit trail — does not affect SLA logic
        "hardware_selector": hw_selector_decision,
    }

    return decision

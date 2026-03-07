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
Decision Layer Integration

Provides hardware-aware runtime override logic for deployment decisions.
"""


from typing import Any


def apply_hardware_aware_override(
    best_runtime: str,
    model_size: str,
    batch_size: int = 1,
) -> tuple[str, dict[str, Any]]:
    """
    Apply hardware-aware runtime selection override.
    
    Args:
        best_runtime: The currently selected best runtime
        model_size: Model size classification (small, medium, large)
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (resolved_runtime, decision_dict)
    """
    decision = {
        "override_applied": False,
        "override_reason": None,
        "selected_provider": best_runtime,
        "calibration_latency_ms": None,
        "calibration_gpu_memory_mb": None,
    }
    
    if model_size == "large" and batch_size <= 1:
        if best_runtime in ("ONNX_CPU",):
            decision["override_applied"] = True
            decision["override_reason"] = "Large model prefers GPU acceleration"
            decision["selected_provider"] = "ONNX_CUDA"
            return ("ONNX_CUDA", decision)
    
    return (best_runtime, decision)

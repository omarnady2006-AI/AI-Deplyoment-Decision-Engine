"""
core/scoring/calibration_validator.py

Pure validation of calibration input data.
No IO. No state. No imports from services or api.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationValidationResult:
    valid: bool
    error: str | None = None


def validate_calibration_input(
    latencies: list[float],
    model_sizes: list[int],
) -> CalibrationValidationResult:
    """
    Validate calibration latency/model_size arrays.

    Returns CalibrationValidationResult(valid=True) or (valid=False, error=<reason>).
    """
    if len(latencies) != len(model_sizes):
        return CalibrationValidationResult(
            valid=False,
            error="latencies and model_sizes must have the same length",
        )
    if not latencies:
        return CalibrationValidationResult(
            valid=False,
            error="latencies must not be empty",
        )
    if any(not math.isfinite(x) or x <= 0 for x in latencies):
        return CalibrationValidationResult(
            valid=False,
            error="latencies must contain positive finite numbers",
        )
    if any(x < 0 for x in model_sizes):
        return CalibrationValidationResult(
            valid=False,
            error="model_sizes must contain non-negative integers",
        )
    return CalibrationValidationResult(valid=True)

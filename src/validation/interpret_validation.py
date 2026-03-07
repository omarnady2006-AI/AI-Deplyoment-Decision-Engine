"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

from src.core.decision import DeploymentConstraint
from src.core.input_profile import InputProfile
from src.validation.runtime_validator import RuntimeValidationResult


def _variant_is_outside_profile(variant_name: str, profile: InputProfile) -> bool:
    if variant_name == "batch_spike":
        return profile.batch is not None and profile.batch.max < 4
    if variant_name == "skinny":
        height_excludes = profile.height is not None and profile.height.min > 1
        width_excludes = profile.width is not None and profile.width.min > 1
        return height_excludes and width_excludes
    return False


def _domain_failure_overlaps_profile(
    failing_variants: list[str] | None, profile: InputProfile | None
) -> bool:
    if not failing_variants:
        return True
    if profile is None:
        return True

    for variant_name in failing_variants:
        if not _variant_is_outside_profile(variant_name, profile):
            return True
    return False


def interpret_validation_result(
    result: RuntimeValidationResult,
    profile: InputProfile | None = None,
) -> list[DeploymentConstraint]:
    error_message = result.error_message or ""
    error_lower = error_message.lower()

    constraints: list[DeploymentConstraint] = []

    if "unstable_domain" in error_lower and _domain_failure_overlaps_profile(
        result.failing_variants, profile
    ):
        constraints.append(DeploymentConstraint.INPUT_SIZE_LIMITED)

    if (
        "shape changes across runs" in error_lower
        or "output shape changed between runs" in error_lower
    ):
        constraints.append(DeploymentConstraint.BATCH_LIMITED)

    if "nan" in error_lower or "inf" in error_lower:
        constraints.append(DeploymentConstraint.NUMERICAL_RISK)

    return constraints

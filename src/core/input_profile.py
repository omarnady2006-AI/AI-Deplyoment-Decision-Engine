"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionRange:
    min: int
    max: int


@dataclass(frozen=True)
class InputProfile:
    batch: DimensionRange | None
    height: DimensionRange | None
    width: DimensionRange | None
    sequence: DimensionRange | None


def _parse_range(payload: object, field_name: str) -> DimensionRange | None:
    if payload is None:
        return None
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError(f"profile field '{field_name}' must be [min, max]")
    min_value, max_value = payload
    if not isinstance(min_value, int) or not isinstance(max_value, int):
        raise ValueError(f"profile field '{field_name}' must contain integers")
    if min_value <= 0 or max_value <= 0:
        raise ValueError(f"profile field '{field_name}' values must be positive")
    if min_value > max_value:
        raise ValueError(f"profile field '{field_name}' has min greater than max")
    return DimensionRange(min=min_value, max=max_value)


def input_profile_from_dict(data: dict[str, object]) -> InputProfile:
    return InputProfile(
        batch=_parse_range(data.get("batch"), "batch"),
        height=_parse_range(data.get("height"), "height"),
        width=_parse_range(data.get("width"), "width"),
        sequence=_parse_range(data.get("sequence"), "sequence"),
    )

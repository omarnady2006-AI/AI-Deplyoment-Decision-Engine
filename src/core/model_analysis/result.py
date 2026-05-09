from __future__ import annotations
"""
Copyright (c) 2026 Omar Nady — Deployment Decision Engine.
See LICENSE in the project root for full terms.
"""
"""core/model_analysis/result.py — typed result contracts."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorShape:
    name: str
    shape: list[int | str | None]
    dtype: str = "float32"


@dataclass
class ModelAnalysisResult:
    success: bool
    model_path: str
    file_hash: str = ""
    operator_count: int = 0
    operators: list[str] = field(default_factory=list)
    operator_counts: dict[str, int] = field(default_factory=dict)
    parameter_count: int = 0
    has_dynamic_shapes: bool = False
    input_count: int = 0
    output_count: int = 0
    input_shapes: dict[str, list[int | str | None]] = field(default_factory=dict)
    unsupported_ops: list[str] = field(default_factory=list)
    sequential_depth: int = 0
    ir_version: int = 0
    opset_version: int = 0
    error: str | None = None
    graph: Any = None


_REQUIRED_ANALYSIS_FIELDS: frozenset[str] = frozenset({
    "model_path","file_hash","operator_count","operators","operator_counts",
    "parameter_count","has_dynamic_shapes","input_shapes","unsupported_ops",
    "sequential_depth","input_count","output_count","success",
})

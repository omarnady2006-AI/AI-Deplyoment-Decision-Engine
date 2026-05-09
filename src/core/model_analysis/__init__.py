from __future__ import annotations
"""Copyright (c) 2026 Omar Nady — Deployment Decision Engine. See LICENSE."""
"""
core/model_analysis/__init__.py

Backward-compatible re-exports so all existing imports
`from src.core.model_analysis import X` continue to work unchanged.
"""
from src.core.model_analysis.result    import ModelAnalysisResult, TensorShape
from src.core.model_analysis.validator import validate_onnx_model
from src.core.model_analysis.analyzer  import (
    analyze_model, detect_unsupported_ops, detect_dynamic_shapes,
    count_parameters, get_operator_counts, get_input_shapes, get_sequential_depth,
)
from src.core.model_analysis.cache     import get_validated_cached_model
from src.core.model_analysis.constants import _ONNX_BASELINE_OPS

__all__ = [
    "ModelAnalysisResult", "TensorShape", "validate_onnx_model", "analyze_model",
    "detect_unsupported_ops", "detect_dynamic_shapes", "count_parameters",
    "get_operator_counts", "get_input_shapes", "get_sequential_depth",
    "get_validated_cached_model", "_ONNX_BASELINE_OPS",
]

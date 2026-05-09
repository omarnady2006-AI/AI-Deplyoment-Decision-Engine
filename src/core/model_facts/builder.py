"""
core/model_facts/builder.py

Pure fact extraction from raw format_router output.

INVARIANTS:
  - All output keys use canonical names (NO dot-prefix)
  - This module is called ONLY by model_facts/analyzer.py
  - No IO. No state. No imports from services or api.

Key name changes from old version:
  OLD: "model.has_non_max_suppression"   NEW: "has_non_max_suppression"
  OLD: "model.has_resize"                NEW: "has_resize"
  OLD: "model.has_conv_transpose"        NEW: "has_conv_transpose"
  OLD: "model.uses_layer_normalization"  NEW: "uses_layer_normalization"
  OLD: "model.uses_batch_normalization"  NEW: "uses_batch_normalization"
  OLD: "model.has_conv"                  NEW: "has_conv"
  OLD: "model.has_attention"             NEW: "has_attention"
  OLD: "model.has_dynamic_shapes"        NEW: "has_dynamic_shapes"
"""
from __future__ import annotations

from typing import Any


def build_model_facts(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Extract canonical architecture facts from a raw format_router result dict.

    Input:  raw dict produced by format_router (may contain dot-prefixed keys
            or canonical keys — we read only the canonical source fields).
    Output: flat dict with ONLY canonical key names, consumed by
            _normalize_to_model_analysis() in analyzer.py.

    All boolean flags default to False; numeric defaults to 0.
    """
    op_types: set[str] = set(analysis.get("operators", []))
    operator_counts: dict[str, int] = dict(analysis.get("operator_counts", {}))
    param_count: int = int(analysis.get("parameter_count", 0))

    # Architecture booleans — canonical names only, no dot-prefix
    has_non_max_suppression  = "NonMaxSuppression" in op_types
    has_resize               = "Resize" in op_types
    has_conv_transpose       = "ConvTranspose" in op_types
    uses_layer_normalization = "LayerNormalization" in op_types
    uses_batch_normalization = "BatchNormalization" in op_types
    has_conv                 = "Conv" in op_types or "ConvTranspose" in op_types
    has_attention            = "Attention" in op_types or "MultiHeadAttention" in op_types

    # has_dynamic_shapes: prefer the field from the router result directly;
    # the router emits it without prefix; some old paths emitted "model.has_dynamic_shapes"
    # — we handle both here and nowhere else.
    has_dynamic_shapes: bool = bool(
        analysis.get("has_dynamic_shapes")
        or analysis.get("model.has_dynamic_shapes")
        or False
    )

    sequential_depth_estimate: int = sum(operator_counts.values())

    if param_count >= 25_000_000:
        parameter_scale_class = "large"
    elif param_count >= 8_000_000:
        parameter_scale_class = "medium"
    else:
        parameter_scale_class = "small"

    return {
        # Numeric
        "parameter_count":           param_count,
        "operator_counts":           operator_counts,
        "sequential_depth_estimate": sequential_depth_estimate,
        # Classification
        "parameter_scale_class":     parameter_scale_class,
        # Booleans — canonical names, no dot-prefix
        "has_dynamic_shapes":        has_dynamic_shapes,
        "has_non_max_suppression":   has_non_max_suppression,
        "has_resize":                has_resize,
        "has_conv_transpose":        has_conv_transpose,
        "uses_layer_normalization":  uses_layer_normalization,
        "uses_batch_normalization":  uses_batch_normalization,
        "has_conv":                  has_conv,
        "has_attention":             has_attention,
    }

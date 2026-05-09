"""
core/contracts/model_analysis.py

Canonical typed contract for model analysis.

INVARIANTS (must hold at all times):
  - ModelAnalysis is ONLY constructed by core/model_facts/analyzer.py
  - Fields are FROZEN — no mutation permitted after construction
  - No dot-prefixed keys exist anywhere in this schema
  - `error is None` iff `success is True`
  - All collection fields use tuple (immutable) not list

Layer dependency:
    format_router  →  [normalization gate]  →  ModelAnalysis  →  pipeline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ModelAnalysis:
    """
    Immutable canonical result of parsing a model file.

    Attributes (units where applicable):
        model_path              Absolute filesystem path to the model file.
        file_hash               SHA-based content+path hash for change detection.
        framework               Format identifier: "onnx" | "tflite" | "pytorch" |
                                "tensorrt" | "openvino" | "unknown".
        operator_count          Total number of nodes/operators in the graph.
        operators               Unique operator type names (de-duplicated).
        operator_counts         Mapping of op_name → occurrence count (immutable).
        parameter_count         Total trainable parameter count.
        has_dynamic_shapes      True if any input dimension is symbolic/dynamic.
        input_count             Number of model inputs.
        output_count            Number of model outputs.
        model_size_mb           File size in megabytes.

        parameter_scale_class   Coarse size bucket: "small" | "medium" | "large".
        sequential_depth_estimate  Proxy depth = sum of operator counts.
        unsupported_ops         Op names not present in the runtime baseline set.

        has_non_max_suppression Whether the graph uses NonMaxSuppression.
        has_resize              Whether the graph uses Resize.
        has_conv_transpose      Whether the graph uses ConvTranspose.
        uses_layer_normalization Whether the graph uses LayerNormalization.
        uses_batch_normalization Whether the graph uses BatchNormalization.
        has_conv                Whether the graph uses Conv or ConvTranspose.
        has_attention           Whether the graph uses Attention/MultiHeadAttention.

        error                   Non-None only when success=False.
        success                 True iff parsing and normalization completed cleanly.
    """
    success: bool
    model_path: str

    # Identity
    file_hash: str                          = ""
    framework: str                          = "unknown"

    # Structural metrics
    operator_count: int                     = 0
    operators: tuple[str, ...]              = ()
    operator_counts: "MappingProxyType[str, int]" = field(
        default_factory=lambda: MappingProxyType({})
    )
    parameter_count: int                    = 0
    has_dynamic_shapes: bool                = False
    input_count: int                        = 0
    output_count: int                       = 0
    model_size_mb: float                    = 0.0

    # Derived facts
    parameter_scale_class: str              = "small"
    sequential_depth_estimate: int          = 0
    unsupported_ops: tuple[str, ...]        = ()

    # Architecture flags
    has_non_max_suppression: bool           = False
    has_resize: bool                        = False
    has_conv_transpose: bool                = False
    uses_layer_normalization: bool          = False
    uses_batch_normalization: bool          = False
    has_conv: bool                          = False
    has_attention: bool                     = False

    # Error path
    error: "str | None"                     = None

    def to_dict(self) -> "dict[str, Any]":
        """Canonical serialisation for API responses and APP_STATE storage.

        All keys use canonical names (no dot-prefix). This is the ONLY
        serialisation method — do not add aliases.
        """
        return {
            "success":                  self.success,
            "model_path":               self.model_path,
            "file_hash":                self.file_hash,
            "framework":                self.framework,
            "operator_count":           self.operator_count,
            "operators":                list(self.operators),
            "operator_counts":          dict(self.operator_counts),
            "parameter_count":          self.parameter_count,
            "has_dynamic_shapes":       self.has_dynamic_shapes,
            "input_count":              self.input_count,
            "output_count":             self.output_count,
            "model_size_mb":            self.model_size_mb,
            "parameter_scale_class":    self.parameter_scale_class,
            "sequential_depth_estimate": self.sequential_depth_estimate,
            "unsupported_ops":          list(self.unsupported_ops),
            "has_non_max_suppression":  self.has_non_max_suppression,
            "has_resize":               self.has_resize,
            "has_conv_transpose":       self.has_conv_transpose,
            "uses_layer_normalization": self.uses_layer_normalization,
            "uses_batch_normalization": self.uses_batch_normalization,
            "has_conv":                 self.has_conv,
            "has_attention":            self.has_attention,
            "error":                    self.error,
        }

    def to_risk_dict(self) -> "dict[str, Any]":
        """Minimal dict consumed by the risk engine.

        Keys match exactly what evaluate_deployment_risk() expects.
        No dot-prefix. No aliases.
        """
        return {
            "has_dynamic_shapes":       self.has_dynamic_shapes,
            "parameter_scale_class":    self.parameter_scale_class,
            "operator_count":           self.operator_count,
            "has_non_max_suppression":  self.has_non_max_suppression,
            "has_resize":               self.has_resize,
            "has_conv_transpose":       self.has_conv_transpose,
            "uses_layer_normalization": self.uses_layer_normalization,
            "uses_batch_normalization": self.uses_batch_normalization,
            "has_conv":                 self.has_conv,
            "has_attention":            self.has_attention,
        }

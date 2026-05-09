"""
core/model_facts/analyzer.py

Single entry point for model file analysis.

NORMALIZATION BOUNDARY — enforced here and nowhere else:
  - format_router output (raw dict, may contain dot-prefixed keys) is consumed
  - _normalize_to_model_analysis() converts it to a canonical ModelAnalysis
  - No raw dict, no dot-prefixed key, no format_router output passes beyond this file

INVARIANTS:
  - analyze_model_file() always returns ModelAnalysis (never raises)
  - On parse failure: success=False, error=<message>
  - On normalization failure: success=False, error=<invariant message>
  - No IO beyond reading the model file
  - No imports from services or api
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.core.contracts.model_analysis import ModelAnalysis
from src.core.model_facts.builder import build_model_facts
from src.core.invariants import invariant_fail


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_model_file(
    model_path: str,
    precomputed_hash: "str | None" = None,
) -> ModelAnalysis:
    """
    Parse a model file and return a typed ModelAnalysis.

    Supports .onnx, .tflite, .pt/.pth, .pb, .engine/.trt, .xml (OpenVINO).
    Returns ModelAnalysis(success=False, error=...) on any failure — never raises.

    Args:
        model_path:        Absolute path to the model file.
        precomputed_hash:  If provided, skip hashing (use caller-computed value).
    """
    try:
        from src.format_router import route_model  # type: ignore[import]
        raw = route_model(model_path)
    except Exception as exc:
        return ModelAnalysis(
            success=False,
            model_path=model_path,
            error=f"format_router failed: {exc}",
        )

    if not raw.get("analysis_success", False):
        return ModelAnalysis(
            success=False,
            model_path=model_path,
            error=raw.get("error", "Unknown parse error"),
        )

    try:
        file_hash = precomputed_hash or _compute_file_hash(model_path)
        return _normalize_to_model_analysis(raw, model_path, file_hash)
    except Exception as exc:
        return ModelAnalysis(
            success=False,
            model_path=model_path,
            error=f"normalization failed: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Normalization Gate
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_to_model_analysis(
    raw: dict[str, Any],
    model_path: str,
    file_hash: str,
) -> ModelAnalysis:
    """
    Convert raw format_router output into a canonical ModelAnalysis.

    This is the ONLY function that reads format_router output.
    After this call, no raw dict and no dot-prefixed key exists downstream.

    Steps:
      1. Extract structural fields from raw dict (canonical names, with fallbacks)
      2. Derive architecture facts via build_model_facts() — already canonical
      3. Validate mandatory fields; invariant_fail() on contract violation
      4. Construct and return frozen ModelAnalysis

    Args:
        raw:        Raw dict produced by format_router.route_model().
        model_path: Absolute path passed through for identity.
        file_hash:  Pre-computed or computed content hash.
    """
    # ── Step 1: Extract structural fields ─────────────────────────────────────
    # format_router may emit either canonical or dot-prefixed has_dynamic_shapes.
    # We accept both here and normalise; this is the last place it is tolerated.
    has_dynamic_shapes: bool = bool(
        raw.get("has_dynamic_shapes")
        or raw.get("model.has_dynamic_shapes")
        or False
    )

    operator_count: int     = int(raw.get("operator_count", 0))
    operators_list: list    = list(raw.get("operators", []))
    operator_counts_raw     = raw.get("operator_counts", {})
    parameter_count: int    = int(raw.get("parameter_count", 0))
    input_count: int        = int(raw.get("input_count", 0))
    output_count: int       = int(raw.get("output_count", 0))
    model_size_mb: float    = float(raw.get("model_size_mb", 0.0))
    framework: str          = str(raw.get("framework", "unknown"))
    unsupported_raw: list   = list(raw.get("unsupported_ops", []))

    # Validate types for critical numeric fields
    if not isinstance(operator_counts_raw, dict):
        invariant_fail(
            f"normalize: operator_counts must be dict, got {type(operator_counts_raw).__name__}"
        )

    # ── Step 2: Derive architecture facts ─────────────────────────────────────
    facts = build_model_facts(raw)

    # ── Step 3: Validate mandatory fields ─────────────────────────────────────
    if not model_path:
        invariant_fail("normalize: model_path must not be empty")
    if not file_hash:
        invariant_fail("normalize: file_hash must not be empty")
    if not isinstance(facts.get("parameter_scale_class"), str):
        invariant_fail("normalize: parameter_scale_class must be str")
    if facts["parameter_scale_class"] not in ("small", "medium", "large"):
        invariant_fail(
            f"normalize: parameter_scale_class {facts['parameter_scale_class']!r} "
            "not in ('small', 'medium', 'large')"
        )

    # ── Step 4: Construct frozen ModelAnalysis ─────────────────────────────────
    return ModelAnalysis(
        success                  = True,
        model_path               = model_path,
        file_hash                = file_hash,
        framework                = framework,
        operator_count           = operator_count,
        operators                = tuple(operators_list),
        operator_counts          = MappingProxyType(dict(operator_counts_raw)),
        parameter_count          = facts["parameter_count"],
        has_dynamic_shapes       = facts["has_dynamic_shapes"],
        input_count              = input_count,
        output_count             = output_count,
        model_size_mb            = model_size_mb,
        parameter_scale_class    = facts["parameter_scale_class"],
        sequential_depth_estimate= facts["sequential_depth_estimate"],
        unsupported_ops          = tuple(unsupported_raw),
        has_non_max_suppression  = bool(facts["has_non_max_suppression"]),
        has_resize               = bool(facts["has_resize"]),
        has_conv_transpose       = bool(facts["has_conv_transpose"]),
        uses_layer_normalization = bool(facts["uses_layer_normalization"]),
        uses_batch_normalization = bool(facts["uses_batch_normalization"]),
        has_conv                 = bool(facts["has_conv"]),
        has_attention            = bool(facts["has_attention"]),
        error                    = None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# File Hashing
# ─────────────────────────────────────────────────────────────────────────────

def _compute_file_hash(model_path: str) -> str:
    """Compute a stable content+path identity hash for change detection."""
    try:
        model_bytes   = Path(model_path).read_bytes()
        content_hash  = hashlib.sha256(model_bytes).hexdigest()
        hash_input    = f"{Path(model_path).resolve()}:{content_hash}".encode("utf-8")
        return hashlib.sha1(hash_input).hexdigest()
    except Exception:
        # Fallback: use filename only (weaker, but never raises)
        return Path(model_path).name

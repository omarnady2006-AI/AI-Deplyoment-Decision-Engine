"""
src/core/feature_extractor.py

Single entry point for extracting the ML decision feature vector.

STRICT INVARIANTS:
  1. Input sources: model analysis facts + deployment profile ONLY.
     Forbidden inputs: best_runtime_name, evaluation objects, previous decisions,
     any output of runtime selection or benchmarking.
  2. Feature vector is fixed-length (FEATURE_COUNT = 20), deterministic, purely numeric.
  3. Missing required input field → raises ValueError immediately.
  4. No silent fallbacks. No default values for required fields.
  5. Optional deployment profile fields (vram_gb, target_latency_ms, memory_limit_mb)
     encode as 0.0 when absent — 0.0 is the correct numeric representation of
     "no GPU VRAM / no latency constraint / no memory limit".
  6. This module has zero imports from services, routes, or any pipeline output.

Feature schema (index → name → source):
  0  parameter_count          model_facts (raw count of trainable parameters)
  1  operator_count           model_facts (total operator/node count)
  2  has_dynamic_shapes       model_facts (bool → 0.0 / 1.0)
  3  parameter_scale_encoded  model_facts (small=0.0, medium=1.0, large=2.0)
  4  model_size_mb            model_facts (file size in megabytes)
  5  sequential_depth         model_facts (sum of operator counts ≈ graph depth)
  6  has_conv                 model_facts (bool → 0.0 / 1.0)
  7  has_attention            model_facts (bool → 0.0 / 1.0)
  8  has_resize               model_facts (bool → 0.0 / 1.0)
  9  uses_batch_norm          model_facts (bool → 0.0 / 1.0)
  10 uses_layer_norm          model_facts (bool → 0.0 / 1.0)
  11 has_nms                  model_facts (bool → 0.0 / 1.0)
  12 has_conv_transpose       model_facts (bool → 0.0 / 1.0)
  13 cpu_cores                deployment_profile (int)
  14 ram_gb                   deployment_profile (float)
  15 gpu_available            deployment_profile (bool → 0.0 / 1.0)
  16 cuda_available           deployment_profile (bool → 0.0 / 1.0)
  17 vram_gb                  deployment_profile (float; 0.0 when no GPU)
  18 target_latency_ms        deployment_profile (float; 0.0 when unconstrained)
  19 memory_limit_mb          deployment_profile (float; 0.0 when unconstrained)
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ── Public constants ──────────────────────────────────────────────────────────

FEATURE_COUNT: int = 20

FEATURE_NAMES: tuple[str, ...] = (
    "parameter_count",
    "operator_count",
    "has_dynamic_shapes",
    "parameter_scale_encoded",
    "model_size_mb",
    "sequential_depth",
    "has_conv",
    "has_attention",
    "has_resize",
    "uses_batch_norm",
    "uses_layer_norm",
    "has_nms",
    "has_conv_transpose",
    "cpu_cores",
    "ram_gb",
    "gpu_available",
    "cuda_available",
    "vram_gb",
    "target_latency_ms",
    "memory_limit_mb",
)

assert len(FEATURE_NAMES) == FEATURE_COUNT, (
    f"FEATURE_NAMES length mismatch: {len(FEATURE_NAMES)} != {FEATURE_COUNT}"
)

# Required fields from model_facts (missing → ValueError)
_REQUIRED_MODEL_FIELDS: tuple[str, ...] = (
    "parameter_count",
    "operator_count",
    "has_dynamic_shapes",
    "parameter_scale_class",
    "model_size_mb",
    "sequential_depth_estimate",
    "has_conv",
    "has_attention",
    "has_resize",
    "uses_batch_normalization",
    "uses_layer_normalization",
    "has_non_max_suppression",
    "has_conv_transpose",
)

# Required fields from deployment_profile (missing → ValueError)
_REQUIRED_PROFILE_FIELDS: tuple[str, ...] = (
    "cpu_cores",
    "ram_gb",
    "gpu_available",
    "cuda_available",
)

# Optional profile fields encoded as 0.0 when None/absent
_OPTIONAL_PROFILE_FIELDS: tuple[str, ...] = (
    "vram_gb",
    "target_latency_ms",
    "memory_limit_mb",
)

_SCALE_CLASS_ENCODING: dict[str, float] = {
    "small":  0.0,
    "medium": 1.0,
    "large":  2.0,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _require(source: dict[str, Any], key: str, source_name: str) -> Any:
    """Return source[key] or raise ValueError with a precise message."""
    if key not in source:
        raise ValueError(
            f"feature_extractor: missing required field {key!r} in {source_name}. "
            f"Available keys: {sorted(source.keys())}"
        )
    return source[key]


def _as_float(value: Any, field_name: str) -> float:
    """Coerce value to float or raise ValueError."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"feature_extractor: field {field_name!r} cannot be coerced to float: "
            f"{value!r} ({type(value).__name__})"
        ) from exc


def _as_bool_float(value: Any, field_name: str) -> float:
    """Coerce value to 0.0 / 1.0 or raise ValueError."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if value else 0.0
    raise ValueError(
        f"feature_extractor: field {field_name!r} must be bool/int/float, "
        f"got {type(value).__name__}: {value!r}"
    )


def _encode_scale_class(scale_class: Any, field_name: str = "parameter_scale_class") -> float:
    """Map scale class string to numeric encoding or raise ValueError."""
    if not isinstance(scale_class, str):
        raise ValueError(
            f"feature_extractor: field {field_name!r} must be str, "
            f"got {type(scale_class).__name__}: {scale_class!r}"
        )
    encoded = _SCALE_CLASS_ENCODING.get(scale_class)
    if encoded is None:
        raise ValueError(
            f"feature_extractor: field {field_name!r} has unknown value {scale_class!r}; "
            f"must be one of {list(_SCALE_CLASS_ENCODING.keys())}"
        )
    return encoded


def _opt_float(source: dict[str, Any], key: str) -> float:
    """
    Return 0.0 when the key is absent or its value is None.
    Otherwise coerce to float.
    This is the correct encoding for genuinely optional fields whose absence
    means 'no constraint' or 'no hardware' — 0.0 is not a silent default
    but the explicit numeric representation of that absence.
    """
    val = source.get(key)
    if val is None:
        return 0.0
    return _as_float(val, key)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(
    model_facts: dict[str, Any],
    deployment_profile: dict[str, Any],
) -> np.ndarray:
    """
    Extract a fixed-length, deterministic, purely numeric feature vector.

    Args:
        model_facts:        Flat dict of raw model analysis facts.
                            Accepted sources:
                              - ModelAnalysis.to_dict()
                              - analysis_state["model_facts"] (extended schema)
        deployment_profile: Flat dict of hardware/SLA configuration.
                            Accepted source: deployment_profile dict from route layer.

    Returns:
        np.ndarray of shape (FEATURE_COUNT,) with dtype float64.

    Raises:
        ValueError: If any required field is missing or has an invalid type/value.
    """
    if not isinstance(model_facts, dict):
        raise ValueError(
            f"feature_extractor: model_facts must be dict, got {type(model_facts).__name__}"
        )
    if not isinstance(deployment_profile, dict):
        raise ValueError(
            f"feature_extractor: deployment_profile must be dict, "
            f"got {type(deployment_profile).__name__}"
        )

    # ── Validate all required fields are present upfront ─────────────────────
    for field in _REQUIRED_MODEL_FIELDS:
        _require(model_facts, field, "model_facts")
    for field in _REQUIRED_PROFILE_FIELDS:
        _require(deployment_profile, field, "deployment_profile")

    # ── Build feature vector ──────────────────────────────────────────────────
    features: list[float] = [
        # 0: parameter_count
        _as_float(_require(model_facts, "parameter_count", "model_facts"),
                  "parameter_count"),
        # 1: operator_count
        _as_float(_require(model_facts, "operator_count", "model_facts"),
                  "operator_count"),
        # 2: has_dynamic_shapes → 0.0 / 1.0
        _as_bool_float(_require(model_facts, "has_dynamic_shapes", "model_facts"),
                       "has_dynamic_shapes"),
        # 3: parameter_scale_class → 0.0 / 1.0 / 2.0
        _encode_scale_class(_require(model_facts, "parameter_scale_class", "model_facts")),
        # 4: model_size_mb
        _as_float(_require(model_facts, "model_size_mb", "model_facts"),
                  "model_size_mb"),
        # 5: sequential_depth_estimate
        _as_float(_require(model_facts, "sequential_depth_estimate", "model_facts"),
                  "sequential_depth_estimate"),
        # 6: has_conv
        _as_bool_float(_require(model_facts, "has_conv", "model_facts"), "has_conv"),
        # 7: has_attention
        _as_bool_float(_require(model_facts, "has_attention", "model_facts"), "has_attention"),
        # 8: has_resize
        _as_bool_float(_require(model_facts, "has_resize", "model_facts"), "has_resize"),
        # 9: uses_batch_normalization
        _as_bool_float(_require(model_facts, "uses_batch_normalization", "model_facts"),
                       "uses_batch_normalization"),
        # 10: uses_layer_normalization
        _as_bool_float(_require(model_facts, "uses_layer_normalization", "model_facts"),
                       "uses_layer_normalization"),
        # 11: has_non_max_suppression
        _as_bool_float(_require(model_facts, "has_non_max_suppression", "model_facts"),
                       "has_non_max_suppression"),
        # 12: has_conv_transpose
        _as_bool_float(_require(model_facts, "has_conv_transpose", "model_facts"),
                       "has_conv_transpose"),
        # 13: cpu_cores
        _as_float(_require(deployment_profile, "cpu_cores", "deployment_profile"),
                  "cpu_cores"),
        # 14: ram_gb
        _as_float(_require(deployment_profile, "ram_gb", "deployment_profile"),
                  "ram_gb"),
        # 15: gpu_available
        _as_bool_float(_require(deployment_profile, "gpu_available", "deployment_profile"),
                       "gpu_available"),
        # 16: cuda_available
        _as_bool_float(_require(deployment_profile, "cuda_available", "deployment_profile"),
                       "cuda_available"),
        # 17: vram_gb (optional; 0.0 = no GPU)
        _opt_float(deployment_profile, "vram_gb"),
        # 18: target_latency_ms (optional; 0.0 = unconstrained)
        _opt_float(deployment_profile, "target_latency_ms"),
        # 19: memory_limit_mb (optional; 0.0 = unconstrained)
        _opt_float(deployment_profile, "memory_limit_mb"),
    ]

    # ── Structural assertion: always exactly FEATURE_COUNT elements ───────────
    if len(features) != FEATURE_COUNT:
        raise RuntimeError(
            f"feature_extractor: internal error — produced {len(features)} features, "
            f"expected {FEATURE_COUNT}. This is a bug in feature_extractor.py."
        )

    vec = np.array(features, dtype=np.float64)

    # ── Validate: no NaN or Inf values ────────────────────────────────────────
    if not np.all(np.isfinite(vec)):
        bad_indices = np.where(~np.isfinite(vec))[0].tolist()
        bad_names = [FEATURE_NAMES[i] for i in bad_indices]
        raise ValueError(
            f"feature_extractor: non-finite values in feature vector at indices "
            f"{bad_indices} ({bad_names}). Inputs must be finite real numbers."
        )

    return vec

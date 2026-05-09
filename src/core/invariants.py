"""
core/invariants.py

Global fail-fast mechanism and signal integrity enforcement.

DESIGN CONTRACT:
  - No silent failures anywhere in this module.
  - Every function raises immediately on violation.
  - All checks are lightweight — no heavy I/O, no logging overhead.
  - _run_internal_validation() validates the ML feature extractor and decision
    engine ONLY. All references to archived risk_based_decision removed.

PUBLIC API:
  invariant_fail(msg)           — global fail-fast
  validate_signals(signals)     — signal integrity (legacy, kept for callers)
  _run_internal_validation()    — ML system validation suite
  REQUIRED_SIGNALS              — kept for backward compat; no longer enforced on ML path
"""
from __future__ import annotations

import statistics
from typing import Any


# =============================================================================
# Global Fail-Fast Mechanism
# =============================================================================

def invariant_fail(msg: str) -> None:
    """
    Raise immediately on any invariant violation.
    Never returns. Always raises RuntimeError.
    """
    raise RuntimeError(f"Invariant violation: {msg}")


# =============================================================================
# Signal Integrity (kept for legacy callers; not on ML decision path)
# =============================================================================

REQUIRED_SIGNALS: frozenset = frozenset({
    "memory_risk",
    "latency_risk",
    "operator_compatibility_risk",
    "numerical_instability_risk",
    "device_capacity_risk",
})


def validate_signals(signals: Any) -> None:
    """
    Enforce signal integrity for legacy callers.

    Note: this is NOT called on the ML decision path.
    """
    if not isinstance(signals, dict):
        invariant_fail(f"signals must be a dict, got {type(signals).__name__}")
    if not signals:
        invariant_fail("signals cannot be empty")
    for k, v in signals.items():
        if not isinstance(v, (int, float)):
            invariant_fail(f"signal {k!r} has invalid type: {type(v).__name__}")
        if not (0.0 <= float(v) <= 1.0):
            invariant_fail(f"signal {k!r} out of bounds: {v!r}")
    missing = REQUIRED_SIGNALS - signals.keys()
    if missing:
        invariant_fail(f"missing required signals: {missing}")


# =============================================================================
# ML Decision Targets Validation
# =============================================================================

VALID_DEPLOYMENT_TARGETS: frozenset[str] = frozenset({
    "edge_int8",
    "edge_fp16",
    "cloud_cpu",
    "cloud_gpu",
})


def assert_valid_deployment_target(target: str) -> None:
    """
    Assert that a deployment target string is one of the four ML targets.
    Raises RuntimeError immediately via invariant_fail if violated.
    """
    if target not in VALID_DEPLOYMENT_TARGETS:
        invariant_fail(
            f"invalid deployment_decision: {target!r} not in "
            f"{sorted(VALID_DEPLOYMENT_TARGETS)}"
        )


# Keep old name as alias for any remaining callers
VALID_DECISIONS: frozenset = VALID_DEPLOYMENT_TARGETS


def assert_valid_decision(decision: str) -> None:
    """Legacy alias — delegates to assert_valid_deployment_target."""
    assert_valid_deployment_target(decision)


# =============================================================================
# Internal Validation Suite — ML Decision Engine
# =============================================================================

def _run_internal_validation() -> None:
    """
    Validate the ML decision engine and feature extractor at startup.

    Tests:
      1. feature_extractor produces correct shape and dtype.
      2. Different inputs produce different feature vectors.
      3. ml_decision_engine.predict returns a valid (label, confidence) pair.
      4. Different feature vectors can produce different decisions.
      5. Confidence is always in (0.0, 1.0].
      6. Output labels are always members of VALID_DEPLOYMENT_TARGETS.

    Raises RuntimeError immediately on any failure.
    """
    import numpy as np
    from src.core.feature_extractor import extract_features, FEATURE_COUNT
    from src.core.ml_decision_engine import predict

    # ── Representative model facts for two clearly different models ───────────
    _small_static_facts = {
        "parameter_count":           500_000,
        "operator_count":            20,
        "has_dynamic_shapes":        False,
        "parameter_scale_class":     "small",
        "model_size_mb":             2.0,
        "sequential_depth_estimate": 25,
        "has_conv":                  True,
        "has_attention":             False,
        "has_resize":                False,
        "uses_batch_normalization":  True,
        "uses_layer_normalization":  False,
        "has_non_max_suppression":   False,
        "has_conv_transpose":        False,
    }
    _large_dynamic_facts = {
        "parameter_count":           150_000_000,
        "operator_count":            400,
        "has_dynamic_shapes":        True,
        "parameter_scale_class":     "large",
        "model_size_mb":             600.0,
        "sequential_depth_estimate": 450,
        "has_conv":                  True,
        "has_attention":             True,
        "has_resize":                True,
        "uses_batch_normalization":  False,
        "uses_layer_normalization":  True,
        "has_non_max_suppression":   False,
        "has_conv_transpose":        False,
    }
    _cpu_profile = {
        "cpu_cores": 4, "ram_gb": 8.0,
        "gpu_available": False, "cuda_available": False,
    }
    _gpu_profile = {
        "cpu_cores": 16, "ram_gb": 32.0,
        "gpu_available": True, "cuda_available": True, "vram_gb": 8.0,
    }

    # ── 1. Feature extraction produces correct shape and dtype ────────────────
    fv1 = extract_features(_small_static_facts, _cpu_profile)
    if fv1.shape != (FEATURE_COUNT,):
        invariant_fail(
            f"feature_extractor produced shape {fv1.shape}, expected ({FEATURE_COUNT},)"
        )
    if fv1.dtype != np.float64:
        invariant_fail(
            f"feature_extractor dtype={fv1.dtype!r}, expected float64"
        )

    # ── 2. Different inputs → different feature vectors ───────────────────────
    fv2 = extract_features(_large_dynamic_facts, _gpu_profile)
    if np.array_equal(fv1, fv2):
        invariant_fail(
            "feature_extractor invariant violated: different inputs produced "
            "identical feature vectors."
        )

    # ── 3. ML engine returns valid (label, confidence) pairs ─────────────────
    for fv, name in ((fv1, "small/cpu"), (fv2, "large/gpu")):
        label, conf = predict(fv)
        if label not in VALID_DEPLOYMENT_TARGETS:
            invariant_fail(
                f"ml_decision_engine returned invalid label {label!r} for {name} input"
            )
        if not isinstance(conf, float):
            invariant_fail(
                f"ml_decision_engine confidence must be float, got {type(conf).__name__}"
            )
        if not (0.0 < conf <= 1.0):
            invariant_fail(
                f"ml_decision_engine confidence={conf!r} out of (0.0, 1.0] for {name}"
            )

    # ── 4. System is not constant — different inputs can yield different labels ─
    label1, _ = predict(fv1)
    label2, _ = predict(fv2)
    # We test 4 diverse inputs to guarantee non-constant output
    _test_vectors = [
        extract_features(_small_static_facts, _cpu_profile),
        extract_features(_large_dynamic_facts, _gpu_profile),
        extract_features(_small_static_facts, _gpu_profile),
        extract_features(_large_dynamic_facts, _cpu_profile),
    ]
    _all_labels = {predict(fv)[0] for fv in _test_vectors}
    if len(_all_labels) < 2:
        invariant_fail(
            f"ML system produces constant outputs {_all_labels!r} across "
            "diverse inputs. System is invalid."
        )

    # ── 5. Determinism — same input → identical output ────────────────────────
    l1a, c1a = predict(fv1)
    l1b, c1b = predict(fv1)
    # Labels must be identical; confidence allows ≤ 1e-12 floating-point rounding
    if l1a != l1b or abs(c1a - c1b) > 1e-12:
        invariant_fail(
            f"ml_decision_engine is non-deterministic: "
            f"({l1a!r}, {c1a!r}) != ({l1b!r}, {c1b!r})"
        )

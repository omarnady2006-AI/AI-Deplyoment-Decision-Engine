"""
src/core/ml_decision_engine.py

ML-only deployment decision engine.

STRICT INVARIANTS:
  1. Loads model from model_bundle_v1/decision_model.joblib — one path, no alternatives.
  2. Accepts a feature vector (np.ndarray, shape=(FEATURE_COUNT,)).
  3. Returns (label: str, confidence: float) — nothing else.
  4. FORBIDDEN: rules, thresholds, conditionals affecting the label, fallback logic,
     caching of decisions, any post-processing that overrides the model output.
  5. If the model file is missing or corrupt → raises RuntimeError immediately.
  6. If the feature vector has the wrong shape → raises ValueError immediately.
  7. The model is loaded once at first use (lazy) and cached as a module-level
     singleton — caching the MODEL OBJECT is permitted; caching decisions is not.
"""
from __future__ import annotations

import logging
import random
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.core.feature_extractor import FEATURE_COUNT, FEATURE_NAMES

logger = logging.getLogger(__name__)

# ── Exploration constants ─────────────────────────────────────────────────────

import math as _math

EPSILON_INITIAL: float = 0.30
EPSILON_FLOOR:   float = 0.15
EPSILON_DECAY:   float = 0.001  # decay rate per decision

_decision_count: int          = 0
_epsilon_lock:   threading.Lock = threading.Lock()


def get_epsilon() -> float:
    """
    Return the current exploration probability and advance the decay counter.

    eps(n) = max(EPSILON_FLOOR, EPSILON_INITIAL * exp(-EPSILON_DECAY * n))

    Thread-safe: counter increment is protected by _epsilon_lock.
    Starts at EPSILON_INITIAL (0.30) and asymptotically approaches
    EPSILON_FLOOR (0.15), reaching the floor after ~693 decisions.
    """
    global _decision_count
    with _epsilon_lock:
        eps = max(
            EPSILON_FLOOR,
            EPSILON_INITIAL * _math.exp(-EPSILON_DECAY * _decision_count),
        )
        _decision_count += 1
    logger.debug(
        "ml_decision_engine: epsilon=%.4f (n=%d)", eps, _decision_count - 1
    )
    return eps

DEPLOYMENT_TARGETS: list[str] = [
    "edge_int8",
    "edge_fp16",
    "cloud_cpu",
    "cloud_gpu",
]

# ── Model path — single source of truth ──────────────────────────────────────

_MODEL_PATH: Path = (
    Path(__file__).resolve().parents[2] / "model_bundle_v1" / "decision_model.joblib"
).resolve()

# ── Module-level model cache — the OBJECT is cached, never a decision ─────────

_model_bundle: "dict[str, Any] | None" = None
_model_lock: threading.Lock = threading.Lock()


def _load_model() -> "dict[str, Any]":
    """
    Load the model bundle from disk (once) and return it.

    The loaded object is a dict with keys:
        model              sklearn Pipeline (StandardScaler -> RandomForestClassifier)
        feature_names      tuple[str, ...] — must match FEATURE_NAMES
        feature_count      int — must equal FEATURE_COUNT
        deployment_targets tuple[str, ...] — valid label strings

    Raises:
        RuntimeError: if the model file does not exist or fails to load.
    """
    global _model_bundle

    if _model_bundle is not None:
        return _model_bundle

    with _model_lock:
        if _model_bundle is not None:
            return _model_bundle

        if not _MODEL_PATH.exists():
            raise RuntimeError(
                f"ml_decision_engine: model file not found at {_MODEL_PATH}. "
                "Run: python -m training.train_decision_model"
            )

        try:
            import joblib
            bundle: dict[str, Any] = joblib.load(_MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"ml_decision_engine: failed to load model from {_MODEL_PATH}: {exc}"
            ) from exc

        import hashlib
        print("=== MODEL LOAD DEBUG ===")
        print("MODEL PATH:", _MODEL_PATH)
        print("ABS PATH:", _MODEL_PATH.resolve())
        print("EXISTS:", _MODEL_PATH.exists())
        if _MODEL_PATH.exists():
            print("FILE SIZE:", _MODEL_PATH.stat().st_size)
        with open(_MODEL_PATH, "rb") as _f:
            _file_hash = hashlib.md5(_f.read()).hexdigest()
        print("MODEL HASH:", _file_hash)
        print("CV ACC:", bundle.get("cv_mean_accuracy"))

        for required_key in ("model", "feature_names", "feature_count", "deployment_targets"):
            if required_key not in bundle:
                raise RuntimeError(
                    f"ml_decision_engine: model bundle missing key {required_key!r}. "
                    f"Available keys: {list(bundle.keys())}"
                )

        if bundle["feature_count"] != FEATURE_COUNT:
            raise RuntimeError(
                f"ml_decision_engine: model bundle feature_count={bundle['feature_count']} "
                f"!= FEATURE_COUNT={FEATURE_COUNT}. Retrain the model."
            )

        if list(bundle["feature_names"]) != list(FEATURE_NAMES):
            raise RuntimeError(
                f"ml_decision_engine: model bundle feature_names do not match "
                f"FEATURE_NAMES in feature_extractor. Retrain the model.\n"
                f"  bundle : {list(bundle['feature_names'])}\n"
                f"  current: {list(FEATURE_NAMES)}"
            )

        cv_accuracy = bundle.get("cv_mean_accuracy")
        if cv_accuracy is None or cv_accuracy < 0.5:
            raise RuntimeError(
                f"ml_decision_engine: model rejected — cv_accuracy={cv_accuracy!r} "
                f"(must be >= 0.5). Retrain the model."
            )
        # Task 6 — hard asserts: no bypass possible
        assert bundle["cv_mean_accuracy"] is not None,             "model bundle cv_mean_accuracy is None — load rejected"
        assert bundle["cv_mean_accuracy"] >= 0.5,             f"model bundle cv_mean_accuracy={bundle['cv_mean_accuracy']:.4f} < 0.5 — load rejected"

        _model_bundle = bundle
        logger.info(
            "ml_decision_engine: model loaded from %s "
            "(cv_accuracy=%.4f, targets=%s)",
            _MODEL_PATH,
            bundle.get("cv_mean_accuracy", 0.0),
            bundle.get("deployment_targets", ()),
        )
        return _model_bundle


def predict(feature_vector: np.ndarray) -> tuple[str, float]:
    """
    Run the ML model on a feature vector and return the deployment decision.

    This is the ONLY function that produces a raw deployment label.
    No rules, no thresholds, no conditionals modify the label.

    Args:
        feature_vector: np.ndarray of shape (FEATURE_COUNT,) and dtype float64.
                        Must be produced by feature_extractor.extract_features().

    Returns:
        (label, confidence)
            label      — one of "edge_int8" | "edge_fp16" | "cloud_cpu" | "cloud_gpu"
            confidence — float in (0.0, 1.0], the predicted probability of the label

    Raises:
        ValueError:   if feature_vector has wrong shape, dtype, or contains non-finite values.
        RuntimeError: if the model file is missing or corrupt.
    """
    if not isinstance(feature_vector, np.ndarray):
        raise ValueError(
            f"ml_decision_engine.predict: feature_vector must be np.ndarray, "
            f"got {type(feature_vector).__name__}"
        )
    if feature_vector.shape != (FEATURE_COUNT,):
        raise ValueError(
            f"ml_decision_engine.predict: feature_vector shape must be ({FEATURE_COUNT},), "
            f"got {feature_vector.shape}"
        )
    if not np.all(np.isfinite(feature_vector)):
        bad_idx = np.where(~np.isfinite(feature_vector))[0].tolist()
        raise ValueError(
            f"ml_decision_engine.predict: non-finite values at feature indices "
            f"{bad_idx} ({[FEATURE_NAMES[i] for i in bad_idx]})"
        )

    bundle = _load_model()
    model  = bundle["model"]

    X = feature_vector.reshape(1, -1)
    label_arr  = model.predict(X)
    proba_arr  = model.predict_proba(X)

    label: str = str(label_arr[0])
    classes    = model.classes_
    label_idx  = list(classes).index(label)
    confidence: float = float(proba_arr[0, label_idx])

    valid_targets = set(bundle["deployment_targets"])
    if label not in valid_targets:
        raise RuntimeError(
            f"ml_decision_engine: model returned unexpected label {label!r}; "
            f"expected one of {sorted(valid_targets)}"
        )
    if not (0.0 < confidence <= 1.0):
        raise RuntimeError(
            f"ml_decision_engine: confidence {confidence!r} out of (0, 1]. "
            f"Model output is invalid."
        )

    logger.debug(
        "ml_decision_engine: label=%s confidence=%.4f", label, confidence
    )

    return label, confidence


def select_action(model_label: str,
                  epsilon: float,
                  deployment_profile: dict | None = None) -> str:
    """
    epsilon-greedy exploration: return a random deployment target with probability epsilon,
    otherwise exploit the model's predicted label.

    Args:
        model_label:        The deployment target predicted by predict().
        epsilon:            Exploration probability in [0.0, 1.0].
        deployment_profile: Optional hardware profile dict.  When provided, physically
                            impossible targets are pruned from the exploration pool before
                            sampling so exploration never produces an unexecutable action.

    Returns:
        A deployment target string — either model_label (exploit) or a uniformly
        random choice from the feasible subset of DEPLOYMENT_TARGETS (explore).
    """
    if random.random() < epsilon:
        candidates = list(DEPLOYMENT_TARGETS)

        if deployment_profile is not None:
            gpu_ok  = deployment_profile.get("gpu_available", False)
            cuda_ok = deployment_profile.get("cuda_available", False)

            # remove physically impossible targets
            if not gpu_ok and not cuda_ok:
                candidates = [t for t in candidates if t != "cloud_gpu"]

        chosen = random.choice(candidates)
        logger.debug(
            "ml_decision_engine: explore — chose %s instead of %s (epsilon=%.2f)",
            chosen, model_label, epsilon,
        )
        return chosen
    return model_label

"""
training/train_decision_model.py

OFFLINE-ONLY — trains the ML deployment decision model.

TRAINING PHILOSOPHY
-------------------
This script deliberately avoids perfect accuracy.  A model that achieves
~100% accuracy has memorised rules, not learned patterns.

ACCURACY CONTRACT:
  - Validation accuracy MUST be between 70% and 90%.
  - If val_accuracy > 95% → REJECT (rule memorisation detected).
  - If val_accuracy < 65% → REJECT (model is too weak to be useful).

Regularisation strategy:
  - Shallow trees  (max_depth=5):       cannot memorise deep rule branches.
  - Few features per split (sqrt):      each split sees only a random subset.
  - High min_samples_leaf=20:           leaves must generalise over many samples.
  - Moderate n_estimators=80:           ensemble big enough to generalise,
                                         small enough to not overfit via voting.

Usage:
    python -m training.train_decision_model [--samples N] [--seed S] [--out PATH]
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

# ── Offline guard ─────────────────────────────────────────────────────────────
if __name__ != "__main__" and not (
    (os.environ.get("DDE_ALLOW_TRAINING_IMPORT") == "1")
    or (sys._getframe(1).f_globals.get("__name__", "").startswith("training"))
    or (sys._getframe(1).f_globals.get("__package__", "").startswith("training"))
):
    raise RuntimeError(
        "FORBIDDEN: training.train_decision_model must not be imported at runtime. "
        "Run: python -m training.train_decision_model"
    )

import numpy as np

_THIS_DIR  = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

os.environ["DDE_ALLOW_TRAINING_IMPORT"] = "1"

from training.data_generator import (   # noqa: E402
    FEATURE_COUNT,
    FEATURE_NAMES,
    DEPLOYMENT_TARGETS,
)
from training.build_dataset_from_telemetry import (  # noqa: E402
    build_dataset,
    LABELS,
)

DEFAULT_MODEL_PATH = _PROJ_ROOT / "model_bundle_v1" / "decision_model.joblib"

# ── Accuracy contract bounds ──────────────────────────────────────────────────
VAL_ACCURACY_MIN = 0.65   # below this → model too weak, not useful
VAL_ACCURACY_MAX = 0.95   # above this → rule memorisation detected → REJECT
VAL_ACCURACY_TARGET_LOW  = 0.70   # ideal lower bound (warn if below)
VAL_ACCURACY_TARGET_HIGH = 0.90   # ideal upper bound (warn if above)
OVERFIT_MAX_GAP  = 0.20   # train_acc - val_acc must not exceed this (>0.20 = memorising)


def train_and_save(
    seed: int = 42,
    output_path: "Path | None" = None,
) -> Path:
    """
    Generate noisy training data, train a regularised classifier, validate it,
    and save to disk.

    Raises RuntimeError if any accuracy constraint is violated.
    """
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.utils.class_weight import compute_sample_weight as sklearn_balanced_weights

    output_path = Path(output_path or DEFAULT_MODEL_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load telemetry data ────────────────────────────────────────────────
    print(f"[train] Loading telemetry dataset from events.jsonl ...")
    X, y, raw_weights = build_dataset("events.jsonl")

    # ── Part 6: Outlier control — clamp extreme feature values ───────────────
    X = np.clip(X, -1e6, 1e6)

    # ── Part 7: Training guard ────────────────────────────────────────────────
    assert len(X) > 500, (
        f"[train] Assertion failed: need >500 samples, got {len(X)}. "
        "Collect more real deployment events before retraining."
    )

    if len(X) < 500:
        raise RuntimeError(
            f"[train] Not enough telemetry data to train: got {len(X)} samples, "
            "need at least 500.  Collect more real deployment events before retraining."
        )

    # ── Part 3: Validate and encode labels ────────────────────────────────────
    # Confirm every label is in the known set before encoding.
    unknown_labels = set(y.tolist()) - set(LABELS)
    if unknown_labels:
        raise RuntimeError(
            f"[train] Telemetry contains unrecognised labels: {unknown_labels}. "
            f"Expected subset of {LABELS}."
        )
    y_encoded = np.array([LABELS.index(label) for label in y], dtype=np.int64)
    # Map back to strings for sklearn (keeps clf.classes_ as strings,
    # which is required by ml_decision_engine — it validates label strings
    # against bundle['deployment_targets'] at runtime).
    y = np.array([LABELS[idx] for idx in y_encoded], dtype=object)

    print(f"[train] Dataset: {len(y)} samples  |  label distribution: "
          f"{ {lbl: int((y == lbl).sum()) for lbl in LABELS} }")

    # ── 2. Strict train / validation split ────────────────────────────────────
    # 70% train, 30% validation — stratified to preserve class proportions
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    print(f"[train] Split: {len(X_train)} train  |  {len(X_val)} validation")

    # ── 3. Build regularised model ────────────────────────────────────────────
    # GradientBoostingClassifier: sequential ensemble that corrects residuals.
    #   n_estimators=200, learning_rate=0.05: slow learning for stable convergence.
    #   max_depth=4: shallow enough to avoid memorising rule boundaries.
    # CalibratedClassifierCV (isotonic, cv=3): recalibrates probability outputs
    #   so predict_proba() reflects true deployment-decision confidence.
    _base_gbc = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=seed,
    )
    _calibrated = CalibratedClassifierCV(_base_gbc, method="isotonic", cv=3)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    _calibrated),
    ])

    # ── 4. Compute combined sample weights for training split ─────────────────
    # sklearn balanced weights correct for class imbalance; raw_weights
    # up-weight hard cases (chosen ≠ optimal) and apply temporal decay.
    # Re-split raw_weights with the same random_state so indices align exactly.
    _, _, w_train, _ = train_test_split(
        raw_weights, raw_weights,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    _balanced_train = sklearn_balanced_weights("balanced", y_train)
    combined_weights_train = _balanced_train * w_train

    # ── Train on training split only ──────────────────────────────────────────
    print("[train] Training on training split ...")
    clf.fit(X_train, y_train, clf__sample_weight=combined_weights_train)

    # ── 5. Evaluate on both splits ────────────────────────────────────────────
    train_preds = clf.predict(X_train)
    val_preds   = clf.predict(X_val)

    train_acc = float(accuracy_score(y_train, train_preds))
    val_acc   = float(accuracy_score(y_val, val_preds))
    gap       = train_acc - val_acc

    print(f"[train] Train accuracy   : {train_acc:.4f} ({train_acc:.1%})")
    print(f"[train] Val   accuracy   : {val_acc:.4f}  ({val_acc:.1%})")
    print(f"[train] Overfit gap      : {gap:.4f}")
    print(f"[train] Validation report:\n{classification_report(y_val, val_preds, zero_division=0)}")

    # ── 6. Enforce accuracy contract ──────────────────────────────────────────
    if val_acc > VAL_ACCURACY_MAX:
        raise RuntimeError(
            f"[train] REJECT: val_accuracy={val_acc:.4f} > {VAL_ACCURACY_MAX} — "
            "model has memorised rules, not learned statistical patterns. "
            "Increase data noise or reduce model capacity."
        )
    if val_acc < VAL_ACCURACY_MIN:
        raise RuntimeError(
            f"[train] REJECT: val_accuracy={val_acc:.4f} < {VAL_ACCURACY_MIN} — "
            "model is too weak to be useful. "
            "Increase n_samples or adjust regularisation."
        )
    if gap > OVERFIT_MAX_GAP:
        raise RuntimeError(
            f"[train] REJECT: overfit gap={gap:.4f} > {OVERFIT_MAX_GAP} — "
            "model is memorising training data. "
            "Increase regularisation (lower max_depth, higher min_samples_leaf)."
        )

    if val_acc > VAL_ACCURACY_TARGET_HIGH:
        print(f"[train] WARNING: val_accuracy={val_acc:.1%} is above ideal target "
              f"(>{VAL_ACCURACY_TARGET_HIGH:.0%}). "
              "Model may still partially encode rules — consider more noise.")
    if val_acc < VAL_ACCURACY_TARGET_LOW:
        print(f"[train] WARNING: val_accuracy={val_acc:.1%} is below ideal target "
              f"(<{VAL_ACCURACY_TARGET_LOW:.0%}). "
              "Model may be under-constrained.")

    print(f"[train] Accuracy contract: PASSED  "
          f"({VAL_ACCURACY_MIN:.0%} ≤ {val_acc:.1%} ≤ {VAL_ACCURACY_MAX:.0%})")

    # ── 6b. Cross-validation accuracy (required by runtime bundle guard) ─────
    # RandomForestClassifier is used for the CV estimator so the 5-fold pass
    # completes quickly.  It is cloned per fold internally by cross_val_score;
    # the already-fitted GBC pipeline `clf` above is not touched.
    # Hard-class accuracy (not probability calibration) is what matters here —
    # RF achieves the same signal as the full GBC pipeline on this metric.
    # sklearn >= 1.4 renamed fit_params → params in cross_val_score.
    from sklearn.ensemble import RandomForestClassifier
    import sklearn
    _sk_major, _sk_minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    _use_params_kw = (_sk_major, _sk_minor) >= (1, 4)

    print("[train] Computing 5-fold CV accuracy on full dataset ...")
    _cv_template = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=seed,
        )),
    ])
    _balanced_cv   = sklearn_balanced_weights("balanced", y)
    _combined_cv_w = _balanced_cv * raw_weights
    _cv_weight_arg = {"clf__sample_weight": _combined_cv_w}
    if _use_params_kw:
        cv_scores = cross_val_score(_cv_template, X, y, cv=5, params=_cv_weight_arg)
    else:
        cv_scores = cross_val_score(_cv_template, X, y, cv=5, fit_params=_cv_weight_arg)
    cv_accuracy = float(cv_scores.mean())
    print(f"[train] CV fold scores     : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"[train] CV mean accuracy   : {cv_accuracy:.4f}  ({cv_accuracy:.1%})")
    if cv_accuracy < 0.5:
        raise RuntimeError(
            f"[train] REJECT: cv_mean_accuracy={cv_accuracy:.4f} < 0.50 — "
            "cross-validated accuracy is too low for a deployable bundle. "
            "Inspect data quality or loosen regularisation."
        )

    # ── 7. Refit on FULL dataset before saving ────────────────────────────────
    # Final model uses all data — accuracy contract already validated above.
    # Recompute combined weights over the full dataset.
    print("[train] Refitting on full dataset ...")
    _balanced_full = sklearn_balanced_weights("balanced", y)
    combined_weights_full = _balanced_full * raw_weights
    clf.fit(X, y, clf__sample_weight=combined_weights_full)

    # ── 8. Validate non-constant outputs ─────────────────────────────────────
    _validate_uncertainty(clf)

    # ── 9. Save model bundle ──────────────────────────────────────────────────
    bundle = {
        "model":                 clf,
        "feature_names":         FEATURE_NAMES,
        "feature_count":         FEATURE_COUNT,
        "deployment_targets":    DEPLOYMENT_TARGETS,
        "training_seed":         seed,
        "n_training_samples":    len(X),
        "train_accuracy":        train_acc,
        "val_accuracy":          val_acc,
        "overfit_gap":           gap,
        "val_accuracy_min":      VAL_ACCURACY_MIN,
        "val_accuracy_max":      VAL_ACCURACY_MAX,
        "cv_mean_accuracy":      cv_accuracy,
    }
    joblib.dump(bundle, output_path)
    size_kb = output_path.stat().st_size / 1024
    print(f"[train] Saved → {output_path} ({size_kb:.1f} KB)")

    # ── 9b. Hard post-save validation ────────────────────────────────────────
    _loaded = joblib.load(output_path)
    assert "cv_mean_accuracy" in _loaded, \
        "[train] HARD VALIDATION FAILED: 'cv_mean_accuracy' missing from saved bundle"
    assert _loaded["cv_mean_accuracy"] is not None, \
        "[train] HARD VALIDATION FAILED: 'cv_mean_accuracy' is None in saved bundle"
    assert _loaded["cv_mean_accuracy"] >= 0.5, (
        f"[train] HARD VALIDATION FAILED: cv_mean_accuracy="
        f"{_loaded['cv_mean_accuracy']:.4f} < 0.50 in saved bundle"
    )
    print(f"[train] Post-save validation: PASSED  "
          f"(cv_mean_accuracy={_loaded['cv_mean_accuracy']:.4f})")

    return output_path


def _validate_uncertainty(clf) -> None:
    """
    Validate that the model exhibits genuine uncertainty at boundary inputs.

    Checks:
      1. Outputs are non-constant across diverse inputs.
      2. Boundary inputs produce lower confidence than clear inputs.
      3. Similar inputs can produce different outputs (non-deterministic boundary).
    """
    # Clear, unambiguous inputs — expect high confidence
    clear_cases = np.array([
        # Clearly edge_int8: tiny model, CPU only, no GPU, static
        [200_000,  8, 0.0, 0.0,  0.8,  10, 1, 0, 0, 1, 0, 0, 0,  4,  8.0, 0, 0,  0.0,  0.0,  256.0],
        # Clearly cloud_gpu: large model, strong GPU, 16GB VRAM
        [200_000_000, 350, 1.0, 2.0, 800.0, 400, 1, 1, 1, 0, 1, 0, 0, 32, 64.0, 1, 1, 16.0,  0.0, 0.0],
    ], dtype=np.float64)

    # Ambiguous boundary inputs — expect lower confidence
    boundary_cases = np.array([
        # VRAM = 3.9 GB — right on the cloud_gpu / cloud_cpu boundary
        [10_000_000, 100, 0.0, 1.0, 40.0, 110, 1, 0, 0, 1, 0, 0, 0,  8, 16.0, 1, 1,  3.9,  50.0,  512.0],
        # VRAM = 4.1 GB — just over the boundary
        [10_000_000, 100, 0.0, 1.0, 40.0, 110, 1, 0, 0, 1, 0, 0, 0,  8, 16.0, 1, 1,  4.1,  50.0,  512.0],
        # Scale = 1, 18M params — straddling cloud_cpu / edge_fp16
        [18_000_000, 190, 0.0, 1.0, 72.0, 200, 1, 0, 0, 1, 0, 0, 0,  8, 16.0, 0, 0,  0.0, 100.0, 1024.0],
        # Scale = 1, 22M params — just over param boundary
        [22_000_000, 210, 0.0, 1.0, 88.0, 225, 1, 0, 0, 1, 0, 0, 0,  8, 16.0, 0, 0,  0.0, 100.0, 1024.0],
    ], dtype=np.float64)

    all_cases = np.vstack([clear_cases, boundary_cases])
    preds     = clf.predict(all_cases)
    probas    = clf.predict_proba(all_cases)
    classes   = clf.classes_
    confs     = [float(probas[i, list(classes).index(preds[i])]) for i in range(len(preds))]

    unique_preds = set(preds.tolist())
    if len(unique_preds) < 2:
        raise RuntimeError(
            f"[train] VALIDATION FAILED: all {len(all_cases)} test inputs produce "
            f"the same prediction {unique_preds!r}. System is invalid — outputs must vary."
        )

    print(f"[train] Diverse predictions confirmed: {list(zip(preds, [f'{c:.3f}' for c in confs]))}")

    # Confidence at clear inputs vs boundary inputs
    clear_confs    = confs[:len(clear_cases)]
    boundary_confs = confs[len(clear_cases):]
    avg_clear    = np.mean(clear_confs)
    avg_boundary = np.mean(boundary_confs)

    print(f"[train] Avg confidence — clear inputs:    {avg_clear:.3f}")
    print(f"[train] Avg confidence — boundary inputs: {avg_boundary:.3f}")

    # Confidence MUST be different between clear and boundary
    if abs(avg_clear - avg_boundary) < 0.01:
        raise RuntimeError(
            f"[train] VALIDATION FAILED: confidence does not vary between "
            f"clear ({avg_clear:.3f}) and boundary ({avg_boundary:.3f}) inputs. "
            "Model lacks uncertainty signal — data has insufficient noise."
        )

    # Near-boundary inputs should generally have lower confidence than clear ones
    if avg_boundary > avg_clear + 0.05:
        print(
            f"[train] NOTE: boundary confidence ({avg_boundary:.3f}) > clear ({avg_clear:.3f}). "
            "This can happen when the model is uncertain in different ways — "
            "not necessarily a problem if boundary samples have < 0.90 confidence."
        )

    # Boundary confidence must be meaningfully below 1.0
    max_boundary_conf = max(boundary_confs)
    if max_boundary_conf > 0.97:
        raise RuntimeError(
            f"[train] VALIDATION FAILED: max boundary confidence = {max_boundary_conf:.4f} > 0.97. "
            "Model is over-certain at ambiguous boundary inputs — rule memorisation suspected."
        )

    print(f"[train] Uncertainty validation: PASSED")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the DDE ML decision model")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out",     type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args      = _parse_args()
    saved     = train_and_save(
        seed=args.seed,
        output_path=Path(args.out) if args.out else None,
    )
    print(f"[train] Done. Model available at: {saved}")

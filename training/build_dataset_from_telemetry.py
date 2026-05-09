"""
training/build_dataset_from_telemetry.py

OFFLINE-ONLY — builds a supervised training dataset from real telemetry.

LABEL SOURCE
------------
Labels come EXCLUSIVELY from the `optimal_action` field written by
compute_optimal_action() in src/core/telemetry.py.  That function derives
the optimal deployment target from real execution outcomes (latency, memory,
execution_success) — never from chosen_action or model predictions.

INVARIANTS
----------
- chosen_action is NEVER read.
- synthetic data is NEVER used.
- rule-based or heuristic labels are NEVER applied.
- Shadow events (shadow=True) ARE included: they provide counterfactual
  outcomes from background benchmarks and are legitimate training signal.

FILTER CRITERIA (all three must hold)
--------------------------------------
- schema_version == 3
- label_finalized == True
- len(evaluated_actions) == 4

FEATURE ORDER
-------------
Must exactly match src/core/feature_extractor.FEATURE_NAMES (20 features):
  0  parameter_count
  1  operator_count
  2  has_dynamic_shapes
  3  parameter_scale_encoded
  4  model_size_mb
  5  sequential_depth
  6  has_conv
  7  has_attention
  8  has_resize
  9  uses_batch_norm
  10 uses_layer_norm
  11 has_nms
  12 has_conv_transpose
  13 cpu_cores
  14 ram_gb
  15 gpu_available
  16 cuda_available
  17 vram_gb
  18 target_latency_ms
  19 memory_limit_mb
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Offline guard ─────────────────────────────────────────────────────────────
if __name__ != "__main__" and not (
    (os.environ.get("DDE_ALLOW_TRAINING_IMPORT") == "1")
    or (sys._getframe(1).f_globals.get("__name__", "").startswith("training"))
    or (sys._getframe(1).f_globals.get("__package__", "").startswith("training"))
):
    raise RuntimeError(
        "FORBIDDEN: training.build_dataset_from_telemetry must not be imported "
        "at runtime.  Run: python -m training.train_decision_model"
    )

# ── Constants ─────────────────────────────────────────────────────────────────

LABELS: list[str] = ["edge_int8", "edge_fp16", "cloud_cpu", "cloud_gpu"]

# Feature names in the exact order they appear in the feature vector.
# Must stay in sync with src/core/feature_extractor.FEATURE_NAMES.
_FEATURE_NAMES: tuple[str, ...] = (
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

_FEATURE_COUNT: int = len(_FEATURE_NAMES)
assert _FEATURE_COUNT == 20


# ── Sample weighting helpers ──────────────────────────────────────────────────

def _compute_sample_weight(event: dict) -> float:
    """Up-weight events where the model's chosen action differed from optimal."""
    if event.get("optimal_action") != event.get("chosen_action"):
        return 2.0
    return 1.0


def _temporal_weight(
    timestamp: "datetime | None",
    now: datetime,
    tau_days: float = 30,
) -> float:
    """Exponential decay: recent events count more than old ones."""
    if timestamp is None:
        return 1.0
    delta = (now - timestamp).total_seconds()
    tau = tau_days * 24 * 3600
    return float(np.exp(-delta / tau))


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(log_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a supervised training dataset from a telemetry JSONL file.

    Args:
        log_path: Path to the events.jsonl file written by log_analysis_event().

    Returns:
        X:       float64 array of shape (N, 20) — feature vectors.
        y:       object  array of shape (N,)    — optimal_action strings.
        weights: float64 array of shape (N,)    — per-sample training weights
                 combining hard-case up-weighting and temporal decay.

    Raises:
        FileNotFoundError: if log_path does not exist.
        ValueError:        if a passed event has an unrecognised feature key or
                           an optimal_action value outside LABELS.
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"build_dataset: telemetry file not found: {path}")

    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    w_rows: list[float] = []
    n_skipped: int = 0
    now: datetime = datetime.now(tz=timezone.utc)

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue

            try:
                event: dict = json.loads(raw)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            # ── Filter ────────────────────────────────────────────────────────
            if event.get("schema_version") != 3:
                n_skipped += 1
                continue
            if not event.get("label_finalized", False):
                n_skipped += 1
                continue
            evaluated_actions = event.get("evaluated_actions", [])
            if len(evaluated_actions) != 4:
                n_skipped += 1
                continue

            # ── Data-quality filters ───────────────────────────────────────────
            # 1. Remove low-coverage events (unreliable benchmark conditions).
            if event.get("low_coverage"):
                n_skipped += 1
                continue

            # 2. Require at least 2 actions with confirmed successful execution.
            if sum(a.get("execution_success", False) for a in evaluated_actions) < 2:
                n_skipped += 1
                continue

            # 3. Remove noisy labels: skip when the two fastest actions are
            #    within 5 % of each other (ambiguous optimal target).
            latencies = [
                a["latency_ms"]
                for a in evaluated_actions
                if a.get("latency_ms") is not None
            ]
            if len(latencies) >= 2:
                lat_sorted = sorted(latencies)
                if abs(lat_sorted[0] - lat_sorted[1]) < 0.05 * lat_sorted[0]:
                    n_skipped += 1
                    continue

            # ── Label — ONLY from optimal_action ──────────────────────────────
            optimal_action: str = event.get("optimal_action", "")
            if optimal_action not in LABELS:
                n_skipped += 1
                continue

            # ── Features ──────────────────────────────────────────────────────
            features_dict: dict = event.get("features", {})
            if not features_dict:
                n_skipped += 1
                continue

            try:
                feature_row: list[float] = [
                    float(features_dict[name])
                    for name in _FEATURE_NAMES
                ]
            except KeyError as exc:
                raise ValueError(
                    f"build_dataset: event at line {lineno} is missing feature "
                    f"key {exc}.  Check that the telemetry schema version matches "
                    f"src/core/feature_extractor.FEATURE_NAMES."
                ) from exc

            # chosen_action is never read — skip it entirely.
            # shadow events are INCLUDED (counterfactual signal).

            # ── Per-sample weight ─────────────────────────────────────────────
            ts_raw = event.get("timestamp")
            try:
                ts = (
                    datetime.fromisoformat(ts_raw).replace(tzinfo=timezone.utc)
                    if ts_raw and not isinstance(ts_raw, datetime)
                    else (ts_raw if isinstance(ts_raw, datetime) else None)
                )
            except (ValueError, TypeError):
                ts = None
            weight = _compute_sample_weight(event) * _temporal_weight(ts, now)

            X_rows.append(feature_row)
            y_rows.append(optimal_action)
            w_rows.append(weight)

    if n_skipped:
        print(
            f"[build_dataset] Skipped {n_skipped} events "
            f"(schema mismatch / unfinalized / malformed)."
        )

    if not X_rows:
        raise RuntimeError(
            "build_dataset: no valid events found in telemetry log. "
            "Ensure schema_version=3, label_finalized=True, and "
            "len(evaluated_actions)==4 before training."
        )

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=object)
    weights = np.array(w_rows, dtype=np.float64)

    print(
        f"[build_dataset] Loaded {len(y)} samples  |  "
        f"features: {X.shape[1]}  |  "
        f"label distribution: { {lbl: int((y == lbl).sum()) for lbl in LABELS} }"
    )

    return X, y, weights

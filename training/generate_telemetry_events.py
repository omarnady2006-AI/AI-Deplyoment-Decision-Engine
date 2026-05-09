"""
training/generate_telemetry_events.py — generates clean, learnable events.jsonl

Labels come from argmax(_soft_label_probs) + 5% noise.
All events satisfy build_dataset_from_telemetry filters.
"""
from __future__ import annotations

import json, os, random, sys, uuid, argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

_THIS_DIR  = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

os.environ["DDE_ALLOW_TRAINING_IMPORT"] = "1"

import numpy as np
from training.data_generator import (
    FEATURE_NAMES, DEPLOYMENT_TARGETS,
    _sample_core_features, _soft_label_probs,
)

LABELS           = list(DEPLOYMENT_TARGETS)
LABEL_NOISE_RATE = 0.05

_BASE_LATENCY = {"edge_int8": 8.0, "edge_fp16": 15.0, "cloud_cpu": 45.0, "cloud_gpu": 5.0}


def _argmax_label(probs):
    return max(probs, key=lambda k: probs[k])

def _flip_label(label, rng):
    return rng.choice([l for l in LABELS if l != label])

def _make_actions(optimal, rng):
    opt_lat = _BASE_LATENCY[optimal] * rng.uniform(0.85, 1.15)
    actions = []
    for label in LABELS:
        lat     = opt_lat if label == optimal else opt_lat * rng.uniform(1.08, 3.5)
        success = True if label == optimal else rng.random() < 0.65
        actions.append({"action": label, "latency_ms": round(lat, 3),
                        "memory_mb": round(rng.uniform(50, 2048), 1),
                        "execution_success": success, "outcome_type": "measured"})
    if sum(a["execution_success"] for a in actions) < 2:
        sorted(actions, key=lambda a: a["latency_ms"])[1]["execution_success"] = True
    return actions


def generate_events(n_samples=6000, seed=42, out_path=None):
    out_path = out_path or (_PROJ_ROOT / "events.jsonl")
    print(f"[gen] Generating {n_samples} events (seed={seed}) …")
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    now = datetime.now(tz=timezone.utc)
    counts = {l: 0 for l in LABELS}
    lines = []
    for _ in range(n_samples):
        feat  = _sample_core_features(rng, nprng)
        label = _argmax_label(_soft_label_probs(feat))
        if rng.random() < LABEL_NOISE_RATE:
            label = _flip_label(label, rng)
        counts[label] += 1
        ts = (now - timedelta(seconds=rng.uniform(0, 60*24*3600))).isoformat()
        event = {
            "schema_version": 3, "event_id": str(uuid.uuid4()),
            "group_id": str(uuid.uuid4()), "hardware_context_id": uuid.uuid4().hex[:16],
            "timestamp_utc": ts, "timestamp": ts,
            "features": {name: float(feat[j]) for j, name in enumerate(FEATURE_NAMES)},
            "optimal_action": label, "model_prediction": rng.choice(LABELS),
            "chosen_action": label, "confidence": round(rng.uniform(0.45, 0.95), 4),
            "is_exploration": rng.random() < 0.15, "label_finalized": True,
            "low_coverage": False, "evaluated_actions": _make_actions(label, rng),
        }
        lines.append(json.dumps(event))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(counts.values())
    print(f"[gen] Labels: { {k: f'{v}({v/total:.0%})' for k,v in counts.items()} }")
    print(f"[gen] Wrote {len(lines)} events → {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=6000)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out",     type=str, default=None)
    a = p.parse_args()
    generate_events(a.samples, a.seed, Path(a.out) if a.out else None)

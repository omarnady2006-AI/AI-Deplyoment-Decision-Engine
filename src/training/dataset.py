"""
src/training/dataset.py

PyTorch Dataset for training the DeploymentDecisionNet.

INPUT SOURCES
=============
1. Benchmark Parquet file (from data_collector.py):
   Each row: model_path, runtime, latency_ms, memory_mb, failed, best_runtime, ...

2. Model feature dicts (from existing feature_extractor.extract_features):
   Each model_path → 20-dim float64 array.

FEATURE VECTOR (33 dimensions)
===============================
The extended feature vector concatenates:
  [0:20]   model/hardware features  (from feature_extractor.FEATURE_NAMES)
  [20:24]  latency_ms per runtime   (4 selected runtimes: ONNX_CPU, ONNX_CUDA, TF_CPU, TFLite_CPU)
  [24:28]  memory_mb per runtime    (same 4 runtimes)
  [28:32]  failed (0/1) per runtime (same 4 runtimes)
  [32]     benchmark_available (0.0 if no benchmarks ran, 1.0 otherwise)

Benchmark signals are normalised (log1p for latency/memory, binary for failed).
Missing runtime results are filled with MAX_LATENCY_MS / MAX_MEMORY_MB / 1.0.

TARGETS
=======
  runtime_label   : int in [0, N_RUNTIMES)  — classification
  latency_target  : float[N_RUNTIMES]       — log(ms+1) per runtime; log(MAX+1) if failed
  memory_target   : float[N_RUNTIMES]       — log(MB+1) per runtime; log(MAX+1) if failed
  failure_target  : float[N_RUNTIMES]       — 0.0 or 1.0 per runtime
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RUNTIMES: tuple[str, ...] = (
    "ONNX_CPU",
    "ONNX_CUDA",
    "TensorRT",
    "TFLite_CPU",
    "TFLite_GPU",
    "TF_CPU",
    "TF_GPU",
    "TensorRT_Native",
    "OpenVINO_CPU",
    "OpenVINO_GPU",
)
N_RUNTIMES = len(RUNTIMES)
RUNTIME_TO_IDX = {r: i for i, r in enumerate(RUNTIMES)}

# Benchmark signal runtimes (subset used as features — pick the 4 most common)
BENCHMARK_SIGNAL_RUNTIMES: tuple[str, ...] = (
    "ONNX_CPU",
    "ONNX_CUDA",
    "TF_CPU",
    "TFLite_CPU",
)
N_BENCHMARK_SIGNAL_RUNTIMES = len(BENCHMARK_SIGNAL_RUNTIMES)

# FEATURE_DIM = 20 (model/hw) + 4×3 (benchmark signals) + 1 (benchmark_available)
FEATURE_DIM = 20 + N_BENCHMARK_SIGNAL_RUNTIMES * 3 + 1   # = 33

from src.training.data_collector import MAX_LATENCY_MS, MAX_MEMORY_MB


class DeploymentDataset(Dataset):
    """
    PyTorch Dataset mapping each model instance to a multi-target training example.

    One example per unique model. Each example contains:
      - features (33,)
      - runtime_label: integer class index
      - latency_target: (N_RUNTIMES,) log-normalised
      - memory_target: (N_RUNTIMES,) log-normalised
      - failure_target: (N_RUNTIMES,) binary

    Args:
        parquet_path:    Path to benchmark Parquet file from DataCollector.
        model_features:  Dict mapping model_path (str) → np.ndarray (20,) float64.
                         These are the outputs of feature_extractor.extract_features().
        augment:         If True, apply mild Gaussian noise to continuous features
                         during __getitem__ (training only).
    """

    def __init__(
        self,
        parquet_path: Path,
        model_features: dict[str, np.ndarray],
        augment: bool = False,
    ) -> None:
        import pandas as pd

        self.augment = augment
        df = pd.read_parquet(parquet_path)

        # ── Validate required columns ─────────────────────────────────────────
        required_cols = {"model_path", "runtime", "latency_ms", "memory_mb",
                         "failed", "best_runtime"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DeploymentDataset: missing columns: {missing}")

        # ── Validate runtime labels ───────────────────────────────────────────
        invalid_runtimes = set(df["best_runtime"].unique()) - set(RUNTIMES)
        if invalid_runtimes:
            raise ValueError(
                f"DeploymentDataset: best_runtime contains unknown values: {invalid_runtimes}. "
                f"Expected subset of: {RUNTIMES}"
            )

        # ── Build per-model pivot: one row per model ──────────────────────────
        # pivot: model_path → {runtime → {latency_ms, memory_mb, failed}}
        self.model_paths = sorted(df["model_path"].unique())
        unknown_models = set(self.model_paths) - set(model_features.keys())
        if unknown_models:
            logger.warning(
                "DeploymentDataset: %d models in parquet have no feature dict — skipping.",
                len(unknown_models),
            )
            self.model_paths = [mp for mp in self.model_paths if mp in model_features]

        self.model_features = model_features

        # Group benchmark data by model_path
        self.benchmark_by_model: dict[str, dict[str, dict[str, float]]] = {}
        self.label_by_model: dict[str, int] = {}

        for mp in self.model_paths:
            subset = df[df["model_path"] == mp]
            rt_data: dict[str, dict[str, float]] = {}
            for _, row in subset.iterrows():
                rt = row["runtime"]
                rt_data[rt] = {
                    "latency_ms": float(row["latency_ms"]),
                    "memory_mb":  float(row["memory_mb"]),
                    "failed":     float(bool(row["failed"])),
                }
            self.benchmark_by_model[mp] = rt_data
            best = subset["best_runtime"].iloc[0]
            self.label_by_model[mp] = RUNTIME_TO_IDX[best]

        logger.info(
            "DeploymentDataset: loaded %d model examples, %d runtimes, %d features",
            len(self.model_paths), N_RUNTIMES, FEATURE_DIM,
        )

    def __len__(self) -> int:
        return len(self.model_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mp = self.model_paths[idx]
        base_features = self.model_features[mp].copy()   # (20,) float64

        rt_data = self.benchmark_by_model[mp]
        benchmark_available = 1.0 if rt_data else 0.0

        # ── Build benchmark signal features (12 + 1 = 13 dims) ───────────────
        bench_signals = []
        for rt in BENCHMARK_SIGNAL_RUNTIMES:
            if rt in rt_data:
                d = rt_data[rt]
                lat = d["latency_ms"] if not d["failed"] else MAX_LATENCY_MS
                mem = d["memory_mb"]  if not d["failed"] else MAX_MEMORY_MB
                fail = d["failed"]
            else:
                lat  = MAX_LATENCY_MS
                mem  = MAX_MEMORY_MB
                fail = 1.0
            bench_signals.extend([
                np.log1p(lat),    # log-normalised latency
                np.log1p(mem),    # log-normalised memory
                float(fail),      # binary failed
            ])
        bench_signals.append(benchmark_available)

        features = np.concatenate([
            base_features.astype(np.float32),
            np.array(bench_signals, dtype=np.float32),
        ])   # (FEATURE_DIM,) = (33,)

        # ── Optional training augmentation ────────────────────────────────────
        if self.augment:
            # Continuous features: mild Gaussian noise (std = 1% of feature range)
            # Boolean features (indices 2,6,7,8,9,10,11,12,15,16,28,29,30,31):
            # small random flip probability
            noise = np.random.randn(*features.shape).astype(np.float32) * 0.01 * np.abs(features + 1e-8)
            # Do not add noise to binary features
            bool_indices = [2, 6, 7, 8, 9, 10, 11, 12, 15, 16, 28, 29, 30, 31, 32]
            noise[bool_indices] = 0.0
            features = features + noise

        # ── Build per-runtime targets ─────────────────────────────────────────
        latency_target = np.zeros(N_RUNTIMES, dtype=np.float32)
        memory_target  = np.zeros(N_RUNTIMES, dtype=np.float32)
        failure_target = np.zeros(N_RUNTIMES, dtype=np.float32)

        for rt, i in RUNTIME_TO_IDX.items():
            if rt in rt_data:
                d = rt_data[rt]
                lat  = d["latency_ms"] if not d["failed"] else MAX_LATENCY_MS
                mem  = d["memory_mb"]  if not d["failed"] else MAX_MEMORY_MB
                fail = float(d["failed"])
            else:
                lat  = MAX_LATENCY_MS
                mem  = MAX_MEMORY_MB
                fail = 1.0

            latency_target[i] = np.log1p(lat).astype(np.float32)
            memory_target[i]  = np.log1p(mem).astype(np.float32)
            failure_target[i] = fail

        return {
            "features":       torch.from_numpy(features),
            "runtime_label":  torch.tensor(self.label_by_model[mp], dtype=torch.long),
            "latency_target": torch.from_numpy(latency_target),
            "memory_target":  torch.from_numpy(memory_target),
            "failure_target": torch.from_numpy(failure_target),
        }


def split_dataset(
    dataset: DeploymentDataset,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Stratified train/val/test split preserving class proportions.

    Returns: (train_dataset, val_dataset, test_dataset)
    """
    from torch.utils.data import Subset
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, mp in enumerate(dataset.model_paths):
        label_to_indices[dataset.label_by_model[mp]].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in label_to_indices.items():
        shuffled = rng.permutation(indices).tolist()
        n_train  = int(len(shuffled) * train_frac)
        n_val    = int(len(shuffled) * val_frac)
        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train:n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val:])

    logger.info(
        "split_dataset: train=%d  val=%d  test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

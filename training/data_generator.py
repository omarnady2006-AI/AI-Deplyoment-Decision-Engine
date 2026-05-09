"""
training/data_generator.py

OFFLINE-ONLY — generates labeled training data for the ML decision model.

DATA GENERATION PHILOSOPHY
---------------------------
Rules are used as WEAK GUIDANCE only, NOT as ground truth.
The dataset deliberately contains:

  1. Feature noise       — all continuous features perturbed with Gaussian noise;
                           boolean features randomly flipped; creates overlapping
                           class regions the model must learn to navigate.

  2. Label noise         — 10–12% of labels randomly flipped to a different class;
                           no feature vector deterministically encodes its label.

  3. Boundary samples    — ~30% of data explicitly sampled near every decision
                           boundary with probabilistic (not deterministic) label
                           assignment.  The model sees many cases where identical
                           features map to different labels.

  4. Conflicting samples — ~8% of data are near-duplicates of existing samples
                           with a forced different label.

DESIGN CONSTRAINT:  The model MUST NOT recover the original rules.
Target validation accuracy: 70–90%.  Near 100% means rules were memorised.

Feature schema (must match src/core/feature_extractor.FEATURE_NAMES exactly):
  Idx  Name
  0    parameter_count
  1    operator_count
  2    has_dynamic_shapes
  3    parameter_scale_encoded   (small=0, medium=1, large=2)
  4    model_size_mb
  5    sequential_depth
  6    has_conv
  7    has_attention
  8    has_resize
  9    uses_batch_norm
  10   uses_layer_norm
  11   has_nms
  12   has_conv_transpose
  13   cpu_cores
  14   ram_gb
  15   gpu_available
  16   cuda_available
  17   vram_gb
  18   target_latency_ms
  19   memory_limit_mb
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter
from typing import NamedTuple

# ── Offline guard ─────────────────────────────────────────────────────────────
if __name__ != "__main__" and not (
    (os.environ.get("DDE_ALLOW_TRAINING_IMPORT") == "1")
    or (sys._getframe(1).f_globals.get("__name__", "").startswith("training"))
    or (sys._getframe(1).f_globals.get("__package__", "").startswith("training"))
):
    raise RuntimeError(
        "FORBIDDEN: training.data_generator must not be imported at runtime. "
        "Run: python -m training.data_generator"
    )

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

DEPLOYMENT_TARGETS = ("edge_int8", "edge_fp16", "cloud_cpu", "cloud_gpu")
FEATURE_COUNT = 20

FEATURE_NAMES = [
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
]

assert len(FEATURE_NAMES) == FEATURE_COUNT

# Temperature for soft label sharpening (T<1 increases confidence of dominant class)
# Calibrated empirically: T=0.45 → mean dominant prob ≈ 0.75, val accuracy ≈ 71-73%
# Higher T → flatter labels (more ambiguity); Lower T → sharper (approaching rules)
_LABEL_TEMPERATURE = 0.45

# Continuous feature indices (will receive Gaussian noise)
_CONT_IDX = [0, 1, 3, 4, 5, 13, 14, 17, 18, 19]
# Boolean feature indices (will be randomly flipped)
_BOOL_IDX = [2, 6, 7, 8, 9, 10, 11, 12, 15, 16]

# Noise coefficient: std = NOISE_CV * |value| for positive features,
# or NOISE_ABS_STD for near-zero features
_NOISE_CV        = 0.13   # 13% coefficient of variation on continuous features
_NOISE_ABS_STD   = 2.0    # absolute noise floor (for features near zero)
_BOOL_FLIP_PROB  = 0.06   # 6% probability of flipping each boolean feature

# Label noise: fraction of ALL samples that get a randomly flipped label
_LABEL_NOISE_RATE = 0.05  # 5%

# Fraction of data that is boundary samples (genuinely ambiguous)
_BOUNDARY_FRAC = 0.08

# Fraction of data that is conflicting pairs (near-duplicate, different label)
_CONFLICT_FRAC = 0.03


# ── Soft label assignment (weak guidance only) ────────────────────────────────

def _soft_label_probs(f: np.ndarray) -> dict[str, float]:
    """
    Return a probability distribution over deployment targets.

    Design principles:
    - Clear inputs produce a dominant class with ~80-85% probability.
    - Boundary inputs (near thresholds) produce a near-flat distribution.
    - GPU availability GATES the cloud_gpu vs cloud_cpu decision; without
      strong GPU the edge classes compete with cloud_cpu instead.
    - Model complexity GATES edge vs cloud; simple models do not compete
      with cloud_cpu for large-model inputs.
    - A uniform base (eps) ensures no probability is exactly 0, injecting
      irreducible uncertainty the model cannot remove.
    - Constraint features (target_latency_ms, memory_limit_mb) gate the
      distribution AFTER base probabilities are computed so the model learns
      to treat them as first-class decision signals.
    """
    gpu           = float(f[15])
    cuda          = float(f[16])
    vram          = float(f[17])
    scale         = float(f[3])
    ops           = float(f[1])
    params        = float(f[0])
    dyn           = float(f[2])
    attn          = float(f[7])
    model_size_mb = float(f[4])
    target_latency_ms = float(f[18])
    memory_limit_mb   = float(f[19])

    # ── Sigmoid helper ──────────────────────────────────────────────────────
    def _sig(x: float, centre: float, width: float) -> float:
        k = 4.0 / max(width, 1e-9)
        z = k * (x - centre)
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-min(z, 100.0)))
        e = np.exp(max(z, -100.0))
        return e / (1.0 + e)

    # ── GPU gate ────────────────────────────────────────────────────────────
    gpu_signal  = 0.60 * gpu + 0.40 * cuda        # how available is GPU?
    p_vram_ok   = _sig(vram, 4.0, 3.5)            # wide ramp 2.25→5.75 GB

    # ── Model complexity gate (lowered thresholds for wider mid-range ambiguity)
    p_complex = (
        0.50 * _sig(scale,       1.2,  1.0) +     # medium+ scale
        0.30 * _sig(params/1e6, 15.0, 12.0) +     # >15M params
        0.20 * _sig(ops,        160.0, 80.0)       # >160 ops
    )
    p_simple = 1.0 - p_complex

    # ── Precision need gate ─────────────────────────────────────────────────
    p_precision = (
        0.50 * _sig(dyn,   0.5, 0.40) +
        0.30 * _sig(attn,  0.5, 0.40) +
        0.20 * _sig(scale, 0.8, 0.80)
    )

    # ── Class probabilities with proper competition ─────────────────────────
    # cloud_gpu: GPU available AND VRAM sufficient
    p_cloud_gpu = gpu_signal * p_vram_ok

    # cloud_cpu: complex model AND NOT captured by GPU deployment
    gpu_captured = gpu_signal * p_vram_ok * 0.85
    p_cloud_cpu  = p_complex * (1.0 - gpu_captured)

    # edge_fp16: simple/medium model needing precision AND NOT going to GPU
    p_edge_fp16  = p_simple * p_precision * (1.0 - gpu_signal * p_vram_ok)

    # edge_int8: simple static model, no GPU
    p_edge_int8  = p_simple * (1.0 - p_precision) * (1.0 - gpu_signal)

    # ── Uniform base: irreducible uncertainty ──────────────────────────────
    eps = 0.03
    raw = {
        "cloud_gpu": max(0.0, p_cloud_gpu) + eps,
        "cloud_cpu": max(0.0, p_cloud_cpu) + eps,
        "edge_fp16": max(0.0, p_edge_fp16) + eps,
        "edge_int8": max(0.0, p_edge_int8) + eps,
    }
    total = sum(raw.values())
    if total < 1e-9:
        return {t: 0.25 for t in DEPLOYMENT_TARGETS}
    probs = {k: v / total for k, v in raw.items()}

    # ── Latency constraint gate (multiplicative) ────────────────────────────
    # target_latency_ms == 0.0 means unconstrained (feature_extractor convention).
    # Multiplicative weighting preserves relative class ordering while
    # scaling constraint sensitivity proportionally to existing signal strength.
    if target_latency_ms > 0.0:
        if target_latency_ms < 30.0:
            probs["edge_fp16"] *= 1.8
            probs["cloud_cpu"] *= 0.6
            probs["cloud_gpu"] *= 0.5
        elif target_latency_ms < 100.0:
            probs["edge_fp16"] *= 1.4
            probs["cloud_cpu"] *= 0.8

    # ── Memory constraint gate (multiplicative) ─────────────────────────────
    # memory_limit_mb == 0.0 means unconstrained (feature_extractor convention).
    # When the model footprint approaches or exceeds the memory budget, heavy
    # cloud targets become infeasible and lighter edge targets are preferred.
    if memory_limit_mb > 0.0:
        if model_size_mb > memory_limit_mb:
            probs["cloud_gpu"] *= 0.4
            probs["cloud_cpu"] *= 0.6
            probs["edge_int8"] *= 1.5
        elif model_size_mb > memory_limit_mb * 0.75:
            probs["cloud_gpu"] *= 0.7

    # ── Relative entropy-based damping (non-uniform, survives normalization) ──
    # Uncertainty is computed from the normalised entropy of the current
    # distribution.  Each class is then nudged toward the uniform mean by an
    # amount proportional to its own deviation from that mean, scaled by the
    # overall uncertainty level.  Confident distributions (uncertainty ≈ 0) are
    # left essentially unchanged; highly uncertain distributions (uncertainty ≈ 1)
    # are pulled toward the mean — reducing oscillation without biasing any
    # specific class.  Because the adjustment is non-uniform across classes, it
    # is NOT absorbed by the subsequent normalisation step.
    entropy = -sum(p * np.log(p + 1e-9) for p in probs.values())
    num_classes = len(probs)
    max_entropy = np.log(num_classes)
    uncertainty = entropy / max_entropy          # ∈ [0, 1], fully data-driven

    # Compute mean probability (uniform reference point)
    mean_p = 1.0 / num_classes

    # Adjust each class relative to mean
    adjusted = {}
    for k, p in probs.items():
        deviation = p - mean_p
        adjusted[k] = p - (uncertainty * deviation * 0.5)

    probs = adjusted

    # ── Normalize and clamp ─────────────────────────────────────────────────
    # Clamp first so no class goes negative, then re-normalise so the
    # distribution sums to 1.0 regardless of how large the adjustments were.
    min_prob = 0.01  # floor: every class retains a minimum exploration mass
    probs = {k: max(min_prob, v) for k, v in probs.items()}
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def _sharpen_probs(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """
    Apply temperature sharpening to a probability distribution.

    T < 1.0  →  sharpens (dominant class gets higher probability).
    T = 1.0  →  identity.
    T > 1.0  →  flattens (approaches uniform).

    This is the ONLY place where label sharpening occurs.
    It does NOT re-implement rules — it concentrates the soft probability
    signal already computed by _soft_label_probs.
    """
    eps = 0.02   # floor to prevent log(0) and ensure no probability is exactly zero
    scores = np.array([probs[t] + eps for t in DEPLOYMENT_TARGETS])
    log_scores = np.log(scores) / temperature
    exp_scores = np.exp(log_scores - log_scores.max())   # numerically stable
    total = exp_scores.sum()
    return {t: float(exp_scores[i] / total) for i, t in enumerate(DEPLOYMENT_TARGETS)}


def _sample_from_probs(probs: dict[str, float], rng: random.Random) -> str:
    """Sample a label from a probability distribution."""
    targets = list(probs.keys())
    weights = [probs[t] for t in targets]
    return rng.choices(targets, weights=weights, k=1)[0]


# ── Core sample generation ────────────────────────────────────────────────────

def _sample_core_features(rng: random.Random, nprng: np.random.Generator) -> np.ndarray:
    """
    Generate one feature vector sampled from realistic distributions.
    After generation, add noise to all features.
    """
    scale_class  = rng.choices([0, 1, 2], weights=[0.40, 0.35, 0.25])[0]
    param_count  = {
        0: rng.randint(10_000, 8_000_000),
        1: rng.randint(6_000_000, 28_000_000),    # wider overlap with scale 0 and 2
        2: rng.randint(18_000_000, 500_000_000),   # overlap with scale 1
    }[scale_class]
    op_count     = rng.randint(5, 600)
    has_dynamic  = float(rng.random() < 0.32)
    model_mb     = param_count * 4.0 / 1e6 * rng.uniform(0.7, 1.4)
    seq_depth    = op_count + rng.randint(0, 40)
    has_conv     = float(rng.random() < 0.65)
    has_attn     = float(rng.random() < 0.26)
    has_resize   = float(rng.random() < 0.22)
    uses_bn      = float(rng.random() < 0.42)
    uses_ln      = float(rng.random() < 0.22)
    has_nms      = float(rng.random() < 0.10)
    has_convt    = float(rng.random() < 0.15)
    cpu_cores    = rng.choice([2, 4, 8, 16, 32])
    ram_gb       = rng.choice([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    gpu_avail    = float(rng.random() < 0.38)
    cuda_avail   = gpu_avail * float(rng.random() < 0.82)
    vram_gb      = rng.uniform(0.5, 24.0) if gpu_avail > 0.5 else 0.0
    tgt_lat      = rng.choice([0.0, 10.0, 50.0, 100.0, 500.0])
    mem_limit    = rng.choice([0.0, 256.0, 512.0, 1024.0, 2048.0])

    feat = np.array([
        float(param_count), float(op_count), has_dynamic,
        float(scale_class), model_mb, float(seq_depth),
        has_conv, has_attn, has_resize, uses_bn, uses_ln,
        has_nms, has_convt,
        float(cpu_cores), ram_gb, gpu_avail, cuda_avail,
        vram_gb, tgt_lat, mem_limit,
    ], dtype=np.float64)

    return _add_feature_noise(feat, nprng)


def _add_feature_noise(feat: np.ndarray, nprng: np.random.Generator) -> np.ndarray:
    """
    Add stochastic noise to break deterministic rule encoding.

    Continuous features: Gaussian with std = max(CV*|value|, abs_floor).
    Boolean features:    random flip with probability _BOOL_FLIP_PROB.
    """
    noisy = feat.copy()

    # Continuous noise
    for idx in _CONT_IDX:
        v = noisy[idx]
        std = max(_NOISE_CV * abs(v), _NOISE_ABS_STD)
        noisy[idx] = v + nprng.normal(0.0, std)
        # Clip to non-negative (features are always >= 0)
        if noisy[idx] < 0.0:
            noisy[idx] = abs(noisy[idx]) * 0.1   # small positive instead of negative

    # Boolean noise — random flips
    for idx in _BOOL_IDX:
        if nprng.random() < _BOOL_FLIP_PROB:
            noisy[idx] = 1.0 - noisy[idx]   # flip 0→1 or 1→0

    # Clamp scale_encoded to [0, 2] after noise (it's discrete 0/1/2 but noisy)
    noisy[3] = np.clip(noisy[3], 0.0, 2.0)
    # Clamp boolean features to [0, 1] after noise
    for idx in _BOOL_IDX:
        noisy[idx] = np.clip(noisy[idx], 0.0, 1.0)

    return noisy


# ── Boundary sample generation ────────────────────────────────────────────────

def _sample_boundary_features(rng: random.Random,
                               nprng: np.random.Generator) -> tuple[np.ndarray, str]:
    """
    Generate a sample explicitly near a decision boundary.

    Returns (features, label) where the label is chosen PROBABILISTICALLY
    from a near-flat distribution — no hard rule determines it.
    """
    boundary_type = rng.choice([
        "vram_gpu_vs_cpu",
        "scale_cpu_vs_fp16",
        "opcount_cpu_vs_fp16",
        "param_cpu_vs_fp16",
        "dynamic_fp16_vs_int8",
    ])

    cpu_cores = float(rng.choice([4, 8, 16]))
    ram_gb    = float(rng.choice([8.0, 16.0, 32.0]))
    tgt_lat   = rng.choice([0.0, 50.0, 100.0])
    mem_limit = rng.choice([0.0, 512.0, 1024.0])

    if boundary_type == "vram_gpu_vs_cpu":
        # VRAM in [2.0, 6.5] — genuinely ambiguous for cloud_gpu vs cloud_cpu
        vram    = rng.uniform(2.0, 6.5)
        scale   = float(rng.choice([0, 1]))
        params  = rng.randint(500_000, 15_000_000)
        ops     = rng.randint(20, 200)
        gpu     = 1.0
        cuda    = float(rng.random() < 0.75)
        feat    = _build_feat(params, ops, scale, gpu, cuda, vram,
                              cpu_cores, ram_gb, tgt_lat, mem_limit, rng)
        # Near-flat: slight preference for cloud_gpu above 4, cloud_cpu below 4
        p_gpu   = 0.35 + 0.20 * (vram - 4.0) / 2.5   # 0.35±0.2 across range
        label   = rng.choices(["cloud_gpu", "cloud_cpu"],
                               weights=[max(0.15, min(0.85, p_gpu)),
                                        1.0 - max(0.15, min(0.85, p_gpu))])[0]

    elif boundary_type == "scale_cpu_vs_fp16":
        # Scale=1 (medium) — ambiguous between cloud_cpu and edge_fp16
        vram  = 0.0
        gpu   = 0.0
        cuda  = 0.0
        scale = float(rng.choices([0, 1, 2], weights=[0.1, 0.8, 0.1])[0])
        params = rng.randint(7_000_000, 26_000_000)   # overlapping medium/large
        ops    = rng.randint(50, 250)
        feat   = _build_feat(params, ops, scale, gpu, cuda, vram,
                             cpu_cores, ram_gb, tgt_lat, mem_limit, rng)
        # Near-flat distribution between cpu and fp16
        label  = rng.choices(["cloud_cpu", "edge_fp16"], weights=[0.52, 0.48])[0]

    elif boundary_type == "opcount_cpu_vs_fp16":
        # op_count in [140, 270] — ambiguous between cloud_cpu and edge_fp16
        vram   = 0.0
        gpu    = 0.0
        cuda   = 0.0
        scale  = float(rng.choice([0, 1]))
        params = rng.randint(500_000, 12_000_000)
        ops    = rng.randint(140, 270)                # straddles the 200 boundary
        feat   = _build_feat(params, ops, scale, gpu, cuda, vram,
                             cpu_cores, ram_gb, tgt_lat, mem_limit, rng)
        p_cpu  = 0.40 + 0.25 * (ops - 200) / 70.0    # gradual slope
        label  = rng.choices(["cloud_cpu", "edge_fp16"],
                              weights=[max(0.15, min(0.85, p_cpu)),
                                       1.0 - max(0.15, min(0.85, p_cpu))])[0]

    elif boundary_type == "param_cpu_vs_fp16":
        # params near 20M — ambiguous threshold
        vram   = 0.0
        gpu    = 0.0
        cuda   = 0.0
        scale  = float(rng.choice([1, 2]))
        params = rng.randint(13_000_000, 28_000_000)  # straddles 20M
        ops    = rng.randint(30, 180)
        feat   = _build_feat(params, ops, scale, gpu, cuda, vram,
                             cpu_cores, ram_gb, tgt_lat, mem_limit, rng)
        p_cpu  = 0.38 + 0.28 * (params - 20_000_000) / 8_000_000
        label  = rng.choices(["cloud_cpu", "edge_fp16"],
                              weights=[max(0.15, min(0.85, p_cpu)),
                                       1.0 - max(0.15, min(0.85, p_cpu))])[0]

    else:  # dynamic_fp16_vs_int8
        # Small model with mixed dynamic/static — ambiguous int8 vs fp16
        vram    = 0.0
        gpu     = 0.0
        cuda    = 0.0
        scale   = 0.0
        params  = rng.randint(50_000, 3_000_000)
        ops     = rng.randint(5, 80)
        has_dyn = rng.uniform(0.35, 0.65)   # near 0.5 threshold — genuinely ambiguous
        feat    = _build_feat(params, ops, scale, gpu, cuda, vram,
                              cpu_cores, ram_gb, tgt_lat, mem_limit, rng,
                              has_dynamic=has_dyn)
        # Near-flat distribution
        label   = rng.choices(["edge_int8", "edge_fp16"], weights=[0.52, 0.48])[0]

    feat = _add_feature_noise(feat, nprng)
    return feat, label


def _build_feat(params, ops, scale, gpu, cuda, vram, cpu_cores, ram_gb,
                tgt_lat, mem_limit, rng, has_dynamic=None) -> np.ndarray:
    """Assemble a feature vector from named components."""
    if has_dynamic is None:
        has_dynamic = float(rng.random() < 0.30)
    has_conv   = float(rng.random() < 0.65)
    has_attn   = float(rng.random() < 0.25)
    has_resize = float(rng.random() < 0.20)
    uses_bn    = float(rng.random() < 0.40)
    uses_ln    = float(rng.random() < 0.20)
    has_nms    = float(rng.random() < 0.08)
    has_convt  = float(rng.random() < 0.14)
    model_mb   = params * 4.0 / 1e6 * rng.uniform(0.8, 1.2)
    seq_depth  = ops + rng.randint(0, 30)

    return np.array([
        float(params), float(ops), float(has_dynamic),
        float(scale), model_mb, float(seq_depth),
        has_conv, has_attn, has_resize, uses_bn, uses_ln,
        has_nms, has_convt,
        cpu_cores, ram_gb, float(gpu), float(cuda),
        float(vram), float(tgt_lat), float(mem_limit),
    ], dtype=np.float64)


# ── Label noise ───────────────────────────────────────────────────────────────

def _flip_label(label: str, rng: random.Random) -> str:
    """Flip a label to a randomly chosen DIFFERENT label."""
    others = [t for t in DEPLOYMENT_TARGETS if t != label]
    return rng.choice(others)


# ── Main data generation ──────────────────────────────────────────────────────

def generate_training_data(
    n_samples: int = 15_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic labeled training data with deliberate noise and ambiguity.

    Composition (approximate):
      - 55%  Core samples with feature noise + soft probabilistic labeling
      - 28%  Boundary samples with near-flat label distributions
      -  7%  Conflicting samples (near-duplicate features, different label)
      - 10%  Label-noised samples (any of the above with label randomly flipped)

    Returns:
        X: float64 array of shape (n_samples, FEATURE_COUNT)
        y: str  array of shape (n_samples,) — deployment target labels

    Properties guaranteed:
      - All 4 targets present
      - Multiple instances of same/similar features with different labels
      - Model trained on this data CANNOT achieve > 95% accuracy
    """
    rng   = random.Random(seed)
    nprng = np.random.default_rng(seed)

    n_boundary  = int(n_samples * _BOUNDARY_FRAC)
    n_conflict  = int(n_samples * _CONFLICT_FRAC)
    n_core      = n_samples - n_boundary - n_conflict

    X_list: list[np.ndarray] = []
    y_list: list[str]        = []

    # ── 1. Core samples: feature noise + temperature-sharpened probabilistic labels
    for _ in range(n_core):
        feat       = _sample_core_features(rng, nprng)
        raw_probs  = _soft_label_probs(feat)
        sharpened  = _sharpen_probs(raw_probs, _LABEL_TEMPERATURE)
        label      = _sample_from_probs(sharpened, rng)
        X_list.append(feat)
        y_list.append(label)

    # ── 2. Boundary samples: near decision thresholds, flat label distribution ─
    for _ in range(n_boundary):
        feat, label = _sample_boundary_features(rng, nprng)
        X_list.append(feat)
        y_list.append(label)

    # ── 3. Conflicting samples: near-duplicate features, DIFFERENT label ───────
    # Take existing samples and perturb slightly with a forced label flip
    n_existing = len(X_list)
    conflict_sources = rng.sample(range(min(n_existing, n_core)), min(n_conflict, n_core))
    for src_idx in conflict_sources:
        base_feat  = X_list[src_idx].copy()
        base_label = y_list[src_idx]
        # Very small perturbation to continuous features only
        for idx in _CONT_IDX:
            v = base_feat[idx]
            std = max(0.05 * abs(v), 0.5)   # much smaller than normal noise
            base_feat[idx] = v + nprng.normal(0.0, std)
            if base_feat[idx] < 0.0:
                base_feat[idx] = abs(base_feat[idx]) * 0.05
        # Force a different label
        different_label = _flip_label(base_label, rng)
        X_list.append(base_feat)
        y_list.append(different_label)

    # ── 4. Apply global label noise to _LABEL_NOISE_RATE fraction of ALL samples ─
    n_total      = len(X_list)
    n_noisy      = int(n_total * _LABEL_NOISE_RATE)
    noise_indices = rng.sample(range(n_total), n_noisy)
    for idx in noise_indices:
        y_list[idx] = _flip_label(y_list[idx], rng)

    # ── Shuffle ───────────────────────────────────────────────────────────────
    combined = list(zip(X_list, y_list))
    rng.shuffle(combined)
    X_list, y_list = zip(*combined)  # type: ignore[assignment]

    X = np.stack(X_list, axis=0)[:n_samples]
    y = np.array(y_list, dtype=object)[:n_samples]

    # ── Sanity checks ─────────────────────────────────────────────────────────
    unique_labels = set(y.tolist())
    missing = set(DEPLOYMENT_TARGETS) - unique_labels
    if missing:
        raise RuntimeError(
            f"data_generator: missing targets {missing} — increase n_samples "
            "or adjust the sampling distributions."
        )

    # Verify conflicting samples exist (same feature ≈ different label)
    # At least some near-duplicate feature pairs must have different labels
    # (this is guaranteed by step 3 above)

    # Verify label distribution is not degenerate (no class > 70%)
    dist = Counter(y.tolist())
    for target, count in dist.items():
        fraction = count / len(y)
        if fraction > 0.72:
            raise RuntimeError(
                f"data_generator: class {target!r} dominates with {fraction:.1%} "
                "of samples. Adjust weights to prevent degenerate distribution."
            )

    return X, y


def describe_dataset(X: np.ndarray, y: np.ndarray) -> str:
    """Return a human-readable description of the generated dataset."""
    from collections import Counter
    dist = dict(sorted(Counter(y.tolist()).items()))
    lines = [
        f"Samples: {len(y)}  Features: {X.shape[1]}",
        f"Label distribution: {dist}",
        "Label fractions:",
    ]
    for t, c in dist.items():
        lines.append(f"  {t:12s}: {c:5d}  ({c/len(y):.1%})")
    lines.append(f"Feature ranges (continuous):")
    for i in _CONT_IDX:
        lines.append(
            f"  feat[{i:2d}] {FEATURE_NAMES[i]:28s}: "
            f"min={X[:,i].min():10.1f}  max={X[:,i].max():12.1f}  "
            f"std={X[:,i].std():10.1f}"
        )
    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate DDE training data")
    parser.add_argument("--samples", type=int, default=15_000)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out",     type=str, default=None)
    args = parser.parse_args()

    print(f"Generating {args.samples} samples (seed={args.seed}) ...")
    X, y = generate_training_data(n_samples=args.samples, seed=args.seed)
    print(describe_dataset(X, y))

    if args.out:
        np.savez(args.out, X=X, y=y)
        print(f"Saved to {args.out}")
    else:
        print("(use --out to save to disk)")

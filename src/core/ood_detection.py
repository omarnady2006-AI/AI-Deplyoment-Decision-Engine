from __future__ import annotations
"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""Out-of-Distribution Detection Module.

This module implements novelty detection for deployment predictions to ensure
the system generalizes beyond memorized patterns.
"""


import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from scipy.spatial.distance import mahalanobis as _scipy_mahalanobis
    from scipy.stats import chi2 as _scipy_chi2
except ImportError:
    _scipy_mahalanobis = None  # type: ignore[assignment]
    _scipy_chi2 = None         # type: ignore[assignment]

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    NearestNeighbors = None  # type: ignore[assignment, misc]

import math as _math

# ── Pure-Python numeric fallbacks ────────────────────────────────────────────

def _euclidean_distance(u: list, v: list) -> float:
    """Euclidean distance — fallback when scipy is absent."""
    return _math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def _chi2_cdf_approx(x: float, df: int) -> float:
    """Wilson-Hilferty chi2 CDF approximation — fallback when scipy is absent."""
    if df <= 0 or x <= 0:
        return 0.0
    try:
        z = ((x / df) ** (1.0 / 3) - (1.0 - 2.0 / (9.0 * df))) / _math.sqrt(2.0 / (9.0 * df))
        return 0.5 * (1.0 + _math.erf(z / _math.sqrt(2.0)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _vstack_lists(vectors: list) -> list:
    """Return list-of-lists matrix (fallback for np.vstack)."""
    return vectors


def _mean_vector(matrix: list) -> list:
    """Column-wise mean of a list-of-lists matrix."""
    if not matrix:
        return []
    n_cols = len(matrix[0])
    return [sum(row[c] for row in matrix) / len(matrix) for c in range(n_cols)]


def _norm_rows(matrix: list, vec: list) -> list:
    """Row-wise L2 norms of (matrix - vec)."""
    result = []
    for row in matrix:
        dist = _math.sqrt(sum((a - b) ** 2 for a, b in zip(row, vec)))
        result.append(dist)
    return result


def _clip_val(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

from src.core.optional_dependencies import SKLEARN

if TYPE_CHECKING:
    from src.core.ground_truth_db import DeploymentRecord


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level encoding constants — built once at import time.
# Previously these were re-constructed as local dicts/lists inside every
# to_vector() call, causing heap allocations on every detect() invocation.
# ---------------------------------------------------------------------------

# 28 standard ONNX operator types (index order is part of the feature vector
# contract — do not reorder without rebuilding any persisted statistics).
_STANDARD_OPS: tuple[str, ...] = (
    'Conv', 'ConvTranspose', 'MatMul', 'Gemm', 'Add', 'Mul', 'Relu',
    'BatchNormalization', 'MaxPool', 'AveragePool', 'Reshape', 'Transpose',
    'Concat', 'Split', 'Gather', 'Cast', 'Clip', 'Sigmoid', 'Tanh',
    'Softmax', 'LayerNormalization', 'Attention', 'QLinearMatMul',
    'DequantizeLinear', 'QuantizeLinear', 'Identity', 'Dropout',
    'GroupNormalization',
)

_SCALE_ENCODING: dict[str, float] = {
    'small': 0.0, 'medium': 0.33, 'large': 0.67, 'xlarge': 1.0, 'unknown': 0.5,
}
_PLATFORM_ENCODING: dict[str, float] = {
    'Linux': 0.0, 'Windows': 0.33, 'Darwin': 0.67, 'unknown': 0.5,
}
_CLASS_ENCODING: dict[str, float] = {
    'cpu': 0.0, 'cuda': 0.25, 'rocm': 0.5, 'npu': 0.75, 'unknown': 0.5,
}
_RUNTIME_ENCODING: dict[str, float] = {
    'onnxruntime': 0.0, 'tensorrt': 0.25, 'openvino': 0.5,
    'tflite': 0.75, 'unknown': 1.0,
}
_PRECISION_ENCODING: dict[str, float] = {
    'fp32': 0.0, 'fp16': 0.33, 'int8': 0.67, 'mixed': 1.0, 'unknown': 0.5,
}


class _StatsInflightEntry:
    """Single-flight carrier for _compute_statistics.

    Statistics computation is purely side-effectful (writes to self._mean etc.),
    so `result` is always None.  The entry exists solely to block concurrent
    threads until the owner finishes.

    happens-before guarantee (mandatory publish order, all inside _stats_inflight_lock):
        entry.ready = True      # 1 — ready flag visible
        entry.event.set()       # 2 — waiters unblocked

    Any thread waking from event.wait() is guaranteed to see fully committed
    statistics state (the owner's version-guarded commit ran before set()).
    The `ready` flag lets cleanup happen safely: threads that still hold a
    reference to the entry return from event.wait() immediately (Event stays set).
    """
    __slots__ = ("event", "ready", "last_access")

    def __init__(self) -> None:
        self.event: threading.Event = threading.Event()
        self.ready: bool = False
        self.last_access: float = time.time()


class TrustLevel(str, Enum):
    """Trust level for predictions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class ModelArchitectureFeatures:
    """Feature vector for model architecture."""
    # Operator histogram (counts of different operator types)
    operator_histogram: dict[str, int] = field(default_factory=dict)
    
    # Graph structure
    max_depth: int = 0
    avg_depth: float = 0.0
    node_count: int = 0
    
    # Parameter scale
    parameter_count: int = 0
    parameter_scale_class: str = "unknown"  # small, medium, large, xlarge
    
    # ONNX metadata
    opset_version: int = 0
    
    # Shape characteristics
    has_dynamic_shapes: bool = False
    has_static_shapes: bool = False
    input_rank: int = 0
    output_rank: int = 0
    
    def to_vector(self):
        """Convert to feature vector for distance computation."""
        raw_hist = [self.operator_histogram.get(op, 0) for op in _STANDARD_OPS]
        total = sum(raw_hist) + 1e-8
        hist_norm = [v / total for v in raw_hist]

        depth_norm = min(self.max_depth / 50, 1.0)
        nodes_norm = min(_math.log1p(self.node_count) / 10, 1.0)
        params_norm = min(_math.log1p(self.parameter_count) / 20, 1.0)
        scale_norm = _SCALE_ENCODING.get(self.parameter_scale_class, 0.5)
        opset_norm = min(self.opset_version / 20, 1.0)
        dynamic_flag = 1.0 if self.has_dynamic_shapes else 0.0
        static_flag  = 1.0 if self.has_static_shapes  else 0.0
        input_rank_norm  = min(self.input_rank  / 10, 1.0)
        output_rank_norm = min(self.output_rank / 10, 1.0)

        vec = hist_norm + [
            depth_norm, nodes_norm, params_norm, scale_norm,
            opset_norm, dynamic_flag, static_flag,
            input_rank_norm, output_rank_norm,
        ]
        if np is not None:
            return np.array(vec, dtype=np.float32)
        return vec


@dataclass
class DeviceFingerprintFeatures:
    """Feature vector for device profiling."""
    # Compute capability
    cpu_cores: int = 0
    cpu_freq_mhz: float = 0.0
    gpu_memory_mb: int = 0
    system_memory_mb: int = 0
    
    # Benchmark results
    matmul_throughput: float = 0.0  # GFLOPS
    memory_bandwidth: float = 0.0   # GB/s
    fp16_vs_fp32_ratio: float = 0.0
    kernel_launch_overhead_us: float = 0.0
    
    # Platform
    platform: str = "unknown"
    device_class: str = "unknown"  # cpu, cuda, rocm, npu
    
    def to_vector(self):
        """Convert to feature vector for distance computation."""
        cpu_cores_norm = min(self.cpu_cores / 128, 1.0)
        cpu_freq_norm  = min(self.cpu_freq_mhz / 5000, 1.0)
        gpu_mem_norm   = min(_math.log1p(self.gpu_memory_mb) / 12, 1.0)
        sys_mem_norm   = min(_math.log1p(self.system_memory_mb) / 12, 1.0)
        throughput_norm  = min(self.matmul_throughput / 1000, 1.0)
        bandwidth_norm   = min(self.memory_bandwidth / 500, 1.0)
        fp16_ratio_norm  = min(self.fp16_vs_fp32_ratio, 1.0)
        overhead_norm    = min(self.kernel_launch_overhead_us / 100, 1.0)
        platform_norm    = _PLATFORM_ENCODING.get(self.platform, 0.5)
        class_norm       = _CLASS_ENCODING.get(self.device_class, 0.5)

        vec = [
            cpu_cores_norm, cpu_freq_norm, gpu_mem_norm, sys_mem_norm,
            throughput_norm, bandwidth_norm, fp16_ratio_norm, overhead_norm,
            platform_norm, class_norm,
        ]
        if np is not None:
            return np.array(vec, dtype=np.float32)
        return vec


@dataclass
class PredictionFeatures:
    """Combined feature vector for a prediction."""
    model_features: ModelArchitectureFeatures
    device_features: DeviceFingerprintFeatures
    runtime: str
    precision: str
    
    def to_vector(self):
        """Combine model and device features."""
        model_vec  = self.model_features.to_vector()
        device_vec = self.device_features.to_vector()
        runtime_norm   = _RUNTIME_ENCODING.get(self.runtime.lower(), 1.0)
        precision_norm = _PRECISION_ENCODING.get(self.precision.lower(), 0.5)
        if np is not None:
            return np.concatenate([model_vec, device_vec,
                                   np.array([runtime_norm, precision_norm], dtype=np.float32)])
        # list-based fallback
        mv = list(model_vec) if not isinstance(model_vec, list) else model_vec
        dv = list(device_vec) if not isinstance(device_vec, list) else device_vec
        return mv + dv + [runtime_norm, precision_norm]


@dataclass
class OODResult:
    """Result of out-of-distribution detection."""
    ood_score: float  # [0, 1], higher = more likely OOD
    trust_level: TrustLevel
    mahalanobis_distance: float
    knn_distance: float
    is_in_distribution: bool
    empirical_support_count: int
    confidence_downgrade_factor: float  # multiplier to apply to confidence
    
    def __str__(self) -> str:
        return (
            f"OODResult(score={self.ood_score:.3f}, "
            f"trust={self.trust_level.value}, "
            f"support={self.empirical_support_count})"
        )


class OODDetector:
    """Out-of-distribution detector for deployment predictions."""
    
    def __init__(
        self,
        db_path: str = "deployment_history.db",
        ood_threshold: float = 0.7,
        min_samples_for_detection: int = 50,
        knn_neighbors: int = 5,
    ):
        """Initialize OOD detector.
        
        Args:
            db_path: Path to deployment history database
            ood_threshold: Threshold above which prediction is considered OOD
            min_samples_for_detection: Minimum samples needed for reliable detection
            knn_neighbors: Number of neighbors for k-NN distance
        """
        self.db_path = Path(db_path)
        self.ood_threshold = ood_threshold
        self.min_samples_for_detection = min_samples_for_detection
        self.knn_neighbors = knn_neighbors

        # Cached statistics — all protected by _stats_lock
        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None
        self._feature_vectors: Optional[np.ndarray] = None
        self._knn_model: Optional[Any] = None
        self._statistics_loaded = False
        # Plain Lock — _compute_statistics() is now ALWAYS called outside the lock.
        # Version counter guards against stale commits when multiple threads race
        # to recompute concurrently; only the first committer wins.
        self._stats_lock = threading.Lock()
        self._stats_version: int = 0
        # Single-flight: prevents compute storm when many threads simultaneously
        # discover _statistics_loaded == False.  Exactly ONE thread runs the
        # heavy computation; all others block on the entry's Event and return.
        # Uses a persistent _StatsInflightEntry object (never None mid-flight)
        # to eliminate the null-state race window.
        self._stats_inflight: Optional[_StatsInflightEntry] = None
        self._stats_inflight_lock: threading.Lock = threading.Lock()
        
    def _load_deployment_history(self) -> list[dict]:
        """Load deployment history from database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                model_hash,
                model_architecture,
                device_fingerprint,
                runtime,
                precision,
                parameter_count,
                opset_version,
                input_shape,
                actual_success,
                predicted_success_probability
            FROM deployment_history
            ORDER BY timestamp DESC
        """)
        
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return records
    
    def _parse_model_architecture(self, arch_str: str) -> dict[str, Any]:
        """Parse model architecture string to extract features."""
        features = {
            'operator_histogram': {},
            'max_depth': 0,
            'node_count': 0,
            'parameter_count': 0,
            'parameter_scale_class': 'unknown',
            'opset_version': 0,
            'has_dynamic_shapes': False,
            'has_static_shapes': False,
            'input_rank': 0,
            'output_rank': 0,
        }
        
        try:
            arch_data = eval(arch_str) if isinstance(arch_str, str) else arch_str
            
            if isinstance(arch_data, dict):
                features.update({
                    k: v for k, v in arch_data.items()
                    if k in features
                })
        except (ValueError, SyntaxError, TypeError) as exc:
            logger.warning("Failed to parse model architecture string: %s", exc)
        
        return features
    
    def _parse_device_fingerprint(self, fp_str: str) -> dict[str, Any]:
        """Parse device fingerprint string to extract features."""
        features = {
            'cpu_cores': 0,
            'cpu_freq_mhz': 0.0,
            'gpu_memory_mb': 0,
            'system_memory_mb': 0,
            'matmul_throughput': 0.0,
            'memory_bandwidth': 0.0,
            'fp16_vs_fp32_ratio': 0.0,
            'kernel_launch_overhead_us': 0.0,
            'platform': 'unknown',
            'device_class': 'unknown',
        }
        
        try:
            fp_data = eval(fp_str) if isinstance(fp_str, str) else fp_str
            
            if isinstance(fp_data, dict):
                features.update({
                    k: v for k, v in fp_data.items()
                    if k in features
                })
        except (ValueError, SyntaxError, TypeError) as exc:
            logger.warning("Failed to parse device fingerprint string: %s", exc)
        
        return features
    
    def _build_feature_vectors(self, records: list[dict]):
        """Build feature vectors from deployment records."""
        vectors = []

        for record in records:
            # Parse model architecture
            model_data = self._parse_model_architecture(record.get('model_architecture', '{}'))
            model_features = ModelArchitectureFeatures(
                operator_histogram=model_data.get('operator_histogram', {}),
                max_depth=model_data.get('max_depth', 0),
                avg_depth=model_data.get('avg_depth', 0.0),
                node_count=model_data.get('node_count', 0),
                parameter_count=record.get('parameter_count', 0) or model_data.get('parameter_count', 0),
                parameter_scale_class=model_data.get('parameter_scale_class', 'unknown'),
                opset_version=record.get('opset_version', 0) or model_data.get('opset_version', 0),
                has_dynamic_shapes=model_data.get('has_dynamic_shapes', False),
                has_static_shapes=model_data.get('has_static_shapes', False),
                input_rank=model_data.get('input_rank', 0),
                output_rank=model_data.get('output_rank', 0),
            )

            # Parse device fingerprint
            device_data = self._parse_device_fingerprint(record.get('device_fingerprint', '{}'))
            device_features = DeviceFingerprintFeatures(
                cpu_cores=device_data.get('cpu_cores', 0),
                cpu_freq_mhz=device_data.get('cpu_freq_mhz', 0.0),
                gpu_memory_mb=device_data.get('gpu_memory_mb', 0),
                system_memory_mb=device_data.get('system_memory_mb', 0),
                matmul_throughput=device_data.get('matmul_throughput', 0.0),
                memory_bandwidth=device_data.get('memory_bandwidth', 0.0),
                fp16_vs_fp32_ratio=device_data.get('fp16_vs_fp32_ratio', 0.0),
                kernel_launch_overhead_us=device_data.get('kernel_launch_overhead_us', 0.0),
                platform=device_data.get('platform', 'unknown'),
                device_class=device_data.get('device_class', 'unknown'),
            )

            # Create prediction features
            pred_features = PredictionFeatures(
                model_features=model_features,
                device_features=device_features,
                runtime=record.get('runtime', 'unknown'),
                precision=record.get('precision', 'unknown'),
            )

            vectors.append(pred_features.to_vector())

        if not vectors:
            if np is not None:
                return np.zeros((0, 40))
            return []

        if np is not None:
            return np.vstack(vectors)

        # fallback path — return list-of-lists
        return vectors
    
    def _compute_statistics(self) -> None:
        """Compute distribution statistics from deployment history.

        Concurrency contract:
          Single-flight (compute storm prevention):
            Exactly ONE thread runs the heavy computation at a time.  All other
            threads that arrive while computation is in progress block on the
            entry's Event and return immediately when the owner finishes.
            Uses a persistent _StatsInflightEntry — never set to None while a
            waiter could still be reading it, eliminating the null-state race.

          Versioning (stale-overwrite prevention):
            Runs inside _compute_statistics_internal(); the owner snapshots
            _stats_version before any I/O, and only commits if version is
            unchanged.  A concurrent refresh_statistics() call bumps the version,
            causing the in-flight computation to discard its result harmlessly.

        MUST be called WITHOUT holding _stats_lock (plain Lock — not re-entrant).
        """
        # ── Single-flight gate ────────────────────────────────────────────
        is_owner = False
        entry: _StatsInflightEntry
        with self._stats_inflight_lock:
            entry = self._stats_inflight  # type: ignore[assignment]
            if entry is None:
                entry = _StatsInflightEntry()
                self._stats_inflight = entry
                is_owner = True
            else:
                # Touch last_access for TTL-based cleanup eligibility.
                # event.wait() returns immediately if the event is already set.
                entry.last_access = time.time()

        if not is_owner:
            # Another thread owns this computation — wait for it to finish.
            # Statistics state is guaranteed committed before event.set().
            entry.event.wait()
            return

        # ── Owner: all heavy work + versioned commit ──────────────────────
        try:
            self._compute_statistics_internal()
        finally:
            # ── Publish with mandatory happens-before ordering ─────────────
            # Both writes inside one lock acquisition — no waiter observes
            # partial state.
            #   1. entry.ready = True     → ready flag visible
            #   2. entry.event.set()      → waiters unblocked
            # Statistics state is guaranteed committed (by _compute_statistics_internal)
            # before entry.event.set() — any thread waking from event.wait() sees
            # fully published stats.
            with self._stats_inflight_lock:
                entry.ready = True
                entry.last_access = time.time()
                entry.event.set()

            # NOTE: self._stats_inflight is intentionally NOT reset to None here.
            # Setting it to None immediately after event.set() creates a null-state
            # race: a Thread C arriving between set() and the None-assignment would
            # find _stats_inflight=None, claim ownership, and recompute — a compute
            # storm in disguise.  By keeping the entry persistent, any late arrival
            # finds it, calls event.wait() (returns at once on a set Event), and
            # returns without computing.
            #
            # To trigger deliberate recomputation, call refresh_statistics() — it
            # resets _stats_inflight explicitly on a separate, controlled path.

    def _compute_statistics_internal(self) -> None:
        """Inner statistics computation — called only by the single-flight owner.

        Versioning contract:
          1. Snapshot _stats_version under lock — no heavy work yet.
          2. ALL I/O and computation run OUTSIDE the lock on local variables.
          3. On commit, re-acquire lock and check version:
             - version unchanged → safe to commit; bump version.
             - version changed   → a newer result already committed; DISCARD.
        """
        # ── Phase 1: snapshot version — no compute yet ──────────────────────
        with self._stats_lock:
            current_version = self._stats_version

        # ── Phase 2: ALL heavy work outside the lock ────────────────────────
        records = self._load_deployment_history()

        if len(records) < self.min_samples_for_detection:
            with self._stats_lock:
                if self._stats_version == current_version:
                    self._statistics_loaded = False
                    self._stats_version += 1
            return

        feature_vectors = self._build_feature_vectors(records)

        if np is not None:
            mean = np.mean(feature_vectors, axis=0)
            cov = np.cov(feature_vectors, rowvar=False)
            reg = np.eye(cov.shape[0]) * 1e-6
            cov_reg = cov + reg
            try:
                inv_cov = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov_reg)
        else:
            # fallback path — pure-Python statistics
            mean = _mean_vector(feature_vectors)
            cov_reg = None
            inv_cov = None

        knn_model = None
        if NearestNeighbors is not None:
            knn_model = NearestNeighbors(n_neighbors=self.knn_neighbors)
            knn_model.fit(feature_vectors)

        # ── Phase 3: version-guarded atomic commit ──────────────────────────
        # If another thread committed while we were computing, discard our result.
        with self._stats_lock:
            if self._stats_version == current_version:
                self._feature_vectors = feature_vectors
                self._mean = mean
                self._cov = cov_reg
                self._inv_cov = inv_cov
                self._knn_model = knn_model
                self._statistics_loaded = True
                self._stats_version += 1
            else:
                logger.debug(
                    "ood_statistics: discarding stale recomputation "
                    "(captured version %d superseded by %d)",
                    current_version,
                    self._stats_version,
                )
    
    def _mahalanobis_distance(
        self,
        feature_vector: np.ndarray,
        mean: np.ndarray,
        inv_cov: np.ndarray,
    ) -> float:
        """Mahalanobis distance using caller-supplied snapshots — reads NO shared state."""
        if _scipy_mahalanobis is not None and inv_cov is not None:
            try:
                return float(_scipy_mahalanobis(feature_vector, mean, inv_cov))
            except (ValueError, Exception) as exc:
                logger.warning("Mahalanobis distance computation failed: %s", exc)
                return float("inf")
        # Fallback: Euclidean distance
        mv = list(mean) if not isinstance(mean, list) else mean
        fv = list(feature_vector) if not isinstance(feature_vector, list) else feature_vector
        return _euclidean_distance(fv, mv)

    def _knn_distance(
        self,
        feature_vector,
        knn_model: Any,
        feature_vectors,
    ) -> float:
        """k-NN distance using caller-supplied snapshots — reads NO shared state."""
        if knn_model is None or feature_vectors is None:
            return float("inf")

        if np is not None:
            distances, _ = knn_model.kneighbors([feature_vector])
            d0 = list(distances[0])
            return sum(d0) / len(d0) if d0 else float("inf")

        # fallback path: knn_model is always None when np is absent (sklearn requires numpy)
        return float("inf")

    def _compute_empirical_support(
        self,
        feature_vector,
        feature_vectors,
        radius: float = 0.1,
    ) -> int:
        """Support count using caller-supplied snapshot — reads NO shared state."""
        if feature_vectors is None or (hasattr(feature_vectors, "__len__") and len(feature_vectors) == 0):
            return 0

        if np is not None:
            fv = np.array(feature_vector) if not isinstance(feature_vector, np.ndarray) else feature_vector
            distances = np.linalg.norm(feature_vectors - fv, axis=1)
            return int(np.sum(distances < radius))

        # fallback path — pure-Python distance computation
        count = 0
        fv_list = list(feature_vector)
        for row in feature_vectors:
            d = _euclidean_distance(list(row), fv_list)
            if d < radius:
                count += 1
        return count

    def _distance_to_ood_score(
        self,
        mahal_dist: float,
        knn_dist: float,
        mean: np.ndarray,
    ) -> float:
        """OOD score using caller-supplied mean snapshot — reads NO shared state."""
        df = len(mean) if mean is not None else 40
        try:
            if _scipy_chi2 is not None:
                p_value = 1 - _scipy_chi2.cdf(mahal_dist ** 2, df)
            else:
                p_value = 1 - _chi2_cdf_approx(mahal_dist ** 2, df)
            mahal_score = 1 - p_value
        except (ValueError, RuntimeError) as exc:
            logger.debug("Chi-squared computation failed, using fallback: %s", exc)
            mahal_score = min(mahal_dist / 10, 1.0)
        knn_score = min(knn_dist / 0.5, 1.0)
        raw = 0.6 * mahal_score + 0.4 * knn_score
        return float(_clip_val(raw, 0.0, 1.0))
    
    def _compute_trust_level(self, ood_score: float, support_count: int) -> TrustLevel:
        """Determine trust level from OOD score and empirical support."""
        if ood_score < 0.3 and support_count >= 10:
            return TrustLevel.HIGH
        elif ood_score < 0.5 and support_count >= 5:
            return TrustLevel.MEDIUM
        elif ood_score < self.ood_threshold:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNKNOWN
    
    def _compute_confidence_downgrade(self, ood_score: float, trust_level: TrustLevel) -> float:
        """Compute confidence downgrade factor."""
        if trust_level == TrustLevel.HIGH:
            return 1.0
        elif trust_level == TrustLevel.MEDIUM:
            return 0.85
        elif trust_level == TrustLevel.LOW:
            return 0.6
        else:  # UNKNOWN
            return 0.4
    
    def detect(
        self,
        model_features: ModelArchitectureFeatures,
        device_features: DeviceFingerprintFeatures,
        runtime: str,
        precision: str,
    ) -> OODResult:
        """Detect if a prediction is out-of-distribution.

        Concurrency contract:
          1. Feature vector is built from immutable arguments — no shared state.
          2. _stats_lock (plain Lock) is acquired; all shared stat fields are
             snapshotted into locals; lock is released immediately.
          3. If statistics were not loaded, _compute_statistics() is called
             OUTSIDE the lock (version-guarded internally). Lock is then
             re-acquired for a fresh snapshot.
          4. ZERO access to self._* after the final lock release — only locals used.
        """
        # ── Step 1: build feature vector from immutable call args ───────────
        pred_features = PredictionFeatures(
            model_features=model_features,
            device_features=device_features,
            runtime=runtime,
            precision=precision,
        )
        feature_vector = pred_features.to_vector()

        # ── Step 2: snapshot shared state under lock (no heavy work here) ───
        with self._stats_lock:
            _stats_loaded    = self._statistics_loaded
            _mean            = self._mean
            _inv_cov         = self._inv_cov
            _feature_vectors = self._feature_vectors
            _knn_model       = self._knn_model

        # ── Step 2b: if not loaded, compute OUTSIDE lock then re-snapshot ───
        if not _stats_loaded:
            self._compute_statistics()          # version-guarded internally
            with self._stats_lock:              # fresh snapshot after compute
                _stats_loaded    = self._statistics_loaded
                _mean            = self._mean
                _inv_cov         = self._inv_cov
                _feature_vectors = self._feature_vectors
                _knn_model       = self._knn_model

        # ── Step 3: guard — insufficient data ───────────────────────────────
        if not _stats_loaded or _mean is None or _inv_cov is None:
            return OODResult(
                ood_score=1.0,
                trust_level=TrustLevel.UNKNOWN,
                mahalanobis_distance=float("inf"),
                knn_distance=float("inf"),
                is_in_distribution=False,
                empirical_support_count=0,
                confidence_downgrade_factor=0.4,
            )

        # ── Step 4: all computation uses ONLY locals — zero self._* reads ───
        mahal_dist    = self._mahalanobis_distance(feature_vector, _mean, _inv_cov)
        knn_dist      = self._knn_distance(feature_vector, _knn_model, _feature_vectors)
        ood_score     = self._distance_to_ood_score(mahal_dist, knn_dist, _mean)
        support_count = self._compute_empirical_support(feature_vector, _feature_vectors)
        trust_level   = self._compute_trust_level(ood_score, support_count)
        downgrade     = self._compute_confidence_downgrade(ood_score, trust_level)

        return OODResult(
            ood_score=ood_score,
            trust_level=trust_level,
            mahalanobis_distance=mahal_dist,
            knn_distance=knn_dist,
            is_in_distribution=ood_score < self.ood_threshold,
            empirical_support_count=support_count,
            confidence_downgrade_factor=downgrade,
        )
    
    def detect_from_record(self, record: DeploymentRecord) -> OODResult:
        """Detect OOD from a deployment record.
        
        Args:
            record: Deployment record with model and device info
            
        Returns:
            OODResult with trust level and OOD score
        """
        # Parse model architecture
        model_data = self._parse_model_architecture(record.model_architecture)
        model_features = ModelArchitectureFeatures(
            operator_histogram=model_data.get('operator_histogram', {}),
            max_depth=model_data.get('max_depth', 0),
            avg_depth=model_data.get('avg_depth', 0.0),
            node_count=model_data.get('node_count', 0),
            parameter_count=record.parameter_count or model_data.get('parameter_count', 0),
            parameter_scale_class=model_data.get('parameter_scale_class', 'unknown'),
            opset_version=record.opset_version or model_data.get('opset_version', 0),
            has_dynamic_shapes=model_data.get('has_dynamic_shapes', False),
            has_static_shapes=model_data.get('has_static_shapes', False),
            input_rank=model_data.get('input_rank', 0),
            output_rank=model_data.get('output_rank', 0),
        )
        
        # Parse device fingerprint
        device_data = self._parse_device_fingerprint(record.device_fingerprint)
        device_features = DeviceFingerprintFeatures(
            cpu_cores=device_data.get('cpu_cores', 0),
            cpu_freq_mhz=device_data.get('cpu_freq_mhz', 0.0),
            gpu_memory_mb=device_data.get('gpu_memory_mb', 0),
            system_memory_mb=device_data.get('system_memory_mb', 0),
            matmul_throughput=device_data.get('matmul_throughput', 0.0),
            memory_bandwidth=device_data.get('memory_bandwidth', 0.0),
            fp16_vs_fp32_ratio=device_data.get('fp16_vs_fp32_ratio', 0.0),
            kernel_launch_overhead_us=device_data.get('kernel_launch_overhead_us', 0.0),
            platform=device_data.get('platform', 'unknown'),
            device_class=device_data.get('device_class', 'unknown'),
        )
        
        return self.detect(
            model_features=model_features,
            device_features=device_features,
            runtime=record.runtime,
            precision=record.precision,
        )
    
    def refresh_statistics(self) -> None:
        """Refresh distribution statistics from database.

        This is the ONLY legitimate place where _stats_inflight is reset to None.
        It is a deliberate, controlled admin call — not a hot-path arrival.

        Sequence:
          1. Bump _stats_version under _stats_lock → invalidates any in-flight
             versioned commit (that thread will discard its stale result).
          2. Reset _stats_inflight to None under _stats_inflight_lock → allows
             _compute_statistics() to claim a fresh entry.  Safe here because
             refresh_statistics() is an explicit "I want new stats" call, not a
             concurrent late arrival.  Any thread that previously obtained the old
             entry already has its Event reference and can still call event.wait()
             (which returns immediately — the old event stays set permanently).
          3. Call _compute_statistics() outside both locks — single-flight +
             versioning prevent duplicate computation.
        """
        with self._stats_lock:
            self._statistics_loaded = False
            self._stats_version += 1          # invalidate any concurrent in-flight compute

        # Reset the inflight slot so _compute_statistics() can start a fresh entry.
        # This is NOT hot-path cleanup — it is an explicit administrative reset on
        # a fully separate code path from the publish block in _compute_statistics().
        with self._stats_inflight_lock:
            self._stats_inflight = None

        self._compute_statistics()            # called outside both locks — version-guarded

    def get_statistics_info(self) -> dict[str, Any]:
        """Snapshot all stat fields under lock and return a plain dict."""
        with self._stats_lock:
            return {
                "statistics_loaded":      self._statistics_loaded,
                "mean_is_none":           self._mean is None,
                "cov_is_none":            self._cov is None,
                "feature_vectors_is_none": self._feature_vectors is None,
                "knn_model_is_none":      self._knn_model is None,
                "feature_vector_dim":     (len(self._mean) if self._mean is not None else 0),
                "sample_count":           (len(self._feature_vectors) if self._feature_vectors is not None else 0),
            }


def create_ood_detector(
    db_path: str = "deployment_history.db",
    ood_threshold: float = 0.7,
    min_samples: int = 50,
) -> OODDetector:
    """Factory function to create OOD detector."""
    detector = OODDetector(
        db_path=db_path,
        ood_threshold=ood_threshold,
        min_samples_for_detection=min_samples,
    )
    detector.refresh_statistics()   # uses lock — safe under concurrency
    return detector

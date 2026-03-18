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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

from src.core.optional_dependencies import SKLEARN

if TYPE_CHECKING:
    from src.core.device_profiler import DeviceProfile
    from src.core.ground_truth_db import DeploymentRecord


logger = logging.getLogger(__name__)


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
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for distance computation."""
        # Standard operator types to include
        standard_ops = [
            'Conv', 'ConvTranspose', 'MatMul', 'Gemm', 'Add', 'Mul', 'Relu',
            'BatchNormalization', 'MaxPool', 'AveragePool', 'Reshape', 'Transpose',
            'Concat', 'Split', 'Gather', 'Cast', 'Clip', 'Sigmoid', 'Tanh',
            'Softmax', 'LayerNormalization', 'Attention', 'QLinearMatMul',
            'DequantizeLinear', 'QuantizeLinear', 'Identity', 'Dropout'
        ]
        
        # Build histogram vector
        hist_vector = [self.operator_histogram.get(op, 0) for op in standard_ops]
        
        # Normalize histogram by total node count
        total = sum(hist_vector) + 1e-8
        hist_vector = [x / total for x in hist_vector]
        
        # Add structural features (normalized)
        depth_norm = min(self.max_depth / 50, 1.0)
        nodes_norm = min(np.log1p(self.node_count) / 10, 1.0)
        params_norm = min(np.log1p(self.parameter_count) / 20, 1.0)
        
        # Encode parameter scale
        scale_encoding = {
            'small': 0.0, 'medium': 0.33, 'large': 0.67, 'xlarge': 1.0, 'unknown': 0.5
        }
        scale_norm = scale_encoding.get(self.parameter_scale_class, 0.5)
        
        # Opset version (normalized to common range)
        opset_norm = min(self.opset_version / 20, 1.0)
        
        # Shape flags
        dynamic_flag = 1.0 if self.has_dynamic_shapes else 0.0
        static_flag = 1.0 if self.has_static_shapes else 0.0
        
        # Rank features
        input_rank_norm = min(self.input_rank / 10, 1.0)
        output_rank_norm = min(self.output_rank / 10, 1.0)
        
        return np.array([
            *hist_vector,  # 28 dimensions
            depth_norm,    # 1
            nodes_norm,    # 1
            params_norm,   # 1
            scale_norm,    # 1
            opset_norm,    # 1
            dynamic_flag,  # 1
            static_flag,   # 1
            input_rank_norm,  # 1
            output_rank_norm, # 1
        ], dtype=np.float32)


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
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for distance computation."""
        # Normalize compute features
        cpu_cores_norm = min(self.cpu_cores / 128, 1.0)
        cpu_freq_norm = min(self.cpu_freq_mhz / 5000, 1.0)
        gpu_mem_norm = min(np.log1p(self.gpu_memory_mb) / 12, 1.0)
        sys_mem_norm = min(np.log1p(self.system_memory_mb) / 12, 1.0)
        
        # Normalize benchmarks
        throughput_norm = min(self.matmul_throughput / 1000, 1.0)
        bandwidth_norm = min(self.memory_bandwidth / 500, 1.0)
        fp16_ratio_norm = min(self.fp16_vs_fp32_ratio, 1.0)
        overhead_norm = min(self.kernel_launch_overhead_us / 100, 1.0)
        
        # Encode platform
        platform_encoding = {
            'Linux': 0.0, 'Windows': 0.33, 'Darwin': 0.67, 'unknown': 0.5
        }
        platform_norm = platform_encoding.get(self.platform, 0.5)
        
        # Encode device class
        class_encoding = {
            'cpu': 0.0, 'cuda': 0.25, 'rocm': 0.5, 'npu': 0.75, 'unknown': 0.5
        }
        class_norm = class_encoding.get(self.device_class, 0.5)
        
        return np.array([
            cpu_cores_norm,
            cpu_freq_norm,
            gpu_mem_norm,
            sys_mem_norm,
            throughput_norm,
            bandwidth_norm,
            fp16_ratio_norm,
            overhead_norm,
            platform_norm,
            class_norm,
        ], dtype=np.float32)


@dataclass
class PredictionFeatures:
    """Combined feature vector for a prediction."""
    model_features: ModelArchitectureFeatures
    device_features: DeviceFingerprintFeatures
    runtime: str
    precision: str
    
    def to_vector(self) -> np.ndarray:
        """Combine model and device features."""
        model_vec = self.model_features.to_vector()
        device_vec = self.device_features.to_vector()
        
        # Encode runtime
        runtime_encoding = {
            'onnxruntime': 0.0, 'tensorrt': 0.25, 'openvino': 0.5,
            'tflite': 0.75, 'unknown': 1.0
        }
        runtime_norm = runtime_encoding.get(self.runtime.lower(), 1.0)
        
        # Encode precision
        precision_encoding = {
            'fp32': 0.0, 'fp16': 0.33, 'int8': 0.67, 'mixed': 1.0, 'unknown': 0.5
        }
        precision_norm = precision_encoding.get(self.precision.lower(), 0.5)
        
        return np.concatenate([
            model_vec,
            device_vec,
            [runtime_norm, precision_norm]
        ])


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
        
        # Cached statistics
        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None
        self._feature_vectors: Optional[np.ndarray] = None
        self._knn_model: Optional[Any] = None
        self._statistics_loaded = False
        
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
    
    def _build_feature_vectors(self, records: list[dict]) -> np.ndarray:
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
            return np.zeros((0, 40))
        
        return np.vstack(vectors)
    
    def _compute_statistics(self):
        """Compute distribution statistics from deployment history."""
        records = self._load_deployment_history()
        
        if len(records) < self.min_samples_for_detection:
            self._statistics_loaded = False
            return
        
        self._feature_vectors = self._build_feature_vectors(records)
        
        # Compute mean and covariance
        self._mean = np.mean(self._feature_vectors, axis=0)
        
        # Add regularization to covariance matrix
        cov = np.cov(self._feature_vectors, rowvar=False)
        reg = np.eye(cov.shape[0]) * 1e-6
        self._cov = cov + reg
        
        # Compute inverse covariance for Mahalanobis distance
        try:
            self._inv_cov = np.linalg.inv(self._cov)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            self._inv_cov = np.linalg.pinv(self._cov)
        
        # Initialize k-NN model if available
        if SKLEARN.available:
            from sklearn.neighbors import NearestNeighbors
            self._knn_model = NearestNeighbors(n_neighbors=self.knn_neighbors)
            self._knn_model.fit(self._feature_vectors)
        
        self._statistics_loaded = True
    
    def _mahalanobis_distance(self, feature_vector: np.ndarray) -> float:
        """Compute Mahalanobis distance to distribution."""
        if self._mean is None or self._inv_cov is None:
            return float('inf')
        
        diff = feature_vector - self._mean
        try:
            dist = mahalanobis(feature_vector, self._mean, self._inv_cov)
            return float(dist)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("Mahalanobis distance computation failed: %s", exc)
            return float('inf')
    
    def _knn_distance(self, feature_vector: np.ndarray) -> float:
        """Compute average distance to k nearest neighbors."""
        if self._knn_model is None or self._feature_vectors is None:
            return float('inf')
        
        distances, _ = self._knn_model.kneighbors([feature_vector])
        return float(np.mean(distances[0]))
    
    def _compute_empirical_support(self, feature_vector: np.ndarray, radius: float = 0.1) -> int:
        """Count number of similar historical samples."""
        if self._feature_vectors is None:
            return 0
        
        distances = np.linalg.norm(self._feature_vectors - feature_vector, axis=1)
        return int(np.sum(distances < radius))
    
    def _distance_to_ood_score(self, mahal_dist: float, knn_dist: float) -> float:
        """Convert distances to OOD score in [0, 1]."""
        df = len(self._mean) if self._mean is not None else 40
        try:
            p_value = 1 - chi2.cdf(mahal_dist ** 2, df)
            mahal_score = 1 - p_value
        except (ValueError, RuntimeError) as exc:
            logger.debug("Chi-squared computation failed, using fallback: %s", exc)
            mahal_score = min(mahal_dist / 10, 1.0)
        
        # Normalize k-NN distance
        knn_score = min(knn_dist / 0.5, 1.0)
        
        # Combine scores (weighted average)
        combined_score = 0.6 * mahal_score + 0.4 * knn_score
        
        return float(np.clip(combined_score, 0.0, 1.0))
    
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
        
        Args:
            model_features: Model architecture features
            device_features: Device profiling features
            runtime: Runtime being used
            precision: Precision being used
            
        Returns:
            OODResult with trust level and OOD score
        """
        # Load statistics if not already loaded
        if not self._statistics_loaded:
            self._compute_statistics()
        
        # Build feature vector
        pred_features = PredictionFeatures(
            model_features=model_features,
            device_features=device_features,
            runtime=runtime,
            precision=precision,
        )
        feature_vector = pred_features.to_vector()
        
        # If insufficient data, return unknown trust
        if not self._statistics_loaded or self._mean is None:
            return OODResult(
                ood_score=1.0,
                trust_level=TrustLevel.UNKNOWN,
                mahalanobis_distance=float('inf'),
                knn_distance=float('inf'),
                is_in_distribution=False,
                empirical_support_count=0,
                confidence_downgrade_factor=0.4,
            )
        
        # Compute distances
        mahal_dist = self._mahalanobis_distance(feature_vector)
        knn_dist = self._knn_distance(feature_vector)
        
        # Compute OOD score
        ood_score = self._distance_to_ood_score(mahal_dist, knn_dist)
        
        # Compute empirical support
        support_count = self._compute_empirical_support(feature_vector)
        
        # Determine trust level
        trust_level = self._compute_trust_level(ood_score, support_count)
        
        # Compute confidence downgrade
        downgrade_factor = self._compute_confidence_downgrade(ood_score, trust_level)
        
        return OODResult(
            ood_score=ood_score,
            trust_level=trust_level,
            mahalanobis_distance=mahal_dist,
            knn_distance=knn_dist,
            is_in_distribution=ood_score < self.ood_threshold,
            empirical_support_count=support_count,
            confidence_downgrade_factor=downgrade_factor,
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
    
    def refresh_statistics(self):
        """Refresh distribution statistics from database."""
        self._statistics_loaded = False
        self._compute_statistics()
    
    def get_statistics_info(self) -> dict[str, Any]:
        """Get information about current statistics."""
        return {
            'statistics_loaded': self._statistics_loaded,
            'mean_is_none': self._mean is None,
            'cov_is_none': self._cov is None,
            'feature_vectors_is_none': self._feature_vectors is None,
            'knn_model_is_none': self._knn_model is None,
            'feature_vector_dim': len(self._mean) if self._mean is not None else 0,
            'sample_count': len(self._feature_vectors) if self._feature_vectors is not None else 0,
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
    detector._compute_statistics()
    return detector

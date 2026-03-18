"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""Confidence calibration using isotonic regression and reliability curves."""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression
import pickle
from pathlib import Path

from src.core.ground_truth_db import GroundTruthDatabase, DeploymentRecord


def adjust_confidence_with_history(
    confidence_score: float,
    runtime: str,
    stats: Dict[str, Any],
) -> float:
    """
    Adjust confidence score based on historical calibration statistics.
    
    Args:
        confidence_score: Raw confidence score
        runtime: Runtime name
        stats: Calibration statistics from compute_calibration_stats
        
    Returns:
        Adjusted confidence score
    """
    # Get runtime-specific calibration data
    runtime_stats = stats.get("runtime_stats", {})
    runtime_data = runtime_stats.get(runtime, {})
    
    # Get calibration metrics
    observed_accuracy = runtime_data.get("observed_accuracy", 0.5)
    predicted_accuracy = runtime_data.get("predicted_accuracy", 0.5)
    sample_count = runtime_data.get("sample_count", 0)
    
    # If we have enough samples, adjust confidence
    if sample_count >= 10:
        # Weight adjustment by sample count (more samples = more adjustment)
        weight = min(sample_count / 100.0, 1.0)
        
        # Adjust towards observed accuracy
        adjustment = (observed_accuracy - predicted_accuracy) * weight
        adjusted = confidence_score + adjustment
        
        # Clamp to valid range
        return max(0.0, min(1.0, adjusted))
    
    # Not enough data, return original
    return confidence_score


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""
    raw_confidence: float
    calibrated_confidence: float
    adjustment: float
    bin_assignment: Optional[int] = None
    bin_observed_frequency: Optional[float] = None


@dataclass
class CalibrationMetrics:
    """Metrics for calibration quality."""
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score_raw: float
    brier_score_calibrated: float
    improvement: float
    sample_size: int


class ConfidenceCalibrator:
    """Calibrate prediction confidence using empirical data."""
    
    def __init__(self, db: GroundTruthDatabase, recalibration_threshold: int = 50):
        self.db = db
        self.recalibration_threshold = recalibration_threshold
        self.isotonic_regressor: Optional[IsotonicRegression] = None
        self.bin_calibrations: Dict[int, float] = {}
        self.n_bins = 10
        self._last_calibration_count = 0
        self._calibration_cache_path = Path("calibration_cache.pkl")
        
        # Try to load cached calibration
        self._load_cache()
    
    def calibrate(self, raw_confidence: float) -> CalibrationResult:
        """Calibrate a raw confidence score."""
        if self.isotonic_regressor is not None:
            # Use isotonic regression
            calibrated = self._isotonic_calibrate(raw_confidence)
            bin_idx = int(raw_confidence * self.n_bins) % self.n_bins
            bin_obs = self.bin_calibrations.get(bin_idx, raw_confidence)
        else:
            # Use bin-based calibration
            calibrated = self._bin_based_calibrate(raw_confidence)
            bin_idx = int(raw_confidence * self.n_bins) % self.n_bins
            bin_obs = self.bin_calibrations.get(bin_idx, raw_confidence)
        
        # Clamp to valid range
        calibrated = max(0.01, min(0.99, calibrated))
        
        return CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            adjustment=calibrated - raw_confidence,
            bin_assignment=bin_idx,
            bin_observed_frequency=bin_obs
        )
    
    def _isotonic_calibrate(self, raw_confidence: float) -> float:
        """Calibrate using isotonic regression."""
        try:
            calibrated = self.isotonic_regressor.predict([raw_confidence])[0]
            return float(calibrated)
        except (AttributeError, ValueError, RuntimeError) as exc:
            logger.warning("Isotonic calibration failed, using raw confidence: %s", exc)
            return raw_confidence
    
    def _bin_based_calibrate(self, raw_confidence: float) -> float:
        """Calibrate using bin-based method."""
        bin_idx = int(raw_confidence * self.n_bins)
        bin_idx = min(bin_idx, self.n_bins - 1)
        
        if bin_idx in self.bin_calibrations:
            return self.bin_calibrations[bin_idx]
        
        return raw_confidence
    
    def recalibrate(self) -> CalibrationMetrics:
        """Recalibrate based on current database."""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        # Get all records
        cursor.execute("""
            SELECT predicted_success_probability, actual_success
            FROM deployment_history
            WHERE predicted_success_probability IS NOT NULL
        """)
        
        rows = cursor.fetchall()
        
        if not rows:
            return self._empty_metrics()
        
        predictions = np.array([float(r[0]) for r in rows])
        outcomes = np.array([1.0 if r[1] else 0.0 for r in rows])
        
        # Compute raw metrics
        raw_ece = self._compute_ece(predictions, outcomes)
        raw_brier = self._compute_brier(predictions, outcomes)
        
        try:
            self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_regressor.fit(predictions, outcomes)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Isotonic regression fitting failed: %s", exc)
            self.isotonic_regressor = None
        
        # Compute bin calibrations
        self._compute_bin_calibrations(predictions, outcomes)
        
        # Calibrate predictions
        if self.isotonic_regressor is not None:
            calibrated = self.isotonic_regressor.predict(predictions)
        else:
            calibrated = np.array([self._bin_based_calibrate(p) for p in predictions])
        
        # Compute calibrated metrics
        calibrated_ece = self._compute_ece(calibrated, outcomes)
        calibrated_brier = self._compute_brier(calibrated, outcomes)
        
        self._last_calibration_count = len(rows)
        self._save_cache()
        
        return CalibrationMetrics(
            expected_calibration_error=calibrated_ece,
            maximum_calibration_error=self._compute_mce(calibrated, outcomes),
            brier_score_raw=raw_brier,
            brier_score_calibrated=calibrated_brier,
            improvement=raw_brier - calibrated_brier,
            sample_size=len(rows)
        )
    
    def _compute_bin_calibrations(self, predictions: np.ndarray, outcomes: np.ndarray):
        """Compute calibration for each confidence bin."""
        self.bin_calibrations = {}
        
        for i in range(self.n_bins):
            bin_low = i / self.n_bins
            bin_high = (i + 1) / self.n_bins
            
            mask = (predictions >= bin_low) & (predictions < bin_high)
            
            if mask.sum() > 0:
                observed = outcomes[mask].mean()
                self.bin_calibrations[i] = float(observed)
    
    def _compute_ece(self, predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bins = [[] for _ in range(n_bins)]
        
        for pred, outcome in zip(predictions, outcomes):
            bin_idx = min(int(pred * n_bins), n_bins - 1)
            bins[bin_idx].append(outcome)
        
        weighted_errors = []
        total = len(predictions)
        
        for i, bin_data in enumerate(bins):
            if bin_data:
                conf_low = i / n_bins
                conf_high = (i + 1) / n_bins
                conf_mid = (conf_low + conf_high) / 2
                observed = np.mean(bin_data)
                
                weight = len(bin_data) / total
                error = abs(conf_mid - observed)
                weighted_errors.append(weight * error)
        
        return float(sum(weighted_errors))
    
    def _compute_mce(self, predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Compute Maximum Calibration Error."""
        bins = [[] for _ in range(n_bins)]
        
        for pred, outcome in zip(predictions, outcomes):
            bin_idx = min(int(pred * n_bins), n_bins - 1)
            bins[bin_idx].append(outcome)
        
        max_error = 0.0
        
        for i, bin_data in enumerate(bins):
            if bin_data:
                conf_low = i / n_bins
                conf_high = (i + 1) / n_bins
                conf_mid = (conf_low + conf_high) / 2
                observed = np.mean(bin_data)
                
                error = abs(conf_mid - observed)
                max_error = max(max_error, error)
        
        return float(max_error)
    
    def _compute_brier(self, predictions: np.ndarray, outcomes: np.ndarray) -> float:
        """Compute Brier score."""
        return float(np.mean((predictions - outcomes) ** 2))
    
    def _empty_metrics(self) -> CalibrationMetrics:
        """Return empty metrics."""
        return CalibrationMetrics(
            expected_calibration_error=0.0,
            maximum_calibration_error=0.0,
            brier_score_raw=0.0,
            brier_score_calibrated=0.0,
            improvement=0.0,
            sample_size=0
        )
    
    def should_recalibrate(self) -> bool:
        """Check if recalibration is needed."""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM deployment_history")
        count = cursor.fetchone()[0]
        
        return count - self._last_calibration_count >= self.recalibration_threshold
    
    def get_reliability_curve(self) -> List[Dict[str, float]]:
        """Get reliability curve data."""
        return self.db.compute_reliability_curve(self.n_bins)
    
    def get_calibration_summary(self) -> Dict[str, any]:
        """Get summary of current calibration state."""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(predicted_success_probability) as avg_pred,
                SUM(CASE WHEN actual_success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as obs_rate
            FROM deployment_history
        """)
        
        row = cursor.fetchone()
        
        return {
            "total_records": row[0],
            "average_predicted_probability": row[1] or 0.0,
            "observed_success_rate": row[2] or 0.0,
            "calibration_gap": abs((row[1] or 0.0) - (row[2] or 0.0)),
            "calibration_method": "isotonic_regression" if self.isotonic_regressor else "bin_based",
            "bins_calibrated": len(self.bin_calibrations),
            "last_calibration_count": self._last_calibration_count
        }
    
    def _save_cache(self):
        """Save calibration to cache."""
        try:
            cache_data = {
                "isotonic_regressor": self.isotonic_regressor,
                "bin_calibrations": self.bin_calibrations,
                "last_calibration_count": self._last_calibration_count
            }
            with open(self._calibration_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except (OSError, pickle.PickleError, AttributeError) as exc:
            logger.warning("Failed to save calibration cache: %s", exc)
    
    def _load_cache(self):
        """Load calibration from cache."""
        try:
            if self._calibration_cache_path.exists():
                with open(self._calibration_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.isotonic_regressor = cache_data.get("isotonic_regressor")
                    self.bin_calibrations = cache_data.get("bin_calibrations", {})
                    self._last_calibration_count = cache_data.get("last_calibration_count", 0)
        except (OSError, pickle.PickleError, json.JSONDecodeError, AttributeError) as exc:
            logger.debug("Failed to load calibration cache, starting fresh: %s", exc)


class TemperatureScaling:
    """Temperature scaling for calibration (simpler alternative to isotonic)."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        scaled = logits / self.temperature
        # Apply sigmoid for binary classification
        return 1.0 / (1.0 + np.exp(-scaled))
    
    def fit(self, predictions: np.ndarray, outcomes: np.ndarray):
        """Find optimal temperature."""
        from scipy.optimize import minimize
        
        def loss(temp):
            self.temperature = temp
            calibrated = self.calibrate(predictions)
            return self._brier_loss(calibrated, outcomes)
        
        result = minimize(loss, x0=1.0, bounds=[(0.1, 10.0)])
        self.temperature = result.x[0]
    
    def _brier_loss(self, predictions: np.ndarray, outcomes: np.ndarray) -> float:
        """Compute Brier loss."""
        return float(np.mean((predictions - outcomes) ** 2))


def calibrate_prediction(
    raw_confidence: float,
    calibrator: ConfidenceCalibrator
) -> Tuple[float, Dict[str, any]]:
    """
    Calibrate a single prediction and return metadata.
    
    Returns:
        Tuple of (calibrated_confidence, metadata)
    """
    result = calibrator.calibrate(raw_confidence)
    
    metadata = {
        "raw_confidence": raw_confidence,
        "calibrated_confidence": result.calibrated_confidence,
        "adjustment": result.adjustment,
        "bin": result.bin_assignment,
        "bin_observed": result.bin_observed_frequency
    }
    
    return result.calibrated_confidence, metadata

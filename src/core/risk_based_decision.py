from __future__ import annotations
"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""
Risk-Based Decision Engine

This module implements a unified risk assessment system for deployment decisions.
Instead of using independent rule checks, it computes a unified risk score from
multiple normalized risk components.

Key principles:
1. Risk is a continuous value in [0, 1], not a binary decision
2. Decision thresholds are based on risk score
3. Confidence is calibrated from historical accuracy
4. Risk estimates evolve based on deployment outcomes
5. Fail-closed behavior when signals are missing
"""


import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

import numpy as np


# =============================================================================
# Decision Enumerations
# =============================================================================

class DeploymentDecision(str, Enum):
    """Deployment decision based on risk assessment."""
    ALLOW = "ALLOW"
    ALLOW_WITH_CONDITIONS = "ALLOW_WITH_CONDITIONS"
    BLOCK = "BLOCK"
    
    # Backward compatibility aliases
    SUPPORTED = ALLOW
    SUPPORTED_WITH_WARNINGS = ALLOW_WITH_CONDITIONS
    UNSUPPORTED = BLOCK


class FailureMode(str, Enum):
    """Expected failure mode for blocked deployments."""
    OOM = "out_of_memory"
    TIMEOUT = "timeout"
    NUMERICAL_ERROR = "numerical_error"
    INCOMPATIBLE = "incompatible"
    DEVICE_ERROR = "device_error"
    UNKNOWN = "unknown"


# =============================================================================
# Risk Components
# =============================================================================

@dataclass
class RiskComponents:
    """Normalized risk components, each in range [0, 1]."""
    memory_risk: float = 0.0
    latency_risk: float = 0.0
    operator_compatibility_risk: float = 0.0
    numerical_instability_risk: float = 0.0
    device_capacity_risk: float = 0.0
    
    def validate(self) -> bool:
        """Validate all components are in [0, 1]."""
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                return False
        return True
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RiskWeights:
    """Weights for combining risk components."""
    memory_weight: float = 0.25
    latency_weight: float = 0.20
    operator_weight: float = 0.25
    numerical_weight: float = 0.15
    device_weight: float = 0.15
    
    def validate(self) -> bool:
        """Validate weights are non-negative and sum to approximately 1."""
        weights = [self.memory_weight, self.latency_weight, 
                  self.operator_weight, self.numerical_weight, self.device_weight]
        
        if any(w < 0 for w in weights):
            return False
        
        total = sum(weights)
        return 0.9 <= total <= 1.1  # Allow small floating point errors
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# Risk Assessment Result
# =============================================================================

@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    risk_score: float  # Overall risk in [0, 1]
    components: RiskComponents
    weights: RiskWeights
    dominant_risk: str  # Name of the highest risk component
    expected_failure_mode: FailureMode
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "components": self.components.to_dict(),
            "weights": self.weights.to_dict(),
            "dominant_risk": self.dominant_risk,
            "expected_failure_mode": self.expected_failure_mode.value,
        }


# =============================================================================
# Decision Result with Mandatory Explanations
# =============================================================================

@dataclass
class DecisionResult:
    """Complete decision result with mandatory explanations."""
    decision: DeploymentDecision
    risk_assessment: RiskAssessment
    confidence: float  # Calibrated probability in [0, 1]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "risk_score": self.risk_assessment.risk_score,
            "confidence": self.confidence,
            "dominant_risk": self.risk_assessment.dominant_risk,
            "expected_failure_mode": self.risk_assessment.expected_failure_mode.value,
            "risk_components": self.risk_assessment.components.to_dict(),
            "timestamp": self.timestamp,
        }
    
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        return f"""Decision: {self.decision.value}
Risk Score: {self.risk_assessment.risk_score:.3f}
Confidence: {self.confidence:.3f}
Dominant Risk: {self.risk_assessment.dominant_risk}
Expected Failure: {self.risk_assessment.expected_failure_mode.value}"""


# =============================================================================
# Risk Calculator
# =============================================================================

class RiskCalculator:
    """
    Computes unified risk score from normalized risk components.
    
    Risk calculation:
    risk = weighted_sum(components)
    
    Decision thresholds:
    - risk < 0.35 → ALLOW
    - 0.35 ≤ risk < 0.65 → ALLOW_WITH_CONDITIONS
    - risk ≥ 0.65 → BLOCK
    """
    
    # Decision thresholds
    THRESHOLD_ALLOW = 0.35
    THRESHOLD_CONDITIONAL = 0.65
    
    def __init__(self, weights: Optional[RiskWeights] = None):
        self.weights = weights or RiskWeights()
        assert self.weights.validate(), "Invalid risk weights"
    
    def compute_risk_score(self, components: RiskComponents) -> float:
        """Compute unified risk score from components."""
        assert components.validate(), "Invalid risk components"
        
        # Weighted sum
        risk = (
            self.weights.memory_weight * components.memory_risk +
            self.weights.latency_weight * components.latency_risk +
            self.weights.operator_weight * components.operator_compatibility_risk +
            self.weights.numerical_weight * components.numerical_instability_risk +
            self.weights.device_weight * components.device_capacity_risk
        )
        
        return np.clip(risk, 0.0, 1.0)
    
    def get_dominant_risk(self, components: RiskComponents) -> str:
        """Get the name of the highest risk component."""
        component_dict = components.to_dict()
        return max(component_dict.items(), key=lambda x: x[1])[0]
    
    def determine_failure_mode(self, components: RiskComponents) -> FailureMode:
        """Determine expected failure mode from risk components."""
        dominant = self.get_dominant_risk(components)
        
        failure_map = {
            "memory_risk": FailureMode.OOM,
            "latency_risk": FailureMode.TIMEOUT,
            "operator_compatibility_risk": FailureMode.INCOMPATIBLE,
            "numerical_instability_risk": FailureMode.NUMERICAL_ERROR,
            "device_capacity_risk": FailureMode.DEVICE_ERROR,
        }
        
        return failure_map.get(dominant, FailureMode.UNKNOWN)
    
    def assess_risk(self, components: RiskComponents) -> RiskAssessment:
        """Perform complete risk assessment."""
        risk_score = self.compute_risk_score(components)
        dominant_risk = self.get_dominant_risk(components)
        failure_mode = self.determine_failure_mode(components)
        
        return RiskAssessment(
            risk_score=risk_score,
            components=components,
            weights=self.weights,
            dominant_risk=dominant_risk,
            expected_failure_mode=failure_mode
        )
    
    def make_decision(self, risk_assessment: RiskAssessment) -> DeploymentDecision:
        """Make deployment decision based on risk assessment."""
        # Special case: if operator compatibility risk is very high (>0.4), always block
        # This indicates true incompatibility (cannot run at all)
        if risk_assessment.components.operator_compatibility_risk > 0.4:
            return DeploymentDecision.BLOCK
        
        if risk_assessment.risk_score < self.THRESHOLD_ALLOW:
            return DeploymentDecision.ALLOW
        elif risk_assessment.risk_score < self.THRESHOLD_CONDITIONAL:
            return DeploymentDecision.ALLOW_WITH_CONDITIONS
        else:
            return DeploymentDecision.BLOCK


# =============================================================================
# Confidence Calibrator
# =============================================================================

class ConfidenceCalibrator:
    """
    Calibrates confidence as probability of correct prediction.
    
    Confidence = P(prediction is correct)
    
    Computed from historical accuracy of similar risk bucket.
    """
    
    def __init__(self, history_path: Optional[Path] = None):
        self.history_path = history_path or Path("risk_assessment_history.jsonl")
        self._lock = threading.Lock()
        
        # Risk buckets for calibration
        self._num_buckets = 10
        self._bucket_edges = np.linspace(0, 1, self._num_buckets + 1)
        
        # Calibration data: {bucket_index: (correct, total)}
        self._calibration_data: Dict[int, Tuple[int, int]] = {}
        
        # Load existing history
        self._load_history()
    
    def _load_history(self) -> None:
        """Load calibration history from file."""
        if not self.history_path.exists():
            return
        
        try:
            with open(self.history_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        self._update_calibration_from_record(record)
        except (json.JSONDecodeError, OSError) as e:
            import logging
            logging.getLogger(__name__).warning(f"Calibration history corrupted, starting fresh: {e}")
    
    def _get_bucket_index(self, risk_score: float) -> int:
        """Get bucket index for a given risk score."""
        return min(int(risk_score * self._num_buckets), self._num_buckets - 1)
    
    def _update_calibration_from_record(self, record: Dict[str, Any]) -> None:
        """Update calibration data from a historical record."""
        risk_score = record.get("risk_score", 0.5)
        was_correct = record.get("prediction_correct", False)
        
        bucket_idx = self._get_bucket_index(risk_score)
        
        if bucket_idx not in self._calibration_data:
            self._calibration_data[bucket_idx] = (0, 0)
        
        correct, total = self._calibration_data[bucket_idx]
        self._calibration_data[bucket_idx] = (
            correct + (1 if was_correct else 0),
            total + 1
        )
    
    def get_confidence(self, risk_score: float) -> float:
        """
        Get calibrated confidence for a risk score.
        
        Returns probability that prediction is correct.
        """
        bucket_idx = self._get_bucket_index(risk_score)
        
        if bucket_idx not in self._calibration_data:
            # No data for this bucket - use prior
            return 0.5
        
        correct, total = self._calibration_data[bucket_idx]
        
        if total == 0:
            return 0.5
        
        # Laplace smoothing for small samples
        confidence = (correct + 1) / (total + 2)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def record_outcome(
        self,
        risk_score: float,
        decision: DeploymentDecision,
        actual_success: bool
    ) -> None:
        """
        Record deployment outcome for calibration.
        
        A prediction is "correct" if:
        - Decision was ALLOW and deployment succeeded
        - Decision was BLOCK and deployment would have failed
        """
        # Determine if prediction was correct
        predicted_success = decision == DeploymentDecision.ALLOW
        prediction_correct = (predicted_success == actual_success)
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_score": risk_score,
            "decision": decision.value,
            "actual_success": actual_success,
            "prediction_correct": prediction_correct,
        }
        
        with self._lock:
            # Update calibration data
            self._update_calibration_from_record(record)
            
            # Append to history file
            with open(self.history_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        stats = {}
        
        for bucket_idx in range(self._num_buckets):
            if bucket_idx in self._calibration_data:
                correct, total = self._calibration_data[bucket_idx]
                stats[f"bucket_{bucket_idx}"] = {
                    "range": f"{bucket_idx/self._num_buckets:.1f}-{(bucket_idx+1)/self._num_buckets:.1f}",
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total > 0 else 0.0,
                }
        
        return stats


# =============================================================================
# Adaptive Risk Weights
# =============================================================================

class AdaptiveRiskWeights:
    """
    Adapts risk weights based on deployment outcomes.
    
    Uses exponential moving average to update per-feature reliability.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize adaptive weights.
        
        Args:
            alpha: Learning rate for EMA (0 < alpha <= 1)
        """
        self.alpha = alpha
        self._weights = RiskWeights()
        self._feature_reliability: Dict[str, float] = {
            "memory_risk": 0.5,
            "latency_risk": 0.5,
            "operator_compatibility_risk": 0.5,
            "numerical_instability_risk": 0.5,
            "device_capacity_risk": 0.5,
        }
    
    def update_from_outcome(
        self,
        components: RiskComponents,
        actual_success: bool
    ) -> None:
        """
        Update feature reliability based on deployment outcome.
        
        If deployment succeeded, decrease weight of high-risk features.
        If deployment failed, increase weight of high-risk features.
        """
        component_dict = components.to_dict()
        
        for feature_name, risk_value in component_dict.items():
            current_reliability = self._feature_reliability[feature_name]
            
            # Update reliability using EMA
            if actual_success:
                # Success: high risk was wrong, decrease reliability
                error = risk_value  # High risk should have meant failure
                new_reliability = (1 - self.alpha) * current_reliability + self.alpha * (1 - error)
            else:
                # Failure: high risk was correct, increase reliability
                error = risk_value
                new_reliability = (1 - self.alpha) * current_reliability + self.alpha * error
            
            self._feature_reliability[feature_name] = np.clip(new_reliability, 0.1, 0.9)
        
        # Renormalize weights based on reliability
        self._renormalize_weights()
    
    def _renormalize_weights(self) -> None:
        """Renormalize weights to sum to 1."""
        # Higher reliability = higher weight
        raw_weights = {
            "memory_weight": self._feature_reliability["memory_risk"],
            "latency_weight": self._feature_reliability["latency_risk"],
            "operator_weight": self._feature_reliability["operator_compatibility_risk"],
            "numerical_weight": self._feature_reliability["numerical_instability_risk"],
            "device_weight": self._feature_reliability["device_capacity_risk"],
        }
        
        total = sum(raw_weights.values())
        if total > 0:
            self._weights = RiskWeights(
                memory_weight=raw_weights["memory_weight"] / total,
                latency_weight=raw_weights["latency_weight"] / total,
                operator_weight=raw_weights["operator_weight"] / total,
                numerical_weight=raw_weights["numerical_weight"] / total,
                device_weight=raw_weights["device_weight"] / total,
            )
    
    def get_weights(self) -> RiskWeights:
        """Get current adaptive weights."""
        return self._weights


# =============================================================================
# Risk-Based Decision Engine
# =============================================================================

class RiskBasedDecisionEngine:
    """
    Main decision engine using unified risk assessment.
    
    Features:
    1. Unified risk score from normalized components
    2. Calibrated confidence from historical accuracy
    3. Adaptive weights that evolve with outcomes
    4. Fail-closed behavior for missing signals
    5. Mandatory explanations in output
    """
    
    def __init__(
        self,
        weights: Optional[RiskWeights] = None,
        history_path: Optional[Path] = None,
        adaptive_alpha: float = 0.1
    ):
        self.calculator = RiskCalculator(weights)
        self.calibrator = ConfidenceCalibrator(history_path)
        self.adaptive_weights = AdaptiveRiskWeights(adaptive_alpha)
    
    def assess_decision(
        self,
        components: RiskComponents,
        has_history: bool = True,
        has_profiling: bool = True,
        has_device_metrics: bool = True,
        deployment_profile: Optional[Dict[str, Any]] = None,
        predicted_latency_ms: Optional[float] = None,
    ) -> "DecisionResult":
        """
        Make deployment decision with risk assessment.
        
        Implements fail-closed behavior:
        - If signals missing, decision becomes ALLOW_WITH_CONDITIONS
        - Confidence capped at 0.5 when signals missing
        """
        # Check for missing signals (fail-closed)
        signals_complete = has_history and has_profiling and has_device_metrics
        
        # Apply SLA latency penalty to latency_risk component if deployment_profile provided
        if deployment_profile is not None and predicted_latency_ms is not None:
            target_latency_ms = deployment_profile.get("target_latency_ms")
            if target_latency_ms is not None and float(target_latency_ms) > 0:
                if predicted_latency_ms > float(target_latency_ms):
                    overshoot_ratio = (predicted_latency_ms - float(target_latency_ms)) / float(target_latency_ms)
                    latency_penalty = min(overshoot_ratio * 0.5, 1.0)
                    # Increase latency_risk proportionally
                    new_latency_risk = min(1.0, components.latency_risk + latency_penalty)
                    components = RiskComponents(
                        memory_risk=components.memory_risk,
                        latency_risk=new_latency_risk,
                        operator_compatibility_risk=components.operator_compatibility_risk,
                        numerical_instability_risk=components.numerical_instability_risk,
                        device_capacity_risk=components.device_capacity_risk,
                    )

        # Perform risk assessment
        risk_assessment = self.calculator.assess_risk(components)
        
        # Get calibrated confidence
        raw_confidence = self.calibrator.get_confidence(risk_assessment.risk_score)
        
        # Apply fail-closed cap to confidence
        if not signals_complete:
            confidence = min(raw_confidence, 0.5)
        else:
            confidence = raw_confidence
        
        # Apply SLA utility reduction to confidence if SLA violated
        if deployment_profile is not None and predicted_latency_ms is not None:
            target_latency_ms = deployment_profile.get("target_latency_ms")
            if target_latency_ms is not None and float(target_latency_ms) > 0:
                if predicted_latency_ms > float(target_latency_ms):
                    overshoot_ratio = (predicted_latency_ms - float(target_latency_ms)) / float(target_latency_ms)
                    confidence_penalty = min(overshoot_ratio * 0.3, 0.3)
                    confidence = max(0.0, confidence - confidence_penalty)

        # Make decision
        if not signals_complete:
            # Fail-closed: downgrade to ALLOW_WITH_CONDITIONS
            decision = DeploymentDecision.ALLOW_WITH_CONDITIONS
        else:
            decision = self.calculator.make_decision(risk_assessment)
        
        return DecisionResult(
            decision=decision,
            risk_assessment=risk_assessment,
            confidence=confidence
        )
    
    def record_outcome(
        self,
        risk_score: float,
        decision: DeploymentDecision,
        actual_success: bool,
        components: Optional[RiskComponents] = None
    ) -> None:
        """Record deployment outcome for learning."""
        # Update confidence calibration
        self.calibrator.record_outcome(risk_score, decision, actual_success)
        
        # Update adaptive weights if components provided
        if components is not None:
            self.adaptive_weights.update_from_outcome(components, actual_success)
            
            # Update calculator with new weights
            self.calculator.weights = self.adaptive_weights.get_weights()
    
    def get_updated_weights(self) -> RiskWeights:
        """Get current adaptive weights."""
        return self.adaptive_weights.get_weights()
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        return self.calibrator.get_calibration_stats()


# =============================================================================
# Helper Functions for Computing Risk Components
# =============================================================================

def compute_memory_risk(
    model_memory_mb: float,
    available_memory_mb: float,
    safety_margin: float = 0.2
) -> float:
    """
    Compute memory risk component.
    
    Risk increases as model memory approaches available memory.
    """
    if available_memory_mb <= 0:
        return 1.0  # Maximum risk when no memory info
    
    # Memory usage with safety margin
    required_memory = model_memory_mb * (1 + safety_margin)
    
    if required_memory >= available_memory_mb:
        return 1.0
    
    # Linear risk from 0 to 1
    risk = required_memory / available_memory_mb
    return np.clip(risk, 0.0, 1.0)


def compute_latency_risk(
    estimated_latency_ms: float,
    max_acceptable_latency_ms: float
) -> float:
    """
    Compute latency risk component.
    
    Risk increases as estimated latency approaches max acceptable.
    """
    if max_acceptable_latency_ms <= 0:
        return 0.5  # Medium risk when no latency requirement
    
    if estimated_latency_ms >= max_acceptable_latency_ms:
        return 1.0
    
    # Linear risk from 0 to 1
    risk = estimated_latency_ms / max_acceptable_latency_ms
    return np.clip(risk, 0.0, 1.0)


def compute_operator_compatibility_risk(
    unsupported_ops: int,
    total_ops: int
) -> float:
    """
    Compute operator compatibility risk.
    
    Risk based on proportion of unsupported operations.
    """
    if total_ops == 0:
        return 0.0
    
    # Risk is proportion of unsupported ops
    risk = unsupported_ops / total_ops
    return np.clip(risk, 0.0, 1.0)


def compute_numerical_instability_risk(
    has_large_constants: bool = False,
    has_div_operations: bool = False,
    has_exp_log: bool = False,
    precision: str = "fp32"
) -> float:
    """
    Compute numerical instability risk.
    
    Based on model characteristics that may cause numerical issues.
    """
    risk = 0.0
    
    # Large constants can cause overflow
    if has_large_constants:
        risk += 0.3
    
    # Division operations can cause division by zero
    if has_div_operations:
        risk += 0.2
    
    # Exp/log can cause overflow/underflow
    if has_exp_log:
        risk += 0.2
    
    # Lower precision is more unstable
    if precision == "fp16":
        risk += 0.3
    elif precision == "bf16":
        risk += 0.2
    
    return np.clip(risk, 0.0, 1.0)


def compute_device_capacity_risk(
    device_utilization: float,
    device_memory_utilization: float,
    thermal_throttling: bool = False
) -> float:
    """
    Compute device capacity risk.
    
    Based on current device state.
    """
    risk = 0.0
    
    # High utilization means less capacity
    risk += 0.4 * device_utilization
    risk += 0.4 * device_memory_utilization
    
    # Thermal throttling indicates stress
    if thermal_throttling:
        risk += 0.2
    
    return np.clip(risk, 0.0, 1.0)

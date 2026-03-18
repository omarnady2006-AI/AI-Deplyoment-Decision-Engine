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
Decision Schema Module

This module provides versioned, deterministic scoring schemas for deployment decisions.
The schema versioning ensures reproducibility and allows for backward compatibility
when decision logic evolves.
"""


from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Optional
import hashlib
import json
from datetime import datetime

if TYPE_CHECKING:
    from src.core.confidence import DecisionConfidence
    from src.core.decision import DeploymentDecision
    from src.diagnostics.report import Diagnostic


from src.core.ood_detection import TrustLevel


class SchemaVersion(str, Enum):
    """Supported decision schema versions."""
    V1_0_0 = "1.0.0"
    V1_1_0 = "1.1.0"
    LATEST = "1.1.0"


# Current schema version - update when decision logic changes
CURRENT_SCHEMA_VERSION: Final[SchemaVersion] = SchemaVersion.LATEST


@dataclass(frozen=True)
class ScoringWeights:
    """Deterministic weights for scoring components."""
    # Decision base scores
    supported_base: float = 1.0
    supported_with_warnings_base: float = 0.6
    unsupported_base: float = 0.0
    
    # Confidence multipliers
    high_confidence_multiplier: float = 1.0
    medium_confidence_multiplier: float = 0.85
    low_confidence_multiplier: float = 0.6
    
    # Diagnostic severity penalties
    fail_penalty: float = -0.5
    warn_penalty: float = -0.1
    info_penalty: float = 0.0
    
    # Constraint penalties
    constraint_penalty: float = -0.15
    
    # Utility score bounds
    min_utility: float = 0.0
    max_utility: float = 1.0
    
    # Global decision thresholds
    safe_threshold: float = 0.75
    risky_threshold: float = 0.45
    
    # Calibration adjustment bounds
    max_calibration_adjustment: float = 0.2


@dataclass(frozen=True)
class DecisionSchema:
    """A versioned decision schema with deterministic scoring rules."""
    version: SchemaVersion
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    schema_hash: str = field(default="", init=False)
    
    def __post_init__(self):
        """Compute schema hash for verification."""
        schema_dict = {
            "version": self.version.value,
            "weights": {
                "supported_base": self.weights.supported_base,
                "supported_with_warnings_base": self.weights.supported_with_warnings_base,
                "unsupported_base": self.weights.unsupported_base,
                "high_confidence_multiplier": self.weights.high_confidence_multiplier,
                "medium_confidence_multiplier": self.weights.medium_confidence_multiplier,
                "low_confidence_multiplier": self.weights.low_confidence_multiplier,
                "fail_penalty": self.weights.fail_penalty,
                "warn_penalty": self.weights.warn_penalty,
                "info_penalty": self.weights.info_penalty,
                "constraint_penalty": self.weights.constraint_penalty,
                "min_utility": self.weights.min_utility,
                "max_utility": self.weights.max_utility,
                "safe_threshold": self.weights.safe_threshold,
                "risky_threshold": self.weights.risky_threshold,
                "max_calibration_adjustment": self.weights.max_calibration_adjustment,
            }
        }
        schema_json = json.dumps(schema_dict, sort_keys=True)
        schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()[:16]
        object.__setattr__(self, "schema_hash", schema_hash)


@dataclass(frozen=True)
class DeterministicScore:
    """A deterministic score with full provenance tracking."""
    value: float
    schema_version: SchemaVersion
    schema_hash: str
    components: dict[str, float]
    confidence_level: str
    decision: str
    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        """Convert score to dictionary for serialization."""
        return {
            "value": round(self.value, 6),
            "schema_version": self.schema_version.value,
            "schema_hash": self.schema_hash,
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "confidence_level": self.confidence_level,
            "decision": self.decision,
            "computed_at": self.computed_at,
        }


@lru_cache(maxsize=1)
def get_schema(version: SchemaVersion | None = None) -> DecisionSchema:
    """Get the decision schema for a given version (cached)."""
    if version is None:
        version = CURRENT_SCHEMA_VERSION
    
    # Version-specific schemas
    schemas: dict[SchemaVersion, DecisionSchema] = {
        SchemaVersion.V1_0_0: DecisionSchema(
            version=SchemaVersion.V1_0_0,
            weights=ScoringWeights(
                supported_base=1.0,
                supported_with_warnings_base=0.6,
                unsupported_base=0.0,
                high_confidence_multiplier=1.0,
                medium_confidence_multiplier=0.85,
                low_confidence_multiplier=0.6,
                fail_penalty=-0.5,
                warn_penalty=-0.1,
                info_penalty=0.0,
                constraint_penalty=-0.15,
                min_utility=0.0,
                max_utility=1.0,
                safe_threshold=0.75,
                risky_threshold=0.45,
                max_calibration_adjustment=0.2,
            )
        ),
        SchemaVersion.V1_1_0: DecisionSchema(
            version=SchemaVersion.V1_1_0,
            weights=ScoringWeights(
                supported_base=1.0,
                supported_with_warnings_base=0.6,
                unsupported_base=0.0,
                high_confidence_multiplier=1.0,
                medium_confidence_multiplier=0.85,
                low_confidence_multiplier=0.6,
                fail_penalty=-0.5,
                warn_penalty=-0.1,
                info_penalty=0.0,
                constraint_penalty=-0.15,
                min_utility=0.0,
                max_utility=1.0,
                safe_threshold=0.75,
                risky_threshold=0.45,
                max_calibration_adjustment=0.2,
            )
        ),
    }
    
    return schemas.get(version, schemas[SchemaVersion.LATEST])


def compute_deterministic_score(
    decision: DeploymentDecision,
    diagnostics: list[Diagnostic],
    constraints: list[str],
    confidence: "DecisionConfidence | None",
    schema: DecisionSchema | None = None,
) -> DeterministicScore:
    """
    Compute a deterministic score based on the decision schema.
    
    Args:
        decision: The deployment decision
        diagnostics: List of diagnostics
        constraints: List of deployment constraints
        confidence: Optional confidence assessment
        schema: Optional schema (uses current if not provided)
    
    Returns:
        A DeterministicScore with full provenance
    """
    if schema is None:
        schema = get_schema()
    
    weights = schema.weights
    components: dict[str, float] = {}
    
    # Base score from decision
    if decision.value == "SUPPORTED":
        base_score = weights.supported_base
    elif decision.value == "SUPPORTED_WITH_WARNINGS":
        base_score = weights.supported_with_warnings_base
    else:
        base_score = weights.unsupported_base
    components["base_score"] = base_score
    
    # Diagnostic penalties
    fail_count = sum(1 for d in diagnostics if d.severity == "FAIL")
    warn_count = sum(1 for d in diagnostics if d.severity == "WARN")
    info_count = sum(1 for d in diagnostics if d.severity == "INFO")
    
    diagnostic_penalty = (
        fail_count * weights.fail_penalty +
        warn_count * weights.warn_penalty +
        info_count * weights.info_penalty
    )
    components["diagnostic_penalty"] = diagnostic_penalty
    
    # Constraint penalties
    constraint_penalty = len(constraints) * weights.constraint_penalty
    components["constraint_penalty"] = constraint_penalty
    
    # Confidence multiplier
    if confidence is not None:
        if confidence.level == "HIGH":
            confidence_multiplier = weights.high_confidence_multiplier
        elif confidence.level == "MEDIUM":
            confidence_multiplier = weights.medium_confidence_multiplier
        else:
            confidence_multiplier = weights.low_confidence_multiplier
        components["confidence_multiplier"] = confidence_multiplier
    else:
        confidence_multiplier = 1.0
        components["confidence_multiplier"] = 1.0
    
    # Compute raw score
    raw_score = base_score + diagnostic_penalty + constraint_penalty
    components["raw_score"] = raw_score
    
    # Apply confidence multiplier
    final_score = raw_score * confidence_multiplier
    components["final_score_before_clamp"] = final_score
    
    # Clamp to bounds
    final_score = max(weights.min_utility, min(weights.max_utility, final_score))
    
    return DeterministicScore(
        value=final_score,
        schema_version=schema.version,
        schema_hash=schema.schema_hash,
        components=components,
        confidence_level=confidence.level if confidence else "UNKNOWN",
        decision=decision.value,
    )


@dataclass(frozen=True)
class PredictionOutput:
    """Enhanced prediction output with trust metrics."""
    # Core prediction
    raw_probability: float
    calibrated_probability: float
    
    # Trust metrics
    trust_level: TrustLevel
    ood_score: float
    empirical_support_count: int
    generalization_risk: float
    
    # Generalization metrics (from validation reports)
    device_generalization_accuracy: float = 0.0
    family_generalization_accuracy: float = 0.0
    
    # Decision type
    decision_type: str = "STANDARD"  # STANDARD, CONDITIONAL_DECISION
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    schema_version: SchemaVersion = CURRENT_SCHEMA_VERSION
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "raw_probability": round(self.raw_probability, 4),
            "calibrated_probability": round(self.calibrated_probability, 4),
            "trust_level": self.trust_level.value,
            "ood_score": round(self.ood_score, 4),
            "empirical_support_count": self.empirical_support_count,
            "generalization_risk": round(self.generalization_risk, 4),
            "device_generalization_accuracy": round(self.device_generalization_accuracy, 4),
            "family_generalization_accuracy": round(self.family_generalization_accuracy, 4),
            "decision_type": self.decision_type,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version.value,
        }
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"Prediction probability: {self.calibrated_probability:.2f}\n"
            f"In-distribution: {self.trust_level in (TrustLevel.HIGH, TrustLevel.MEDIUM)}\n"
            f"Device generalization accuracy: {self.device_generalization_accuracy:.2f}\n"
            f"Family generalization accuracy: {self.family_generalization_accuracy:.2f}\n"
            f"Recommendation trust: {self.trust_level.value}"
        )
    
    def requires_verification(self) -> bool:
        """Check if this prediction requires verification before use."""
        return self.trust_level in (TrustLevel.LOW, TrustLevel.UNKNOWN) or self.decision_type == "CONDITIONAL_DECISION"


def create_prediction_output(
    raw_probability: float,
    trust_level: TrustLevel,
    ood_score: float,
    empirical_support_count: int,
    generalization_risk: float,
    device_generalization_accuracy: float = 0.0,
    family_generalization_accuracy: float = 0.0,
    confidence_downgrade_factor: float = 1.0,
) -> PredictionOutput:
    """Create a PredictionOutput with proper calibration and decision type.
    
    Args:
        raw_probability: The raw predicted probability
        trust_level: The assessed trust level
        ood_score: The out-of-distribution score
        empirical_support_count: Number of similar historical samples
        generalization_risk: Assessed generalization risk
        device_generalization_accuracy: Device generalization accuracy from validation
        family_generalization_accuracy: Family generalization accuracy from validation
        confidence_downgrade_factor: Factor to downgrade confidence based on OOD
        
    Returns:
        PredictionOutput with all trust metrics
    """
    # Apply confidence downgrade based on OOD
    calibrated_probability = raw_probability * confidence_downgrade_factor
    calibrated_probability = max(0.0, min(1.0, calibrated_probability))
    
    # Determine decision type
    if trust_level in (TrustLevel.LOW, TrustLevel.UNKNOWN):
        decision_type = "CONDITIONAL_DECISION"
    else:
        decision_type = "STANDARD"
    
    return PredictionOutput(
        raw_probability=raw_probability,
        calibrated_probability=calibrated_probability,
        trust_level=trust_level,
        ood_score=ood_score,
        empirical_support_count=empirical_support_count,
        generalization_risk=generalization_risk,
        device_generalization_accuracy=device_generalization_accuracy,
        family_generalization_accuracy=family_generalization_accuracy,
        decision_type=decision_type,
    )


def compute_global_decision_score(
    utility_score: float,
    schema: DecisionSchema | None = None,
) -> tuple[str, str]:
    """
    Compute global decision status and reason from utility score.
    
    Args:
        utility_score: The computed utility score
        schema: Optional schema (uses current if not provided)
    
    Returns:
        Tuple of (status, reason) where status is one of: SAFE, RISKY, UNSAFE
    """
    if schema is None:
        schema = get_schema()
    
    weights = schema.weights
    
    if utility_score >= weights.safe_threshold:
        return "SAFE", f"Utility {utility_score:.3f} >= {weights.safe_threshold} threshold for safe deployment"
    elif utility_score >= weights.risky_threshold:
        return "RISKY", f"Utility {utility_score:.3f} in risky range [{weights.risky_threshold}, {weights.safe_threshold})"
    else:
        return "UNSAFE", f"Utility {utility_score:.3f} below {weights.risky_threshold} threshold for safe deployment"


def verify_schema_compatibility(
    schema_hash: str,
    required_version: SchemaVersion | None = None,
) -> bool:
    """
    Verify that a schema hash is compatible with a required version.
    
    Args:
        schema_hash: The schema hash to verify
        required_version: Optional required version (uses latest if not provided)
    
    Returns:
        True if compatible, False otherwise
    """
    if required_version is None:
        required_version = CURRENT_SCHEMA_VERSION
    
    schema = get_schema(required_version)
    return schema.schema_hash == schema_hash


def export_schema(schema: DecisionSchema | None = None) -> str:
    """
    Export a decision schema as JSON.
    
    Args:
        schema: Optional schema (uses current if not provided)
    
    Returns:
        JSON string representation of the schema
    """
    if schema is None:
        schema = get_schema()
    
    schema_dict = {
        "version": schema.version.value,
        "schema_hash": schema.schema_hash,
        "created_at": schema.created_at,
        "weights": {
            "decision_base_scores": {
                "SUPPORTED": schema.weights.supported_base,
                "SUPPORTED_WITH_WARNINGS": schema.weights.supported_with_warnings_base,
                "UNSUPPORTED": schema.weights.unsupported_base,
            },
            "confidence_multipliers": {
                "HIGH": schema.weights.high_confidence_multiplier,
                "MEDIUM": schema.weights.medium_confidence_multiplier,
                "LOW": schema.weights.low_confidence_multiplier,
            },
            "diagnostic_penalties": {
                "FAIL": schema.weights.fail_penalty,
                "WARN": schema.weights.warn_penalty,
                "INFO": schema.weights.info_penalty,
            },
            "constraint_penalty": schema.weights.constraint_penalty,
            "utility_bounds": {
                "min": schema.weights.min_utility,
                "max": schema.weights.max_utility,
            },
            "global_thresholds": {
                "safe": schema.weights.safe_threshold,
                "risky": schema.weights.risky_threshold,
            },
            "max_calibration_adjustment": schema.weights.max_calibration_adjustment,
        }
    }
    
    return json.dumps(schema_dict, indent=2)


def import_schema(schema_json: str) -> DecisionSchema:
    """
    Import a decision schema from JSON.
    
    Args:
        schema_json: JSON string representation of the schema
    
    Returns:
        A DecisionSchema instance
    """
    schema_dict = json.loads(schema_json)
    weights_dict = schema_dict["weights"]
    
    weights = ScoringWeights(
        supported_base=weights_dict["decision_base_scores"]["SUPPORTED"],
        supported_with_warnings_base=weights_dict["decision_base_scores"]["SUPPORTED_WITH_WARNINGS"],
        unsupported_base=weights_dict["decision_base_scores"]["UNSUPPORTED"],
        high_confidence_multiplier=weights_dict["confidence_multipliers"]["HIGH"],
        medium_confidence_multiplier=weights_dict["confidence_multipliers"]["MEDIUM"],
        low_confidence_multiplier=weights_dict["confidence_multipliers"]["LOW"],
        fail_penalty=weights_dict["diagnostic_penalties"]["FAIL"],
        warn_penalty=weights_dict["diagnostic_penalties"]["WARN"],
        info_penalty=weights_dict["diagnostic_penalties"]["INFO"],
        constraint_penalty=weights_dict["constraint_penalty"],
        min_utility=weights_dict["utility_bounds"]["min"],
        max_utility=weights_dict["utility_bounds"]["max"],
        safe_threshold=weights_dict["global_thresholds"]["safe"],
        risky_threshold=weights_dict["global_thresholds"]["risky"],
        max_calibration_adjustment=weights_dict["max_calibration_adjustment"],
    )
    
    return DecisionSchema(
        version=SchemaVersion(schema_dict["version"]),
        weights=weights,
    )

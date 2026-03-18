"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""Continuous learning from historical deployment outcomes."""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import statistics

from src.core.ground_truth_db import GroundTruthDatabase, DeploymentRecord


@dataclass
class SimilarCase:
    """A similar historical case."""
    record: DeploymentRecord
    similarity_score: float
    outcome: bool


@dataclass
class LearnedAdjustment:
    """Adjustment learned from historical data."""
    base_probability: float
    adjusted_probability: float
    confidence: float
    sample_size: int
    similar_cases: List[SimilarCase]


class ContinuousLearner:
    """Learn from historical deployment outcomes to improve predictions."""
    
    def __init__(self, db: GroundTruthDatabase):
        self.db = db
        self._cache = {}
        self._cache_size = 1000
    
    def adjust_prediction(
        self,
        model_architecture: str,
        runtime: str,
        device_fingerprint: str,
        parameter_count: Optional[int] = None,
        opset_version: Optional[int] = None,
        base_probability: float = 0.5
    ) -> LearnedAdjustment:
        """Adjust prediction based on historical outcomes."""
        
        # Find similar cases
        similar = self._find_similar_cases(
            model_architecture=model_architecture,
            runtime=runtime,
            device_fingerprint=device_fingerprint,
            parameter_count=parameter_count,
            opset_version=opset_version
        )
        
        if not similar:
            return LearnedAdjustment(
                base_probability=base_probability,
                adjusted_probability=base_probability,
                confidence=0.0,
                sample_size=0,
                similar_cases=[]
            )
        
        # Compute empirical success rate
        successful = sum(1 for c in similar if c.outcome)
        empirical_rate = successful / len(similar)
        
        # Blend base probability with empirical rate
        # Weight by sample size (more samples = more weight to empirical)
        sample_weight = min(len(similar) / 50.0, 1.0)  # Cap at 50 samples
        adjusted = base_probability * (1 - sample_weight) + empirical_rate * sample_weight
        
        # Compute confidence based on sample size and variance
        confidence = self._compute_confidence(similar)
        
        return LearnedAdjustment(
            base_probability=base_probability,
            adjusted_probability=adjusted,
            confidence=confidence,
            sample_size=len(similar),
            similar_cases=similar[:10]  # Top 10 for reference
        )
    
    def _find_similar_cases(
        self,
        model_architecture: str,
        runtime: str,
        device_fingerprint: str,
        parameter_count: Optional[int] = None,
        opset_version: Optional[int] = None,
        limit: int = 100
    ) -> List[SimilarCase]:
        """Find similar historical cases."""
        
        # Build cache key
        cache_key = (
            model_architecture,
            runtime,
            device_fingerprint,
            parameter_count,
            opset_version
        )
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Query database
        records = self.db.get_similar_deployments(
            model_architecture=model_architecture,
            parameter_count=parameter_count,
            opset_version=opset_version,
            device_fingerprint=device_fingerprint,
            runtime=runtime,
            limit=limit
        )
        
        # Convert to SimilarCase with similarity scores
        similar = []
        for r_dict in records:
            record = DeploymentRecord(
                model_hash=r_dict["model_hash"],
                model_architecture=r_dict["model_architecture"],
                device_fingerprint=r_dict["device_fingerprint"],
                runtime=r_dict["runtime"],
                precision=r_dict["precision"],
                predicted_success_probability=r_dict["predicted_success_probability"],
                actual_success=bool(r_dict["actual_success"]),
                predicted_latency=r_dict.get("predicted_latency"),
                measured_latency=r_dict.get("measured_latency"),
                predicted_memory=r_dict.get("predicted_memory"),
                measured_memory=r_dict.get("measured_memory"),
                accuracy_drop=r_dict.get("accuracy_drop"),
                timestamp=r_dict["timestamp"],
                engine_version=r_dict["engine_version"],
                parameter_count=r_dict.get("parameter_count"),
                opset_version=r_dict.get("opset_version"),
                input_shape=r_dict.get("input_shape"),
                error_type=r_dict.get("error_type"),
                error_message=r_dict.get("error_message")
            )
            
            similarity = self._compute_similarity(
                record,
                model_architecture,
                runtime,
                device_fingerprint,
                parameter_count,
                opset_version
            )
            
            similar.append(SimilarCase(
                record=record,
                similarity_score=similarity,
                outcome=record.actual_success
            ))
        
        # Sort by similarity
        similar.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Cache result
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = similar
        
        return similar
    
    def _compute_similarity(
        self,
        record: DeploymentRecord,
        model_architecture: str,
        runtime: str,
        device_fingerprint: str,
        parameter_count: Optional[int],
        opset_version: Optional[int]
    ) -> float:
        """Compute similarity score between record and query."""
        score = 0.0
        max_score = 0.0
        
        # Architecture match (high weight)
        max_score += 3.0
        if record.model_architecture == model_architecture:
            score += 3.0
        
        # Runtime match (high weight)
        max_score += 3.0
        if record.runtime == runtime:
            score += 3.0
        
        # Device match (medium weight)
        max_score += 2.0
        if record.device_fingerprint == device_fingerprint:
            score += 2.0
        
        # Parameter count similarity (medium weight)
        max_score += 1.5
        if parameter_count and record.parameter_count:
            param_ratio = min(parameter_count, record.parameter_count) / max(parameter_count, record.parameter_count)
            score += param_ratio * 1.5
        
        # Opset version match (low weight)
        max_score += 0.5
        if opset_version and record.opset_version:
            if opset_version == record.opset_version:
                score += 0.5
            elif abs(opset_version - record.opset_version) <= 2:
                score += 0.25
        
        return score / max_score if max_score > 0 else 0.0
    
    def _compute_confidence(self, similar: List[SimilarCase]) -> float:
        """Compute confidence in the adjustment."""
        if not similar:
            return 0.0
        
        n = len(similar)
        
        # Base confidence from sample size
        # More samples = higher confidence, capped at 50 samples
        sample_confidence = min(n / 50.0, 1.0)
        
        # Adjust for variance in outcomes
        outcomes = [1.0 if c.outcome else 0.0 for c in similar]
        if len(outcomes) > 1:
            variance = statistics.variance(outcomes)
            # Lower variance = higher confidence
            variance_penalty = min(variance, 0.25) / 0.25
            confidence = sample_confidence * (1 - variance_penalty * 0.5)
        else:
            confidence = sample_confidence * 0.5
        
        return max(0.0, min(1.0, confidence))
    
    def get_success_probability_distribution(
        self,
        model_architecture: str,
        runtime: str,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Get distribution of success probabilities for similar cases."""
        records = self.db.get_similar_deployments(
            model_architecture=model_architecture,
            runtime=runtime,
            limit=500
        )
        
        if not records:
            return {
                "mean": 0.5,
                "std": 0.0,
                "min": 0.5,
                "max": 0.5,
                "median": 0.5,
                "histogram": [],
                "sample_size": 0
            }
        
        probs = [r["predicted_success_probability"] for r in records]
        
        # Compute statistics
        mean_prob = statistics.mean(probs)
        median_prob = statistics.median(probs)
        
        if len(probs) > 1:
            std_prob = statistics.stdev(probs)
        else:
            std_prob = 0.0
        
        # Build histogram
        hist_bins = [0] * n_bins
        for p in probs:
            bin_idx = min(int(p * n_bins), n_bins - 1)
            hist_bins[bin_idx] += 1
        
        histogram = [
            {
                "bin": i,
                "range": (i / n_bins, (i + 1) / n_bins),
                "count": hist_bins[i]
            }
            for i in range(n_bins)
        ]
        
        return {
            "mean": mean_prob,
            "std": std_prob,
            "min": min(probs),
            "max": max(probs),
            "median": median_prob,
            "histogram": histogram,
            "sample_size": len(probs)
        }
    
    def get_error_patterns(
        self,
        model_architecture: Optional[str] = None,
        runtime: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze error patterns in historical data."""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if model_architecture:
            conditions.append("model_architecture = ?")
            params.append(model_architecture)
        
        if runtime:
            conditions.append("runtime = ?")
            params.append(runtime)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Get failed deployments
        cursor.execute(f"""
            SELECT error_type, error_message, COUNT(*) as count
            FROM deployment_history
            WHERE {where_clause} AND actual_success = 0
              AND error_type IS NOT NULL
            GROUP BY error_type
            ORDER BY count DESC
        """, params)
        
        error_types = [
            {"type": row[0], "count": row[2]}
            for row in cursor.fetchall()
        ]
        
        # Get common error messages
        cursor.execute(f"""
            SELECT error_message, COUNT(*) as count
            FROM deployment_history
            WHERE {where_clause} AND actual_success = 0
              AND error_message IS NOT NULL
            GROUP BY error_message
            ORDER BY count DESC
            LIMIT 10
        """, params)
        
        common_errors = [
            {"message": row[0], "count": row[1]}
            for row in cursor.fetchall()
        ]
        
        return {
            "error_types": error_types,
            "common_errors": common_errors
        }
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get metrics about learning progress."""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM deployment_history")
        total = cursor.fetchone()[0]
        
        # Records per architecture
        cursor.execute("""
            SELECT model_architecture, COUNT(*) as count
            FROM deployment_history
            GROUP BY model_architecture
            ORDER BY count DESC
        """)
        
        per_arch = [
            {"architecture": row[0], "count": row[1]}
            for row in cursor.fetchall()
        ]
        
        # Records per runtime
        cursor.execute("""
            SELECT runtime, COUNT(*) as count
            FROM deployment_history
            GROUP BY runtime
            ORDER BY count DESC
        """)
        
        per_runtime = [
            {"runtime": row[0], "count": row[1]}
            for row in cursor.fetchall()
        ]
        
        # Success rate over time (recent vs all)
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN actual_success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as rate
            FROM deployment_history
        """)
        overall_rate = cursor.fetchone()[0] or 0.0
        
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN actual_success = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as rate
            FROM (
                SELECT * FROM deployment_history
                ORDER BY timestamp DESC
                LIMIT 50
            )
        """)
        recent_rate = cursor.fetchone()[0] or 0.0
        
        return {
            "total_records": total,
            "per_architecture": per_arch,
            "per_runtime": per_runtime,
            "overall_success_rate": overall_rate,
            "recent_success_rate": recent_rate,
            "improvement": recent_rate - overall_rate,
            "cache_size": len(self._cache)
        }
    
    def predict_with_confidence_interval(
        self,
        model_architecture: str,
        runtime: str,
        device_fingerprint: str,
        parameter_count: Optional[int] = None,
        base_probability: float = 0.5
    ) -> Dict[str, Any]:
        """Predict with confidence interval using historical data."""
        adjustment = self.adjust_prediction(
            model_architecture=model_architecture,
            runtime=runtime,
            device_fingerprint=device_fingerprint,
            parameter_count=parameter_count,
            base_probability=base_probability
        )
        
        if adjustment.sample_size == 0:
            return {
                "predicted_probability": base_probability,
                "lower_bound": max(0.0, base_probability - 0.1),
                "upper_bound": min(1.0, base_probability + 0.1),
                "confidence": 0.0,
                "sample_size": 0
            }
        
        # Compute Wilson score interval
        import math
        z = 1.96  # 95% confidence
        n = adjustment.sample_size
        p = adjustment.adjusted_probability
        
        denominator = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denominator
        margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return {
            "predicted_probability": adjustment.adjusted_probability,
            "lower_bound": lower,
            "upper_bound": upper,
            "confidence": adjustment.confidence,
            "sample_size": adjustment.sample_size,
            "base_probability": base_probability,
            "adjustment": adjustment.adjusted_probability - base_probability
        }
    
    def clear_cache(self):
        """Clear similarity cache."""
        self._cache.clear()

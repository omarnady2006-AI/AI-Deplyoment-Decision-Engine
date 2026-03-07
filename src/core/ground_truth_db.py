"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

"""Ground truth database for storing deployment outcomes."""
import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import threading

from src.core.model_hash import compute_model_hash


@dataclass
class DeploymentRecord:
    """Single deployment outcome record."""
    model_hash: str
    model_architecture: str
    device_fingerprint: str
    runtime: str
    precision: str
    predicted_success_probability: float
    actual_success: bool
    predicted_latency: Optional[float]
    measured_latency: Optional[float]
    predicted_memory: Optional[float]
    measured_memory: Optional[float]
    accuracy_drop: Optional[float]
    timestamp: float
    engine_version: str
    
    # Additional fields
    parameter_count: Optional[int] = None
    opset_version: Optional[int] = None
    input_shape: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class GroundTruthDatabase:
    """Persistent SQLite database for deployment outcomes."""
    
    def __init__(self, db_path: str = "deployment_history.db"):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deployment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_hash TEXT NOT NULL,
                model_architecture TEXT NOT NULL,
                device_fingerprint TEXT NOT NULL,
                runtime TEXT NOT NULL,
                precision TEXT NOT NULL,
                predicted_success_probability REAL NOT NULL,
                actual_success BOOLEAN NOT NULL,
                predicted_latency REAL,
                measured_latency REAL,
                predicted_memory REAL,
                measured_memory REAL,
                accuracy_drop REAL,
                timestamp REAL NOT NULL,
                engine_version TEXT NOT NULL,
                parameter_count INTEGER,
                opset_version INTEGER,
                input_shape TEXT,
                error_type TEXT,
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_hash 
            ON deployment_history(model_hash)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_device_fingerprint 
            ON deployment_history(device_fingerprint)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runtime 
            ON deployment_history(runtime)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_architecture 
            ON deployment_history(model_architecture)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON deployment_history(timestamp)
        """)
        
        conn.commit()
    
    def record_deployment(self, record: DeploymentRecord) -> int:
        """Record a deployment outcome."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO deployment_history (
                model_hash, model_architecture, device_fingerprint,
                runtime, precision, predicted_success_probability,
                actual_success, predicted_latency, measured_latency,
                predicted_memory, measured_memory, accuracy_drop,
                timestamp, engine_version, parameter_count,
                opset_version, input_shape, error_type, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.model_hash,
            record.model_architecture,
            record.device_fingerprint,
            record.runtime,
            record.precision,
            record.predicted_success_probability,
            record.actual_success,
            record.predicted_latency,
            record.measured_latency,
            record.predicted_memory,
            record.measured_memory,
            record.accuracy_drop,
            record.timestamp,
            record.engine_version,
            record.parameter_count,
            record.opset_version,
            record.input_shape,
            record.error_type,
            record.error_message
        ))
        
        conn.commit()
        return cursor.lastrowid if cursor.lastrowid is not None else 0
    
    def get_similar_deployments(
        self,
        model_architecture: Optional[str] = None,
        parameter_count: Optional[int] = None,
        opset_version: Optional[int] = None,
        device_fingerprint: Optional[str] = None,
        runtime: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find similar historical deployments."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if model_architecture:
            conditions.append("model_architecture = ?")
            params.append(model_architecture)
        
        if parameter_count:
            # Find within 20% parameter count range
            conditions.append("parameter_count BETWEEN ? AND ?")
            params.append(int(parameter_count * 0.8))
            params.append(int(parameter_count * 1.2))
        
        if opset_version:
            conditions.append("opset_version = ?")
            params.append(opset_version)
        
        if device_fingerprint:
            conditions.append("device_fingerprint = ?")
            params.append(device_fingerprint)
        
        if runtime:
            conditions.append("runtime = ?")
            params.append(runtime)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        
        cursor.execute(f"""
            SELECT * FROM deployment_history
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_success_rate(
        self,
        model_architecture: Optional[str] = None,
        runtime: Optional[str] = None,
        device_fingerprint: Optional[str] = None,
        recent_n: Optional[int] = None
    ) -> float:
        """Calculate empirical success rate."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if model_architecture:
            conditions.append("model_architecture = ?")
            params.append(model_architecture)
        
        if runtime:
            conditions.append("runtime = ?")
            params.append(runtime)
        
        if device_fingerprint:
            conditions.append("device_fingerprint = ?")
            params.append(device_fingerprint)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        if recent_n:
            cursor.execute(f"""
                SELECT 
                    CAST(SUM(CASE WHEN actual_success = 1 THEN 1 ELSE 0 END) AS REAL) / 
                    COUNT(*) as success_rate
                FROM (
                    SELECT * FROM deployment_history
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
            """, params + [recent_n])
        else:
            cursor.execute(f"""
                SELECT 
                    CAST(SUM(CASE WHEN actual_success = 1 THEN 1 ELSE 0 END) AS REAL) / 
                    COUNT(*) as success_rate
                FROM deployment_history
                WHERE {where_clause}
            """, params)
        
        result = cursor.fetchone()
        return result[0] if result and result[0] is not None else 0.5
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Get overall calibration metrics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_deployments,
                SUM(CASE WHEN actual_success = 1 THEN 1 ELSE 0 END) as successful,
                AVG(CASE WHEN actual_success = 1 THEN predicted_success_probability ELSE NULL END) as avg_confidence_when_success,
                AVG(CASE WHEN actual_success = 0 THEN predicted_success_probability ELSE NULL END) as avg_confidence_when_failure,
                AVG(predicted_success_probability) as avg_predicted_probability,
                CAST(SUM(CASE WHEN actual_success = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as observed_success_rate
            FROM deployment_history
        """)
        
        row = cursor.fetchone()
        
        return {
            "total_deployments": row[0] or 0,
            "successful_deployments": row[1] or 0,
            "avg_confidence_when_success": row[2] or 0.0,
            "avg_confidence_when_failure": row[3] or 0.0,
            "avg_predicted_probability": row[4] or 0.0,
            "observed_success_rate": row[5] or 0.0,
            "calibration_error": abs((row[4] or 0.0) - (row[5] or 0.0))
        }
    
    def get_latency_error_stats(self) -> Dict[str, float]:
        """Get latency prediction error statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                predicted_latency,
                measured_latency
            FROM deployment_history
            WHERE predicted_latency IS NOT NULL 
              AND measured_latency IS NOT NULL
              AND measured_latency > 0
        """)
        
        rows = cursor.fetchall()
        
        if not rows:
            return {"count": 0, "mape": 0.0, "mae": 0.0, "rmse": 0.0}
        
        errors = []
        for pred, meas in rows:
            if meas > 0:
                errors.append(abs(pred - meas) / meas)
        
        import statistics
        import math
        
        mape = statistics.mean(errors) if errors else 0.0
        
        # MAE and RMSE
        abs_errors = [abs(p - m) for p, m in rows]
        mae = statistics.mean(abs_errors) if abs_errors else 0.0
        rmse = math.sqrt(statistics.mean([e ** 2 for e in abs_errors])) if abs_errors else 0.0
        
        return {
            "count": len(rows),
            "mape": mape,
            "mae": mae,
            "rmse": rmse
        }
    
    def get_recent_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent deployment records."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM deployment_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_false_positives(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get cases where high confidence predictions failed."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM deployment_history
            WHERE predicted_success_probability >= ?
              AND actual_success = 0
            ORDER BY predicted_success_probability DESC
        """, (threshold,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_false_negatives(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Get cases where low confidence predictions succeeded."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM deployment_history
            WHERE predicted_success_probability <= ?
              AND actual_success = 1
            ORDER BY predicted_success_probability ASC
        """, (threshold,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def compute_reliability_curve(self, n_bins: int = 10) -> List[Dict[str, Any]]:
        """Compute reliability diagram data for calibration assessment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT predicted_success_probability, actual_success
            FROM deployment_history
            WHERE predicted_success_probability IS NOT NULL
        """)
        
        rows = cursor.fetchall()
        
        if not rows:
            return []
        
        bins = [[] for _ in range(n_bins)]
        
        for pred, actual in rows:
            bin_idx = min(int(pred * n_bins), n_bins - 1)
            bins[bin_idx].append(actual)
        
        result = []
        for i, bin_data in enumerate(bins):
            if bin_data:
                conf_low = i / n_bins
                conf_high = (i + 1) / n_bins
                observed = sum(bin_data) / len(bin_data)
                count = len(bin_data)
                
                result.append({
                    "bin": i,
                    "confidence_low": conf_low,
                    "confidence_high": conf_high,
                    "confidence_mid": (conf_low + conf_high) / 2,
                    "observed_frequency": observed,
                    "count": count,
                    "calibration_error": abs((conf_low + conf_high) / 2 - observed)
                })
        
        return result
    
    def export_to_json(self, output_path: str):
        """Export database to JSON file."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM deployment_history")
        rows = cursor.fetchall()
        
        data = [dict(row) for row in rows]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()


def get_engine_version() -> str:
    """Get current engine version."""
    try:
        import deployment_decision_engine
        return getattr(deployment_decision_engine, '__version__', '0.0.0')
    except (ImportError, AttributeError):
        return "0.0.0"

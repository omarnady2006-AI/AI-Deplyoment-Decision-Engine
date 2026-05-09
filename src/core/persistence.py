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
Persistence Layer for Deployment Decision Engine

Provides SQLite-based persistence for models, decisions, and calibrations.
"""


import json
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Optional

# AFTER: path derived from BASE_DIR via paths module — no CWD reliance.
from src.core.paths import persistence_db_path, enforce_quota, INSTANCE_DIR

_DB_PATH: Path = persistence_db_path()

# AFTER: schema-init flag keyed by the resolved db path string,
# eliminating any cross-instance bleed when _DB_PATH is overridden.
_schema_initialized: dict[str, bool] = {}
_schema_locks: dict[str, RLock] = {}
_schema_locks_lock: RLock = RLock()


import logging
logger = logging.getLogger(__name__)

def _canonicalize_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def _get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    # AFTER: accepts explicit db_path; falls back to module-level _DB_PATH.
    target = db_path if db_path is not None else _DB_PATH
    conn = sqlite3.connect(str(target), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # RULE 4: crash-safe SQLite — WAL avoids reader/writer contention;
    # FULL synchronous guarantees durability on every commit.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=FULL;")
    return conn


def _get_schema_lock(db_path: Path) -> RLock:
    key = str(db_path.resolve())
    with _schema_locks_lock:
        if key not in _schema_locks:
            _schema_locks[key] = RLock()
        return _schema_locks[key]


def _ensure_schema(db_path: Path | None = None) -> None:
    # AFTER: keyed by resolved db path — no cross-instance _schema_initialized leakage.
    target = (db_path if db_path is not None else _DB_PATH).resolve()
    key = str(target)
    lock = _get_schema_lock(target)
    with lock:
        if _schema_initialized.get(key):
            return
        conn = _get_connection(target)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    uploaded_at REAL NOT NULL,
                    model_path TEXT,
                    operator_count INTEGER,
                    parameter_count INTEGER
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    runtime TEXT,
                    status TEXT,
                    confidence REAL,
                    risk_score REAL,
                    risk_level TEXT,
                    latency_ms REAL,
                    memory_mb REAL
                );

                CREATE TABLE IF NOT EXISTS calibrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    loaded_at REAL NOT NULL,
                    provider_list TEXT,
                    gpu_name TEXT,
                    raw_json TEXT
                );

                CREATE TABLE IF NOT EXISTS uploaded_models (
                    model_id TEXT PRIMARY KEY,
                    model_hash TEXT NOT NULL,
                    server_path TEXT NOT NULL UNIQUE,
                    owner_user_id INTEGER,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_models_file_hash ON models(file_hash);
                CREATE INDEX IF NOT EXISTS idx_calibrations_loaded_at ON calibrations(loaded_at DESC);
                CREATE INDEX IF NOT EXISTS idx_uploaded_models_created_at ON uploaded_models(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_uploaded_models_hash ON uploaded_models(model_hash);
            """)

            # Backward-compatible schema migration for pre-existing databases.
            try:
                conn.execute("ALTER TABLE models ADD COLUMN model_path TEXT")
            except sqlite3.OperationalError:
                pass

            conn.commit()
            _schema_initialized[key] = True
        finally:
            conn.close()


def save_model_record(
    file_hash: str,
    operator_count: Optional[int],
    parameter_count: Optional[int],
    model_path: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    enforce_quota(INSTANCE_DIR)
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO models (file_hash, uploaded_at, model_path, operator_count, parameter_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                file_hash,
                time.time(),
                _canonicalize_path(model_path) if model_path else None,
                operator_count,
                parameter_count,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


def save_decision_record(
    runtime: str,
    status: str,
    confidence: float,
    risk_score: float,
    risk_level: str,
    latency_ms: Optional[float],
    memory_mb: Optional[float],
    db_path: Optional[Path] = None,
) -> int:
    enforce_quota(INSTANCE_DIR)
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO decisions (timestamp, runtime, status, confidence, risk_score, risk_level, latency_ms, memory_mb)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (time.time(), runtime, status, confidence, risk_score, risk_level, latency_ms, memory_mb),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


def save_calibration_record(
    provider_list: list[str],
    gpu_name: Optional[str],
    raw_json: dict,
    db_path: Optional[Path] = None,
) -> int:
    enforce_quota(INSTANCE_DIR)
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO calibrations (loaded_at, provider_list, gpu_name, raw_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                time.time(),
                json.dumps(provider_list),
                gpu_name,
                json.dumps(raw_json),
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


def load_latest_calibration(db_path: Optional[Path] = None) -> Optional[dict]:
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT loaded_at, provider_list, gpu_name, raw_json
            FROM calibrations
            ORDER BY loaded_at DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "loaded_at": row["loaded_at"],
            "provider_list": json.loads(row["provider_list"]),
            "gpu_name": row["gpu_name"],
            "raw_json": json.loads(row["raw_json"]),
        }
    finally:
        conn.close()


def list_calibrations(limit: int = 500, db_path: Optional[Path] = None) -> list[dict]:
    """Return all calibration records from SQLite, newest first.

    INT-38: Added so get_calibration_stats in misc_routes can read from the
    same store that save_calibration_record writes to, instead of the orphaned
    calibration_history.json file which is never written by the API routes.
    """
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT loaded_at, provider_list, gpu_name, raw_json
            FROM calibrations
            ORDER BY loaded_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "loaded_at": row["loaded_at"],
                "provider_list": json.loads(row["provider_list"]),
                "gpu_name": row["gpu_name"],
                "raw_json": json.loads(row["raw_json"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def list_decisions(limit: int = 50, db_path: Optional[Path] = None) -> list[dict]:
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, timestamp, runtime, status, confidence, risk_score, risk_level, latency_ms, memory_mb
            FROM decisions
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "runtime": row["runtime"],
                "status": row["status"],
                "confidence": row["confidence"],
                "risk_score": row["risk_score"],
                "risk_level": row["risk_level"],
                "latency_ms": row["latency_ms"],
                "memory_mb": row["memory_mb"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def load_model_info(model_hash: str, db_path: Optional[Path] = None) -> Optional[dict]:
    """Load model info by file hash."""
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, file_hash, uploaded_at, model_path, operator_count, parameter_count
            FROM models
            WHERE file_hash = ?
            LIMIT 1
            """,
            (model_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "file_hash": row["file_hash"],
            "uploaded_at": row["uploaded_at"],
            "model_path": row["model_path"],
            "operator_count": row["operator_count"],
            "parameter_count": row["parameter_count"],
        }
    finally:
        conn.close()


def save_uploaded_model_mapping(
    model_id: str,
    model_hash: str,
    server_path: str,
    owner_user_id: int | None = None,
    db_path: Optional[Path] = None,
) -> None:
    enforce_quota(INSTANCE_DIR)
    _ensure_schema(db_path)
    canonical_path = _canonicalize_path(server_path)
    now = time.time()
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO uploaded_models
            (model_id, model_hash, server_path, owner_user_id, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model_id, model_hash, canonical_path, owner_user_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def resolve_uploaded_model_path(model_id: str, db_path: Optional[Path] = None) -> str | None:
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT server_path
            FROM uploaded_models
            WHERE model_id = ?
            LIMIT 1
            """,
            (model_id,),
        ).fetchone()
        if row is None:
            return None

        conn.execute(
            "UPDATE uploaded_models SET last_accessed = ? WHERE model_id = ?",
            (time.time(), model_id),
        )
        conn.commit()
        return str(row["server_path"])
    finally:
        conn.close()


def resolve_uploaded_model_mapping(model_id: str, db_path: Optional[Path] = None) -> dict[str, str] | None:
    """Resolve uploaded model metadata by model_id and update last_accessed."""
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT server_path, model_hash
            FROM uploaded_models
            WHERE model_id = ?
            LIMIT 1
            """,
            (model_id,),
        ).fetchone()
        if row is None:
            return None

        conn.execute(
            "UPDATE uploaded_models SET last_accessed = ? WHERE model_id = ?",
            (time.time(), model_id),
        )
        conn.commit()
        return {
            "server_path": str(row["server_path"]),
            "model_hash": str(row["model_hash"]),
        }
    finally:
        conn.close()


def list_uploaded_model_paths(db_path: Optional[Path] = None) -> set[str]:
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        rows = conn.execute("SELECT server_path FROM uploaded_models").fetchall()
        return {str(r["server_path"]) for r in rows}
    finally:
        conn.close()


def delete_uploaded_model_mapping_by_path(server_path: str, db_path: Optional[Path] = None) -> None:
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        conn.execute(
            "DELETE FROM uploaded_models WHERE server_path = ?",
            (_canonicalize_path(server_path),),
        )
        conn.commit()
    finally:
        conn.close()


def load_incidents(db_path: Optional[Path] = None) -> list[dict]:
    """Load recent incidents (decisions with FAIL status)."""
    _ensure_schema(db_path)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, timestamp, runtime, status, confidence, risk_level
            FROM decisions
            WHERE status = 'REJECTED' OR risk_level = 'HIGH_RISK'
            ORDER BY timestamp DESC
            LIMIT 100
            """
        )
        rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "runtime": row["runtime"],
                "status": row["status"],
                "confidence": row["confidence"],
                "risk_level": row["risk_level"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def load_decision_history(limit: int = 100, db_path: Optional[Path] = None) -> list[dict]:
    """Load decision history."""
    return list_decisions(limit, db_path=db_path)

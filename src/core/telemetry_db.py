from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

_DB_PATH = "telemetry.db"
_db_lock = threading.Lock()


def init_db(db_path: str = _DB_PATH) -> None:
    with _db_lock:
        con = sqlite3.connect(db_path, check_same_thread=False)
        try:
            cur = con.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      TEXT,
                    optimal_action TEXT,
                    chosen_action  TEXT,
                    event_json     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_optimal_action
                    ON events (optimal_action);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                    ON events (timestamp);
                """
            )
            con.commit()
        finally:
            con.close()


def insert_event(event: dict[str, Any], db_path: str = _DB_PATH) -> None:
    row = (
        event.get("timestamp"),
        event.get("optimal_action"),
        event.get("chosen_action"),
        json.dumps(event, ensure_ascii=False),
    )
    with _db_lock:
        con = sqlite3.connect(db_path, check_same_thread=False)
        try:
            con.execute(
                """
                INSERT INTO events (timestamp, optimal_action, chosen_action, event_json)
                VALUES (?, ?, ?, ?)
                """,
                row,
            )
            con.commit()
        finally:
            con.close()

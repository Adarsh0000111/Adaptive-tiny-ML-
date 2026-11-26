from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class EventStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS telemetry (id INTEGER PRIMARY KEY, node_id TEXT NOT NULL, timestamp TEXT NOT NULL, source TEXT NOT NULL, payload TEXT NOT NULL, energy_j REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS telemetry_time ON telemetry(timestamp)")
            db.execute("CREATE TABLE IF NOT EXISTS inferences (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, payload TEXT NOT NULL)")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def add_telemetry(self, payload: dict[str, Any], energy_j: float | None) -> dict[str, Any]:
        with self._lock, self._connect() as db:
            cur = db.execute("INSERT INTO telemetry(node_id,timestamp,source,payload,energy_j) VALUES(?,?,?,?,?)", (payload["node_id"], payload["timestamp"], payload["source"], json.dumps(payload), energy_j))
            return {"id": cur.lastrowid, "cumulative_energy_j": energy_j, **payload}

    def recent_telemetry(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT id,payload,energy_j FROM telemetry ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": row["id"], **json.loads(row["payload"]), "cumulative_energy_j": row["energy_j"]} for row in rows][::-1]

    def add_inference(self, payload: dict[str, Any]) -> None:
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO inferences(timestamp,payload) VALUES(?,?)", (payload["timestamp"], json.dumps(payload)))

    def recent_inferences(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT payload FROM inferences ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row["payload"]) for row in rows][::-1]

"""Long-term memory - SQLite-backed run history (stdlib sqlite3)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class LongTermMemory:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, game TEXT, goal TEXT,
                started_at REAL, finished_at REAL, outcome TEXT
            );
            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, iteration INTEGER,
                skill TEXT, action_type TEXT, verification TEXT, screen TEXT,
                confidence REAL, screenshot_ref TEXT, timestamp REAL
            );
            """
        )
        self._conn.commit()

    def start_run(self, run_id: str, game: str, goal: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, game, goal, started_at) VALUES (?,?,?,?)",
            (run_id, game, goal, time.time()),
        )
        self._conn.commit()

    def finish_run(self, run_id: str, outcome: str) -> None:
        self._conn.execute("UPDATE runs SET finished_at=?, outcome=? WHERE run_id=?",
                           (time.time(), outcome, run_id))
        self._conn.commit()

    def record_iteration(self, run_id: str, data: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT INTO iterations
               (run_id, iteration, skill, action_type, verification, screen,
                confidence, screenshot_ref, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (run_id, data.get("iteration", 0), data.get("skill", ""),
             data.get("action_type", ""), data.get("verification", ""),
             data.get("screen", ""), float(data.get("confidence", 0.0)),
             data.get("screenshot_ref", ""), time.time()),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(zip([d[0] for d in cur.description], row))

    def export_run(self, run_id: str) -> str:
        run = self.get_run(run_id) or {}
        cur = self._conn.execute(
            "SELECT iteration, skill, action_type, verification, screen, confidence "
            "FROM iterations WHERE run_id=? ORDER BY iteration", (run_id,))
        cols = [d[0] for d in cur.description]
        iterations = [dict(zip(cols, r)) for r in cur.fetchall()]
        return json.dumps({"run": run, "iterations": iterations}, indent=2)

    def close(self) -> None:
        self._conn.close()

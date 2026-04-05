from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from ai_lab.paths import ensure_lab_directories, storage_root


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  config_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  artifact_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  variant TEXT NOT NULL,
  score REAL NOT NULL,
  details_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_results_run_id ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class LabStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else storage_root()
        self.root.mkdir(parents=True, exist_ok=True)
        for subdir in ["docs", "datasets", "runs", "artifacts", "eval_results", "synthetic", "indexes"]:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "lab.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def record_run(
        self,
        *,
        name: str,
        mode: str,
        config: dict[str, Any],
        metrics: dict[str, Any],
        artifact_path: str | Path,
        status: str = "completed",
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> str:
        run_id = run_id or _run_id(name)
        created_at = created_at or now_iso()
        artifact = str(artifact_path)
        payload = (
            run_id,
            name,
            mode,
            status,
            created_at,
            json.dumps(config, sort_keys=True),
            json.dumps(metrics, sort_keys=True),
            artifact,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, name, mode, status, created_at, config_json, metrics_json, artifact_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return run_id

    def record_eval_results(self, run_id: str, results: Iterable[dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO eval_results (run_id, task_id, variant, score, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        str(result["task_id"]),
                        str(result["variant"]),
                        float(result["score"]),
                        json.dumps(result, sort_keys=True),
                        str(result.get("created_at", now_iso())),
                    )
                    for result in results
                ],
            )

    def latest_run(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs ORDER BY datetime(created_at) DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def recent_runs(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY datetime(created_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def eval_results_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def write_json_artifact(self, subdir: str, name: str, payload: dict[str, Any]) -> Path:
        target_dir = self.root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path

    def write_jsonl_artifact(self, subdir: str, name: str, rows: Iterable[dict[str, Any]]) -> Path:
        target_dir = self.root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / name
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        return path


def _run_id(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    return f"{slug[:24] or 'run'}-{uuid4().hex[:8]}"

"""SQLite persistence for reproducible AquaDoc analysis runs."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .schemas import AnalysisResult
from .serialization import analysis_result_to_dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "aquadoc.sqlite3"


@dataclass(frozen=True)
class AnalysisRun:
    run_id: str
    created_at: str
    start_date: str
    end_date: str
    size_m: int
    result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "size_m": self.size_m,
            "result": self.result,
        }


class AnalysisRunRepository:
    """Store and retrieve immutable analysis runs in a local SQLite database."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    size_m INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    captured_at TEXT,
                    resolution_m REAL,
                    risk_score REAL NOT NULL,
                    safety_label TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_runs_location_created "
                "ON analysis_runs(latitude, longitude, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_runs_captured "
                "ON analysis_runs(captured_at DESC)"
            )

    def save(
        self,
        result: AnalysisResult,
        start_date: str,
        end_date: str,
        size_m: int,
        run_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> AnalysisRun:
        self.initialize()
        selected_run_id = run_id or str(uuid.uuid4())
        selected_created_at = created_at or datetime.now(timezone.utc).isoformat()
        result_dict = analysis_result_to_dict(result)
        result_json = json.dumps(result_dict, sort_keys=True, separators=(",", ":"))

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_runs (
                    run_id, created_at, latitude, longitude, start_date, end_date,
                    size_m, source, captured_at, resolution_m, risk_score,
                    safety_label, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    selected_run_id,
                    selected_created_at,
                    result.latitude,
                    result.longitude,
                    start_date,
                    end_date,
                    size_m,
                    result.source,
                    result.captured_at,
                    result.resolution_m,
                    result.risk_score,
                    result.safety_label,
                    result_json,
                ),
            )

        return AnalysisRun(
            run_id=selected_run_id,
            created_at=selected_created_at,
            start_date=start_date,
            end_date=end_date,
            size_m=size_m,
            result=result_dict,
        )

    def get(self, run_id: str) -> Optional[AnalysisRun]:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id, created_at, start_date, end_date, size_m, result_json "
                "FROM analysis_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def list(self, limit: int = 20) -> List[AnalysisRun]:
        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, created_at, start_date, end_date, size_m, result_json "
                "FROM analysis_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> AnalysisRun:
        return AnalysisRun(
            run_id=row["run_id"],
            created_at=row["created_at"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            size_m=row["size_m"],
            result=json.loads(row["result_json"]),
        )

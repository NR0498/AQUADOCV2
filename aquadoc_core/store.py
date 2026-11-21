"""Persist analysis results and their PNG layers.

* Local development: SQLite file (default ``data/aquadoc.sqlite3``).
* Deployment: PostgreSQL when ``DATABASE_URL`` is set (e.g. Neon from the Vercel Marketplace).

Serverless instances share nothing and may disappear at any time, so anything that must
survive between requests (results, images, the request cache) lives here, not in memory.
PNG layers are small (~10-150 KB each), so they are stored alongside the result rather than
in a separate object store.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "aquadoc.sqlite3"

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS water_analyses (
        id TEXT PRIMARY KEY,
        request_key TEXT NOT NULL,
        created_at TEXT NOT NULL,
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        water_body TEXT,
        scene_datetime TEXT,
        result TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS water_analyses_request_key ON water_analyses (request_key, created_at)",
    "CREATE INDEX IF NOT EXISTS water_analyses_created_at ON water_analyses (created_at)",
    """CREATE TABLE IF NOT EXISTS water_analysis_layers (
        analysis_id TEXT NOT NULL,
        name TEXT NOT NULL,
        png {blob} NOT NULL,
        PRIMARY KEY (analysis_id, name)
    )""",
)


class AnalysisStore(Protocol):
    def save(self, analysis_id: str, request_key: str, result: dict, images: dict[str, bytes]) -> None: ...
    def get(self, analysis_id: str) -> dict | None: ...
    def get_layer(self, analysis_id: str, name: str) -> bytes | None: ...
    def find_recent(self, request_key: str, max_age: timedelta) -> dict | None: ...
    def list_recent(self, limit: int) -> list[dict]: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _summary_row(result: dict) -> tuple:
    body = result.get("water_body") or {}
    request = result.get("request", {})
    return (request.get("latitude"), request.get("longitude"), body.get("name"),
            (result.get("scene") or {}).get("datetime"))


class _SqlStore:
    """Shared SQL for SQLite and PostgreSQL; subclasses supply connections and placeholders."""

    placeholder = "?"
    blob_type = "BLOB"

    def _connect(self):
        raise NotImplementedError

    def _sql(self, statement: str) -> str:
        return statement.replace("?", self.placeholder)

    def init_schema(self) -> None:
        with closing(self._connect()) as connection:
            cursor = connection.cursor()
            for statement in _SCHEMA:
                cursor.execute(statement.format(blob=self.blob_type))
            connection.commit()

    def save(self, analysis_id, request_key, result, images):
        latitude, longitude, name, scene_datetime = _summary_row(result)
        with closing(self._connect()) as connection:
            cursor = connection.cursor()
            cursor.execute(self._sql(
                "INSERT INTO water_analyses (id, request_key, created_at, latitude, longitude, water_body, "
                "scene_datetime, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"),
                (analysis_id, request_key, result.get("created_at") or _now(), latitude, longitude, name,
                 scene_datetime, json.dumps(result)))
            for layer, png in images.items():
                cursor.execute(self._sql(
                    "INSERT INTO water_analysis_layers (analysis_id, name, png) VALUES (?, ?, ?)"),
                    (analysis_id, layer, png))
            connection.commit()

    def get(self, analysis_id):
        row = self._one("SELECT result FROM water_analyses WHERE id = ?", (analysis_id,))
        return json.loads(row[0]) if row else None

    def get_layer(self, analysis_id, name):
        row = self._one("SELECT png FROM water_analysis_layers WHERE analysis_id = ? AND name = ?", (analysis_id, name))
        return bytes(row[0]) if row else None

    def find_recent(self, request_key, max_age):
        cutoff = (datetime.now(timezone.utc) - max_age).isoformat(timespec="seconds")
        row = self._one("SELECT result FROM water_analyses WHERE request_key = ? AND created_at >= ? "
                        "ORDER BY created_at DESC LIMIT 1", (request_key, cutoff))
        return json.loads(row[0]) if row else None

    def list_recent(self, limit):
        with closing(self._connect()) as connection:
            cursor = connection.cursor()
            cursor.execute(self._sql(
                "SELECT id, created_at, latitude, longitude, water_body, scene_datetime FROM water_analyses "
                "ORDER BY created_at DESC LIMIT ?"), (int(limit),))
            rows = cursor.fetchall()
        return [{"id": r[0], "created_at": r[1], "latitude": r[2], "longitude": r[3], "water_body": r[4],
                 "scene_datetime": r[5]} for r in rows]

    def _one(self, statement, params):
        with closing(self._connect()) as connection:
            cursor = connection.cursor()
            cursor.execute(self._sql(statement), params)
            return cursor.fetchone()


class SqliteStore(_SqlStore):
    def __init__(self, path: str | Path = DEFAULT_SQLITE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)


class PostgresStore(_SqlStore):
    placeholder = "%s"
    blob_type = "BYTEA"

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.init_schema()

    def _connect(self):
        import psycopg  # imported lazily: only needed when DATABASE_URL is set

        return psycopg.connect(self.dsn, connect_timeout=10)


def store_from_env() -> AnalysisStore:
    """PostgreSQL if DATABASE_URL (or POSTGRES_URL) is set, else SQLite.

    On Vercel without a database the only writable path is /tmp, which is per-instance
    and temporary: results then disappear quickly. Configure a database for deployment.
    """
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if dsn:
        return PostgresStore(dsn)
    if os.environ.get("VERCEL"):
        return SqliteStore("/tmp/aquadoc.sqlite3")
    return SqliteStore(os.environ.get("AQUADOC_DB_PATH", DEFAULT_SQLITE_PATH))

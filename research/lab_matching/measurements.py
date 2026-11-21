"""Field and laboratory measurement models and relational storage."""

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ground_truth import (
    CATEGORICAL_PARAMETERS,
    NUMERIC_PARAMETERS,
    ObservationParameter,
    ObservationQuality,
)
from aquadoc_core.persistence import DEFAULT_DATABASE_PATH


class SamplingEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    sample_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=200)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    collected_at: datetime
    depth_m: Optional[float] = Field(default=None, ge=0)
    spatial_uncertainty_m: float = Field(default=0, ge=0)
    source: str = Field(min_length=1, max_length=200)
    source_reference: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def require_timezone(self) -> "SamplingEvent":
        if self.collected_at.utcoffset() is None:
            raise ValueError("collected_at must include a timezone")
        return self


class MeasurementBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True, allow_inf_nan=False)

    measurement_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_id: str = Field(min_length=1, max_length=200)
    parameter: ObservationParameter
    numeric_value: Optional[float] = None
    categorical_value: Optional[str] = Field(default=None, min_length=1, max_length=100)
    unit: Optional[str] = Field(default=None, min_length=1, max_length=40)
    method: str = Field(min_length=1, max_length=200)
    quality: ObservationQuality = ObservationQuality.PROVISIONAL
    notes: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_value(self) -> "MeasurementBase":
        has_numeric = self.numeric_value is not None
        has_categorical = self.categorical_value is not None
        if has_numeric == has_categorical:
            raise ValueError("provide exactly one of numeric_value or categorical_value")
        if self.parameter in NUMERIC_PARAMETERS and (not has_numeric or not self.unit):
            raise ValueError(f"{self.parameter} requires numeric_value and unit")
        if self.parameter in CATEGORICAL_PARAMETERS:
            if not has_categorical:
                raise ValueError(f"{self.parameter} requires categorical_value")
            if self.unit is not None:
                raise ValueError("categorical measurements must not have a unit")
        return self


class FieldMeasurement(MeasurementBase):
    measurement_type: Literal["field"] = "field"
    instrument: Optional[str] = Field(default=None, max_length=200)


class LaboratoryMeasurement(MeasurementBase):
    measurement_type: Literal["laboratory"] = "laboratory"
    laboratory_name: str = Field(min_length=1, max_length=200)
    analyzed_at: datetime
    detection_limit: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_analysis_timezone(self) -> "LaboratoryMeasurement":
        if self.analyzed_at.utcoffset() is None:
            raise ValueError("analyzed_at must include a timezone")
        return self


class MeasurementRepository:
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sampling_events (
                    sample_id TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    collected_at TEXT NOT NULL,
                    depth_m REAL,
                    spatial_uncertainty_m REAL NOT NULL,
                    source TEXT NOT NULL,
                    source_reference TEXT
                );

                CREATE TABLE IF NOT EXISTS field_measurements (
                    measurement_id TEXT PRIMARY KEY,
                    sample_id TEXT NOT NULL REFERENCES sampling_events(sample_id),
                    parameter TEXT NOT NULL,
                    numeric_value REAL,
                    categorical_value TEXT,
                    unit TEXT,
                    method TEXT NOT NULL,
                    instrument TEXT,
                    quality TEXT NOT NULL,
                    notes TEXT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    CHECK ((numeric_value IS NOT NULL) != (categorical_value IS NOT NULL))
                );

                CREATE TABLE IF NOT EXISTS laboratory_measurements (
                    measurement_id TEXT PRIMARY KEY,
                    sample_id TEXT NOT NULL REFERENCES sampling_events(sample_id),
                    parameter TEXT NOT NULL,
                    numeric_value REAL,
                    categorical_value TEXT,
                    unit TEXT,
                    method TEXT NOT NULL,
                    laboratory_name TEXT NOT NULL,
                    analyzed_at TEXT NOT NULL,
                    detection_limit REAL,
                    quality TEXT NOT NULL,
                    notes TEXT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    CHECK ((numeric_value IS NOT NULL) != (categorical_value IS NOT NULL))
                );

                CREATE INDEX IF NOT EXISTS idx_sampling_events_location_time
                    ON sampling_events(latitude, longitude, collected_at);
                CREATE INDEX IF NOT EXISTS idx_field_parameter
                    ON field_measurements(parameter);
                CREATE INDEX IF NOT EXISTS idx_laboratory_parameter
                    ON laboratory_measurements(parameter);
                """
            )

    def save_field(self, event: SamplingEvent, measurement: FieldMeasurement) -> bool:
        return self._save(event, measurement, "field_measurements")

    def save_laboratory(self, event: SamplingEvent, measurement: LaboratoryMeasurement) -> bool:
        return self._save(event, measurement, "laboratory_measurements")

    def counts(self) -> dict[str, int]:
        self.initialize()
        with self._connect() as connection:
            return {
                "sampling_events": connection.execute("SELECT COUNT(*) FROM sampling_events").fetchone()[0],
                "field_measurements": connection.execute("SELECT COUNT(*) FROM field_measurements").fetchone()[0],
                "laboratory_measurements": connection.execute("SELECT COUNT(*) FROM laboratory_measurements").fetchone()[0],
            }

    def _save(
        self,
        event: SamplingEvent,
        measurement: FieldMeasurement | LaboratoryMeasurement,
        table: str,
    ) -> bool:
        if measurement.sample_id != event.sample_id:
            raise ValueError("measurement sample_id must match sampling event sample_id")
        self.initialize()
        fingerprint = measurement_fingerprint(event, measurement)
        with self._connect() as connection:
            self._insert_or_validate_event(connection, event)
            common = (
                measurement.measurement_id,
                measurement.sample_id,
                measurement.parameter,
                measurement.numeric_value,
                measurement.categorical_value,
                measurement.unit,
                measurement.method,
            )
            if table == "field_measurements":
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO field_measurements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    common + (measurement.instrument, measurement.quality, measurement.notes, fingerprint),
                )
            else:
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO laboratory_measurements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    common
                    + (
                        measurement.laboratory_name,
                        measurement.analyzed_at.isoformat(),
                        measurement.detection_limit,
                        measurement.quality,
                        measurement.notes,
                        fingerprint,
                    ),
                )
        return cursor.rowcount == 1

    @staticmethod
    def _insert_or_validate_event(connection: sqlite3.Connection, event: SamplingEvent) -> None:
        values = (
            event.sample_id,
            event.latitude,
            event.longitude,
            event.collected_at.isoformat(),
            event.depth_m,
            event.spatial_uncertainty_m,
            event.source,
            event.source_reference,
        )
        connection.execute("INSERT OR IGNORE INTO sampling_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values)
        existing = connection.execute(
            "SELECT latitude, longitude, collected_at, depth_m, spatial_uncertainty_m, source, source_reference "
            "FROM sampling_events WHERE sample_id = ?",
            (event.sample_id,),
        ).fetchone()
        if tuple(existing) != values[1:]:
            raise ValueError(f"sample_id '{event.sample_id}' has conflicting event metadata")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def measurement_fingerprint(
    event: SamplingEvent,
    measurement: FieldMeasurement | LaboratoryMeasurement,
) -> str:
    payload = {
        "event": event.model_dump(mode="json", exclude={"sample_id"}),
        "measurement": measurement.model_dump(mode="json", exclude={"measurement_id", "sample_id"}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

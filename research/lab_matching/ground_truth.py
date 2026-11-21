"""Validated ground-truth observations and SQLite persistence."""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aquadoc_core.persistence import DEFAULT_DATABASE_PATH


class ObservationParameter(str, Enum):
    CHLOROPHYLL_A = "chlorophyll_a"
    TURBIDITY = "turbidity"
    TOTAL_SUSPENDED_SOLIDS = "total_suspended_solids"
    DISSOLVED_OXYGEN = "dissolved_oxygen"
    PH = "ph"
    HARMFUL_ALGAE_BLOOM = "harmful_algae_bloom"
    FLOATING_WASTE = "floating_waste"
    OIL_OR_SURFACE_FILM = "oil_or_surface_film"


class ObservationQuality(str, Enum):
    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    SUSPECT = "suspect"
    REJECTED = "rejected"


NUMERIC_PARAMETERS = {
    ObservationParameter.CHLOROPHYLL_A,
    ObservationParameter.TURBIDITY,
    ObservationParameter.TOTAL_SUSPENDED_SOLIDS,
    ObservationParameter.DISSOLVED_OXYGEN,
    ObservationParameter.PH,
}

CATEGORICAL_PARAMETERS = {
    ObservationParameter.HARMFUL_ALGAE_BLOOM,
    ObservationParameter.FLOATING_WASTE,
    ObservationParameter.OIL_OR_SURFACE_FILM,
}


class GroundTruthObservation(BaseModel):
    """One field or laboratory observation used as an ML target."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True, allow_inf_nan=False)

    observation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    observed_at: datetime
    parameter: ObservationParameter
    numeric_value: Optional[float] = None
    categorical_value: Optional[str] = Field(default=None, min_length=1, max_length=100)
    unit: Optional[str] = Field(default=None, min_length=1, max_length=40)
    depth_m: Optional[float] = Field(default=None, ge=0)
    spatial_uncertainty_m: float = Field(default=0, ge=0)
    method: str = Field(min_length=1, max_length=200)
    source: str = Field(min_length=1, max_length=200)
    source_reference: Optional[str] = Field(default=None, max_length=500)
    quality: ObservationQuality = ObservationQuality.PROVISIONAL
    notes: Optional[str] = Field(default=None, max_length=2000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_measurement_shape(self) -> "GroundTruthObservation":
        if self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        if self.created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone")

        has_numeric = self.numeric_value is not None
        has_categorical = self.categorical_value is not None
        if has_numeric == has_categorical:
            raise ValueError("provide exactly one of numeric_value or categorical_value")

        if self.parameter in NUMERIC_PARAMETERS:
            if not has_numeric:
                raise ValueError(f"{self.parameter} requires numeric_value")
            if not self.unit:
                raise ValueError(f"{self.parameter} requires a unit")
        if self.parameter in CATEGORICAL_PARAMETERS:
            if not has_categorical:
                raise ValueError(f"{self.parameter} requires categorical_value")
            if self.unit is not None:
                raise ValueError("categorical observations must not have a unit")
        return self


class GroundTruthRepository:
    """Store validated field and laboratory observations in SQLite."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ground_truth_observations (
                    observation_id TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    observed_at TEXT NOT NULL,
                    parameter TEXT NOT NULL,
                    numeric_value REAL,
                    categorical_value TEXT,
                    unit TEXT,
                    depth_m REAL,
                    spatial_uncertainty_m REAL NOT NULL,
                    method TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_reference TEXT,
                    quality TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    CHECK ((numeric_value IS NOT NULL) != (categorical_value IS NOT NULL))
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ground_truth_parameter_time "
                "ON ground_truth_observations(parameter, observed_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ground_truth_location_time "
                "ON ground_truth_observations(latitude, longitude, observed_at)"
            )

    def save(self, observation: GroundTruthObservation) -> GroundTruthObservation:
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ground_truth_observations (
                    observation_id, latitude, longitude, observed_at, parameter,
                    numeric_value, categorical_value, unit, depth_m,
                    spatial_uncertainty_m, method, source, source_reference,
                    quality, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.observation_id,
                    observation.latitude,
                    observation.longitude,
                    observation.observed_at.isoformat(),
                    observation.parameter,
                    observation.numeric_value,
                    observation.categorical_value,
                    observation.unit,
                    observation.depth_m,
                    observation.spatial_uncertainty_m,
                    observation.method,
                    observation.source,
                    observation.source_reference,
                    observation.quality,
                    observation.notes,
                    observation.created_at.isoformat(),
                ),
            )
        return observation

    def get(self, observation_id: str) -> Optional[GroundTruthObservation]:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ground_truth_observations WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        return self._row_to_observation(row) if row else None

    def list(
        self,
        parameter: ObservationParameter | None = None,
        limit: int = 100,
    ) -> list[GroundTruthObservation]:
        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        self.initialize()
        query = "SELECT * FROM ground_truth_observations"
        values: tuple = ()
        if parameter is not None:
            query += " WHERE parameter = ?"
            values = (parameter.value,)
        query += " ORDER BY observed_at DESC LIMIT ?"
        values += (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row_to_observation(row) for row in rows]

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
    def _row_to_observation(row: sqlite3.Row) -> GroundTruthObservation:
        return GroundTruthObservation(
            observation_id=row["observation_id"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            observed_at=row["observed_at"],
            parameter=row["parameter"],
            numeric_value=row["numeric_value"],
            categorical_value=row["categorical_value"],
            unit=row["unit"],
            depth_m=row["depth_m"],
            spatial_uncertainty_m=row["spatial_uncertainty_m"],
            method=row["method"],
            source=row["source"],
            source_reference=row["source_reference"],
            quality=row["quality"],
            notes=row["notes"],
            created_at=row["created_at"],
        )

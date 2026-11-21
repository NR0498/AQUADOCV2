"""Measurement matching, reproducible dataset export, and baseline training."""

import csv
import hashlib
import io
import json
import pickle
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Mapping, Optional

from aquadoc_core.ingestion import (
    RasterioBandReader,
    SENTINEL2_ASSET_MAP,
    SENTINEL2_L2A_COLLECTION,
    Sentinel2StacClient,
    StacItemSummary,
)
from aquadoc_core.pipeline import analyze_multispectral_scene
from aquadoc_core.schemas import MultispectralScene


FEATURE_VERSION = "spectral-statistics-v2"
REVIEW_STATUSES = ("pending", "approved", "rejected")


@dataclass(frozen=True)
class MeasurementCandidate:
    measurement_type: str
    measurement_id: str
    sample_id: str
    parameter: str
    numeric_value: Optional[float]
    categorical_value: Optional[str]
    unit: Optional[str]
    quality: str
    latitude: float
    longitude: float
    collected_at: str


@dataclass
class MatchReport:
    candidates: int = 0
    matched: int = 0
    already_matched: int = 0
    no_scene: int = 0
    failed: int = 0
    errors: list[dict] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> dict:
        return {
            "candidates": self.candidates,
            "matched": self.matched,
            "already_matched": self.already_matched,
            "no_scene": self.no_scene,
            "failed": self.failed,
            "errors": self.errors,
        }


class TrainingMatchRepository:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS measurement_scene_matches (
                    match_id TEXT PRIMARY KEY,
                    measurement_type TEXT NOT NULL,
                    measurement_id TEXT NOT NULL UNIQUE,
                    sample_id TEXT NOT NULL,
                    parameter TEXT NOT NULL,
                    numeric_value REAL,
                    categorical_value TEXT,
                    unit TEXT,
                    quality TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    collected_at TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    cloud_cover REAL,
                    time_delta_hours REAL NOT NULL,
                    extraction_distance_m REAL NOT NULL,
                    chip_size_m INTEGER NOT NULL,
                    feature_version TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    review_note TEXT,
                    reviewed_at TEXT,
                    CHECK ((numeric_value IS NOT NULL) != (categorical_value IS NOT NULL))
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_matches_parameter ON measurement_scene_matches(parameter)"
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(measurement_scene_matches)")
            }
            if "review_status" not in columns:
                connection.execute(
                    "ALTER TABLE measurement_scene_matches "
                    "ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'"
                )
            if "review_note" not in columns:
                connection.execute("ALTER TABLE measurement_scene_matches ADD COLUMN review_note TEXT")
            if "reviewed_at" not in columns:
                connection.execute("ALTER TABLE measurement_scene_matches ADD COLUMN reviewed_at TEXT")

    def candidates(self, parameter: str, qualities: tuple[str, ...] = ("verified",)) -> list[MeasurementCandidate]:
        self.initialize()
        placeholders = ",".join("?" for _ in qualities)
        common = (
            "m.measurement_id, m.sample_id, m.parameter, m.numeric_value, "
            "m.categorical_value, m.unit, m.quality, e.latitude, e.longitude, e.collected_at "
        )
        query = (
            "SELECT 'field', " + common + "FROM field_measurements m "
            "JOIN sampling_events e ON e.sample_id = m.sample_id "
            f"WHERE m.parameter = ? AND m.quality IN ({placeholders}) "
            "UNION ALL SELECT 'laboratory', " + common + "FROM laboratory_measurements m "
            "JOIN sampling_events e ON e.sample_id = m.sample_id "
            f"WHERE m.parameter = ? AND m.quality IN ({placeholders}) "
            "ORDER BY m.measurement_id"
        )
        values = (parameter, *qualities, parameter, *qualities)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [MeasurementCandidate(*row) for row in rows]

    def has_match(self, measurement_id: str) -> bool:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM measurement_scene_matches WHERE measurement_id = ?",
                (measurement_id,),
            ).fetchone()
        return row is not None

    def save_match(
        self,
        candidate: MeasurementCandidate,
        item: StacItemSummary,
        time_delta_hours: float,
        chip_size_m: int,
        features: Mapping[str, float],
    ) -> str:
        self.initialize()
        match_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{candidate.measurement_id}:{item.item_id}:{FEATURE_VERSION}"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO measurement_scene_matches (
                    match_id, measurement_type, measurement_id, sample_id, parameter,
                    numeric_value, categorical_value, unit, quality, latitude, longitude,
                    collected_at, scene_id, captured_at, cloud_cover, time_delta_hours,
                    extraction_distance_m, chip_size_m, feature_version, features_json, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    match_id,
                    candidate.measurement_type,
                    candidate.measurement_id,
                    candidate.sample_id,
                    candidate.parameter,
                    candidate.numeric_value,
                    candidate.categorical_value,
                    candidate.unit,
                    candidate.quality,
                    candidate.latitude,
                    candidate.longitude,
                    candidate.collected_at,
                    item.item_id,
                    item.datetime,
                    item.cloud_cover,
                    time_delta_hours,
                    0.0,
                    chip_size_m,
                    FEATURE_VERSION,
                    json.dumps(dict(features), sort_keys=True, separators=(",", ":")),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return match_id

    def review_matches(
        self,
        parameter: str | None = None,
        review_status: str = "pending",
        limit: int = 100,
    ) -> list[dict]:
        if review_status not in REVIEW_STATUSES:
            raise ValueError(f"review_status must be one of: {', '.join(REVIEW_STATUSES)}")
        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        self.initialize()
        query = (
            "SELECT match_id, measurement_id, sample_id, parameter, numeric_value, categorical_value, "
            "unit, latitude, longitude, collected_at, scene_id, captured_at, cloud_cover, "
            "time_delta_hours, chip_size_m, feature_version, review_status, review_note, reviewed_at "
            "FROM measurement_scene_matches WHERE review_status = ?"
        )
        values: tuple = (review_status,)
        if parameter:
            query += " AND parameter = ?"
            values += (parameter,)
        query += " ORDER BY collected_at, match_id LIMIT ?"
        values += (limit,)
        with self._connect() as connection:
            columns = [column[0] for column in connection.execute(query, values).description]
            return [dict(zip(columns, row)) for row in connection.execute(query, values).fetchall()]

    def review_match(self, match_id: str, review_status: str, review_note: str | None = None) -> dict:
        if review_status not in {"approved", "rejected"}:
            raise ValueError("review_status must be 'approved' or 'rejected'")
        self.initialize()
        reviewed_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE measurement_scene_matches "
                "SET review_status = ?, review_note = ?, reviewed_at = ? WHERE match_id = ?",
                (review_status, review_note or None, reviewed_at, match_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Match '{match_id}' was not found")
            row = connection.execute(
                "SELECT match_id, review_status, review_note, reviewed_at "
                "FROM measurement_scene_matches WHERE match_id = ?",
                (match_id,),
            ).fetchone()
        return dict(zip(("match_id", "review_status", "review_note", "reviewed_at"), row))

    def export_rows(self, parameter: str) -> list[sqlite3.Row]:
        self.initialize()
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM measurement_scene_matches "
                "WHERE parameter = ? AND review_status = 'approved' "
                "ORDER BY measurement_id, scene_id",
                (parameter,),
            ).fetchall()

    def match_counts(self, parameter: str) -> dict[str, int]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT review_status, COUNT(*) FROM measurement_scene_matches "
                "WHERE parameter = ? GROUP BY review_status",
                (parameter,),
            ).fetchall()
        counts = {status: 0 for status in REVIEW_STATUSES}
        counts.update(dict(rows))
        return counts

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def match_measurements_to_scenes(
    database_path: str | Path,
    parameter: str,
    max_days: int = 3,
    chip_size_m: int = 1000,
    search_client: Sentinel2StacClient | None = None,
    band_reader: RasterioBandReader | None = None,
) -> MatchReport:
    if max_days < 0:
        raise ValueError("max_days must be non-negative")
    if chip_size_m <= 0:
        raise ValueError("chip_size_m must be positive")
    repository = TrainingMatchRepository(database_path)
    client = search_client or Sentinel2StacClient()
    reader = band_reader or RasterioBandReader()
    report = MatchReport()

    for candidate in repository.candidates(parameter):
        report.candidates += 1
        if repository.has_match(candidate.measurement_id):
            report.already_matched += 1
            continue
        collected = _parse_datetime(candidate.collected_at)
        tolerance = timedelta(days=max_days)
        start = (collected - tolerance).isoformat()
        end = (collected + tolerance).isoformat()
        try:
            items = client.search(candidate.latitude, candidate.longitude, start, end, chip_size_m)
            eligible = [
                item for item in items
                if item.datetime
                and abs(_parse_datetime(item.datetime) - collected) <= tolerance
            ]
            selected = choose_closest_scene(eligible, collected)
            if selected is None:
                report.no_scene += 1
                continue
            bands = reader.read_bands(
                selected,
                candidate.latitude,
                candidate.longitude,
                chip_size_m,
                SENTINEL2_ASSET_MAP,
            )
            first_band = next(iter(bands.values()), None)
            resolution = chip_size_m / first_band.shape[0] if first_band is not None and first_band.shape[0] else None
            scene = MultispectralScene(
                latitude=candidate.latitude,
                longitude=candidate.longitude,
                bands=bands,
                source=f"Microsoft Planetary Computer / {SENTINEL2_L2A_COLLECTION} / {selected.item_id}",
                captured_at=selected.datetime,
                resolution_m=resolution,
            )
            result = analyze_multispectral_scene(scene)
            features = flatten_analysis_features(result)
            delta = abs((_parse_datetime(selected.datetime) - collected).total_seconds()) / 3600.0
            repository.save_match(candidate, selected, delta, chip_size_m, features)
            report.matched += 1
        except (ValueError, RuntimeError, OSError) as exc:
            report.failed += 1
            report.errors.append({"measurement_id": candidate.measurement_id, "message": str(exc)})
    return report


def choose_closest_scene(
    items: list[StacItemSummary],
    observed_at: datetime,
) -> Optional[StacItemSummary]:
    valid = [item for item in items if item.datetime]
    if not valid:
        return None
    return min(
        valid,
        key=lambda item: (
            abs((_parse_datetime(item.datetime) - observed_at).total_seconds()),
            item.cloud_cover if item.cloud_cover is not None else 100.0,
            item.item_id,
        ),
    )


def flatten_analysis_features(result) -> dict[str, float]:
    features = {}
    for index_name, statistics in sorted(result.indices.items()):
        for statistic_name in ("mean", "median", "std", "p10", "p90", "valid_fraction"):
            features[f"{index_name}__{statistic_name}"] = float(getattr(statistics, statistic_name))
    return features


def export_training_dataset(
    database_path: str | Path,
    parameter: str,
    output_path: str | Path,
) -> dict:
    repository = TrainingMatchRepository(database_path)
    rows = repository.export_rows(parameter)
    if not rows:
        counts = repository.match_counts(parameter)
        if counts["pending"]:
            raise ValueError(
                f"No approved matches found for parameter '{parameter}'; "
                f"review {counts['pending']} pending match(es) first"
            )
        raise ValueError(f"No approved matched measurements found for parameter '{parameter}'")
    feature_versions = {row["feature_version"] for row in rows}
    if len(feature_versions) != 1:
        raise ValueError("Cannot export mixed feature versions; re-extract matches with one feature definition")
    feature_names = sorted({name for row in rows for name in json.loads(row["features_json"])})
    metadata_names = [
        "match_id", "measurement_type", "measurement_id", "sample_id", "parameter",
        "target_numeric", "target_category", "unit", "quality", "latitude", "longitude",
        "location_group", "collected_at", "scene_id", "captured_at", "cloud_cover",
        "time_delta_hours", "chip_size_m", "feature_version",
    ]
    fieldnames = metadata_names + [f"feature__{name}" for name in feature_names]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            features = json.loads(row["features_json"])
            exported = {
                "match_id": row["match_id"],
                "measurement_type": row["measurement_type"],
                "measurement_id": row["measurement_id"],
                "sample_id": row["sample_id"],
                "parameter": row["parameter"],
                "target_numeric": row["numeric_value"],
                "target_category": row["categorical_value"],
                "unit": row["unit"],
                "quality": row["quality"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "location_group": f"{row['latitude']:.3f},{row['longitude']:.3f}",
                "collected_at": row["collected_at"],
                "scene_id": row["scene_id"],
                "captured_at": row["captured_at"],
                "cloud_cover": row["cloud_cover"],
                "time_delta_hours": row["time_delta_hours"],
                "chip_size_m": row["chip_size_m"],
                "feature_version": row["feature_version"],
            }
            exported.update({f"feature__{name}": features.get(name) for name in feature_names})
            writer.writerow(exported)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "dataset_path": output.name,
        "sha256": digest,
        "parameter": parameter,
        "row_count": len(rows),
        "feature_version": next(iter(feature_versions)),
        "feature_columns": [f"feature__{name}" for name in feature_names],
        "sort_order": ["measurement_id", "scene_id"],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def train_baseline(
    dataset_path: str | Path,
    model_path: str | Path,
    report_path: str | Path,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
        )
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Baseline training requires numpy, pandas, and scikit-learn from requirements.txt"
        ) from exc

    dataset_bytes = Path(dataset_path).read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    manifest_path = Path(dataset_path).with_suffix(Path(dataset_path).suffix + ".manifest.json")
    if not manifest_path.exists():
        raise ValueError("Dataset manifest is required; export the training dataset before training")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != dataset_hash:
        raise ValueError("Dataset SHA-256 does not match its manifest; export the dataset again")

    data = pd.read_csv(io.BytesIO(dataset_bytes))
    required_columns = {
        "location_group", "parameter", "unit", "feature_version",
        "target_numeric", "target_category",
    }
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing_columns)}")
    feature_columns = sorted(column for column in data.columns if column.startswith("feature__"))
    if not feature_columns:
        raise ValueError("Dataset has no feature columns")
    if manifest.get("row_count") != len(data) or manifest.get("feature_columns") != feature_columns:
        raise ValueError("Dataset rows or feature columns do not match its manifest")
    for column in ("parameter", "feature_version"):
        expected = manifest.get(column)
        if not expected or data[column].isna().any() or set(data[column]) != {expected}:
            raise ValueError(f"Dataset must contain one {column} matching its manifest")
    if len(data) < 4 or data["location_group"].nunique() < 2:
        raise ValueError("Baseline training requires at least 4 rows from at least 2 location groups")
    if data["location_group"].isna().any():
        raise ValueError("Every dataset row must have a location group")
    is_regression = data["target_numeric"].notna().all()
    if not is_regression and not data["target_category"].notna().all():
        raise ValueError("Dataset must contain one consistent target type")
    if is_regression and data["target_category"].notna().any():
        raise ValueError("Dataset must contain one consistent target type")
    if not is_regression and data["target_numeric"].notna().any():
        raise ValueError("Dataset must contain one consistent target type")
    if is_regression:
        try:
            data["target_numeric"] = pd.to_numeric(data["target_numeric"], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("Numeric targets must contain finite numbers") from exc
        if not np.isfinite(data["target_numeric"]).all():
            raise ValueError("Numeric targets must contain finite numbers")
        if data["unit"].isna().any() or data["unit"].nunique() != 1:
            raise ValueError("Numeric targets must use one consistent unit")

    try:
        x = data[feature_columns].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Feature columns must contain numbers or missing values") from exc
    x = x.replace([np.inf, -np.inf], np.nan)
    y = data["target_numeric"] if is_regression else data["target_category"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(x, y, groups=data["location_group"]))
    if not is_regression and y.iloc[train_indices].nunique() < 2:
        raise ValueError("Classification training split must contain at least two classes")

    if is_regression:
        estimator = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    else:
        estimator = RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1, class_weight="balanced"
        )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("estimator", estimator),
    ])
    model.fit(x.iloc[train_indices], y.iloc[train_indices])
    predictions = model.predict(x.iloc[test_indices])

    if is_regression:
        metrics = {
            "mae": float(mean_absolute_error(y.iloc[test_indices], predictions)),
            "rmse": float(mean_squared_error(y.iloc[test_indices], predictions) ** 0.5),
            "r2": float(r2_score(y.iloc[test_indices], predictions)) if len(test_indices) >= 2 else None,
        }
        task = "regression"
    else:
        labels = sorted(y.unique().tolist())
        metrics = {
            "accuracy": float(accuracy_score(y.iloc[test_indices], predictions)),
            "precision_weighted": float(precision_score(y.iloc[test_indices], predictions, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y.iloc[test_indices], predictions, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y.iloc[test_indices], predictions, average="weighted", zero_division=0)),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y.iloc[test_indices], predictions, labels=labels).tolist(),
        }
        task = "classification"

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "task": task,
        "dataset_sha256": dataset_hash,
        "random_state": random_state,
    }
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as handle:
        pickle.dump(artifact, handle)
    report = {
        "task": task,
        "dataset_sha256": dataset_hash,
        "rows": len(data),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "train_location_groups": sorted(data.iloc[train_indices]["location_group"].unique().tolist()),
        "test_location_groups": sorted(data.iloc[test_indices]["location_group"].unique().tolist()),
        "feature_count": len(feature_columns),
        "metrics": metrics,
        "random_state": random_state,
    }
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    return parsed.astimezone(timezone.utc)

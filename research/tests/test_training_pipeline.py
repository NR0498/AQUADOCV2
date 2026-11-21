import json
import csv
import hashlib
import pickle
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from contextlib import closing
from unittest.mock import patch

import numpy as np

from aquadoc_core.ingestion import StacItemSummary
from research.lab_matching.measurements import FieldMeasurement, MeasurementRepository, SamplingEvent
from research.lab_matching.training_pipeline import (
    MeasurementCandidate,
    TrainingMatchRepository,
    choose_closest_scene,
    export_training_dataset,
    match_measurements_to_scenes,
    train_baseline,
)


class FakeSearchClient:
    def search(self, latitude, longitude, start_date, end_date, size_m):
        return [
            StacItemSummary("far-clear", "2026-06-17T05:00:00Z", 1.0, {"B02": "x"}),
            StacItemSummary("near-cloudy", "2026-06-15T06:00:00Z", 15.0, {"B02": "x"}),
        ]


class FakeBandReader:
    def read_bands(self, item, latitude, longitude, size_m, asset_map):
        shape = (8, 8)
        return {
            "blue": np.full(shape, 0.08, dtype=np.float32),
            "green": np.full(shape, 0.16, dtype=np.float32),
            "red": np.full(shape, 0.07, dtype=np.float32),
            "red_edge": np.full(shape, 0.08, dtype=np.float32),
            "nir": np.full(shape, 0.04, dtype=np.float32),
            "swir1": np.full(shape, 0.03, dtype=np.float32),
        }


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "training.sqlite3"
        repository = MeasurementRepository(self.database_path)
        event = SamplingEvent(
            sample_id="sample-1",
            latitude=12.9716,
            longitude=77.5946,
            collected_at="2026-06-15T05:00:00Z",
            source="Test programme",
        )
        repository.save_field(
            event,
            FieldMeasurement(
                measurement_id="measurement-1",
                sample_id=event.sample_id,
                parameter="turbidity",
                numeric_value=18.2,
                unit="NTU",
                method="Test meter",
                quality="verified",
            ),
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_closest_time_wins_before_cloud_cover(self):
        selected = choose_closest_scene(
            FakeSearchClient().search(0, 0, "", "", 1000),
            datetime(2026, 6, 15, 5, tzinfo=timezone.utc),
        )
        self.assertEqual(selected.item_id, "near-cloudy")

    def test_match_persists_scene_provenance_and_features(self):
        report = match_measurements_to_scenes(
            self.database_path,
            "turbidity",
            search_client=FakeSearchClient(),
            band_reader=FakeBandReader(),
        )
        rows = TrainingMatchRepository(self.database_path).review_matches("turbidity")

        self.assertEqual(report.matched, 1)
        self.assertEqual(rows[0]["scene_id"], "near-cloudy")
        self.assertEqual(rows[0]["time_delta_hours"], 1.0)
        self.assertEqual(rows[0]["review_status"], "pending")

    def test_out_of_window_scene_is_rejected_before_raster_read(self):
        for captured_at in ("2026-05-12T05:06:49Z", "2026-06-18T05:00:01Z"):
            with self.subTest(captured_at=captured_at):
                search = FakeSearchClient()
                search.search = lambda *args: [StacItemSummary("outside-window", captured_at, 1, {})]
                reader = FakeBandReader()
                with patch.object(reader, "read_bands", wraps=reader.read_bands) as read:
                    report = match_measurements_to_scenes(
                        self.database_path, "turbidity", max_days=3,
                        search_client=search, band_reader=reader,
                    )
                self.assertEqual(report.no_scene, 1)
                self.assertEqual(report.matched, 0)
                read.assert_not_called()


class ExportAndBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.database_path = self.root / "training.sqlite3"
        repository = TrainingMatchRepository(self.database_path)
        for index in range(8):
            location = index // 2
            candidate = MeasurementCandidate(
                measurement_type="field",
                measurement_id=f"measurement-{index}",
                sample_id=f"sample-{index}",
                parameter="turbidity",
                numeric_value=float(index * 2 + 1),
                categorical_value=None,
                unit="NTU",
                quality="verified",
                latitude=10.0 + location,
                longitude=70.0 + location,
                collected_at=f"2026-06-{index + 1:02d}T05:00:00Z",
            )
            item = StacItemSummary(
                item_id=f"scene-{index}",
                datetime=f"2026-06-{index + 1:02d}T06:00:00Z",
                cloud_cover=2.0,
                assets={},
            )
            match_id = repository.save_match(
                candidate,
                item,
                time_delta_hours=1.0,
                chip_size_m=1000,
                features={"ndwi__median": float(index), "ndwi__p90": float(index + 0.5)},
            )
            repository.review_match(match_id, "approved", "Synthetic test fixture")

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_export_is_deterministic_and_has_manifest_hash(self):
        first = self.root / "first.csv"
        second = self.root / "second.csv"
        manifest_one = export_training_dataset(self.database_path, "turbidity", first)
        manifest_two = export_training_dataset(self.database_path, "turbidity", second)

        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(manifest_one["sha256"], manifest_two["sha256"])
        self.assertEqual(manifest_one["row_count"], 8)
        self.assertTrue(first.with_suffix(".csv.manifest.json").exists())

    def test_export_rejects_mixed_feature_versions(self):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                "UPDATE measurement_scene_matches SET feature_version = ? WHERE measurement_id = ?",
                ("spectral-statistics-v1", "measurement-0"),
            )
        with self.assertRaisesRegex(ValueError, "mixed feature versions"):
            export_training_dataset(self.database_path, "turbidity", self.root / "mixed.csv")
        self.assertFalse((self.root / "mixed.csv").exists())

    def test_export_preserves_the_version_of_historical_rows(self):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("UPDATE measurement_scene_matches SET feature_version = 'spectral-statistics-v1'")
        manifest = export_training_dataset(self.database_path, "turbidity", self.root / "historical.csv")
        self.assertEqual(manifest["feature_version"], "spectral-statistics-v1")

    def test_pending_match_is_excluded_until_approved(self):
        candidate = MeasurementCandidate(
            measurement_type="field",
            measurement_id="pending-measurement",
            sample_id="pending-sample",
            parameter="chlorophyll_a",
            numeric_value=42.0,
            categorical_value=None,
            unit="ug/L",
            quality="verified",
            latitude=1.0,
            longitude=2.0,
            collected_at="2026-06-01T00:00:00Z",
        )
        repository = TrainingMatchRepository(self.database_path)
        match_id = repository.save_match(
            candidate,
            StacItemSummary("pending-scene", "2026-06-01T01:00:00Z", 1.0, {}),
            time_delta_hours=1.0,
            chip_size_m=1000,
            features={"ndwi__median": 0.2},
        )
        with self.assertRaisesRegex(ValueError, "review 1 pending"):
            export_training_dataset(self.database_path, "chlorophyll_a", self.root / "pending.csv")

        reviewed = repository.review_match(match_id, "approved", "Time and location checked")
        manifest = export_training_dataset(self.database_path, "chlorophyll_a", self.root / "approved.csv")

        self.assertEqual(reviewed["review_status"], "approved")
        self.assertEqual(manifest["row_count"], 1)

    def test_train_regression_baseline_saves_model_and_grouped_report(self):
        dataset = self.root / "training.csv"
        model = self.root / "model.joblib"
        report_path = self.root / "report.json"
        export_training_dataset(self.database_path, "turbidity", dataset)

        report = train_baseline(dataset, model, report_path, test_size=0.25, random_state=7)
        with model.open("rb") as handle:
            artifact = pickle.load(handle)

        self.assertEqual(report["task"], "regression")
        self.assertTrue(model.exists())
        self.assertTrue(report_path.exists())
        self.assertEqual(artifact["dataset_sha256"], report["dataset_sha256"])
        self.assertTrue(
            set(report["train_location_groups"]).isdisjoint(report["test_location_groups"])
        )
        self.assertIn("mae", report["metrics"])

    def test_training_rejects_missing_or_changed_manifest(self):
        dataset = self.root / "training.csv"
        manifest = dataset.with_suffix(".csv.manifest.json")
        export_training_dataset(self.database_path, "turbidity", dataset)
        original_manifest = manifest.read_bytes()
        manifest.unlink()
        with self.assertRaisesRegex(ValueError, "manifest is required"):
            train_baseline(dataset, self.root / "model.pkl", self.root / "report.json")
        manifest.write_bytes(original_manifest)
        dataset.write_bytes(dataset.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            train_baseline(dataset, self.root / "model.pkl", self.root / "report.json")
        self.assertFalse((self.root / "model.pkl").exists())

    def test_imputation_uses_only_training_locations_and_survives_reload(self):
        from sklearn.model_selection import GroupShuffleSplit
        dataset = self.root / "missing.csv"
        export_training_dataset(self.database_path, "turbidity", dataset)
        with dataset.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            rows = list(reader)
        groups = [row["location_group"] for row in rows]
        train, test = next(GroupShuffleSplit(test_size=0.25, random_state=7).split(rows, groups=groups))
        for position, index in enumerate(train):
            rows[index]["feature__ndwi__median"] = str(position * 2)
        rows[train[0]]["feature__ndwi__median"] = ""
        for index in test:
            rows[index]["feature__ndwi__median"] = "1000000"
        with dataset.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        manifest_path = dataset.with_suffix(".csv.manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        train_baseline(dataset, self.root / "model.pkl", self.root / "report.json", random_state=7)
        with (self.root / "model.pkl").open("rb") as handle:
            artifact = pickle.load(handle)
        pipeline = artifact["model"]
        self.assertEqual(pipeline.named_steps["imputer"].statistics_[0], 6.0)
        import pandas as pd
        features = pd.DataFrame([[np.nan, 3]], columns=artifact["feature_columns"])
        self.assertTrue(np.isfinite(pipeline.predict(features)).all())


if __name__ == "__main__":
    unittest.main()

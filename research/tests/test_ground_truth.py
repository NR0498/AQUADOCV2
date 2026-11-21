import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from research.lab_matching.ground_truth import (
    GroundTruthObservation,
    GroundTruthRepository,
    ObservationParameter,
)


def turbidity_observation(**overrides) -> GroundTruthObservation:
    values = {
        "observation_id": "observation-1",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "observed_at": datetime(2026, 6, 15, 5, 0, tzinfo=timezone.utc),
        "parameter": "turbidity",
        "numeric_value": 18.2,
        "unit": "NTU",
        "depth_m": 0.2,
        "spatial_uncertainty_m": 5.0,
        "method": "ISO 7027 nephelometric field meter",
        "source": "Example monitoring programme",
        "quality": "verified",
        "created_at": datetime(2026, 7, 11, 10, 0, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return GroundTruthObservation(**values)


class GroundTruthSchemaTests(unittest.TestCase):
    def test_nonfinite_observation_is_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                turbidity_observation(numeric_value=value)

    def test_numeric_observation_requires_unit(self):
        with self.assertRaisesRegex(ValidationError, "requires a unit"):
            turbidity_observation(unit=None)

    def test_observation_requires_exactly_one_value_type(self):
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            turbidity_observation(categorical_value="high")

    def test_categorical_observation_rejects_unit(self):
        with self.assertRaisesRegex(ValidationError, "must not have a unit"):
            GroundTruthObservation(
                latitude=12,
                longitude=77,
                observed_at="2026-06-15T05:00:00Z",
                parameter="harmful_algae_bloom",
                categorical_value="confirmed",
                unit="yes/no",
                method="Microscopy",
                source="Laboratory report",
            )

    def test_observation_time_requires_timezone(self):
        with self.assertRaisesRegex(ValidationError, "must include a timezone"):
            turbidity_observation(observed_at=datetime(2026, 6, 15, 5, 0))


class GroundTruthRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.repository = GroundTruthRepository(
            Path(self.temp_directory.name) / "ground-truth.sqlite3"
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_save_get_and_filter_observations(self):
        turbidity = self.repository.save(turbidity_observation())
        self.repository.save(
            GroundTruthObservation(
                observation_id="observation-2",
                latitude=12.9716,
                longitude=77.5946,
                observed_at="2026-06-15T05:00:00Z",
                parameter="harmful_algae_bloom",
                categorical_value="not_observed",
                method="Visual survey",
                source="Example monitoring programme",
                quality="provisional",
            )
        )

        loaded = self.repository.get(turbidity.observation_id)
        filtered = self.repository.list(ObservationParameter.TURBIDITY)

        self.assertEqual(loaded.numeric_value, 18.2)
        self.assertEqual(loaded.unit, "NTU")
        self.assertEqual([item.observation_id for item in filtered], ["observation-1"])


if __name__ == "__main__":
    unittest.main()

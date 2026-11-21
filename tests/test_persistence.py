import tempfile
import unittest
from pathlib import Path

from aquadoc_core.persistence import AnalysisRunRepository
from aquadoc_core.schemas import AnalysisResult, BandStatistics


def sample_result() -> AnalysisResult:
    statistics = BandStatistics(0.2, 0.2, 0.01, 0.1, 0.3, 0.9)
    return AnalysisResult(
        latitude=12.9716,
        longitude=77.5946,
        source="sentinel-test-item",
        captured_at="2026-06-15T05:00:00Z",
        resolution_m=10.0,
        indices={"ndwi": statistics},
        risk_score=0.25,
        safety_label="likely_safe",
        signals={"water_mask": "Water signal present."},
        limitations=("Test limitation",),
    )


class AnalysisRunRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "runs.sqlite3"
        self.repository = AnalysisRunRepository(self.database_path)

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_save_and_get_preserve_result_and_query_metadata(self):
        saved = self.repository.save(
            sample_result(),
            "2026-01-01",
            "2026-07-01",
            3000,
            run_id="fixed-run-id",
            created_at="2026-07-11T10:00:00+00:00",
        )
        loaded = self.repository.get(saved.run_id)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "fixed-run-id")
        self.assertEqual(loaded.start_date, "2026-01-01")
        self.assertEqual(loaded.result["indices"]["ndwi"]["median"], 0.2)
        self.assertEqual(loaded.result["safety_label"], "likely_safe")

    def test_list_returns_newest_run_first(self):
        result = sample_result()
        self.repository.save(
            result, "2026-01-01", "2026-02-01", 1000,
            run_id="older", created_at="2026-02-01T00:00:00+00:00",
        )
        self.repository.save(
            result, "2026-02-01", "2026-03-01", 1000,
            run_id="newer", created_at="2026-03-01T00:00:00+00:00",
        )

        self.assertEqual([run.run_id for run in self.repository.list()], ["newer", "older"])

    def test_list_rejects_unsafe_limit(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 1000"):
            self.repository.list(0)


if __name__ == "__main__":
    unittest.main()

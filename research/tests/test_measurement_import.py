import tempfile
import unittest
from pathlib import Path

from research.lab_matching.csv_import import import_measurements_csv
from research.lab_matching.measurements import MeasurementRepository


HEADER = (
    "measurement_type,sample_id,latitude,longitude,collected_at,parameter,"
    "numeric_value,categorical_value,unit,depth_m,spatial_uncertainty_m,method,"
    "instrument,laboratory_name,analyzed_at,detection_limit,source,"
    "source_reference,quality,notes\n"
)


class MeasurementCsvImportTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        self.csv_path = root / "measurements.csv"
        self.repository = MeasurementRepository(root / "measurements.sqlite3")

    def tearDown(self):
        self.temp_directory.cleanup()

    def write_rows(self, rows: str) -> None:
        self.csv_path.write_text(HEADER + rows, encoding="utf-8")

    def test_nonfinite_chlorophyll_is_rejected_without_saving_sample(self):
        self.write_rows(
            "laboratory,s1,12.9,77.5,2026-06-15T05:00:00Z,chlorophyll_a,nan,,ug/L,0.2,5,Fluorometry,,Lab A,2026-06-15T12:00:00Z,0.1,Agency,ref-1,verified,\n"
        )
        report = import_measurements_csv(self.csv_path, self.repository)
        self.assertEqual(report.rejected_rows, 1)
        self.assertEqual(report.imported_rows, 0)
        self.assertEqual(self.repository.counts()["sampling_events"], 0)

    def test_imports_field_and_laboratory_rows(self):
        self.write_rows(
            "field,s1,12.9,77.5,2026-06-15T05:00:00Z,turbidity,18.2,,NTU,0.2,5,ISO 7027,Meter X,,,,Agency,ref-1,verified,\n"
            "laboratory,s2,12.9,77.5,2026-06-15T05:00:00Z,chlorophyll_a,42,,ug/L,0.2,5,Fluorometry,,Lab A,2026-06-15T12:00:00Z,0.1,Agency,ref-2,verified,\n"
        )

        report = import_measurements_csv(self.csv_path, self.repository)

        self.assertEqual(report.imported_rows, 2)
        self.assertEqual(report.rejected_rows, 0)
        self.assertEqual(
            self.repository.counts(),
            {"sampling_events": 2, "field_measurements": 1, "laboratory_measurements": 1},
        )

    def test_reports_invalid_rows_without_stopping_valid_rows(self):
        self.write_rows(
            "field,bad,12.9,77.5,2026-06-15T05:00:00Z,turbidity,abc,,NTU,,,Method,Meter,,,,Agency,,verified,\n"
            "field,good,12.9,77.5,2026-06-15T05:00:00Z,turbidity,10,,NTU,,,Method,Meter,,,,Agency,,verified,\n"
        )

        report = import_measurements_csv(self.csv_path, self.repository)

        self.assertEqual(report.imported_rows, 1)
        self.assertEqual(report.rejected_rows, 1)
        self.assertEqual(report.errors[0].row_number, 2)
        self.assertIn("numeric_value must be a number", report.errors[0].message)

    def test_reimport_detects_duplicates(self):
        self.write_rows(
            "field,s1,12.9,77.5,2026-06-15T05:00:00Z,turbidity,18.2,,NTU,,,Method,Meter,,,,Agency,ref,verified,\n"
        )
        first = import_measurements_csv(self.csv_path, self.repository)
        second = import_measurements_csv(self.csv_path, self.repository)

        self.assertEqual(first.imported_rows, 1)
        self.assertEqual(second.duplicate_rows, 1)
        self.assertEqual(self.repository.counts()["field_measurements"], 1)

    def test_converts_chlorophyll_to_canonical_unit(self):
        self.write_rows(
            "laboratory,s1,12.9,77.5,2026-06-15T05:00:00Z,chlorophyll_a,0.042,,mg/L,,,Fluorometry,,Lab A,2026-06-15T12:00:00Z,0.0001,Agency,ref,verified,\n"
        )
        report = import_measurements_csv(self.csv_path, self.repository)

        self.assertEqual(report.imported_rows, 1)
        with self.repository._connect() as connection:
            row = connection.execute(
                "SELECT numeric_value, unit, detection_limit FROM laboratory_measurements"
            ).fetchone()
        self.assertEqual(row, (42.0, "ug/L", 0.1))

    def test_rejects_unknown_unit(self):
        self.write_rows(
            "field,s1,12.9,77.5,2026-06-15T05:00:00Z,turbidity,18.2,,ppm,,,Method,Meter,,,,Agency,ref,verified,\n"
        )
        report = import_measurements_csv(self.csv_path, self.repository)
        self.assertEqual(report.rejected_rows, 1)
        self.assertIn("unsupported unit", report.errors[0].message)

    def test_rejects_missing_required_header(self):
        self.csv_path.write_text("latitude,longitude\n12,77\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            import_measurements_csv(self.csv_path, self.repository)


if __name__ == "__main__":
    unittest.main()

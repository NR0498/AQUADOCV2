import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from aquadoc_core.cli import main
from aquadoc_core.pipeline import analyze_multispectral_scene
from aquadoc_core.schemas import MultispectralScene


class CliTests(unittest.TestCase):
    @staticmethod
    def fake_fetcher(**kwargs):
        shape = (6, 6)
        return MultispectralScene(
            latitude=kwargs["latitude"],
            longitude=kwargs["longitude"],
            bands={
                "blue": np.full(shape, 0.08, dtype=np.float32),
                "green": np.full(shape, 0.16, dtype=np.float32),
                "red": np.full(shape, 0.07, dtype=np.float32),
                "red_edge": np.full(shape, 0.08, dtype=np.float32),
                "nir": np.full(shape, 0.04, dtype=np.float32),
                "swir1": np.full(shape, 0.03, dtype=np.float32),
            },
            source="test-scene",
            captured_at="2026-06-15T05:00:00Z",
            resolution_m=10.0,
        )

    def test_analyze_command_runs_complete_backend_path(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            database_path = f"{directory}/test.sqlite3"
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "analyze", "--lat", "12.9716", "--lon", "77.5946",
                        "--start", "2026-01-01", "--end", "2026-07-01",
                        "--db", database_path,
                    ],
                    fetcher=self.fake_fetcher,
                    analyzer=analyze_multispectral_scene,
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertIn("run_id", payload)
        self.assertEqual(payload["result"]["source"], "test-scene")
        self.assertEqual(payload["result"]["latitude"], 12.9716)
        self.assertIn(payload["result"]["safety_label"], {"likely_safe", "caution", "unsafe"})
        self.assertIn("ndwi", payload["result"]["indices"])
        self.assertIn("limitations", payload["result"])

    def test_saved_run_can_be_listed_and_shown(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = f"{directory}/test.sqlite3"
            analyze_output = io.StringIO()
            with redirect_stdout(analyze_output):
                main(
                    [
                        "analyze", "--lat", "12", "--lon", "77",
                        "--start", "2026-01-01", "--end", "2026-07-01",
                        "--db", database_path,
                    ],
                    fetcher=self.fake_fetcher,
                )
            run_id = json.loads(analyze_output.getvalue())["run_id"]

            list_output = io.StringIO()
            with redirect_stdout(list_output):
                list_exit_code = main(["list-runs", "--db", database_path])

            show_output = io.StringIO()
            with redirect_stdout(show_output):
                show_exit_code = main(["show-run", run_id, "--db", database_path])

        self.assertEqual(list_exit_code, 0)
        self.assertEqual(json.loads(list_output.getvalue())["runs"][0]["run_id"], run_id)
        self.assertEqual(show_exit_code, 0)
        self.assertEqual(json.loads(show_output.getvalue())["run_id"], run_id)

    def test_domain_error_is_json_on_stderr(self):
        def failing_fetcher(**kwargs):
            raise RuntimeError("No scene found")

        errors = io.StringIO()
        with redirect_stderr(errors):
            exit_code = main(
                [
                    "analyze", "--lat", "12", "--lon", "77",
                    "--start", "2026-01-01", "--end", "2026-07-01",
                ],
                fetcher=failing_fetcher,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(errors.getvalue()), {"error": "No scene found"})


if __name__ == "__main__":
    unittest.main()

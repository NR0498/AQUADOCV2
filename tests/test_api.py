import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from aquadoc_core.api import create_app
from aquadoc_core.schemas import MultispectralScene


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_directory.name) / "api.sqlite3"
        self.fetch_calls = []

        def fake_fetcher(**kwargs):
            self.fetch_calls.append(kwargs)
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
                source="api-test-scene",
                captured_at="2026-06-15T05:00:00Z",
                resolution_m=10.0,
            )

        self.client = TestClient(create_app(database_path=database_path, fetcher=fake_fetcher))

    def tearDown(self):
        self.client.close()
        self.temp_directory.cleanup()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_create_list_and_get_analysis(self):
        create_response = self.client.post(
            "/analyses",
            json={
                "latitude": 12.9716,
                "longitude": 77.5946,
                "start_date": "2026-01-01",
                "end_date": "2026-07-01",
                "size_m": 3000,
            },
        )
        self.assertEqual(create_response.status_code, 201)
        created = create_response.json()
        self.assertEqual(created["result"]["source"], "api-test-scene")
        self.assertEqual(self.fetch_calls[0]["start_date"], "2026-01-01")

        list_response = self.client.get("/analyses", params={"limit": 1})
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["runs"][0]["run_id"], created["run_id"])

        get_response = self.client.get(f"/analyses/{created['run_id']}")
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(get_response.json()["run_id"], created["run_id"])

    def test_request_validation_happens_before_ingestion(self):
        response = self.client.post(
            "/analyses",
            json={
                "latitude": 95,
                "longitude": 77,
                "start_date": "2026-07-01",
                "end_date": "2026-01-01",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.fetch_calls, [])

    def test_unknown_run_returns_404(self):
        response = self.client.get("/analyses/not-a-real-run")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()

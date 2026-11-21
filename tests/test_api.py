"""API tests with fake satellite/OSM services and a temporary SQLite store."""

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from aquadoc_core.api import create_app
from aquadoc_core.store import SqliteStore
from aquadoc_core.water_pipeline import Services
from tests import test_water_pipeline as fixtures

CANDIDATES = fixtures.PipelineTests.candidates(None)
FakeReader = fixtures.FakeReader


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if not lines:
            continue
        kind = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((kind, data))
    return events


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SqliteStore(Path(self.tmp.name) / "api.sqlite3")
        self.searches = 0

        def services():
            def search(*args, **kwargs):
                self.searches += 1
                return CANDIDATES
            return Services(find_water_body=lambda lat, lon: None, lookup_water_body=lambda r, a, b: None,
                            search_scenes=search, reader=FakeReader({"cloudy"}))

        self.client = TestClient(create_app(store=self.store, services_factory=services, rate_limit=(3, 600)))
        self.body = {"latitude": 17.42, "longitude": 78.47, "end_date": "2026-11-20", "max_size_m": 1000}

    def tearDown(self):
        self.tmp.cleanup()

    def test_health_and_guide(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")
        guide = self.client.get("/api/guide").json()
        self.assertEqual(len(guide["bands"]), 10)
        self.assertIn("chlorophyll_a", guide["model"]["targets"])
        self.assertTrue(guide["not_detectable"])

    def test_stream_saves_result_serves_layers_and_caches(self):
        response = self.client.post("/api/analyses/stream", json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = parse_sse(response.text)
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds[-1], "complete")
        self.assertIn("stage", kinds)
        self.assertTrue(any(d.get("status") == "rejected" for k, d in events if k == "stage"))
        result = events[-1][1]
        self.assertFalse(result["cached"])

        stored = self.client.get(f"/api/analyses/{result['id']}").json()
        self.assertEqual(stored["scene"]["id"], "clear")
        layer = self.client.get(result["layers"][0]["url"])
        self.assertEqual(layer.status_code, 200)
        self.assertEqual(layer.headers["content-type"], "image/png")
        self.assertIn("immutable", layer.headers["cache-control"])
        self.assertEqual(self.client.get(f"/api/analyses/{result['id']}/layers/nope.png").status_code, 404)

        listing = self.client.get("/api/analyses").json()["analyses"]
        self.assertEqual(listing[0]["id"], result["id"])

        again = parse_sse(self.client.post("/api/analyses/stream", json=self.body).text)
        self.assertTrue(again[-1][1]["cached"])
        self.assertEqual(again[-1][1]["id"], result["id"])
        self.assertEqual(self.searches, 1)  # served from the store, no new satellite work

    def test_blocking_endpoint_and_failures(self):
        result = self.client.post("/api/analyses", json=self.body).json()
        self.assertIn("estimates", result)
        self.assertEqual(self.client.get("/api/analyses/missing").status_code, 404)
        self.assertEqual(self.client.post("/api/analyses", json={**self.body, "latitude": 95}).status_code, 422)
        self.assertEqual(self.client.post("/api/analyses", json={**self.body, "osm_ref": "lake/1"}).status_code, 422)
        early = {**self.body, "start_date": "2015-01-01"}
        self.assertEqual(self.client.post("/api/analyses", json=early).status_code, 400)

    def test_pipeline_failure_is_streamed_as_error(self):
        app = create_app(store=self.store, services_factory=lambda: Services(
            find_water_body=lambda lat, lon: None, search_scenes=lambda *a, **k: [], reader=FakeReader()))
        events = parse_sse(TestClient(app).post("/api/analyses/stream", json=self.body).text)
        self.assertEqual(events[-1][0], "error")
        self.assertIn("No Sentinel-2 scenes", events[-1][1]["message"])

    def test_rate_limit(self):
        codes = []
        for day in range(1, 6):  # different dates -> no cache hits
            codes.append(self.client.post("/api/analyses", json={**self.body, "end_date": f"2026-11-1{day}"}).status_code)
        self.assertEqual(codes[:3], [200, 200, 200])
        self.assertEqual(codes[3], 429)


if __name__ == "__main__":
    unittest.main()

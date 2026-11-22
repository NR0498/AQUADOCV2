"""Lakes near a point: directory search (Nominatim faked) and the API's saved-analysis matching."""

import tempfile
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from aquadoc_core import api as api_module
from aquadoc_core.api import create_app
from aquadoc_core.store import SqliteStore
from aquadoc_core.waterbody import lookup_water_body, nearby_water_bodies


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def place(name, osm_id, lat, lon, half_deg=0.005, kind="lake"):
    return {"name": name, "osm_type": "way", "osm_id": osm_id, "type": kind, "lat": str(lat), "lon": str(lon),
            "boundingbox": [str(lat - half_deg), str(lat + half_deg), str(lon - half_deg), str(lon + half_deg)]}


def fake_get(places_by_query):
    def get(url, params, headers, timeout):
        return FakeResponse(places_by_query.get(params["q"], []))
    return get


def test_nearby_filters_merges_and_sorts():
    places = {
        "[water=lake]": [place("Big Lake", 1, 17.45, 78.50, 0.02), place("Far Lake", 2, 18.5, 78.5)],
        "[natural=water]": [
            place("Pond", 3, 17.401, 78.481, 0.0005, "water"),             # ~110 m across: too small
            place("Big Lake basin", 4, 17.4501, 78.5001, 0.004, "water"),  # within 400 m of Big Lake: duplicate
            place(None, 5, 17.41, 78.49),                                  # unnamed: skipped
            place("Near Tank", 6, 17.41, 78.48, 0.003, "water"),
        ],
    }
    lakes = nearby_water_bodies(17.40, 78.48, 25, get=fake_get(places), pause_s=0)
    assert [lake["name"] for lake in lakes] == ["Near Tank", "Big Lake"]  # nearest first; Far Lake is outside 25 km
    assert lakes[0]["distance_km"] < 1.5 and lakes[1]["extent_km"] > 4
    assert 0 <= lakes[1]["bearing_deg"] < 90  # Big Lake is north-east


def test_saved_outline_needs_no_network():
    def no_network(*args, **kwargs):
        raise AssertionError("network used")
    body = lookup_water_body.__wrapped__("way/67551369", 12.816, 80.0458, get=no_network)
    assert body.name.startswith("Potheri Lake") and body.contains_point


def test_nearby_endpoint_rounds_location_and_attaches_saved_analysis():
    tmp = tempfile.TemporaryDirectory()
    store = SqliteStore(Path(tmp.name) / "api.sqlite3")
    result = {"request": {"latitude": 17.4239, "longitude": 78.4738}, "water_body": {"name": "Hussain Sagar"},
              "scene": {"datetime": "2026-11-18T05:11:09Z"}, "pixels": {"clear_fraction": 0.69},
              "estimates": {"chlorophyll_a": {"value": 81.0, "trophic_state": "hypereutrophic"},
                            "secchi_depth": {"value": 0.53}, "turbidity": {"value": 86.2}}}
    store.save("abc123", "key", result, {})
    lake = {"name": "Hussain Sagar", "osm": "relation/2833155", "type": "lake", "latitude": 17.4239, "longitude": 78.4738,
            "bbox": [17.40, 17.45, 78.45, 78.50], "distance_km": 2.6, "bearing_deg": 350, "extent_km": 3.2}
    calls = []

    def fake_nearby(lat, lon, radius_km):
        calls.append((lat, lon, radius_km))
        return [lake]

    with mock.patch.object(api_module, "nearby_water_bodies", fake_nearby):
        client = TestClient(create_app(store=store))
        response = client.get("/api/waterbodies/nearby", params={"latitude": 17.40123, "longitude": 78.48456, "radius_km": 10})
    assert response.status_code == 200
    assert calls == [(17.40, 78.48, 10)]  # rounded to ~1 km before any lookup
    saved = response.json()["lakes"][0]["latest_analysis"]
    assert saved["id"] == "abc123" and saved["chlorophyll_a"] == 81.0 and saved["trophic_state"] == "hypereutrophic"
    tmp.cleanup()

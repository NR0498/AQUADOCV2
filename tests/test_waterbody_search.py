"""Name search: bare place names should still find their lake (no network: Nominatim is faked)."""

from aquadoc_core.waterbody import search_water_bodies


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


TOWN = {"name": "Powai", "display_name": "Powai, Mumbai", "category": "place", "type": "suburb",
        "lat": "19.12", "lon": "72.90", "osm_type": "node", "osm_id": 1, "boundingbox": ["19.1", "19.2", "72.8", "73.0"]}
LAKE = {"name": "Powai Lake", "display_name": "Powai Lake, Mumbai", "category": "natural", "type": "water",
        "lat": "19.127", "lon": "72.905", "osm_type": "relation", "osm_id": 8546709,
        "boundingbox": ["19.12", "19.14", "72.89", "72.92"]}


def fake_get(answers):
    calls = []

    def get(url, params, headers, timeout):
        calls.append(params["q"])
        return FakeResponse(answers.get(params["q"], []))

    return get, calls


def test_bare_name_retries_with_lake_suffix():
    get, calls = fake_get({"Powai": [TOWN], "Powai lake": [LAKE]})
    results = search_water_bodies("Powai", get=get, pause_s=0)
    assert calls == ["Powai", "Powai lake"]
    assert [r["osm"] for r in results] == ["relation/8546709"]


def test_no_retry_when_water_found_or_query_names_a_lake():
    get, calls = fake_get({"Powai Lake": [LAKE]})
    assert search_water_bodies("Powai Lake", get=get, pause_s=0)[0]["name"] == "Powai Lake"
    assert calls == ["Powai Lake"]

    get, calls = fake_get({})
    assert search_water_bodies("Hussain Sagar", get=get, pause_s=0) == []
    assert calls == ["Hussain Sagar"]  # "sagar" already names a water body: no second request

"""Water-body outlines and place search from OpenStreetMap (free, no API key).

* Nominatim search/lookup: find water bodies by name and fetch their polygon by OSM id.
  Fast (~1 s); this is the dashboard's main path.
* Overpass API: which mapped water polygon contains (or is nearest to) a clicked point.
  Public instances are often slow (20-50 s) or overloaded, so several are tried and the
  pipeline runs this lookup in parallel with satellite reads.

Both services are run by volunteers. Send an identifying User-Agent, keep requests
low-volume, and cache results. See https://operations.osmfoundation.org/policies/nominatim/
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping, shape
from shapely.ops import linemerge, polygonize, unary_union

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
OVERPASS_URL = OVERPASS_URLS[0]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_LOOKUP_URL = "https://nominatim.openstreetmap.org/lookup"
WATER_TYPES = {"water", "lake", "reservoir", "river", "riverbank", "bay", "lagoon", "pond", "basin", "strait",
               "wetland", "canal", "oxbow"}
USER_AGENT = "AquaDoc/0.2 (satellite water-quality demo)"

# Outlines saved in the repository for featured lakes that OpenStreetMap leaves unnamed: Nominatim does
# not index them and Overpass is often overloaded, so the dashboard should not depend on either.
KNOWN_OUTLINES_PATH = Path(__file__).with_name("known_outlines.json")

_WATER_FILTERS = ('["natural"="water"]', '["waterway"="riverbank"]', '["landuse"="reservoir"]', '["landuse"="basin"]')


@dataclass(frozen=True)
class WaterBody:
    name: str | None
    osm_type: str
    osm_id: int
    water_type: str | None
    geometry: Polygon | MultiPolygon  # WGS84 lon/lat
    contains_point: bool

    def summary(self) -> dict:
        minx, miny, maxx, maxy = self.geometry.bounds
        return {"name": self.name, "osm": f"{self.osm_type}/{self.osm_id}", "water_type": self.water_type,
                "contains_point": self.contains_point, "bbox": [minx, miny, maxx, maxy],
                "source": "© OpenStreetMap contributors (ODbL)"}

    def geojson(self) -> dict:
        return mapping(self.geometry)


def _query(latitude: float, longitude: float, around_m: int | None) -> str:
    if around_m is None:
        selector = "(pivot.a)"
        prefix = f"is_in({latitude},{longitude})->.a;"
    else:
        selector = f"(around:{around_m},{latitude},{longitude})"
        prefix = ""
    parts = "".join(f"way{selector}{f};relation{selector}{f};" for f in _WATER_FILTERS)
    return f"[out:json][timeout:25];{prefix}({parts});out geom;"


def _way_polygon(element: dict) -> Polygon | None:
    coords = [(node["lon"], node["lat"]) for node in element.get("geometry", [])]
    if len(coords) < 4 or coords[0] != coords[-1]:
        return None
    polygon = Polygon(coords)
    return polygon if polygon.is_valid else polygon.buffer(0)


def _relation_polygon(element: dict) -> Polygon | MultiPolygon | None:
    """Assemble an OSM multipolygon relation from its member ways' line geometry."""
    rings = {"outer": [], "inner": []}
    for member in element.get("members", []):
        if member.get("type") != "way" or "geometry" not in member:
            continue
        role = member.get("role") or "outer"
        coords = [(node["lon"], node["lat"]) for node in member["geometry"]]
        if len(coords) >= 2 and role in rings:
            rings[role].append(LineString(coords))
    if not rings["outer"]:
        return None
    outer = unary_union(list(polygonize(linemerge(rings["outer"]))))
    if outer.is_empty:
        return None
    if rings["inner"]:
        inner = unary_union(list(polygonize(linemerge(rings["inner"]))))
        outer = outer.difference(inner)
    return outer if outer.is_valid else outer.buffer(0)


def _parse(payload: dict, point: Point) -> list[WaterBody]:
    bodies = []
    for element in payload.get("elements", []):
        geometry = _way_polygon(element) if element["type"] == "way" else _relation_polygon(element)
        if geometry is None or geometry.is_empty:
            continue
        tags = element.get("tags", {})
        bodies.append(WaterBody(
            name=tags.get("name:en") or tags.get("name"),
            osm_type=element["type"],
            osm_id=int(element["id"]),
            water_type=tags.get("water") or tags.get("waterway") or tags.get("landuse") or tags.get("natural"),
            geometry=geometry,
            contains_point=geometry.covers(point),
        ))
    return bodies


def find_water_body(
    latitude: float,
    longitude: float,
    search_radius_m: int = 300,
    post: Callable[..., requests.Response] = requests.post,
    timeout_s: float = 45,
) -> WaterBody | None:
    """Most specific water polygon containing the point, else the nearest within the radius.

    Tries each Overpass instance in turn. Raises the last error if every instance fails,
    so the caller can distinguish "no water here" (None) from "service unavailable".
    """
    point = Point(longitude, latitude)
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        try:
            for around in (None, search_radius_m):
                response = post(url, data={"data": _query(latitude, longitude, around)},
                                headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
                response.raise_for_status()
                bodies = _parse(response.json(), point)
                containing = [b for b in bodies if b.contains_point]
                if containing:
                    return min(containing, key=lambda b: b.geometry.area)  # e.g. a bay inside a larger lake
                if around is not None and bodies:
                    return min(bodies, key=lambda b: b.geometry.distance(point))
            return None
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"All Overpass instances failed: {last_error}")


@lru_cache(maxsize=256)
def find_water_body_cached(latitude: float, longitude: float) -> WaterBody | None:
    """In-process cache keyed by coordinates rounded to ~10 m."""
    return find_water_body(round(latitude, 4), round(longitude, 4))


_OSM_REF = re.compile(r"^(node|way|relation)/(\d+)$")


@lru_cache(maxsize=1)
def _known_outlines() -> dict:
    try:
        return json.loads(KNOWN_OUTLINES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _osm_api_way(osm_id: int, latitude: float, longitude: float,
                 get: Callable[..., requests.Response] = requests.get) -> WaterBody | None:
    """A single closed way from the main OpenStreetMap API (reliable for one-off lookups by id)."""
    response = get(f"https://api.openstreetmap.org/api/0.6/way/{osm_id}/full.json",
                   headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    elements = response.json()["elements"]
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}
    way = next(e for e in elements if e["type"] == "way")
    polygon = _way_polygon({"geometry": [{"lon": nodes[n][0], "lat": nodes[n][1]} for n in way["nodes"]]})
    if polygon is None:
        return None
    tags = way.get("tags", {})
    return WaterBody(name=tags.get("name:en") or tags.get("name"), osm_type="way", osm_id=osm_id,
                     water_type=tags.get("water") or tags.get("natural"), geometry=polygon,
                     contains_point=polygon.covers(Point(longitude, latitude)))


def _overpass_by_id(osm_type: str, osm_id: int, latitude: float, longitude: float,
                    post: Callable[..., requests.Response] = requests.post, timeout_s: float = 30) -> WaterBody | None:
    """One OSM way or relation with its geometry. Fetching by id is a light query even on busy servers."""
    if osm_type not in ("way", "relation"):
        return None
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        try:
            response = post(url, data={"data": f"[out:json][timeout:25];{osm_type}({osm_id});out geom;"},
                            headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
            response.raise_for_status()
            bodies = _parse(response.json(), Point(longitude, latitude))
            return bodies[0] if bodies else None
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    if osm_type == "way":  # every Overpass instance failed: one way is cheap to fetch from the main API
        return _osm_api_way(osm_id, latitude, longitude)
    raise RuntimeError(f"All Overpass instances failed: {last_error}")


@lru_cache(maxsize=256)
def lookup_water_body(osm_ref: str, latitude: float, longitude: float,
                      get: Callable[..., requests.Response] = requests.get) -> WaterBody | None:
    """Polygon for an OSM object such as ``relation/2833155`` via Nominatim lookup (~1 s)."""
    match = _OSM_REF.match(osm_ref)
    if not match:
        raise ValueError("osm_ref must look like 'relation/123' or 'way/123'")
    osm_type, osm_id = match.groups()
    known = _known_outlines().get(osm_ref)
    if known:
        geometry = shape(known["geometry"])
        return WaterBody(name=known.get("name"), osm_type=osm_type, osm_id=int(osm_id), water_type=known.get("water_type"),
                         geometry=geometry, contains_point=geometry.covers(Point(longitude, latitude)))
    response = get(NOMINATIM_LOOKUP_URL, params={"osm_ids": f"{osm_type[0].upper()}{osm_id}", "format": "jsonv2",
                                                 "polygon_geojson": 1, "polygon_threshold": 0.00005},
                   headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    results = response.json()
    if not results:
        # Nominatim does not index unnamed features (many small lakes): ask Overpass for this one object.
        return _overpass_by_id(osm_type, int(osm_id), latitude, longitude)
    if results[0].get("geojson", {}).get("type") not in ("Polygon", "MultiPolygon"):
        return None
    place = results[0]
    geometry = shape(place["geojson"])
    geometry = geometry if geometry.is_valid else geometry.buffer(0)
    return WaterBody(name=place.get("name") or place.get("display_name"), osm_type=osm_type, osm_id=int(osm_id),
                     water_type=place.get("type"), geometry=geometry,
                     contains_point=geometry.covers(Point(longitude, latitude)))


def _water_matches(payload: list[dict]) -> list[dict]:
    water_classes = {"natural", "water", "waterway", "landuse"}
    return [{"name": place.get("name") or place.get("display_name"),
             "display_name": place.get("display_name"),
             "latitude": float(place["lat"]), "longitude": float(place["lon"]),
             "type": place.get("type"), "osm": f"{place.get('osm_type')}/{place.get('osm_id')}",
             "bbox": [float(v) for v in place.get("boundingbox", [])]}
            for place in payload
            if place.get("category") in water_classes and place.get("type") in WATER_TYPES]


def search_water_bodies(query: str, limit: int = 8, get: Callable[..., requests.Response] = requests.get,
                        pause_s: float = 1.0) -> list[dict]:
    """Name search via Nominatim, filtered to water-like features.

    A bare name such as "Powai" or "Naini" usually matches the town, not its lake, so when the
    first search finds no water the query is retried once as "<name> lake"."""
    def run(text: str) -> list[dict]:
        response = get(NOMINATIM_URL, params={"q": text, "format": "jsonv2", "limit": 25},
                       headers={"User-Agent": USER_AGENT}, timeout=20)
        response.raise_for_status()
        return _water_matches(response.json())

    results = run(query)
    if not results and not re.search(r"\b(lake|lagoon|reservoir|dam|river|tal|sagar|sarovar|jheel|beel|tso)\b",
                                     query, re.IGNORECASE):
        time.sleep(pause_s)  # Nominatim usage policy: at most one request per second
        results = run(f"{query} lake")
    return results[:limit]


# --------------------------------------------------------------------------- nearby

NEARBY_MIN_EXTENT_M = 150  # smaller than ~7 Sentinel-2 pixels across is not worth analysing
NEARBY_LIMIT = 24


def _distance_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """Great-circle distance (km) and initial bearing (degrees from north)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    distance = 2 * 6371.0 * math.asin(math.sqrt(a))
    bearing = math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                      math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)))
    return distance, (bearing + 360) % 360


def nearby_water_bodies(latitude: float, longitude: float, radius_km: float = 25,
                        get: Callable[..., requests.Response] = requests.get, pause_s: float = 1.0) -> list[dict]:
    """Named lakes, reservoirs and ponds around a point, nearest first.

    Nominatim answers tag queries such as ``[water=lake]`` inside a bounded box in about a
    second. It returns at most 40 places per query, so a small inner box is searched too:
    otherwise a dense city's many ponds could crowd out the lake next door.
    """
    seen: dict[str, dict] = {}
    # Nominatim matches exact tags: a lake tagged water=lake is not returned for [natural=water].
    searches = ((radius_km / 4, "[water=lake]"), (radius_km, "[water=lake]"), (radius_km, "[water=reservoir]"),
                (radius_km / 4, "[natural=water]"), (radius_km, "[natural=water]"))
    for box_km, query in searches:
        if seen:
            time.sleep(pause_s)  # Nominatim usage policy: at most one request per second
        dlat = box_km / 110.574
        dlon = box_km / (111.320 * max(math.cos(math.radians(latitude)), 0.01))
        response = get(NOMINATIM_URL, params={
            "q": query, "format": "jsonv2", "limit": 40, "bounded": 1,
            "viewbox": f"{longitude - dlon},{latitude + dlat},{longitude + dlon},{latitude - dlat}"},
            headers={"User-Agent": USER_AGENT}, timeout=20)
        response.raise_for_status()
        for place in response.json():
            name = place.get("name")
            osm = f"{place.get('osm_type')}/{place.get('osm_id')}"
            if not name or place.get("osm_type") not in ("way", "relation") or osm in seen:
                continue
            south, north, west, east = (float(v) for v in place["boundingbox"])
            height_m = (north - south) * 110_574
            width_m = (east - west) * 111_320 * math.cos(math.radians((north + south) / 2))
            if min(height_m, width_m) < NEARBY_MIN_EXTENT_M:
                continue
            lat, lon = float(place["lat"]), float(place["lon"])
            distance, bearing = _distance_bearing(latitude, longitude, lat, lon)
            if distance > radius_km:
                continue
            seen[osm] = {"name": name, "osm": osm, "type": place.get("type"), "latitude": lat, "longitude": lon,
                         "bbox": [south, north, west, east], "distance_km": round(distance, 2),
                         "bearing_deg": round(bearing), "extent_km": round(max(height_m, width_m) / 1000, 2)}
    # The same water is sometimes mapped twice (two names, or two basins): merge objects that share
    # a name or whose centres are within 400 m, keeping the larger one.
    kept: list[dict] = []
    for lake in sorted(seen.values(), key=lambda item: -item["extent_km"]):
        duplicate = any(lake["name"].strip().lower() == other["name"].strip().lower()
                        or _distance_bearing(lake["latitude"], lake["longitude"],
                                             other["latitude"], other["longitude"])[0] < 0.4
                        for other in kept)
        if not duplicate:
            kept.append(lake)
    # The nearest lakes, plus the largest of the rest: a big reservoir 20 km away is worth showing
    # even when a city has dozens of small ponds closer by.
    by_distance = sorted(kept, key=lambda lake: lake["distance_km"])
    nearest = by_distance[: NEARBY_LIMIT * 2 // 3]
    largest = sorted(by_distance[len(nearest):], key=lambda lake: -lake["extent_km"])[: NEARBY_LIMIT - len(nearest)]
    return sorted(nearest + largest, key=lambda lake: lake["distance_km"])

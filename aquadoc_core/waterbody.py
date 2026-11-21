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

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, mapping, shape
from shapely.ops import linemerge, polygonize, unary_union

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OVERPASS_URL = OVERPASS_URLS[0]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_LOOKUP_URL = "https://nominatim.openstreetmap.org/lookup"
WATER_TYPES = {"water", "lake", "reservoir", "river", "riverbank", "bay", "lagoon", "pond", "basin", "strait",
               "wetland", "canal", "oxbow"}
USER_AGENT = "AquaDoc/0.2 (satellite water-quality demo)"

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


@lru_cache(maxsize=256)
def lookup_water_body(osm_ref: str, latitude: float, longitude: float,
                      get: Callable[..., requests.Response] = requests.get) -> WaterBody | None:
    """Polygon for an OSM object such as ``relation/2833155`` via Nominatim lookup (~1 s)."""
    match = _OSM_REF.match(osm_ref)
    if not match:
        raise ValueError("osm_ref must look like 'relation/123' or 'way/123'")
    osm_type, osm_id = match.groups()
    response = get(NOMINATIM_LOOKUP_URL, params={"osm_ids": f"{osm_type[0].upper()}{osm_id}", "format": "jsonv2",
                                                 "polygon_geojson": 1, "polygon_threshold": 0.00005},
                   headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    results = response.json()
    if not results or results[0].get("geojson", {}).get("type") not in ("Polygon", "MultiPolygon"):
        return None
    place = results[0]
    geometry = shape(place["geojson"])
    geometry = geometry if geometry.is_valid else geometry.buffer(0)
    return WaterBody(name=place.get("name") or place.get("display_name"), osm_type=osm_type, osm_id=int(osm_id),
                     water_type=place.get("type"), geometry=geometry,
                     contains_point=geometry.covers(Point(longitude, latitude)))


def search_water_bodies(query: str, limit: int = 8, get: Callable[..., requests.Response] = requests.get) -> list[dict]:
    """Name search via Nominatim, filtered to water-like features."""
    response = get(NOMINATIM_URL, params={"q": query, "format": "jsonv2", "limit": 25},
                   headers={"User-Agent": USER_AGENT}, timeout=20)
    response.raise_for_status()
    water_classes = {"natural", "water", "waterway", "landuse"}
    results = []
    for place in response.json():
        if place.get("category") in water_classes and place.get("type") in WATER_TYPES:
            results.append({"name": place.get("name") or place.get("display_name"),
                            "display_name": place.get("display_name"),
                            "latitude": float(place["lat"]), "longitude": float(place["lon"]),
                            "type": place.get("type"), "osm": f"{place.get('osm_type')}/{place.get('osm_id')}",
                            "bbox": [float(v) for v in place.get("boundingbox", [])]})
    return results[:limit]

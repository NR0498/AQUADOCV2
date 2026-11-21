"""Find candidate Sentinel-2 L2A scenes for an area of interest.

Scene-level ``eo:cloud_cover`` describes a whole 110 km tile, not the lake. It is only
used to discard hopeless scenes; the pipeline then checks pixel-level clouds over the
water itself and moves on to the next candidate if the lake is obscured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import requests
from shapely.geometry import box, shape

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
COLLECTION = "sentinel-2-l2a"


@dataclass(frozen=True)
class SceneCandidate:
    item: Mapping  # raw STAC item (assets, properties, geometry)
    item_id: str
    datetime: str
    cloud_cover: float | None
    platform: str | None
    aoi_coverage: float  # fraction of the AOI bounding box inside the scene footprint

    def summary(self) -> dict:
        return {"id": self.item_id, "datetime": self.datetime, "cloud_cover_pct": self.cloud_cover,
                "platform": self.platform, "aoi_coverage": round(self.aoi_coverage, 3)}


def search_scenes(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 60.0,
    limit: int = 40,
    post: Callable[..., requests.Response] = requests.post,
) -> list[SceneCandidate]:
    """Return candidates sorted newest first, best-covering tile first within a date."""
    payload = {
        "collections": [COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    response = post(STAC_SEARCH_URL, json=payload, timeout=60)
    response.raise_for_status()
    aoi = box(*bbox)
    candidates = []
    for feature in response.json().get("features", []):
        footprint = shape(feature["geometry"]) if feature.get("geometry") else None
        coverage = footprint.intersection(aoi).area / aoi.area if footprint is not None and aoi.area > 0 else 0.0
        properties = feature.get("properties", {})
        candidates.append(SceneCandidate(
            item=feature,
            item_id=str(feature.get("id", "")),
            datetime=str(properties.get("datetime", "")),
            cloud_cover=properties.get("eo:cloud_cover"),
            platform=properties.get("platform"),
            aoi_coverage=coverage,
        ))
    return rank_candidates(candidates)


def rank_candidates(candidates: list[SceneCandidate], min_coverage: float = 0.5) -> list[SceneCandidate]:
    """Drop tiles barely touching the AOI; keep one tile per acquisition (the best covering)."""
    best: dict[str, SceneCandidate] = {}
    for candidate in candidates:
        if candidate.aoi_coverage < min_coverage:
            continue
        acquisition = candidate.datetime[:16]  # overlapping tiles of one pass share a timestamp
        current = best.get(acquisition)
        if current is None or (candidate.aoi_coverage, -(candidate.cloud_cover or 100)) > (
            current.aoi_coverage, -(current.cloud_cover or 100)
        ):
            best[acquisition] = candidate
    return sorted(best.values(), key=lambda c: c.datetime, reverse=True)

from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import requests

from .schemas import MultispectralScene


PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SENTINEL2_L2A_COLLECTION = "sentinel-2-l2a"

SENTINEL2_ASSET_MAP = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "red_edge": "B05",
    "nir": "B08",
    "swir1": "B11",
}


@dataclass(frozen=True)
class Sentinel2SearchConfig:
    """Search settings for Sentinel-2 Level-2A imagery."""

    max_cloud_cover: float = 20.0
    limit: int = 10
    collection: str = SENTINEL2_L2A_COLLECTION
    stac_search_url: str = PLANETARY_COMPUTER_STAC_URL


@dataclass(frozen=True)
class StacItemSummary:
    """Small item summary so the rest of the code does not depend on PySTAC."""

    item_id: str
    datetime: Optional[str]
    cloud_cover: Optional[float]
    assets: Mapping[str, str]


class Sentinel2StacClient:
    """Minimal STAC client for Sentinel-2 items on Microsoft Planetary Computer."""

    def __init__(self, config: Sentinel2SearchConfig | None = None):
        self.config = config or Sentinel2SearchConfig()

    def search(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        size_m: int,
    ) -> list[StacItemSummary]:
        bbox = coordinate_bbox(latitude, longitude, size_m)
        payload = {
            "collections": [self.config.collection],
            "bbox": list(bbox),
            "datetime": f"{start_date}/{end_date}",
            "limit": self.config.limit,
            "query": {
                "eo:cloud_cover": {
                    "lt": self.config.max_cloud_cover,
                }
            },
            "sortby": [
                {"field": "properties.eo:cloud_cover", "direction": "asc"},
                {"field": "properties.datetime", "direction": "desc"},
            ],
        }

        response = requests.post(self.config.stac_search_url, json=payload, timeout=60)
        response.raise_for_status()
        features = response.json().get("features", [])
        return [self._to_summary(feature) for feature in features]

    def _to_summary(self, feature: Mapping) -> StacItemSummary:
        signed_feature = sign_planetary_computer_item(feature)
        properties = signed_feature.get("properties", {})
        assets = {
            key: value["href"]
            for key, value in signed_feature.get("assets", {}).items()
            if isinstance(value, Mapping) and "href" in value
        }
        return StacItemSummary(
            item_id=str(signed_feature.get("id", "")),
            datetime=properties.get("datetime"),
            cloud_cover=properties.get("eo:cloud_cover"),
            assets=assets,
        )


class RasterioBandReader:
    """Read a coordinate-centered chip from Sentinel-2 COG band assets."""

    def read_bands(
        self,
        item: StacItemSummary,
        latitude: float,
        longitude: float,
        size_m: int,
        asset_map: Mapping[str, str],
    ) -> Dict[str, np.ndarray]:
        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.transform import rowcol
            from rasterio.windows import Window
            from rasterio.warp import transform as transform_coords
        except ImportError as exc:
            raise RuntimeError(
                "Reading Sentinel-2 COG assets requires rasterio. Install the geospatial dependencies in requirements.txt."
            ) from exc

        bands: Dict[str, np.ndarray] = {}
        target_shape: Tuple[int, int] | None = None

        for canonical_name, asset_key in asset_map.items():
            href = item.assets.get(asset_key)
            if not href:
                continue

            with rasterio.open(href) as src:
                xs, ys = transform_coords("EPSG:4326", src.crs, [longitude], [latitude])
                center_row, center_col = rowcol(src.transform, xs[0], ys[0])
                pixel_size_x = abs(src.transform.a)
                pixel_size_y = abs(src.transform.e)
                half_width = max(1, int(round(size_m / (2 * pixel_size_x))))
                half_height = max(1, int(round(size_m / (2 * pixel_size_y))))
                window = Window(
                    col_off=center_col - half_width,
                    row_off=center_row - half_height,
                    width=half_width * 2,
                    height=half_height * 2,
                )

                if target_shape is None:
                    target_shape = (int(window.height), int(window.width))

                data = src.read(
                    1,
                    window=window,
                    out_shape=target_shape,
                    boundless=True,
                    fill_value=0,
                    resampling=Resampling.bilinear,
                )
                bands[canonical_name] = data.astype(np.float32)

        return bands


def fetch_sentinel2_chip(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    size_m: int = 3000,
    search_client: Sentinel2StacClient | None = None,
    band_reader: RasterioBandReader | None = None,
    asset_map: Mapping[str, str] | None = None,
) -> MultispectralScene:
    """Fetch an aligned Sentinel-2 multispectral chip for one coordinate.

    The default implementation searches Microsoft Planetary Computer STAC and
    reads Cloud Optimized GeoTIFF assets with Rasterio. Tests can inject fake
    search and reader objects to avoid network access.
    """

    validate_ingestion_request(latitude, longitude, start_date, end_date, size_m)
    selected_asset_map = asset_map or SENTINEL2_ASSET_MAP
    client = search_client or Sentinel2StacClient()
    reader = band_reader or RasterioBandReader()

    items = client.search(latitude, longitude, start_date, end_date, size_m)
    item = choose_best_item(items)
    if item is None:
        raise RuntimeError(
            "No Sentinel-2 scenes found for this coordinate/date range with the current cloud-cover limit."
        )

    bands = reader.read_bands(item, latitude, longitude, size_m, selected_asset_map)
    scene = MultispectralScene(
        latitude=latitude,
        longitude=longitude,
        bands=bands,
        source=f"Microsoft Planetary Computer / {SENTINEL2_L2A_COLLECTION} / {item.item_id}",
        captured_at=item.datetime,
        resolution_m=_estimate_scene_resolution(size_m, bands),
    )
    scene.validate()
    return scene


def coordinate_bbox(latitude: float, longitude: float, size_m: int) -> Tuple[float, float, float, float]:
    """Create a small WGS84 bounding box around a coordinate."""

    validate_lat_lon(latitude, longitude)
    if size_m <= 0:
        raise ValueError("size_m must be positive")

    half_size_km = size_m / 2000.0
    lat_delta = half_size_km / 110.574
    lon_delta = half_size_km / (111.320 * max(0.01, np.cos(np.radians(latitude))))
    return (
        max(-180.0, longitude - lon_delta),
        max(-90.0, latitude - lat_delta),
        min(180.0, longitude + lon_delta),
        min(90.0, latitude + lat_delta),
    )


def choose_best_item(items: Iterable[StacItemSummary]) -> StacItemSummary | None:
    """Choose the item with lowest cloud cover, then newest datetime."""

    item_list = list(items)
    if not item_list:
        return None

    def sort_key(item: StacItemSummary):
        cloud = item.cloud_cover if item.cloud_cover is not None else 100.0
        dt = item.datetime or ""
        return (cloud, _reverse_datetime_key(dt))

    return sorted(item_list, key=sort_key)[0]


def sign_planetary_computer_item(feature: Mapping) -> Mapping:
    """Sign Planetary Computer asset URLs when the optional SDK is installed."""

    try:
        import planetary_computer
        import pystac
    except ImportError:
        return feature

    item = pystac.Item.from_dict(dict(feature))
    planetary_computer.sign_inplace(item)
    return item.to_dict()


def validate_ingestion_request(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    size_m: int,
) -> None:
    validate_lat_lon(latitude, longitude)
    if size_m <= 0:
        raise ValueError("size_m must be positive")
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date must be before or equal to end_date")


def validate_lat_lon(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180")


def _reverse_datetime_key(value: str) -> str:
    return "".join(chr(255 - ord(char)) for char in value)


def _estimate_scene_resolution(size_m: int, bands: Mapping[str, np.ndarray]) -> Optional[float]:
    if not bands:
        return None
    first_band = next(iter(bands.values()))
    if first_band.shape[0] == 0:
        return None
    return float(size_m / first_band.shape[0])

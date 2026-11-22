"""Read Sentinel-2 L2A bands onto one explicit 20 m analysis grid.

Sentinel-2 bands come at 10, 20 and 60 m. Resizing arrays to equal shapes is not the
same as aligning them geographically. Every band here is warped onto a single grid
(CRS + affine transform + size) with a GDAL WarpedVRT: bilinear resampling for
reflectance, nearest-neighbour for the categorical scene classification (SCL).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

# Asset keys on Planetary Computer -> band names used by the water-quality code.
REFLECTANCE_ASSETS: dict[str, str] = {
    "B02": "B2", "B03": "B3", "B04": "B4", "B05": "B5", "B06": "B6",
    "B07": "B7", "B08": "B8", "B8A": "B8A", "B11": "B11", "B12": "B12",
}
SCL_ASSET = "SCL"

RESOLUTION_M = 20


@dataclass(frozen=True)
class Grid:
    crs: str  # e.g. "EPSG:32617"
    transform: tuple[float, float, float, float, float, float]  # affine a, b, c, d, e, f
    width: int
    height: int
    resolution_m: float

    def affine(self):
        from rasterio.transform import Affine

        return Affine(*self.transform)

    def corners_wgs84(self) -> list[list[float]]:
        """[lon, lat] of top-left, top-right, bottom-right, bottom-left (map image overlay order)."""
        from rasterio.warp import transform

        a, _, c, _, e, f = self.transform  # north-up grid: x = c + col*a, y = f + row*e
        pixel_corners = ((0, 0), (self.width, 0), (self.width, self.height), (0, self.height))
        xs = [c + col * a for col, _ in pixel_corners]
        ys = [f + row * e for _, row in pixel_corners]
        lons, lats = transform(self.crs, "EPSG:4326", xs, ys)
        return [[round(lon, 7), round(lat, 7)] for lon, lat in zip(lons, lats)]

    def to_dict(self) -> dict:
        return {"crs": self.crs, "transform": list(self.transform), "width": self.width,
                "height": self.height, "resolution_m": self.resolution_m,
                "corners_wgs84": self.corners_wgs84()}


def utm_crs_for(latitude: float, longitude: float) -> str:
    zone = min(60, int((longitude + 180) // 6) + 1)
    return f"EPSG:{(32600 if latitude >= 0 else 32700) + zone}"


def make_grid(latitude: float, longitude: float, size_m: float, resolution_m: float = RESOLUTION_M) -> Grid:
    """Square grid centred on a coordinate, in the local UTM zone."""
    from rasterio.warp import transform

    crs = utm_crs_for(latitude, longitude)
    xs, ys = transform("EPSG:4326", crs, [longitude], [latitude])
    pixels = max(3, int(math.ceil(size_m / resolution_m)))
    half = pixels * resolution_m / 2
    left = math.floor((xs[0] - half) / resolution_m) * resolution_m
    top = math.ceil((ys[0] + half) / resolution_m) * resolution_m
    return Grid(crs=crs, transform=(resolution_m, 0.0, left, 0.0, -resolution_m, top),
                width=pixels, height=pixels, resolution_m=resolution_m)


def expand_grid(grid: Grid, margin_px: int) -> Grid:
    """Same pixel lattice, ``margin_px`` larger on every side (for context such as clouds
    outside the analysis window whose shadows fall inside it)."""
    a, b, c, d, e, f = grid.transform
    return Grid(crs=grid.crs, transform=(a, b, c - margin_px * a, d, e, f - margin_px * e),
                width=grid.width + 2 * margin_px, height=grid.height + 2 * margin_px,
                resolution_m=grid.resolution_m)


def reflectance_calibration(item: Mapping, asset_key: str) -> tuple[float, float, str]:
    """Scale/offset converting stored integers to reflectance.

    Prefer STAC ``raster:bands`` metadata; otherwise apply the Sentinel-2 rule that
    processing baseline 04.00+ (from January 2022) stores reflectance with a -0.1 offset.
    """
    asset = item["assets"][asset_key]
    bands = asset.get("raster:bands") or []
    if bands and "scale" in bands[0] and "offset" in bands[0]:
        scale, offset, basis = float(bands[0]["scale"]), float(bands[0]["offset"]), "STAC raster:bands"
    else:
        baseline = item.get("properties", {}).get("s2:processing_baseline")
        if baseline is None:
            raise ValueError(f"No reflectance calibration for {asset_key}")
        scale, offset = 0.0001, (-0.1 if float(baseline) >= 4 else 0.0)
        basis = f"processing baseline {baseline} rule"
    if not (math.isfinite(scale) and math.isfinite(offset) and scale > 0):
        raise ValueError(f"Invalid calibration for {asset_key}")
    return scale, offset, basis


class BandReader(Protocol):
    def read(self, item: Mapping, grid: Grid, asset_keys: Sequence[str]) -> tuple[dict[str, np.ndarray], dict]: ...


class CogBandReader:
    """Windowed reads from Cloud Optimized GeoTIFFs over HTTP, warped onto ``grid``."""

    def __init__(self, max_workers: int = 6):
        self.max_workers = max_workers

    def read(self, item: Mapping, grid: Grid, asset_keys: Sequence[str]) -> tuple[dict[str, np.ndarray], dict]:
        missing = [key for key in asset_keys if key not in item.get("assets", {})]
        if missing:
            raise ValueError(f"Scene {item.get('id')} lacks assets {missing}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = dict(zip(asset_keys, pool.map(lambda key: self._read_with_retry(item, key, grid), asset_keys)))
        arrays, calibration = {}, {}
        for key, (array, cal) in results.items():
            arrays[key] = array
            if cal:
                calibration[key] = cal
        return arrays, calibration

    def _read_with_retry(self, item: Mapping, key: str, grid: Grid):
        """One retry with a freshly issued access token: a long-running server can hold a token that
        Planetary Computer no longer accepts, which shows up as a read error on every image."""
        try:
            return self._read_one(item, key, grid)
        except Exception:
            import planetary_computer.sas

            planetary_computer.sas.TOKEN_CACHE.clear()
            return self._read_one(item, key, grid)

    def _read_one(self, item: Mapping, key: str, grid: Grid):
        import planetary_computer
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT

        href = item["assets"][key]["href"]
        if href.startswith("https://") and "sig=" not in href:
            href = planetary_computer.sign(href)
        categorical = key == SCL_ASSET
        env = dict(GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1",
                   GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE="TRUE")
        with rasterio.Env(**env), rasterio.open(href) as src:
            # Sentinel-2 DN 0 is nodata even when the file omits the nodata tag.
            with WarpedVRT(src, crs=grid.crs, transform=grid.affine(), width=grid.width, height=grid.height,
                           src_nodata=src.nodata if src.nodata is not None else 0, nodata=np.nan,
                           dtype="float32",
                           resampling=Resampling.nearest if categorical else Resampling.bilinear) as vrt:
                data = vrt.read(1, masked=True).filled(np.nan).astype(np.float32)
        if categorical:
            return data, None
        scale, offset, basis = reflectance_calibration(item, key)
        return data * scale + offset, {"scale": scale, "offset": offset, "basis": basis}

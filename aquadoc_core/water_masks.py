"""Decide which pixels are usable open water.

Ordered rules, each counted separately so the dashboard can show where pixels went:

1. no data           - outside the swath or DN 0
2. cloud / shadow / snow / defective, from the Sentinel-2 Scene Classification Layer
3. cloud-adjacent    - within 40 m of a cloud or shadow (thin edges, adjacency)
4. outside water     - outside the (shoreline-buffered) OSM polygon, or not water-like
5. floating material - vegetation-like reflectance (NDVI > 0.3) or SCL vegetation on the
                       water: algal scum, floating plants. Kept separate, because
                       the water-quality model assumes open water.
6. glint / mixed     - SWIR reflectance too high for water (sun glint, land mixing)
remaining pixels     -> valid open water, fed to the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from .s2_reader import Grid

SCL_CLASSES = {
    0: "no_data", 1: "saturated_or_defective", 2: "dark_area", 3: "cloud_shadow", 4: "vegetation",
    5: "not_vegetated", 6: "water", 7: "unclassified", 8: "cloud_medium_probability",
    9: "cloud_high_probability", 10: "thin_cirrus", 11: "snow_or_ice",
}
SCL_BAD = (1, 3, 8, 9, 10, 11)

# Inward shoreline buffers tried in order. Bright land scatters light into nearby water
# pixels (the adjacency effect), so the widest buffer that still leaves MIN_INTERIOR_PX
# pixels is used. Small or narrow water bodies fall back to narrower buffers.
SHORELINE_BUFFERS_M = (100.0, 60.0, 40.0, 20.0, 0.0)
MIN_INTERIOR_PX = 100
CLOUD_BUFFER_M = 100.0       # haze and adjacency around cloud edges
SHADOW_HEIGHTS_M = (200.0, 4000.0)  # plausible cloud-base heights for shadow projection
SHADOW_DARKENING = 0.8       # shadowed water is darker than this fraction of nearby cloud-free water
LOCAL_WINDOW_PX = 61         # ~1.2 km neighbourhood for the local brightness reference
FLOATING_NDVI = 0.3
GLINT_SWIR_REFLECTANCE = 0.05


@dataclass
class MaskResult:
    valid_water: np.ndarray
    floating: np.ndarray
    water_area: np.ndarray
    counts: dict
    layers: dict[str, np.ndarray]  # boolean masks per category, for the classification map
    notes: list[str]


def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.copy()
    padded = np.pad(mask, pixels, constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(-pixels, pixels + 1):
        for dx in range(-pixels, pixels + 1):
            if dx * dx + dy * dy <= pixels * pixels:
                result |= padded[pixels + dy:pixels + dy + mask.shape[0], pixels + dx:pixels + dx + mask.shape[1]]
    return result


def shift(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Move a boolean mask by (rows, cols) pixels, filling with False."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    if abs(rows) >= h or abs(cols) >= w:
        return out
    src = mask[max(0, -rows):h - max(0, rows), max(0, -cols):w - max(0, cols)]
    out[max(0, rows):max(0, rows) + src.shape[0], max(0, cols):max(0, cols) + src.shape[1]] = src
    return out


def project_shadow_zone(cloud: np.ndarray, sun_zenith_deg: float, sun_azimuth_deg: float,
                        resolution_m: float, heights_m: tuple[float, float] = SHADOW_HEIGHTS_M) -> np.ndarray:
    """Where shadows of ``cloud`` can fall for cloud heights in ``heights_m``.

    A cloud at height h casts its shadow h * tan(zenith) away from the sun. With azimuth
    measured clockwise from north and image rows increasing southward, the displacement
    in pixels is (+cos(az), -sin(az)) * distance. Satellite view parallax is ignored.
    """
    tan_zenith = math.tan(math.radians(sun_zenith_deg))
    d_min, d_max = (h * tan_zenith / resolution_m for h in heights_m)
    row_unit, col_unit = math.cos(math.radians(sun_azimuth_deg)), -math.sin(math.radians(sun_azimuth_deg))
    zone = np.zeros_like(cloud)
    seen = set()
    for distance in np.arange(d_min, d_max + 1.0, 1.0):
        offset = (int(round(distance * row_unit)), int(round(distance * col_unit)))
        if offset not in seen:
            seen.add(offset)
            zone |= shift(cloud, *offset)
    return zone & ~cloud


def local_mean(values: np.ndarray, reference_mask: np.ndarray, window: int = LOCAL_WINDOW_PX,
               min_count: int = 20) -> np.ndarray:
    """Local mean of ``values`` over ``reference_mask`` pixels in a square window (box filter
    via cumulative sums). Falls back to the global median where too few reference pixels exist."""
    v = np.where(reference_mask, values, 0.0)
    m = reference_mask.astype(float)
    pad = window // 2

    def box(a):
        c = np.pad(a, pad + 1, mode="constant").cumsum(0).cumsum(1)
        return c[window:, window:] - c[:-window, window:] - c[window:, :-window] + c[:-window, :-window]

    total, count = box(v)[: values.shape[0], : values.shape[1]], box(m)[: values.shape[0], : values.shape[1]]
    global_ref = float(np.median(values[reference_mask])) if reference_mask.any() else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        local = total / count
    return np.where(count >= min_count, local, global_ref)


def rasterize_water_polygon(geometry_wgs84: BaseGeometry, grid: Grid) -> tuple[np.ndarray, float | None, list[str]]:
    """Rasterize the polygon on the grid with the widest workable inward shoreline buffer."""
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom

    projected = shape(transform_geom("EPSG:4326", grid.crs, mapping(geometry_wgs84)))
    a = grid.affine()
    left, top = a.c, a.f
    extent = box(left, top - grid.height * grid.resolution_m, left + grid.width * grid.resolution_m, top)
    clipped = projected.intersection(extent.buffer(2 * SHORELINE_BUFFERS_M[0]))
    best = None
    for buffer_m in SHORELINE_BUFFERS_M:
        interior = clipped.buffer(-buffer_m) if buffer_m else clipped
        if interior.is_empty:
            continue
        mask = geometry_mask([mapping(interior)], (grid.height, grid.width), a, invert=True, all_touched=False)
        count = int(mask.sum())
        if count >= MIN_INTERIOR_PX:
            best = (mask, buffer_m)
            break
        if count >= 4 and best is None:
            best = (mask, buffer_m)  # keep the widest buffer that leaves anything, in case none reaches the minimum
    if best is None:
        return np.zeros((grid.height, grid.width), dtype=bool), None, ["Water polygon covers no full pixels at 20 m."]
    mask, buffer_m = best
    notes = []
    if buffer_m < 60:
        notes.append(f"Small or narrow water body: shoreline buffer is only {buffer_m:.0f} m, so light scattered from "
                     "nearby land (adjacency effect) may bias estimates upward.")
    return mask, buffer_m, notes


def classify_pixels(rho: dict[str, np.ndarray], scl: np.ndarray, polygon_mask: np.ndarray | None,
                    resolution_m: float, sun_zenith_deg: float | None = None,
                    sun_azimuth_deg: float | None = None, context_scl: np.ndarray | None = None,
                    context_margin_px: int = 0) -> MaskResult:
    """``rho`` holds surface reflectance for at least B2, B3, B4, B8 and B11.

    Sun angles enable geometric cloud-shadow detection, which the Sentinel-2 SCL often
    misses over water because shadowed water and clear dark water look alike.
    ``context_scl`` is the SCL on a grid ``context_margin_px`` larger on each side, so
    clouds just outside the window still cast their shadows into it.
    """
    shape_ = scl.shape
    finite = np.isfinite(scl) & (scl != 0)
    for array in rho.values():
        finite &= np.isfinite(array)

    scl_int = np.where(np.isfinite(scl), scl, 0).astype(int)
    cloud = finite & np.isin(scl_int, (8, 9, 10))
    projected_shadow = np.zeros_like(cloud)
    if context_scl is not None:
        context_cloud = np.isin(np.where(np.isfinite(context_scl), context_scl, 0).astype(int), (8, 9, 10))
    else:
        context_cloud, context_margin_px = cloud, 0
    mndwi = _nd(rho["B3"], rho["B11"])
    ndvi = _nd(rho["B8"], rho["B4"])
    notes = []
    if polygon_mask is not None:
        area = polygon_mask
    else:
        area = (scl_int == 6) | (mndwi > 0.1)
        notes.append("No mapped water polygon: water detected from SCL and MNDWI only.")

    if sun_zenith_deg is not None and sun_azimuth_deg is not None and context_cloud.any():
        m = context_margin_px
        zone = project_shadow_zone(context_cloud, sun_zenith_deg, sun_azimuth_deg, resolution_m)
        zone = zone[m:m + shape_[0], m:m + shape_[1]] & ~cloud
        # Compare with the local mean of cloud-free water (shadows included): shadows are
        # markedly darker than their surroundings, natural gradients are smoother.
        brightness = np.where(finite, rho["B2"] + rho["B3"] + rho["B4"], 0.0)
        haze_free = ~dilate(cloud, math.ceil(CLOUD_BUFFER_M / resolution_m))
        reference = area & finite & haze_free & ~np.isin(scl_int, SCL_BAD)
        local = local_mean(brightness, reference)
        projected_shadow = dilate(zone & finite & (brightness < SHADOW_DARKENING * local), 1) & ~cloud
    bad = finite & (np.isin(scl_int, SCL_BAD) | projected_shadow)
    adjacent = dilate(bad, math.ceil(CLOUD_BUFFER_M / resolution_m)) & ~bad & finite
    if polygon_mask is None:
        # Without an outline, cloud/shadow pixels hide whether the surface is water: count
        # them in the area so a clouded lake does not look clear.
        area = area | bad

    usable = area & finite & ~bad & ~adjacent
    floating = usable & ((ndvi > FLOATING_NDVI) | (scl_int == 4))
    open_candidate = usable & ~floating
    not_water_like = open_candidate & ~(mndwi > 0)
    glint = open_candidate & ~not_water_like & (rho["B11"] > GLINT_SWIR_REFLECTANCE)
    valid = open_candidate & ~not_water_like & ~glint

    def count(mask):
        return int(np.count_nonzero(mask))

    in_area = count(area)
    counts = {
        "grid_pixels": int(np.prod(shape_)),
        "water_area_pixels": in_area,
        "no_data": count(area & ~finite),
        "cloud": count(area & np.isin(scl_int, (8, 9, 10))),
        "cloud_shadow": count(area & ((scl_int == 3) | projected_shadow)),
        "snow_ice": count(area & (scl_int == 11)),
        "saturated_or_defective": count(area & (scl_int == 1)),
        "cloud_adjacent": count(area & adjacent),
        "floating_vegetation_or_scum": count(floating),
        "not_water_like": count(not_water_like),
        "glint_or_mixed": count(glint),
        "valid_open_water": count(valid),
    }
    counts["clear_fraction"] = round(counts["valid_open_water"] / in_area, 4) if in_area else 0.0
    layers = {
        "cloud_or_shadow": area & bad,
        "cloud_adjacent": area & adjacent,
        "not_water_like": not_water_like,
        "glint_or_mixed": glint,
        "floating_vegetation_or_scum": floating,
        "valid_open_water": valid,
    }
    return MaskResult(valid_water=valid, floating=floating, water_area=area, counts=counts,
                      layers=layers, notes=notes)

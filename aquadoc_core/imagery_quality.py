"""Aligned Sentinel-2 surface-reflectance extraction and conservative local QA."""
import io
import math
import zipfile
from dataclasses import asdict, dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform, transform_geom
import shapefile
from shapely.geometry import Point, mapping, shape

from .ingestion import SENTINEL2_ASSET_MAP

FEATURE_VERSION = 'nla-s2-local-water-v1'


@dataclass(frozen=True)
class QualityConfig:
    resolution_m: int = 20
    radius_m: int = 60
    shoreline_buffer_m: int = 20
    cloud_buffer_m: int = 40
    min_pixels: int = 9
    min_valid_fraction: float = 0.7

    def __post_init__(self):
        if self.resolution_m != 20:
            raise ValueError('This feature version requires a 20 m grid')
        if not 0 < self.radius_m <= 500 or not 0 <= self.shoreline_buffer_m <= 200 or not 0 <= self.cloud_buffer_m <= 200:
            raise ValueError('Invalid spatial QA distances')
        if self.min_pixels < 1 or not 0 < self.min_valid_fraction <= 1:
            raise ValueError('Invalid pixel quality thresholds')


class LakePolygons:
    """Read the supplied EPA archive in memory, without changing or extracting it."""
    def __init__(self, archive):
        with zipfile.ZipFile(archive) as bundle:
            stems = [name[:-4] for name in bundle.namelist() if name.endswith('.shp')]
            if len(stems) != 1:
                raise ValueError('Expected one lake polygon layer')
            stem = stems[0]
            self.crs = rasterio.crs.CRS.from_wkt(bundle.read(stem + '.prj').decode())
            reader = shapefile.Reader(**{ext: io.BytesIO(bundle.read(stem + '.' + ext)) for ext in ('shp', 'shx', 'dbf')})
            self.sites = {}
            for record in reader.iterShapeRecords():
                site = record.record.as_dict()['SITE_ID'].strip()
                if site in self.sites:
                    raise ValueError(f'Duplicate lake polygon site: {site}')
                self.sites[site] = shape(record.shape.__geo_interface__)

    def get(self, site, latitude, longitude):
        lake = self.sites.get(site)
        if lake is None:
            return None, 'missing_lake_polygon'
        if lake.is_empty or not lake.is_valid:
            return None, 'invalid_lake_polygon'
        xs, ys = transform('EPSG:4326', self.crs, [longitude], [latitude])
        if not lake.covers(Point(xs[0], ys[0])):
            return None, 'sampling_point_outside_lake_polygon'
        return lake, None


def grid(latitude, longitude, config):
    zone = min(60, int((longitude + 180) // 6) + 1)
    crs = rasterio.crs.CRS.from_epsg((32600 if latitude >= 0 else 32700) + zone)
    xs, ys = transform('EPSG:4326', crs, [longitude], [latitude])
    # Extra context avoids treating the chip boundary as an unobserved cloud-free edge.
    half = math.ceil((config.radius_m + config.cloud_buffer_m + config.resolution_m) / config.resolution_m)
    size = 2 * half + 1
    affine = from_origin(xs[0] - (half + .5) * config.resolution_m,
                         ys[0] + (half + .5) * config.resolution_m,
                         config.resolution_m, config.resolution_m)
    return crs, affine, size, xs[0], ys[0]


def spatial_masks(lake, lake_crs, latitude, longitude, config):
    crs, affine, size, x, y = grid(latitude, longitude, config)
    projected = shape(transform_geom(lake_crs, crs, mapping(lake)))
    interior = projected.buffer(-config.shoreline_buffer_m)
    lake_mask = np.zeros((size, size), dtype=bool) if interior.is_empty else geometry_mask(
        [mapping(interior)], (size, size), affine, invert=True, all_touched=False)
    rr, cc = np.indices((size, size))
    distance = np.hypot(rr - size // 2, cc - size // 2) * config.resolution_m
    neighborhood = distance <= config.radius_m
    return crs, affine, lake_mask & neighborhood, neighborhood


def radiometry(item, key):
    """Use declared raster calibration first, otherwise the Sentinel-2 baseline rule."""
    bands = item['assets'][key].get('raster:bands', [])
    if bands and 'scale' in bands[0] and 'offset' in bands[0]:
        scale, offset = float(bands[0]['scale']), float(bands[0]['offset'])
        basis = 'STAC raster:bands'
    else:
        baseline = item.get('properties', {}).get('s2:processing_baseline')
        if baseline is None:
            raise ValueError('Missing reflectance calibration and processing baseline')
        baseline = float(baseline)
        if not math.isfinite(baseline):
            raise ValueError('Invalid processing baseline')
        scale, offset = 0.0001, (-0.1 if baseline >= 4 else 0.0)
        basis = 'Sentinel-2 processing-baseline DN rule'
    if not math.isfinite(scale) or not math.isfinite(offset) or scale <= 0:
        raise ValueError('Invalid reflectance scale/offset')
    return scale, offset, basis


def read_aligned(item, latitude, longitude, config):
    """Read all bands onto one explicit metric grid; SCL always uses nearest neighbour."""
    import planetary_computer
    crs, affine, size, _, _ = grid(latitude, longitude, config)
    arrays, calibration = {}, {}
    asset_map = dict(SENTINEL2_ASSET_MAP, scl='SCL')
    for name, key in asset_map.items():
        if key not in item.get('assets', {}):
            raise ValueError(f'Missing required asset {key}')
        asset = item['assets'][key]
        href = asset['href']
        if href.startswith('https://'):
            href = planetary_computer.sign(href)
        with rasterio.Env(GDAL_HTTP_TIMEOUT='30', GDAL_HTTP_MAX_RETRY='2', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            with rasterio.open(href) as src:
                if src.crs is None:
                    raise ValueError(f'Asset {key} has no CRS')
                # Sentinel-2 DN zero is nodata, even when the TIFF omits its nodata tag.
                with WarpedVRT(src, crs=crs, transform=affine, width=size, height=size,
                               src_nodata=src.nodata if src.nodata is not None else 0,
                               nodata=np.nan, dtype='float32',
                               resampling=Resampling.nearest if name == 'scl' else Resampling.bilinear) as vrt:
                    data = vrt.read(1, masked=True).filled(np.nan)
        if name != 'scl':
            scale, offset, basis = radiometry(item, key)
            data = data * scale + offset
            calibration[key] = dict(scale=scale, offset=offset, basis=basis)
        arrays[name] = data
    return arrays, dict(crs=crs.to_string(), transform=list(affine)[:6], shape=[size, size],
                        calibration=calibration, config=asdict(config))


def dilate(mask, pixels):
    padded = np.pad(mask, pixels, constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(-pixels, pixels + 1):
        for dx in range(-pixels, pixels + 1):
            if dx * dx + dy * dy <= pixels * pixels:
                result |= padded[pixels + dy:pixels + dy + mask.shape[0], pixels + dx:pixels + dx + mask.shape[1]]
    return result


def assess(arrays, lake_mask, neighborhood, config):
    if set(arrays) != set(SENTINEL2_ASSET_MAP) | {'scl'}:
        raise ValueError('All reflectance bands and SCL are required')
    if any(a.shape != lake_mask.shape for a in arrays.values()) or neighborhood.shape != lake_mask.shape:
        raise ValueError('QA arrays must share the same grid')
    scl = arrays['scl']
    finite = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    reflectance_valid = np.logical_and.reduce([(arrays[k] >= 0) & (arrays[k] <= 1) for k in SENTINEL2_ASSET_MAP])
    bad = ~np.isfinite(scl) | np.isin(scl, [0, 1, 2, 3, 8, 9, 10, 11])
    buffered_bad = dilate(bad, math.ceil(config.cloud_buffer_m / config.resolution_m))
    valid = lake_mask & finite & reflectance_valid & (scl == 6) & ~buffered_bad
    total, potential, kept = int(neighborhood.sum()), int(lake_mask.sum()), int(valid.sum())
    metrics = dict(neighborhood_pixels=total, lake_interior_pixels=potential, valid_water_pixels=kept,
                   valid_fraction=kept / total if total else 0,
                   lake_valid_fraction=kept / potential if potential else 0,
                   finite_fraction=float((finite & neighborhood).sum() / total) if total else 0,
                   cloud_shadow_snow_fraction=float((np.isin(scl, [2, 3, 8, 9, 10, 11]) & neighborhood).sum() / total) if total else 0)
    reasons = []
    if kept < config.min_pixels:
        reasons.append('insufficient_valid_water_pixels')
    if metrics['valid_fraction'] < config.min_valid_fraction:
        reasons.append('insufficient_clear_water_coverage')
    features = {}
    if not reasons:
        values = {k: a[valid].astype(float) for k, a in arrays.items() if k != 'scl'}
        for name, numerator, denominator in (
                ('ndci', values['red_edge'] - values['red'], values['red_edge'] + values['red']),
                ('ndwi', values['green'] - values['nir'], values['green'] + values['nir']),
                ('mndwi', values['green'] - values['swir1'], values['green'] + values['swir1'])):
            if np.any(np.abs(denominator) <= 1e-8):
                reasons.append('undefined_spectral_ratio')
                break
            values[name] = numerator / denominator
        if not reasons:
            for name, data in values.items():
                for label, value in zip(('p10', 'median', 'p90'), np.percentile(data, [10, 50, 90])):
                    features[f'{name}_{label}'] = float(value)
    return dict(passed=not reasons, reasons=reasons, metrics=metrics, features=features,
                feature_version=FEATURE_VERSION, review_status='pending' if not reasons else 'rejected')

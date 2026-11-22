"""End-to-end water-quality analysis for one location, as a stream of stage events.

    for event in run_water_analysis(WaterAnalysisRequest(latitude=41.7, longitude=-83.3)):
        print(event.stage, event.status, event.message)

Stages: locate -> search -> scene (one per attempted acquisition) -> estimate ->
explain -> render -> complete. The final ``complete`` event carries the JSON result
in ``data`` and the PNG layers in ``images``. On failure, a ``failed`` event ends the
stream. External services are injectable, so tests run offline.
"""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta, timezone, datetime
from functools import lru_cache
from typing import Callable, Iterator

import numpy as np
from shapely.geometry import mapping

from . import band_guide, render, wq_algorithms
from .s2_reader import REFLECTANCE_ASSETS, SCL_ASSET, BandReader, CogBandReader, Grid, expand_grid, make_grid
from .scene_search import SceneCandidate, search_scenes
from .water_masks import SHADOW_HEIGHTS_M, classify_pixels, local_mean, rasterize_water_polygon
from .waterbody import WaterBody, find_water_body_cached, lookup_water_body
from .wq_features import MODEL_BANDS
from .wq_model import WaterQualityModel

PIPELINE_VERSION = "water-pipeline-v4"  # bump whenever results change shape or meaning (invalidates the API cache)
MIN_SIZE_M, DEFAULT_MAX_SIZE_M, HARD_MAX_SIZE_M = 600, 5000, 10000
# Floating-material anomaly: FDI this far above the scene's open-water median (reflectance
# units) and above 4 robust standard deviations. Absolute thresholds fail in turbid lakes.
DEBRIS_FDI_ANOMALY = 0.02
LOCATE_TIMEOUT_S = 60       # give up on the outline (and use spectral water detection) after this
MAX_SHADOW_MARGIN_PX = 300  # 6 km of extra cloud context at 20 m
SMOOTHING_WINDOW_PX = 3     # 3x3 mean of valid neighbours suppresses per-pixel sensor noise


@dataclass(frozen=True)
class WaterAnalysisRequest:
    latitude: float
    longitude: float
    start_date: str | None = None   # default: end_date - lookback_days
    end_date: str | None = None     # default: today (UTC)
    lookback_days: int = 60
    max_size_m: int = DEFAULT_MAX_SIZE_M
    max_scene_attempts: int = 6
    max_cloud_cover: float = 60.0
    min_valid_pixels: int = 20
    min_clear_fraction: float = 0.2
    osm_ref: str | None = None      # e.g. "relation/2833155" from name search; enables a fast outline lookup

    def resolved_dates(self) -> tuple[str, str]:
        end = date.fromisoformat(self.end_date) if self.end_date else datetime.now(timezone.utc).date()
        start = date.fromisoformat(self.start_date) if self.start_date else end - timedelta(days=self.lookback_days)
        return start.isoformat(), end.isoformat()

    def validate(self) -> None:
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("Coordinate out of range")
        start, end = self.resolved_dates()
        if start > end:
            raise ValueError("start_date must not be after end_date")
        if start < "2017-03-28":
            raise ValueError("Sentinel-2 L2A coverage on Planetary Computer starts in 2017")
        if not MIN_SIZE_M <= self.max_size_m <= HARD_MAX_SIZE_M:
            raise ValueError(f"max_size_m must be between {MIN_SIZE_M} and {HARD_MAX_SIZE_M}")
        if not 1 <= self.max_scene_attempts <= 8:
            raise ValueError("max_scene_attempts must be between 1 and 8")


@dataclass
class PipelineEvent:
    stage: str
    status: str  # started | completed | rejected | skipped | failed
    message: str
    data: dict = field(default_factory=dict)
    elapsed_ms: int = 0
    images: dict[str, bytes] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {"stage": self.stage, "status": self.status, "message": self.message,
                "data": self.data, "elapsed_ms": self.elapsed_ms}


@dataclass
class Services:
    find_water_body: Callable[[float, float], WaterBody | None] = find_water_body_cached
    lookup_water_body: Callable[[str, float, float], WaterBody | None] = lookup_water_body
    search_scenes: Callable[..., list[SceneCandidate]] = search_scenes
    reader: BandReader = field(default_factory=CogBandReader)
    load_model: Callable[[], WaterQualityModel] = None  # type: ignore[assignment]

    def model(self) -> WaterQualityModel:
        return (self.load_model or _default_model)()


@lru_cache(maxsize=1)
def _default_model() -> WaterQualityModel:
    return WaterQualityModel.load()


# --------------------------------------------------------------------------- helpers

log = logging.getLogger("aquadoc.pipeline")


def _short_error(exc: Exception, limit: int = 160) -> str:
    """Exception type and message without URL query strings (signed links carry credentials)."""
    text = re.sub(r"\?[^\s'\"]*", "", str(exc)).strip()
    return f"{type(exc).__name__}: {text}"[:limit] if text else type(exc).__name__

def _meters_extent(bounds, latitude) -> tuple[float, float]:
    minx, miny, maxx, maxy = bounds
    return (maxx - minx) * 111_320 * math.cos(math.radians(latitude)), (maxy - miny) * 110_574


def choose_grid(request: WaterAnalysisRequest, body: WaterBody | None) -> tuple[Grid, str]:
    """Fit small water bodies entirely; for large ones, centre a capped window on the point."""
    if body is not None:
        width_m, height_m = _meters_extent(body.geometry.bounds, request.latitude)
        needed = max(width_m, height_m) + 200
        if needed <= request.max_size_m:
            minx, miny, maxx, maxy = body.geometry.bounds
            size = max(MIN_SIZE_M, needed)
            return make_grid((miny + maxy) / 2, (minx + maxx) / 2, size), "whole water body"
        return make_grid(request.latitude, request.longitude, request.max_size_m), (
            f"{request.max_size_m / 1000:g} km window around the selected point (water body is larger)")
    return make_grid(request.latitude, request.longitude, min(2000, request.max_size_m)), "2 km window (no mapped polygon)"


def _locate_done(event, body: WaterBody | None, grid: Grid, extent_note: str, notes: list[str]) -> PipelineEvent:
    if body is not None:
        message = f"Found {body.name or 'an unnamed water body'} ({body.water_type or 'water'}); analysing {extent_note}"
        if not body.contains_point:
            notes.append("The selected point is not inside a mapped water polygon; using the nearest one.")
    else:
        message = f"No mapped water outline; detecting water spectrally in the {extent_note}"
    return event("locate", "completed", message, {
        "water_body": {**body.summary(), "geometry": _display_geometry(body, grid)} if body else None,
        "grid_corners": grid.corners_wgs84(),
        "grid": {"width": grid.width, "height": grid.height, "resolution_m": grid.resolution_m, "extent": extent_note},
    })


def _bbox_of(grid: Grid) -> tuple[float, float, float, float]:
    corners = grid.corners_wgs84()
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    return min(lons), min(lats), max(lons), max(lats)


def _stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": None, "p10": None, "p90": None}
    p10, median, p90 = np.percentile(finite, [10, 50, 90])
    return {"median": _round(median), "p10": _round(p10), "p90": _round(p90)}


def _round(value: float, digits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(f"{value:.{digits}g}")


def _previews(images: dict[str, bytes]) -> dict[str, str]:
    """PNG layers as data URLs, so a streaming client can show them the moment a stage ends."""
    import base64

    return {name: "data:image/png;base64," + base64.b64encode(png).decode("ascii") for name, png in images.items()}


def _pixel_class_png(masks) -> bytes:
    order = ["cloud_or_shadow", "cloud_adjacent", "not_water_like", "glint_or_mixed",
             "floating_vegetation_or_scum", "valid_open_water"]  # painted bottom -> top
    return render.class_map([(masks.layers[name], render.hex_to_rgba(render.PIXEL_CLASS_COLORS[name], 230))
                             for name in order], masks.valid_water.shape)


def _display_geometry(body: WaterBody, grid: Grid) -> dict | None:
    """Outline clipped to the analysis window and simplified (~20 m), for the map."""
    from shapely.geometry import box

    clipped = body.geometry.intersection(box(*_bbox_of(grid)).buffer(0.002))
    return mapping(clipped.simplify(0.0002, preserve_topology=True)) if not clipped.is_empty else None


BLOCKER_TEXT = {
    "cloud": "cloud, cloud shadow and haze",
    "floating": "floating vegetation or dense algal scum",
    "not_water": "surface that does not look like open water",
    "glint": "sun glint",
}


def _attempt_order(candidates: list, limit: int, newest: int = 2) -> list:
    """The newest images first (the goal is the latest clear view), then the least cloudy of the rest.

    Tile cloud cover describes a 110 km tile, not the lake, but when the newest images are clouded
    over, the clearest older tiles are far more likely to show the lake than the next-newest ones."""
    head = candidates[:newest]
    rest = sorted(candidates[newest:], key=lambda c: (c.cloud_cover if c.cloud_cover is not None else 100.0))
    return (head + rest)[:limit]


def _main_blocker(counts: dict) -> str:
    """Which kind of pixel removed most of the water on a rejected image."""
    removed = {
        "cloud": counts["cloud"] + counts["cloud_shadow"] + counts["cloud_adjacent"] + counts["snow_ice"],
        "floating": counts["floating_vegetation_or_scum"],
        "not_water": counts["not_water_like"],
        "glint": counts["glint_or_mixed"],
    }
    return max(removed, key=removed.get)


def _no_image_message(blockers: list[str]) -> str:
    if blockers and blockers.count("floating") * 2 >= len(blockers):
        return ("On the clear images this water was mostly covered by floating vegetation or dense algal scum, so "
                "there was too little open water to estimate concentrations. The surface cover itself is the finding.")
    if blockers and all(b in ("floating", "not_water") for b in blockers):
        return ("The surface did not look like open water on any image (dense vegetation, scum, very shallow or "
                "unusually coloured water such as a salt lake), so the model was not applied.")
    if not blockers:
        return "No image in this period could be read. Try again, or choose a longer date range."
    return "No image in this period had enough clear open water (clouds or haze). Try a longer date range."


def _trophic_state(chl: float | None) -> str | None:
    if chl is None:
        return None
    for level in band_guide.PARAMETERS["chlorophyll_a"]["scale"]:
        if level["max"] is None or chl < level["max"]:
            return level["label"]
    return None


def satellite_mdsa(model: WaterQualityModel, target: str) -> float | None:
    """Median error measured on real Sentinel-2 match-ups for the input the pipeline uses."""
    validation = model.target_spec(target).get("satellite_validation") or {}
    metrics = validation.get(f"satellite_{model.satellite_input(target)}") or {}
    return metrics.get("mdsa_pct")


def _confidence(in_domain_fraction: float, interval_factor: float | None, valid_pixels: int,
                validated_mdsa: float | None) -> str:
    """low / medium / high.

    Without satellite validation the best possible is "medium". With it, "high" needs a
    measured satellite error of at most 40% (MdSA), inputs inside the training domain and
    enough pixels; a measured error above 100% (worse than a factor of two) means "low".
    """
    if interval_factor is None or valid_pixels < 20 or in_domain_fraction < 0.5:
        return "low"
    if validated_mdsa is None:
        return "low" if interval_factor > 4 else "medium"
    if validated_mdsa > 100:
        return "low"
    if validated_mdsa <= 40 and in_domain_fraction >= 0.9 and valid_pixels >= 100:
        return "high"
    return "medium"


def band_sensitivity(model: WaterQualityModel, rrs_median: dict[str, float], step: float = 0.1) -> dict[str, dict]:
    """Elasticity of each estimate to each band at this location's median spectrum.

    elasticity = d log(estimate) / d log(band): +1 means a 10% brighter band gives a ~10%
    higher estimate. Shares are |elasticity| normalised to 100% per target.
    """
    base = {band: np.array([value]) for band, value in rrs_median.items()}
    results: dict[str, dict] = {target: {} for target in model.targets}
    for band in MODEL_BANDS:
        up = dict(base, **{band: base[band] * (1 + step)})
        down = dict(base, **{band: base[band] * (1 - step)})
        high, low = model.predict_satellite(up), model.predict_satellite(down)
        for target in model.targets:
            elasticity = (math.log(high[target].median[0]) - math.log(low[target].median[0])) / (
                math.log(1 + step) - math.log(1 - step))
            results[target][band] = elasticity
    output = {}
    for target, values in results.items():
        total = sum(abs(v) for v in values.values()) or 1.0
        output[target] = [{"band": band, "elasticity": round(value, 3), "share_pct": round(100 * abs(value) / total, 1)}
                          for band, value in sorted(values.items(), key=lambda kv: -abs(kv[1]))]
    return output


# -------------------------------------------------------------------------- pipeline

def run_water_analysis(request: WaterAnalysisRequest, services: Services | None = None) -> Iterator[PipelineEvent]:
    services = services or Services()
    t0 = time.perf_counter()

    timeline: list[dict] = []  # stored with the result so a shared link can replay the pipeline

    def event(stage, status, message, data=None, images=None):
        built = PipelineEvent(stage, status, message, data or {}, int((time.perf_counter() - t0) * 1000), images or {})
        stored = {} if stage == "complete" else {k: v for k, v in built.data.items() if k != "previews"}
        if "previews" in built.data:
            stored["preview_layers"] = list(built.data["previews"])
        timeline.append({"stage": stage, "status": status, "message": message, "data": stored,
                         "elapsed_ms": built.elapsed_ms})
        return built

    try:
        request.validate()
    except ValueError as exc:
        yield event("validate", "failed", str(exc))
        return
    start_date, end_date = request.resolved_dates()
    notes: list[str] = []

    # 1. Locate the water body ------------------------------------------------------
    body: WaterBody | None = None
    body_future: Future | None = None
    executor: ThreadPoolExecutor | None = None
    if request.osm_ref:
        yield event("locate", "started", f"Fetching the outline of OpenStreetMap {request.osm_ref}")
        try:
            body = services.lookup_water_body(request.osm_ref, request.latitude, request.longitude)
        except Exception as exc:  # the polygon is helpful but optional
            notes.append(f"Water-body lookup failed ({type(exc).__name__}); using spectral water detection.")
        grid, extent_note = choose_grid(request, body)
        yield _locate_done(event, body, grid, extent_note, notes)
    else:
        # Point lookups go through Overpass, which can take tens of seconds: run it in the
        # background on a fixed window around the point while the satellite data loads.
        executor = ThreadPoolExecutor(max_workers=1)
        body_future = executor.submit(services.find_water_body, request.latitude, request.longitude)
        grid, extent_note = make_grid(request.latitude, request.longitude, request.max_size_m), (
            f"{request.max_size_m / 1000:g} km window around the selected point")
        yield event("locate", "started", "Looking up the water body outline in OpenStreetMap "
                    "(in parallel with the satellite search)")

    def resolve_body():
        nonlocal body, body_future
        if body_future is None:
            return None
        try:
            body = body_future.result(timeout=max(1.0, LOCATE_TIMEOUT_S - (time.perf_counter() - t0)))
        except Exception as exc:
            body = None
            notes.append(f"Water-body outline unavailable ({type(exc).__name__}); water detected spectrally.")
        finally:
            body_future = None
            executor.shutdown(wait=False, cancel_futures=True)
        return _locate_done(event, body, grid, extent_note, notes)

    try:
        # 2. Search the catalogue ---------------------------------------------------
        bbox = _bbox_of(grid)
        yield event("search", "started", f"Searching Sentinel-2 L2A scenes {start_date} → {end_date}")
        try:
            candidates = services.search_scenes(bbox, start_date, end_date, max_cloud_cover=request.max_cloud_cover)
        except Exception as exc:
            yield event("search", "failed", f"Satellite catalogue search failed: {exc}")
            return
        if not candidates:
            yield event("search", "failed", "No Sentinel-2 scenes with less than "
                        f"{request.max_cloud_cover:.0f}% tile cloud cover in this period. Try a longer date range.")
            return
        yield event("search", "completed", f"{len(candidates)} candidate acquisitions found, newest first",
                    {"candidates": [c.summary() for c in candidates[:10]]})
        polygon_mask = shoreline_buffer = None
        polygon_ready = False
        asset_keys = [*REFLECTANCE_ASSETS, SCL_ASSET]
        attempts, accepted, blockers = [], None, []
        for candidate in _attempt_order(candidates, request.max_scene_attempts):
            when = candidate.datetime[:10]
            yield event("scene", "started", f"Reading 10 bands + scene classification from {candidate.platform or 'Sentinel-2'} "
                        f"image of {when}", {"scene": candidate.summary()})
            try:
                arrays, calibration = services.reader.read(candidate.item, grid, asset_keys)
            except Exception as exc:
                detail = _short_error(exc)
                log.warning("Reading %s failed: %s", candidate.summary().get("id"), detail)
                attempts.append({**candidate.summary(), "outcome": "read_error", "detail": detail})
                yield event("scene", "rejected", f"Could not read the image of {when}: {detail}",
                            {"scene": candidate.summary()})
                continue
            rho = {REFLECTANCE_ASSETS[key]: arrays[key] for key in REFLECTANCE_ASSETS}
            scene_images = {
                "true_color": render.true_color(rho["B4"], rho["B3"], rho["B2"]),
                "scene_classification": render.scl_map(arrays[SCL_ASSET]),
            }
            yield event("scene", "completed", f"Read a {grid.width}×{grid.height} pixel window (20 m) from the image of {when}",
                        {"scene": candidate.summary(), "previews": _previews(scene_images)})
            if body_future is not None:  # first successful read: the outline is needed now
                yield resolve_body()
            if not polygon_ready:
                polygon_ready = True
                if body is not None:
                    polygon_mask, shoreline_buffer, polygon_notes = rasterize_water_polygon(body.geometry, grid)
                    notes.extend(polygon_notes)
                    if polygon_mask.sum() == 0:
                        polygon_mask = None
            yield event("mask", "started", "Detecting clouds, projecting their shadows and isolating open water")
            properties = candidate.item.get("properties", {})
            zenith, azimuth = properties.get("s2:mean_solar_zenith"), properties.get("s2:mean_solar_azimuth")
            context_scl, margin = None, 0
            if zenith is not None and azimuth is not None:
                # Clouds up to one maximum shadow length outside the window can shade it.
                margin = min(MAX_SHADOW_MARGIN_PX, math.ceil(
                    SHADOW_HEIGHTS_M[1] * math.tan(math.radians(zenith)) / grid.resolution_m))
                try:
                    context, _ = services.reader.read(candidate.item, expand_grid(grid, margin), [SCL_ASSET])
                    context_scl = context[SCL_ASSET]
                except Exception:
                    margin = 0
                    notes.append("Could not read surrounding cloud context; shadows from clouds outside the window may be missed.")
            masks = classify_pixels(rho, arrays[SCL_ASSET], polygon_mask, grid.resolution_m,
                                    sun_zenith_deg=zenith, sun_azimuth_deg=azimuth,
                                    context_scl=context_scl, context_margin_px=margin)
            counts = masks.counts
            mask_images = {"pixel_classes": _pixel_class_png(masks)}
            ok = counts["valid_open_water"] >= request.min_valid_pixels and counts["clear_fraction"] >= request.min_clear_fraction
            attempts.append({**candidate.summary(), "outcome": "accepted" if ok else "too_cloudy", "pixels": counts})
            mask_data = {"scene": candidate.summary(), "pixels": counts, "previews": _previews(mask_images),
                         "sun": {"zenith_deg": zenith, "azimuth_deg": azimuth}}
            if ok:
                accepted = (candidate, rho, arrays[SCL_ASSET], masks, calibration, {**scene_images, **mask_images})
                yield event("mask", "completed", f"{counts['valid_open_water']:,} clear open-water pixels "
                            f"({counts['clear_fraction']:.0%} of the water area); {counts['cloud']:,} cloud and "
                            f"{counts['cloud_shadow']:,} shadow pixels removed", mask_data)
                break
            blocker = _main_blocker(counts)
            blockers.append(blocker)
            yield event("mask", "rejected", f"Only {counts['clear_fraction']:.0%} of the water is clear on {when} "
                        f"(mostly {BLOCKER_TEXT[blocker]}); trying an earlier image", mask_data)
        if accepted is None:
            yield event("mask", "failed", _no_image_message(blockers), {"attempts": attempts})
            return
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    candidate, rho, scl, masks, calibration, images = accepted
    notes.extend(masks.notes)
    valid = masks.valid_water
    n_valid = int(valid.sum())
    layers = [
        {"id": "true_color", "label": "True colour (B4, B3, B2)"},
        {"id": "scene_classification", "label": "Sentinel-2 scene classification (as delivered)",
         "legend": render.SCL_LEGEND},
        {"id": "pixel_classes", "label": "AquaDoc pixel classification", "legend": render.PIXEL_CLASS_LEGEND},
    ]
    model = services.model()

    # 4. Correct reflectance --------------------------------------------------------
    yield event("correct", "started", "Smoothing 3×3 neighbourhoods and correcting Sentinel-2 reflectance over water")
    # Average each band over valid 3x3 neighbours (60 m) before estimating: band-ratio
    # signals such as B5/B4 amplify sensor noise, and averaging 9 pixels cuts it ~3x.
    smoothed = {band: np.where(valid, local_mean(np.where(valid, rho[band], 0.0), valid, SMOOTHING_WINDOW_PX,
                                                 min_count=1), np.nan) for band in (*MODEL_BANDS, "B11")}
    # Sen2Cor is tuned for land; a residual correction learned from satellite/in-situ
    # match-ups (training/validate_satellite.py) is used for the targets where it helped.
    rrs_l2a = {band: smoothed[band][valid] / math.pi for band in (*MODEL_BANDS, "B11")}
    rrs_valid = model.correct_satellite_rrs(rrs_l2a)
    correction = [
        {"band": band, "center_nm": band_guide.BAND_CENTERS[band],
         "rrs_l2a": _round(float(np.median(rrs_l2a[band]))), "rrs_corrected": _round(float(np.median(rrs_valid[band])))}
        for band in MODEL_BANDS
    ]
    using = [t for t in model.targets if model.satellite_input(t) == "corrected"]
    yield event("correct", "completed",
                (f"Removed Sen2Cor's residual offset over water (median B4 Rrs {correction[2]['rrs_l2a']} → "
                 f"{correction[2]['rrs_corrected']} sr⁻¹); used for {len(using)} of {len(model.targets)} parameters")
                if model.satellite_correction else "No satellite correction configured; using Sentinel-2 reflectance as delivered",
                {"reflectance": correction, "targets_using_correction": using})

    # 5. Estimate -----------------------------------------------------------------
    yield event("estimate", "started", f"Running the water-quality model on {n_valid:,} pixels")
    predictions = model.predict_satellite(rrs_l2a)
    estimates, pixel_maps = {}, {}
    for target, prediction in predictions.items():
        spec = model.metadata["targets"][target]
        in_domain_fraction = float(prediction.in_domain.mean()) if prediction.in_domain.size else 0.0
        median_low = float(np.nanmedian(prediction.low))
        median_high = float(np.nanmedian(prediction.high))
        median = _stats(prediction.median)
        factor = math.sqrt(median_high / median_low) if median_low > 0 else None
        full = np.full(valid.shape, np.nan)
        full[valid] = prediction.median
        pixel_maps[target] = full
        estimates[target] = {
            **{k: v for k, v in band_guide.PARAMETERS[target].items() if k in ("label", "what", "bands")},
            "unit": spec["unit"],
            "kind": "model estimate",
            "value": median["median"],
            "spatial_p10": median["p10"],
            "spatial_p90": median["p90"],
            "interval": {"level": prediction.interval_level, "low": _round(median_low), "high": _round(median_high)},
            "in_training_domain_pct": round(100 * in_domain_fraction, 1),
            "confidence": _confidence(in_domain_fraction, factor, n_valid, satellite_mdsa(model, target)),
            "model_test_mdsa_pct": spec["evaluation"]["model"]["mdsa_pct"],
            "satellite_validated_mdsa_pct": satellite_mdsa(model, target),
            "reflectance_input": model.satellite_input(target),
            "interval_basis": spec.get("interval_basis", "simulated in-situ spectra (not validated on satellite data)"),
        }
    if "chlorophyll_a" in estimates:
        estimates["chlorophyll_a"]["trophic_state"] = _trophic_state(estimates["chlorophyll_a"]["value"])
    estimate_images = {}
    for target, values in pixel_maps.items():
        finite = values[np.isfinite(values)]
        vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0.1, 1.0))
        vmin, vmax = max(float(vmin), 1e-3), max(float(vmax), float(vmin) * 1.5)
        colormap = band_guide.PARAMETERS[target]["colormap"]
        estimate_images[target] = render.colorize(values, valid, vmin, vmax, colormap, log=True)
        layers.append({"id": target, "label": estimates[target]["label"], "unit": estimates[target]["unit"],
                       "colormap": colormap, "colors": render.ramp_hex(colormap), "scale": "log10",
                       "vmin": _round(vmin), "vmax": _round(vmax)})
    images.update(estimate_images)

    algorithms = {
        "chlorophyll_a_ndci_mishra2012": _stats(wq_algorithms.chlorophyll_ndci_mishra(rrs_valid["B4"], rrs_valid["B5"])),
        "tss_nechad2010": _stats(wq_algorithms.tss_nechad_665(rrs_valid["B4"])),
        "turbidity_dogliotti2015": _stats(wq_algorithms.turbidity_dogliotti(rrs_valid["B4"], rho["B8A"][valid] / math.pi)),
    }

    fdi = wq_algorithms.floating_debris_index(rho["B6"], rho["B8"], rho["B11"])
    fdi_water = fdi[valid]
    fdi_median = float(np.median(fdi_water))
    fdi_mad = float(np.median(np.abs(fdi_water - fdi_median))) * 1.4826  # robust standard deviation
    fdi_cut = fdi_median + max(DEBRIS_FDI_ANOMALY, 4 * fdi_mad)
    debris_like = valid & (fdi > fdi_cut)
    surface = n_valid + int(masks.floating.sum())
    floating = {
        **{k: v for k, v in band_guide.PARAMETERS["floating_material"].items() if k in ("label", "what", "bands")},
        "kind": "spectral index",
        "vegetation_or_scum_pct": round(100 * masks.floating.sum() / surface, 2) if surface else 0.0,
        "debris_like_pct": round(100 * debris_like.sum() / surface, 2) if surface else 0.0,
        "fdi_water_median": _round(fdi_median),
        "fdi_anomaly_threshold": _round(fdi_cut),
    }
    yield event("estimate", "completed", "Estimated " + ", ".join(
        f"{estimates[t]['label']} ≈ {estimates[t]['value']} {estimates[t]['unit']}" for t in estimates),
        {"estimates": {t: {"value": e["value"], "unit": e["unit"], "confidence": e["confidence"]}
                       for t, e in estimates.items()},
         "previews": _previews(estimate_images)})

    # 6. Explain: spectrum, band sensitivity and index maps --------------------------
    yield event("explain", "started", "Measuring which bands drive each estimate at this location")
    spectrum = []
    for band, meta in ((b["band"], b) for b in band_guide.BANDS):
        values = rho[band][valid]
        p25, p50, p75 = np.percentile(values, [25, 50, 75])
        spectrum.append({"band": band, "name": meta["name"], "center_nm": meta["center_nm"],
                         "reflectance_median": _round(p50), "reflectance_p25": _round(p25), "reflectance_p75": _round(p75)})
    rrs_median = {band: float(np.median(rrs_l2a[band])) for band in (*MODEL_BANDS, "B11")}
    sensitivity = band_sensitivity(model, rrs_median)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndci_map = (rho["B5"] - rho["B4"]) / (rho["B5"] + rho["B4"])
    indices = {
        "ndci_median": _stats(ndci_map[valid])["median"],
        "green_blue_ratio_median": _stats(rho["B3"][valid] / rho["B2"][valid])["median"],
        "red_reflectance_median": _stats(rho["B4"][valid])["median"],
    }
    explain_images = {"ndci": render.colorize(np.where(valid, ndci_map, np.nan), valid, -0.2, 0.3, "algae")}
    layers.append({"id": "ndci", "label": "NDCI (B5 vs B4 red-edge index)", "colormap": "algae",
                   "colors": render.ramp_hex("algae"), "scale": "linear", "vmin": -0.2, "vmax": 0.3})
    fdi_map = np.where(valid | masks.floating, fdi, np.nan)
    fdi_lo, fdi_hi = fdi_median, fdi_cut + 0.03
    explain_images["floating_debris_index"] = render.colorize(fdi_map, valid | masks.floating, fdi_lo, fdi_hi, "debris")
    layers.append({"id": "floating_debris_index", "label": "Floating Debris Index (relative to this water)",
                   "colormap": "debris", "colors": render.ramp_hex("debris"), "scale": "linear",
                   "vmin": _round(fdi_lo), "vmax": _round(fdi_hi)})
    images.update(explain_images)
    yield event("explain", "completed", "Band sensitivity and index maps computed",
                {"top_bands": {t: s[0]["band"] for t, s in sensitivity.items()}, "previews": _previews(explain_images)})

    # 7. Result ---------------------------------------------------------------------
    result = {
        "pipeline_version": PIPELINE_VERSION,
        "request": {"latitude": request.latitude, "longitude": request.longitude,
                    "start_date": start_date, "end_date": end_date, "max_size_m": request.max_size_m},
        "water_body": ({**body.summary(), "geometry": _display_geometry(body, grid)} if body else None),
        "grid": {**grid.to_dict(), "extent": extent_note, "shoreline_buffer_m": shoreline_buffer},
        "scene": {**candidate.summary(),
                  "processing_baseline": candidate.item.get("properties", {}).get("s2:processing_baseline"),
                  "mgrs_tile": candidate.item.get("properties", {}).get("s2:mgrs_tile"),
                  "calibration": calibration.get("B04")},
        "attempts": attempts,
        "pixels": masks.counts,
        "estimates": estimates,
        "floating_material": floating,
        "published_algorithms": algorithms,
        "indices": indices,
        "spectrum": spectrum,
        "band_sensitivity": sensitivity,
        "reflectance_correction": {"bands": correction, "targets_using_correction": using},
        "layers": layers,
        "not_detectable": band_guide.NOT_DETECTABLE,
        "model": {"version": model.metadata.get("model_version"), "feature_version": model.metadata.get("feature_version"),
                  "training_data": model.metadata.get("training_data", {}).get("source"),
                  "satellite_correction": (model.satellite_correction or {}).get("version")},
        "notes": notes,
        "disclaimer": "Satellite model estimates for optically active constituents only; not a substitute for "
                      "laboratory testing or a drinking-water safety assessment.",
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }
    done = event("complete", "completed", "Analysis complete", result, images)
    result["timeline"] = timeline  # includes this final event
    yield done

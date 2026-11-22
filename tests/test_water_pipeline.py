"""Offline tests for the Phase 2 water pipeline: grid, masks, OSM parsing, search, render, end-to-end."""

import struct
import unittest
import zlib

import numpy as np
from shapely.geometry import Polygon, box

from aquadoc_core import render
from aquadoc_core.s2_reader import REFLECTANCE_ASSETS, SCL_ASSET, expand_grid, make_grid, reflectance_calibration
from aquadoc_core.scene_search import SceneCandidate, rank_candidates, search_scenes
from aquadoc_core.water_masks import classify_pixels, project_shadow_zone, rasterize_water_polygon, shift
from aquadoc_core.water_pipeline import Services, WaterAnalysisRequest, band_sensitivity, run_water_analysis
from aquadoc_core.waterbody import WaterBody, _parse, find_water_body, lookup_water_body
from aquadoc_core.wq_model import WaterQualityModel

# Typical turbid-lake surface reflectance (rho) per band.
WATER_RHO = {"B2": 0.03, "B3": 0.05, "B4": 0.04, "B5": 0.045, "B6": 0.02, "B7": 0.015,
             "B8": 0.012, "B8A": 0.011, "B11": 0.004, "B12": 0.002}


def decode_png(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    idat = b""
    position = 8
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        if kind == b"IDAT":
            idat += data[position + 8:position + 8 + length]
        position += 12 + length
    raw = zlib.decompress(idat)
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, 1 + width * 4)[:, 1:]
    return rows.reshape(height, width, 4)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def synthetic_scene(shape=(40, 40), cloud_block=None):
    arrays = {asset: np.full(shape, WATER_RHO[band], dtype=np.float32) for asset, band in REFLECTANCE_ASSETS.items()}
    scl = np.full(shape, 6.0, dtype=np.float32)
    if cloud_block is not None:
        scl[cloud_block] = 9
        for asset in REFLECTANCE_ASSETS:
            arrays[asset][cloud_block] = 0.4
    arrays[SCL_ASSET] = scl
    return arrays


def item(item_id="S2B_TEST", when="2026-11-18T05:00:00Z"):
    return {"id": item_id, "properties": {"datetime": when, "eo:cloud_cover": 5.0, "platform": "Sentinel-2B",
                                          "s2:mean_solar_zenith": 40.0, "s2:mean_solar_azimuth": 150.0,
                                          "s2:processing_baseline": "05.11"},
            "assets": {}}


class GridAndCalibrationTests(unittest.TestCase):
    def test_grid_is_metric_and_expandable(self):
        grid = make_grid(17.42, 78.47, 1000)
        self.assertEqual(grid.crs, "EPSG:32644")
        self.assertEqual((grid.width, grid.height), (50, 50))
        wider = expand_grid(grid, 10)
        self.assertEqual(wider.width, 70)
        self.assertAlmostEqual(wider.transform[2], grid.transform[2] - 200)
        self.assertAlmostEqual(wider.transform[5], grid.transform[5] + 200)
        corners = grid.corners_wgs84()
        self.assertEqual(len(corners), 4)
        self.assertTrue(all(abs(lat - 17.42) < 0.01 for _, lat in corners))

    def test_calibration_prefers_stac_metadata_then_baseline_rule(self):
        stac = {"assets": {"B04": {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}}}
        self.assertEqual(reflectance_calibration(stac, "B04")[:2], (0.0001, -0.1))
        new = {"assets": {"B04": {}}, "properties": {"s2:processing_baseline": "05.10"}}
        old = {"assets": {"B04": {}}, "properties": {"s2:processing_baseline": "03.01"}}
        self.assertEqual(reflectance_calibration(new, "B04")[1], -0.1)
        self.assertEqual(reflectance_calibration(old, "B04")[1], 0.0)


class MaskTests(unittest.TestCase):
    def test_shadow_falls_away_from_sun(self):
        cloud = np.zeros((60, 60), dtype=bool)
        cloud[40:44, 28:32] = True
        # Sun due south (azimuth 180): shadow lies north, i.e. towards smaller row numbers.
        zone = project_shadow_zone(cloud, 45.0, 180.0, 20.0, heights_m=(200.0, 400.0))
        rows = np.where(zone.any(axis=1))[0]
        self.assertTrue(rows.max() < 40 and rows.min() >= 40 - 21)
        self.assertFalse((zone & cloud).any())

    def test_shift_fills_with_false(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        moved = shift(mask, -2, 1)
        self.assertTrue(moved[0, 3])
        self.assertEqual(moved.sum(), 1)

    def test_classification_counts_clouds_shadow_floating_and_glint(self):
        arrays = synthetic_scene((60, 60), cloud_block=(slice(40, 46), slice(20, 30)))
        rho = {band: arrays[asset].astype(float) for asset, band in REFLECTANCE_ASSETS.items()}
        rho_shadow_rows = slice(25, 31)  # ~ north of the cloud for a southern sun
        for band in ("B2", "B3", "B4"):
            rho[band][rho_shadow_rows, 20:30] *= 0.5
        rho["B8"][5:8, 50:53] = 0.2   # floating vegetation: NIR >> red
        rho["B11"][5:8, 5:8] = 0.08   # glint brightens every band, SWIR included
        rho["B3"][5:8, 5:8] = 0.12
        masks = classify_pixels(rho, arrays[SCL_ASSET], None, 20.0, sun_zenith_deg=45.0, sun_azimuth_deg=180.0)
        counts = masks.counts
        self.assertEqual(counts["cloud"], 60)
        self.assertGreater(counts["cloud_shadow"], 30)
        self.assertEqual(counts["floating_vegetation_or_scum"], 9)
        self.assertEqual(counts["glint_or_mixed"], 9)
        self.assertFalse((masks.valid_water & (arrays[SCL_ASSET] == 9)).any())
        self.assertGreater(counts["valid_open_water"], 2000)

    def test_polygon_rasterization_uses_shoreline_buffer(self):
        grid = make_grid(17.42, 78.47, 1000)
        lake = box(78.468, 17.418, 78.472, 17.422)  # ~420 m square
        mask, buffer_m, _ = rasterize_water_polygon(lake, grid)
        self.assertEqual(buffer_m, 100.0)       # ~220 m interior square still has >= 100 pixels
        self.assertTrue(100 <= mask.sum() < 200)
        tiny = box(78.4695, 17.4195, 78.4705, 17.4205)  # ~105 m: needs a narrower buffer
        mask, buffer_m, notes = rasterize_water_polygon(tiny, grid)
        self.assertLess(buffer_m, 60.0)
        self.assertIn("adjacency", notes[0])


class OsmTests(unittest.TestCase):
    def test_relation_with_inner_ring_and_smallest_containing(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        island = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
        bay = [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]

        def nodes(coords):
            return [{"lon": x, "lat": y} for x, y in coords]
        payload = {"elements": [
            {"type": "relation", "id": 1, "tags": {"name": "Big Lake", "natural": "water"},
             "members": [{"type": "way", "role": "outer", "geometry": nodes(outer[:3])},
                         {"type": "way", "role": "outer", "geometry": nodes(outer[2:])},
                         {"type": "way", "role": "inner", "geometry": nodes(island)}]},
            {"type": "way", "id": 2, "tags": {"name": "Bay", "water": "bay"}, "geometry": nodes(bay)},
        ]}
        bodies = _parse(payload, __import__("shapely.geometry", fromlist=["Point"]).Point(1, 1))
        big = next(b for b in bodies if b.name == "Big Lake")
        self.assertAlmostEqual(big.geometry.area, 96.0)
        calls = []

        def post(url, **kwargs):
            calls.append(url)
            return FakeResponse(payload)
        found = find_water_body(1, 1, post=post)
        self.assertEqual(found.name, "Bay")  # most specific polygon containing the point
        self.assertEqual(len(calls), 1)

    def test_all_overpass_instances_failing_raises(self):
        import requests

        def post(url, **kwargs):
            raise requests.ConnectionError("down")
        with self.assertRaises(RuntimeError):
            find_water_body(1, 1, post=post)

    def test_nominatim_lookup_builds_polygon(self):
        payload = [{"name": "Pond", "type": "pond", "geojson": {
            "type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}]
        body = lookup_water_body.__wrapped__("way/5", 0.5, 0.5, get=lambda *a, **k: FakeResponse(payload))
        self.assertEqual((body.osm_type, body.osm_id, body.name), ("way", 5, "Pond"))
        self.assertTrue(body.contains_point)
        with self.assertRaises(ValueError):
            lookup_water_body.__wrapped__("lake/5", 0, 0)


class SceneSearchTests(unittest.TestCase):
    def test_duplicates_and_poor_coverage_removed(self):
        def candidate(i, when, coverage, cloud=10.0):
            return SceneCandidate(item={}, item_id=i, datetime=when, cloud_cover=cloud, platform="S2",
                                  aoi_coverage=coverage)
        ranked = rank_candidates([
            candidate("a", "2026-11-10T05:00:00Z", 0.6), candidate("b", "2026-11-10T05:00:01Z", 1.0),
            candidate("c", "2026-11-15T05:00:00Z", 0.2), candidate("d", "2026-11-05T05:00:00Z", 1.0),
        ])
        self.assertEqual([c.item_id for c in ranked], ["b", "d"])

    def test_search_parses_features(self):
        feature = {"id": "x", "geometry": box(0, 0, 2, 2).__geo_interface__,
                   "properties": {"datetime": "2026-11-01T00:00:00Z", "eo:cloud_cover": 3, "platform": "S2A"}}
        result = search_scenes((0.5, 0.5, 1, 1), "2026-10-01", "2026-11-01",
                               post=lambda *a, **k: FakeResponse({"features": [feature]}))
        self.assertEqual(result[0].aoi_coverage, 1.0)


class RenderTests(unittest.TestCase):
    def test_png_round_trip_and_transparency(self):
        values = np.array([[1.0, 10.0], [100.0, np.nan]])
        mask = np.array([[True, True], [False, True]])
        image = decode_png(render.colorize(values, mask, 1.0, 100.0, "algae", log=True))
        self.assertEqual(image.shape, (2, 2, 4))
        self.assertEqual(image[0, 0, 3], 255)
        self.assertEqual(image[1, 0, 3], 0)   # masked
        self.assertEqual(image[1, 1, 3], 0)   # NaN
        self.assertEqual(tuple(image[0, 0, :3]), render.COLORMAPS["algae"][0])


class FakeReader:
    """Serves synthetic scenes; the first scene can be fully clouded to exercise retries."""

    def __init__(self, cloudy_ids=()):
        self.cloudy_ids = set(cloudy_ids)

    def read(self, item, grid, keys):
        shape = (grid.height, grid.width)
        cloud = (slice(None), slice(None)) if item["id"] in self.cloudy_ids else None
        arrays = synthetic_scene(shape, cloud_block=cloud)
        return {k: arrays[k] for k in keys}, {"B04": {"scale": 0.0001, "offset": -0.1, "basis": "test"}}


class PipelineTests(unittest.TestCase):
    def services(self, candidates, cloudy_ids=(), body=None):
        return Services(
            find_water_body=lambda lat, lon: body,
            lookup_water_body=lambda ref, lat, lon: body,
            search_scenes=lambda *a, **k: candidates,
            reader=FakeReader(cloudy_ids),
        )

    def candidates(self):
        return [SceneCandidate(item=item("cloudy", "2026-11-18T05:00:00Z"), item_id="cloudy",
                               datetime="2026-11-18T05:00:00Z", cloud_cover=40, platform="S2B", aoi_coverage=1),
                SceneCandidate(item=item("clear", "2026-11-13T05:00:00Z"), item_id="clear",
                               datetime="2026-11-13T05:00:00Z", cloud_cover=2, platform="S2A", aoi_coverage=1)]

    def test_end_to_end_retries_cloudy_scene_and_returns_result(self):
        lake = Polygon([(78.465, 17.415), (78.48, 17.415), (78.48, 17.43), (78.465, 17.43)])
        body = WaterBody("Test Lake", "relation", 1, "lake", lake, True)
        request = WaterAnalysisRequest(latitude=17.42, longitude=78.47, end_date="2026-11-20", max_size_m=1000)
        events = list(run_water_analysis(request, self.services(self.candidates(), {"cloudy"}, body)))
        stages = [(e.stage, e.status) for e in events]
        self.assertIn(("mask", "rejected"), stages)
        for stage in ("locate", "search", "scene", "mask", "correct", "estimate", "explain"):
            self.assertIn((stage, "completed"), stages)
        self.assertEqual(stages[-1], ("complete", "completed"))
        # Streaming previews arrive as data URLs at the stage that produced them.
        scene_done = next(e for e in events if (e.stage, e.status) == ("scene", "completed"))
        self.assertTrue(scene_done.data["previews"]["true_color"].startswith("data:image/png;base64,"))
        estimate_done = next(e for e in events if (e.stage, e.status) == ("estimate", "completed"))
        self.assertIn("chlorophyll_a", estimate_done.data["previews"])
        result, images = events[-1].data, events[-1].images
        self.assertEqual(result["scene"]["id"], "clear")
        self.assertEqual(result["water_body"]["name"], "Test Lake")
        self.assertGreater(result["pixels"]["valid_open_water"], 100)
        for target in ("chlorophyll_a", "tss", "cdom_a440", "turbidity", "secchi_depth"):
            estimate = result["estimates"][target]
            self.assertGreater(estimate["value"], 0)
            self.assertLessEqual(estimate["interval"]["low"], estimate["value"])
            self.assertIn(estimate["confidence"], ("low", "medium"))
        self.assertEqual(len(result["spectrum"]), 10)
        self.assertEqual(set(result["band_sensitivity"]["chlorophyll_a"][0]), {"band", "elasticity", "share_pct"})
        self.assertTrue(result["not_detectable"])
        # The stored timeline replays the pipeline without embedded images or the result itself.
        timeline = result["timeline"]
        self.assertEqual(timeline[-1]["stage"], "complete")
        self.assertEqual(timeline[-1]["data"], {})
        scene_entry = next(t for t in timeline if (t["stage"], t["status"]) == ("scene", "completed"))
        self.assertIn("true_color", scene_entry["data"]["preview_layers"])
        self.assertNotIn("previews", scene_entry["data"])
        self.assertEqual(len(result["reflectance_correction"]["bands"]), 6)
        layer_ids = {layer["id"] for layer in result["layers"]}
        self.assertTrue({"true_color", "scene_classification", "pixel_classes", "ndci"} <= layer_ids)
        self.assertEqual(layer_ids, set(images))
        for name, png in images.items():
            self.assertEqual(png[:4], b"\x89PNG", name)
        import json
        json.dumps(result)  # must be JSON-serialisable for the API

    def test_osm_ref_path_and_no_polygon_fallback(self):
        request = WaterAnalysisRequest(latitude=17.42, longitude=78.47, end_date="2026-11-20",
                                       max_size_m=1000, osm_ref="way/9")
        events = list(run_water_analysis(request, self.services(self.candidates()[1:])))
        self.assertEqual(events[0].message, "Fetching the outline of OpenStreetMap way/9")
        self.assertEqual(events[-1].stage, "complete")
        self.assertIn("No mapped water polygon", " ".join(events[-1].data["notes"]))

    def test_failures_are_reported_as_events(self):
        request = WaterAnalysisRequest(latitude=17.42, longitude=78.47, end_date="2026-11-20", max_size_m=1000)
        events = list(run_water_analysis(request, self.services([])))
        self.assertEqual((events[-1].stage, events[-1].status), ("search", "failed"))
        events = list(run_water_analysis(request, self.services(self.candidates()[:1], {"cloudy"})))
        self.assertEqual((events[-1].stage, events[-1].status), ("mask", "failed"))
        bad = WaterAnalysisRequest(latitude=95, longitude=0)
        self.assertEqual(list(run_water_analysis(bad, self.services([])))[0].status, "failed")

    def test_band_sensitivity_highlights_red_edge_for_chlorophyll(self):
        model = WaterQualityModel.load()
        rrs = {band: WATER_RHO[band] / np.pi for band in ("B2", "B3", "B4", "B5", "B6", "B7", "B11")}
        sensitivity = band_sensitivity(model, rrs)
        top_two = {entry["band"] for entry in sensitivity["chlorophyll_a"][:2]}
        self.assertTrue(top_two & {"B4", "B5"})
        self.assertAlmostEqual(sum(e["share_pct"] for e in sensitivity["tss"]), 100, delta=0.5)


if __name__ == "__main__":
    unittest.main()

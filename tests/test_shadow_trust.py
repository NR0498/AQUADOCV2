"""Sentinel-2's cloud-shadow class is only trusted where a real cloud could cast the shadow."""

import numpy as np

from aquadoc_core.water_masks import classify_pixels

WATER = {"B2": 0.03, "B3": 0.05, "B4": 0.04, "B8": 0.012, "B11": 0.004}


def scene(size=60):
    rho = {band: np.full((size, size), value, dtype=np.float32) for band, value in WATER.items()}
    scl = np.full((size, size), 6.0, dtype=np.float32)  # SCL 6 = water
    return rho, scl


def test_shadow_label_ignored_on_cloud_free_image():
    rho, scl = scene()
    scl[20:40, 20:40] = 3  # Sen2Cor calls dark water "cloud shadow" with no cloud anywhere
    masks = classify_pixels(rho, scl, np.ones_like(scl, dtype=bool), 20.0, sun_zenith_deg=40.0, sun_azimuth_deg=150.0)
    assert masks.counts["cloud_shadow"] == 0
    assert masks.counts["valid_open_water"] == 60 * 60
    assert any("no cloud that could cast" in note for note in masks.notes)


def test_shadow_label_kept_next_to_real_clouds():
    rho, scl = scene()
    scl[5:12, 25:32] = 9  # a cloud ...
    scl[13:22, 20:40] = 3  # ... and the SCL shadow around it
    masks = classify_pixels(rho, scl, np.ones_like(scl, dtype=bool), 20.0)  # no sun angles: SCL's call stands
    assert masks.counts["cloud_shadow"] == 9 * 20
    assert not masks.valid_water[13:22, 20:40].any()

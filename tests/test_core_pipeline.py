import unittest

import numpy as np

from aquadoc_core import MultispectralScene, analyze_multispectral_scene


class CorePipelineTests(unittest.TestCase):
    def test_no_water_pixels_does_not_produce_a_low_risk_result(self):
        scene = MultispectralScene(
            latitude=12, longitude=77,
            bands={name: np.zeros((4, 4)) for name in ("blue", "green", "red", "nir")},
        )
        with self.assertRaisesRegex(ValueError, "no usable water pixels"):
            analyze_multispectral_scene(scene)

    def test_nonfinite_band_pixels_are_excluded_from_every_summary(self):
        bands = {
            name: np.full((2, 2), value)
            for name, value in {"blue": 0.04, "green": 0.08, "red": 0.03, "nir": 0.02}.items()
        }
        bands["red"][0, 0] = np.nan
        result = analyze_multispectral_scene(MultispectralScene(latitude=12, longitude=77, bands=bands))
        self.assertTrue(all(stats.valid_fraction == 0.75 for stats in result.indices.values()))

    def test_empty_or_entirely_invalid_scene_is_rejected(self):
        for values in (np.empty((0, 0)), np.full((2, 2), np.nan)):
            with self.subTest(shape=values.shape):
                scene = MultispectralScene(
                    latitude=12, longitude=77,
                    bands={name: values for name in ("blue", "green", "red", "nir")},
                )
                with self.assertRaises(ValueError):
                    analyze_multispectral_scene(scene)

    def test_clean_water_scene_returns_low_risk(self):
        shape = (16, 16)
        scene = MultispectralScene(
            latitude=12.0,
            longitude=77.0,
            source="unit-test",
            bands={
                "blue": np.full(shape, 0.04),
                "green": np.full(shape, 0.08),
                "red": np.full(shape, 0.03),
                "nir": np.full(shape, 0.02),
                "red_edge": np.full(shape, 0.035),
                "swir1": np.full(shape, 0.01),
            },
        )

        result = analyze_multispectral_scene(scene)

        self.assertEqual(result.safety_label, "likely_safe")
        self.assertLess(result.risk_score, 0.4)
        self.assertGreater(result.indices["ndwi"].median, 0)

    def test_bloom_like_scene_increases_risk(self):
        shape = (16, 16)
        scene = MultispectralScene(
            latitude=12.0,
            longitude=77.0,
            source="unit-test",
            bands={
                "blue": np.full(shape, 0.04),
                "green": np.full(shape, 0.06),
                "red": np.full(shape, 0.03),
                "nir": np.full(shape, 0.02),
                "red_edge": np.full(shape, 0.08),
                "swir1": np.full(shape, 0.015),
            },
        )

        result = analyze_multispectral_scene(scene)

        self.assertIn(result.safety_label, {"caution", "unsafe"})
        self.assertIn("algae_bloom", result.signals)


if __name__ == "__main__":
    unittest.main()

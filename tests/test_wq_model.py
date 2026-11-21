"""Tests for water-quality features, published algorithms and NumPy inference."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aquadoc_core import wq_algorithms
from aquadoc_core.wq_features import FEATURE_NAMES, FEATURE_VERSION, MODEL_BANDS, build_features
from aquadoc_core.wq_model import MODELS_DIR, DEFAULT_MODEL_NAME, WaterQualityModel


def sample_rrs(n=4):
    base = {"B2": 0.004, "B3": 0.008, "B4": 0.006, "B5": 0.009, "B6": 0.004, "B7": 0.003}
    return {band: np.full(n, value) for band, value in base.items()}


def write_tiny_model(directory: Path, deployed=True):
    """A one-member linear 'network' whose output equals the first standardized feature."""
    n_features = len(FEATURE_NAMES)
    w0 = np.zeros((n_features, 1)); w0[0, 0] = 1.0
    arrays = {"chlorophyll_a/m0/W0": w0, "chlorophyll_a/m0/b0": np.zeros(1)}
    np.savez(directory / "tiny.npz", **arrays)
    spec = {
        "unit": "ug/L", "deployed": deployed, "n_members": 1, "n_layers": 1,
        "x_mean": [0.0] * n_features, "x_std": [1.0] * n_features,
        "y_mean": 0.0, "y_std": 1.0, "conformal_q": 1.0, "sigma_floor": 0.1,
        "interval_level": 0.8,
        "feature_p01": [-10.0] * n_features, "feature_p99": [10.0] * n_features,
    }
    metadata = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                "targets": {"chlorophyll_a": spec}}
    (directory / "tiny.json").write_text(json.dumps(metadata), encoding="utf-8")


class FeatureTests(unittest.TestCase):
    def test_shape_and_names(self):
        features = build_features(sample_rrs(5))
        self.assertEqual(features.shape, (5, len(FEATURE_NAMES)))
        self.assertAlmostEqual(features[0, 0], np.log10(0.004))

    def test_ndci_feature_matches_definition(self):
        features = build_features(sample_rrs(1))
        ndci = features[0, FEATURE_NAMES.index("nd_B5_B4")]
        self.assertAlmostEqual(ndci, (0.009 - 0.006) / (0.009 + 0.006))

    def test_negative_reflectance_is_floored_and_nonfinite_propagates(self):
        rrs = sample_rrs(3)
        rrs["B2"] = np.array([-0.01, np.nan, 0.004])
        features = build_features(rrs)
        self.assertAlmostEqual(features[0, 0], -5.0)
        self.assertTrue(np.isnan(features[1]).all())
        self.assertTrue(np.isfinite(features[2]).all())

    def test_missing_band_rejected(self):
        rrs = sample_rrs()
        del rrs["B5"]
        with self.assertRaises(ValueError):
            build_features(rrs)


class AlgorithmTests(unittest.TestCase):
    def test_mishra_ndci_zero_gives_intercept(self):
        value = wq_algorithms.chlorophyll_ndci_mishra(np.array([0.01]), np.array([0.01]))
        self.assertAlmostEqual(value[0], 14.039)

    def test_nechad_monotonic_and_saturates(self):
        low, high = wq_algorithms.tss_nechad_665(np.array([0.002, 0.02]))
        self.assertLess(low, high)
        self.assertTrue(np.isnan(wq_algorithms.tss_nechad_665(np.array([0.06]))[0]))

    def test_dogliotti_blends_to_nir_when_turbid(self):
        red_only = wq_algorithms.turbidity_dogliotti(np.array([0.005]), np.array([0.0005]))
        expected_red = 228.1 * (np.pi * 0.005) / (1 - np.pi * 0.005 / 0.1641)
        self.assertAlmostEqual(red_only[0], expected_red)
        turbid = wq_algorithms.turbidity_dogliotti(np.array([0.03]), np.array([0.01]))
        rho_nir = np.pi * 0.01
        self.assertAlmostEqual(turbid[0], 3078.9 * rho_nir / (1 - rho_nir / 0.2112))


class InferenceTests(unittest.TestCase):
    def test_tiny_model_round_trip_and_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            write_tiny_model(Path(directory))
            model = WaterQualityModel.load("tiny", directory)
            rrs = {band: values.reshape(2, 2) for band, values in sample_rrs(4).items()}
            prediction = model.predict(rrs)["chlorophyll_a"]
        expected = 10 ** np.log10(0.004)
        self.assertEqual(prediction.median.shape, (2, 2))
        np.testing.assert_allclose(prediction.median, expected)
        np.testing.assert_allclose(prediction.low, expected / 10 ** 0.1)
        np.testing.assert_allclose(prediction.high, expected * 10 ** 0.1)
        self.assertTrue(prediction.in_domain.all())

    def test_undeployed_targets_are_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            write_tiny_model(Path(directory), deployed=False)
            self.assertEqual(WaterQualityModel.load("tiny", directory).targets, ())

    def test_invalid_pixels_return_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            write_tiny_model(Path(directory))
            model = WaterQualityModel.load("tiny", directory)
            rrs = sample_rrs(2)
            rrs["B4"] = np.array([np.nan, 0.006])
            prediction = model.predict(rrs)["chlorophyll_a"]
        self.assertTrue(np.isnan(prediction.median[0]))
        self.assertFalse(prediction.in_domain[0])
        self.assertTrue(np.isfinite(prediction.median[1]))

    @unittest.skipUnless((MODELS_DIR / f"{DEFAULT_MODEL_NAME}.npz").exists(), "trained model not present")
    def test_shipped_model_predicts_plausible_values(self):
        model = WaterQualityModel.load()
        self.assertIn("chlorophyll_a", model.targets)
        predictions = model.predict(sample_rrs(3))
        for prediction in predictions.values():
            self.assertTrue(np.all(np.isfinite(prediction.median)))
            self.assertTrue(np.all(prediction.low <= prediction.median))
            self.assertTrue(np.all(prediction.median <= prediction.high))


class SatelliteCorrectionTests(unittest.TestCase):
    @unittest.skipUnless((MODELS_DIR / f"{DEFAULT_MODEL_NAME}.json").exists(), "trained model not present")
    def test_per_target_input_selection(self):
        model = WaterQualityModel.load()
        if model.satellite_correction is None:
            self.skipTest("no satellite correction in this model")
        rrs = {**sample_rrs(2), "B11": np.full(2, 0.002)}
        raw, chosen = model.predict(rrs), model.predict_satellite(rrs)
        corrected = model.predict(model.correct_satellite_rrs(rrs))
        for target in model.targets:
            expected = corrected if model.satellite_input(target) == "corrected" else raw
            np.testing.assert_allclose(chosen[target].median, expected[target].median)
        # The correction removes a SWIR-correlated offset: higher B11 means lower corrected Rrs.
        brighter_swir = model.correct_satellite_rrs({**rrs, "B11": np.full(2, 0.004)})
        self.assertTrue(np.all(brighter_swir["B4"] < model.correct_satellite_rrs(rrs)["B4"]))


if __name__ == "__main__":
    unittest.main()

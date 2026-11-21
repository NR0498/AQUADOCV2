"""Offline tests for SRF convolution, metrics and sklearn->NumPy export parity.

Run: python -m unittest discover -s training/tests -t .   (needs the [training] extra)
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aquadoc_core.wq_features import MODEL_BANDS
from aquadoc_core.wq_model import WaterQualityModel
from training.gloria import spatial_group
from training.srf import convolve, parse_acolite_rsr
from training.train import TrainConfig, log_metrics, train_target

RSR_TEXT = """# Sentinel-2A / MSI RSR
# S2A_MSI Band 2
489.0 0.5
490.0 1.0
491.0 0.5
# S2A_MSI Band 8A
864.0 1.0
865.0 1.0
"""


class SrfTests(unittest.TestCase):
    def test_parse_and_convolve_weighted_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S2A_MSI_V4.txt"
            path.write_text(RSR_TEXT, encoding="utf-8")
            srf = parse_acolite_rsr(path)
        self.assertEqual(set(srf), {"B2", "B8A"})
        wavelengths = np.arange(480.0, 501.0)
        spectra = np.vstack([np.full(wavelengths.size, 0.01), wavelengths * 1e-4])
        values = convolve(spectra, wavelengths, srf["B2"])
        self.assertAlmostEqual(values[0], 0.01)
        self.assertAlmostEqual(values[1], 490 * 1e-4)  # symmetric weights -> centre value

    def test_incomplete_spectrum_returns_nan(self):
        wavelengths = np.arange(480.0, 501.0)
        spectrum = np.full((1, wavelengths.size), 0.01)
        spectrum[0, wavelengths == 490.0] = np.nan  # the heaviest-weighted wavelength
        srf = (np.array([489.0, 490.0, 491.0]), np.array([0.5, 1.0, 0.5]))
        self.assertTrue(np.isnan(convolve(spectrum, wavelengths, srf)[0]))


class MetricTests(unittest.TestCase):
    def test_perfect_and_factor_two(self):
        truth = np.array([1.0, 10.0, 100.0])
        self.assertEqual(log_metrics(truth, truth)["mdsa_pct"], 0.0)
        doubled = log_metrics(truth, truth * 2)
        self.assertAlmostEqual(doubled["mdsa_pct"], 100.0)
        self.assertAlmostEqual(doubled["sspb_pct"], 100.0)

    def test_spatial_group_falls_back_to_dataset(self):
        groups = spatial_group(pd.Series([10.1, 10.2, np.nan]), pd.Series([20.1, 20.2, 5.0]),
                               pd.Series(["a", "a", "b"]))
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], "dataset_b")


class ExportParityTests(unittest.TestCase):
    def test_numpy_inference_matches_training_predictions(self):
        rng = np.random.default_rng(0)
        n = 400
        chl = 10 ** rng.uniform(-0.5, 2.0, n)
        table = pd.DataFrame({f"rrs_{band}": 0.002 + 0.001 * rng.random(n) for band in MODEL_BANDS})
        table["rrs_B5"] = table["rrs_B4"] * (1 + 0.004 * chl)  # synthetic red-edge signal
        table["rrs_B8A"] = 0.0005
        table["chlorophyll_a"] = chl
        table["group"] = [f"g{i % 40}" for i in range(n)]
        table["water_body_type"] = "lake"
        config = TrainConfig(hidden_layers=(8,), n_members=2, max_iter=50, augment_copies=1)
        spec, weights, (test, mu_test) = train_target("chlorophyll_a", table, config, rng)
        spec["deployed"] = True
        import json
        from aquadoc_core.wq_features import FEATURE_NAMES, FEATURE_VERSION
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "m.npz", **weights)
            metadata = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                        "targets": {"chlorophyll_a": spec}}
            (Path(directory) / "m.json").write_text(json.dumps(metadata), encoding="utf-8")
            model = WaterQualityModel.load("m", directory)
        rrs = {band: test[f"rrs_{band}"].to_numpy() for band in MODEL_BANDS}
        median = model.predict(rrs)["chlorophyll_a"].median
        np.testing.assert_allclose(np.log10(median), mu_test, atol=1e-4)  # float32 export


if __name__ == "__main__":
    unittest.main()

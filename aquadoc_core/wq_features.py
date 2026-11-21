"""Feature construction shared by offline training and deployed inference.

Inputs are Sentinel-2 *remote-sensing reflectance* (Rrs, units sr^-1) per band.
Sentinel-2 Level-2A surface reflectance (rho, unitless) converts as Rrs = rho / pi.

Training (``training/``) and inference (``wq_model.py``) both call
:func:`build_features`, so the model always sees identically built inputs.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

FEATURE_VERSION = "s2-rrs-b2b7-v1"

# Bands used by the trained model. B8/B8A (NIR) are excluded because about two thirds of
# the GLORIA training spectra stop before 850 nm; B1 is excluded for its 60 m resolution.
MODEL_BANDS: tuple[str, ...] = ("B2", "B3", "B4", "B5", "B6", "B7")

# Central wavelengths (nm), used for documentation and dashboard charts.
BAND_CENTERS_NM: dict[str, float] = {
    "B1": 443, "B2": 490, "B3": 560, "B4": 665, "B5": 705,
    "B6": 740, "B7": 783, "B8": 842, "B8A": 865, "B11": 1610, "B12": 2190,
}

RRS_FLOOR = 1e-5  # sr^-1; reflectance below this is treated as the floor before log.

FEATURE_NAMES: tuple[str, ...] = (
    *(f"log10_{band}" for band in MODEL_BANDS),
    "nd_B3_B2",  # green vs blue: CDOM / chlorophyll pigment absorption in blue
    "nd_B5_B4",  # NDCI: red-edge peak vs red chlorophyll absorption
    "nd_B6_B4",  # far red-edge vs red: particulate scattering / high biomass
)


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(total) > 1e-12, (a - b) / total, 0.0)
    return np.clip(result, -1.0, 1.0)


def build_features(rrs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return an (n, len(FEATURE_NAMES)) float64 feature matrix.

    ``rrs`` maps band names to arrays of identical shape (any shape; flattened here).
    Non-finite inputs produce non-finite feature rows; callers must mask them.
    """
    missing = [band for band in MODEL_BANDS if band not in rrs]
    if missing:
        raise ValueError(f"Missing bands for water-quality features: {missing}")
    arrays = {band: np.asarray(rrs[band], dtype=np.float64).ravel() for band in MODEL_BANDS}
    sizes = {array.size for array in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("All bands must contain the same number of pixels")

    floored = {band: np.maximum(array, RRS_FLOOR) for band, array in arrays.items()}
    nonfinite = np.zeros(next(iter(sizes)), dtype=bool)
    for array in arrays.values():
        nonfinite |= ~np.isfinite(array)

    columns = [np.log10(floored[band]) for band in MODEL_BANDS]
    columns.append(_normalized_difference(floored["B3"], floored["B2"]))
    columns.append(_normalized_difference(floored["B5"], floored["B4"]))
    columns.append(_normalized_difference(floored["B6"], floored["B4"]))
    features = np.column_stack(columns)
    features[nonfinite] = np.nan
    return features

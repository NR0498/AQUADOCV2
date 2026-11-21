from typing import Dict, Mapping

import numpy as np

from .schemas import BandStatistics


EPSILON = 1e-6


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute (a - b) / (a + b) with protection against division by zero."""

    return (a - b) / (a + b + EPSILON)


def compute_spectral_indices(bands: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute water-quality remote-sensing indices from reflectance bands.

    Expected bands use lowercase common names: blue, green, red, nir, red_edge,
    swir1, swir2. Inputs should be surface reflectance or consistently scaled
    reflectance-like values, not RGB visualization pixels.
    """

    blue = _as_float_reflectance(bands["blue"])
    green = _as_float_reflectance(bands["green"])
    red = _as_float_reflectance(bands["red"])
    nir = _as_float_reflectance(bands["nir"])

    indices = {
        "ndwi": normalized_difference(green, nir),
        "ndvi": normalized_difference(nir, red),
        "turbidity_proxy": normalized_difference(red, green),
        "floating_debris_proxy": nir - red,
    }

    if "red_edge" in bands:
        red_edge = _as_float_reflectance(bands["red_edge"])
        indices["ndci"] = normalized_difference(red_edge, red)
        indices["chlorophyll_proxy"] = red_edge / (red + EPSILON)

    if "swir1" in bands:
        swir1 = _as_float_reflectance(bands["swir1"])
        indices["mndwi"] = normalized_difference(green, swir1)
        indices["oil_sheen_proxy"] = swir1 - blue

    return indices


def summarize_index(values: np.ndarray, valid_mask: np.ndarray | None = None) -> BandStatistics:
    """Summarize an index using robust statistics for model features and APIs."""

    total_pixels = values.size
    if valid_mask is not None:
        values = values[valid_mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return BandStatistics(
            mean=0.0,
            median=0.0,
            std=0.0,
            p10=0.0,
            p90=0.0,
            valid_fraction=0.0,
        )

    return BandStatistics(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        p10=float(np.percentile(values, 10)),
        p90=float(np.percentile(values, 90)),
        valid_fraction=float(values.size / total_pixels),
    )


def _as_float_reflectance(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    if np.nanmax(arr) > 2.0:
        arr = arr / 10000.0
    return arr

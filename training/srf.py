"""Sentinel-2 spectral response functions (SRF) and band convolution.

A satellite band does not measure one wavelength: it integrates incoming light weighted
by the band's spectral response. To simulate what Sentinel-2 would see for an in-situ
hyperspectral Rrs spectrum, take the SRF-weighted mean of the spectrum over each band:

    Rrs_band = sum(Rrs(l) * S(l)) / sum(S(l))
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_HEADER = re.compile(r"Band\s+(\w+)\s*$")


def parse_acolite_rsr(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse an ACOLITE-format RSR text file into {"B1": (wavelength_nm, response), ...}."""
    bands: dict[str, list[tuple[float, float]]] = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("#", ";")):
            match = _HEADER.search(line)
            if match:
                current = f"B{match.group(1).upper()}"
                bands[current] = []
            continue
        if current is None:
            continue
        wavelength, response = line.split()[:2]
        bands[current].append((float(wavelength), float(response)))
    parsed = {}
    for band, rows in bands.items():
        array = np.asarray(rows, dtype=np.float64)
        parsed[band] = (array[:, 0], np.clip(array[:, 1], 0.0, None))
    if not parsed:
        raise ValueError(f"No bands found in {path}")
    return parsed


def convolve(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    srf: tuple[np.ndarray, np.ndarray],
    min_weight_coverage: float = 0.99,
) -> np.ndarray:
    """SRF-weighted band values for each row of ``spectra`` (n_samples x n_wavelengths).

    Rows where the finite part of the spectrum covers less than ``min_weight_coverage``
    of the band's total response weight return NaN instead of a biased partial average.
    """
    srf_wavelengths, response = srf
    weights = np.interp(wavelengths, srf_wavelengths, response, left=0.0, right=0.0)
    total = response.sum() * _step(srf_wavelengths) / _step(wavelengths)
    if total <= 0:
        raise ValueError("Spectral response has no positive weight")
    finite = np.isfinite(spectra)
    covered = (finite * weights).sum(axis=1)
    values = (np.where(finite, spectra, 0.0) * weights).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = values / covered
    result[covered < min_weight_coverage * total] = np.nan
    return result


def _step(wavelengths: np.ndarray) -> float:
    return float(np.median(np.diff(wavelengths)))


def load_sentinel2_srfs(directory: str | Path) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load every ``S2?_MSI_V4.txt`` file found in ``directory``, keyed by satellite."""
    directory = Path(directory)
    files = sorted(directory.glob("S2*_MSI_V4.txt"))
    if not files:
        raise FileNotFoundError(f"No Sentinel-2 SRF files in {directory}; run python -m training.download")
    return {path.name.split("_")[0]: parse_acolite_rsr(path) for path in files}


def convolve_sentinel2(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    srfs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    bands: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Band values averaged across the Sentinel-2 satellites (S2A/S2B/S2C).

    Their SRFs differ by a few nanometres; the deployed model serves imagery from all of
    them, so it is trained on the mean simulated response.
    """
    output = {}
    for band in bands:
        per_satellite = [convolve(spectra, wavelengths, sat[band]) for sat in srfs.values()]
        output[band] = np.mean(per_satellite, axis=0)
    return output

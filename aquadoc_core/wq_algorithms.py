"""Published semi-empirical water-quality algorithms.

These serve two purposes: baselines the trained model must beat, and transparent
estimates the dashboard can show next to the model. Inputs are remote-sensing
reflectance Rrs (sr^-1); several algorithms are defined on water-leaving
reflectance rho_w = pi * Rrs.

The coefficients were calibrated for other sensors or wavelengths. Applying them to
Sentinel-2 bands is an approximation, and each function documents the mismatch.
"""

from __future__ import annotations

import numpy as np


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def chlorophyll_ndci_mishra(rrs_b4: np.ndarray, rrs_b5: np.ndarray) -> np.ndarray:
    """Chlorophyll-a (ug/L) from NDCI, Mishra & Mishra (2012), Remote Sens. Environ. 117.

    NDCI = (R708 - R665) / (R708 + R665); Chl = 14.039 + 86.115*NDCI + 194.325*NDCI^2.
    Calibrated with MERIS 708 nm; Sentinel-2 B5 is centred at 705 nm. Intended for
    turbid, productive waters; unreliable for clear oligotrophic water.
    """
    ndci = _nd(np.asarray(rrs_b5, float), np.asarray(rrs_b4, float))
    chl = 14.039 + 86.115 * ndci + 194.325 * ndci**2
    # The quadratic turns upward again for NDCI < -0.22; values there are meaningless.
    chl = np.where(ndci < -0.22, np.nan, chl)
    return np.maximum(chl, 0.01)


def tss_nechad_665(rrs_b4: np.ndarray) -> np.ndarray:
    """Total suspended solids (mg/L), Nechad et al. (2010), Remote Sens. Environ. 114.

    TSS = A * rho_w / (1 - rho_w / C), A = 355.85 g m^-3, C = 0.1728 for 665 nm.
    Single red band; saturates in very turbid water, where NIR algorithms are needed.
    """
    rho = np.pi * np.asarray(rrs_b4, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tss = 355.85 * rho / (1.0 - rho / 0.1728)
    tss = np.where((rho >= 0.1728) | ~np.isfinite(tss), np.nan, tss)
    return np.maximum(tss, 0.01)


def turbidity_dogliotti(rrs_b4: np.ndarray, rrs_b8a: np.ndarray | None = None) -> np.ndarray:
    """Turbidity (FNU), Dogliotti et al. (2015), Remote Sens. Environ. 156.

    T = A * rho_w / (1 - rho_w / C), red: A = 228.1, C = 0.1641 (645 nm);
    NIR: A = 3078.9, C = 0.2112 (859 nm). Uses red below rho_red = 0.05, NIR above 0.07,
    and blends linearly between. Sentinel-2 B4 (665 nm) and B8A (865 nm) stand in for
    the 645/859 nm calibration bands. Without B8A, the red branch is used throughout.
    """
    rho_red = np.pi * np.asarray(rrs_b4, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_red = 228.1 * rho_red / (1.0 - rho_red / 0.1641)
    t_red = np.where(rho_red >= 0.1641, np.nan, t_red)
    if rrs_b8a is None:
        return np.maximum(t_red, 0.01)
    rho_nir = np.pi * np.asarray(rrs_b8a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_nir = 3078.9 * rho_nir / (1.0 - rho_nir / 0.2112)
    t_nir = np.where(rho_nir >= 0.2112, np.nan, t_nir)
    weight = np.clip((rho_red - 0.05) / 0.02, 0.0, 1.0)
    blended = (1.0 - weight) * t_red + weight * t_nir
    # Where the NIR estimate is unavailable, fall back to red only for low-turbidity pixels.
    blended = np.where(np.isfinite(t_nir), blended, np.where(weight == 0.0, t_red, np.nan))
    return np.maximum(blended, 0.01)


def floating_debris_index(rho_b6: np.ndarray, rho_b8: np.ndarray, rho_b11: np.ndarray) -> np.ndarray:
    """Floating Debris Index (FDI), Biermann et al. (2020), Scientific Reports 10:5364.

    FDI = R_NIR - R'_NIR, with the NIR baseline interpolated between red-edge B6 and
    SWIR B11: R'_NIR = R_B6 + (R_B11 - R_B6) * (l_NIR - l_red) / (l_SWIR - l_red) * 10.
    Uses surface reflectance (not Rrs). Floating material (plastics, seaweed, foam, wood,
    scum) raises NIR above the water baseline; FDI alone cannot say which material it is.
    """
    l_red, l_nir, l_swir = 664.6, 832.8, 1612.6
    baseline = rho_b6 + (rho_b11 - rho_b6) * ((l_nir - l_red) / (l_swir - l_red)) * 10.0
    return np.asarray(rho_b8, float) - baseline

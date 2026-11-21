from typing import Dict, Tuple

from .schemas import BandStatistics


def estimate_water_risk(indices: Dict[str, BandStatistics]) -> Tuple[float, str, Dict[str, str]]:
    """Estimate water safety from explainable spectral statistics.

    This is a baseline risk model, not a validated pollution classifier. It is
    useful for sanity checks, feature exploration, and bootstrapping labels.
    """

    score = 0.0
    signals: Dict[str, str] = {}

    ndwi = indices.get("ndwi")
    if ndwi and ndwi.median < 0.05:
        score += 0.15
        signals["water_mask"] = "Weak water signal; scene may include land, clouds, or shallow/turbid water."
    elif ndwi:
        signals["water_mask"] = "Water signal present."

    turbidity = indices.get("turbidity_proxy")
    if turbidity:
        if turbidity.p90 > 0.08:
            score += 0.25
            signals["turbidity"] = "High red-vs-green response suggests suspended sediment or turbid water."
        elif turbidity.p90 > 0.03:
            score += 0.12
            signals["turbidity"] = "Moderate turbidity signal."
        else:
            signals["turbidity"] = "Low turbidity proxy."

    ndci = indices.get("ndci")
    if ndci:
        if ndci.p90 > 0.12:
            score += 0.40
            signals["algae_bloom"] = "High red-edge response suggests possible chlorophyll/algal bloom."
        elif ndci.p90 > 0.04:
            score += 0.15
            signals["algae_bloom"] = "Moderate chlorophyll proxy."
        else:
            signals["algae_bloom"] = "Low chlorophyll proxy."

    debris = indices.get("floating_debris_proxy")
    if debris:
        if debris.p90 > 0.08:
            score += 0.20
            signals["floating_debris"] = "Bright near-infrared patches may indicate vegetation mats or floating debris."
        elif debris.p90 > 0.03:
            score += 0.10
            signals["floating_debris"] = "Some near-infrared floating-material signal."

    oil = indices.get("oil_sheen_proxy")
    if oil:
        if oil.p90 > 0.06:
            score += 0.20
            signals["oil_or_surface_film"] = "SWIR-blue contrast suggests possible surface film, oil, or non-water contamination."
        elif oil.p90 > 0.02:
            score += 0.10
            signals["oil_or_surface_film"] = "Weak surface-film proxy."

    score = max(0.0, min(1.0, score))

    if score >= 0.70:
        label = "unsafe"
    elif score >= 0.40:
        label = "caution"
    else:
        label = "likely_safe"

    return score, label, signals

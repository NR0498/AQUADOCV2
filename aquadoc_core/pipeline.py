import numpy as np

from .risk import estimate_water_risk
from .schemas import AnalysisResult, MultispectralScene
from .spectral_indices import compute_spectral_indices, summarize_index


BASELINE_LIMITATIONS = (
    "This result is an explainable spectral baseline, not a trained or field-validated ML model.",
    "Satellite imagery can suggest algae, turbidity, surface film, or floating material, but lab measurements are needed for chemical safety.",
    "Clouds, shadows, shallow water, sun glint, and mixed land/water pixels can cause false signals.",
)


def analyze_multispectral_scene(scene: MultispectralScene) -> AnalysisResult:
    """Run the first backend analysis pass for a multispectral scene."""

    scene.validate()
    valid_pixels = np.logical_and.reduce([
        np.isfinite(values) for values in scene.bands.values()
    ])
    if not valid_pixels.any():
        raise ValueError("Scene has no finite pixels shared by all supplied bands")
    raw_indices = compute_spectral_indices(scene.bands)
    water_mask = valid_pixels & (raw_indices["ndwi"] > 0.0)
    for values in raw_indices.values():
        water_mask &= np.isfinite(values)
    if not water_mask.any():
        raise ValueError("Scene has no usable water pixels; no screening result can be produced")

    summaries = {
        name: summarize_index(values, water_mask)
        for name, values in raw_indices.items()
    }

    risk_score, safety_label, signals = estimate_water_risk(summaries)

    return AnalysisResult(
        latitude=scene.latitude,
        longitude=scene.longitude,
        source=scene.source,
        captured_at=scene.captured_at,
        resolution_m=scene.resolution_m,
        indices=summaries,
        risk_score=risk_score,
        safety_label=safety_label,
        signals=signals,
        limitations=BASELINE_LIMITATIONS,
    )

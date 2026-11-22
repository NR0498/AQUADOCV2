"""Core backend for AquaDoc multispectral water-quality analysis."""

from .native import ensure_libexpat

# Must run before anything imports rasterio (see native.py). Failure is reported by /api/health.
try:
    LIBEXPAT = ensure_libexpat()
except OSError as _exc:  # pragma: no cover - only on a Linux host with neither copy usable
    LIBEXPAT = f"unavailable: {_exc}"

from .pipeline import analyze_multispectral_scene
from .risk import estimate_water_risk
from .schemas import AnalysisResult, BandStatistics, MultispectralScene
from .spectral_indices import compute_spectral_indices
from .ingestion import fetch_sentinel2_chip
from .serialization import analysis_result_to_dict
from .persistence import AnalysisRun, AnalysisRunRepository

__all__ = [
    "AnalysisResult",
    "BandStatistics",
    "MultispectralScene",
    "analyze_multispectral_scene",
    "compute_spectral_indices",
    "estimate_water_risk",
    "fetch_sentinel2_chip",
    "analysis_result_to_dict",
    "AnalysisRun",
    "AnalysisRunRepository",
]

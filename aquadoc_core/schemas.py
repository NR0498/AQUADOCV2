from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


REQUIRED_BANDS = ("blue", "green", "red", "nir")
OPTIONAL_BANDS = ("red_edge", "swir1", "swir2")


@dataclass(frozen=True)
class MultispectralScene:
    """In-memory representation of a small multispectral satellite chip."""

    latitude: float
    longitude: float
    bands: Mapping[str, np.ndarray]
    source: str = "unknown"
    captured_at: Optional[str] = None
    resolution_m: Optional[float] = None

    def validate(self) -> Tuple[int, int]:
        missing = [band for band in REQUIRED_BANDS if band not in self.bands]
        if missing:
            raise ValueError(f"Missing required bands: {', '.join(missing)}")

        shape = None
        for name, values in self.bands.items():
            if name not in REQUIRED_BANDS + OPTIONAL_BANDS:
                raise ValueError(f"Unsupported band '{name}'")
            if values.ndim != 2:
                raise ValueError(f"Band '{name}' must be a 2D array")
            if values.size == 0:
                raise ValueError(f"Band '{name}' must not be empty")
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError(f"Band '{name}' must contain real numeric values")
            if shape is None:
                shape = values.shape
            elif values.shape != shape:
                raise ValueError("All bands must have the same shape")

        if not -90 <= self.latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180")

        return shape


@dataclass(frozen=True)
class BandStatistics:
    mean: float
    median: float
    std: float
    p10: float
    p90: float
    valid_fraction: float


@dataclass(frozen=True)
class AnalysisResult:
    latitude: float
    longitude: float
    source: str
    captured_at: Optional[str]
    resolution_m: Optional[float]
    indices: Dict[str, BandStatistics]
    risk_score: float
    safety_label: str
    signals: Dict[str, str] = field(default_factory=dict)
    limitations: Tuple[str, ...] = field(default_factory=tuple)

"""NumPy-only inference for the trained water-quality model ensemble.

The model is trained offline (see ``training/``) and exported as two files:

* ``<name>.npz`` - layer weights for each ensemble member, per target.
* ``<name>.json`` - metadata: units, feature scaling, conformal interval width,
  training-domain bounds and evaluation metrics (the model card).

Inference needs only NumPy, which keeps the deployed bundle small.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .wq_features import FEATURE_NAMES, FEATURE_VERSION, build_features

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_NAME = "water_quality_v1"


@dataclass(frozen=True)
class TargetPrediction:
    """Per-pixel predictions for one target. Arrays share the input shape."""

    target: str
    unit: str
    median: np.ndarray
    low: np.ndarray   # lower bound of the calibrated interval
    high: np.ndarray  # upper bound of the calibrated interval
    in_domain: np.ndarray  # False where inputs fall outside the training feature range
    interval_level: float


@dataclass(frozen=True)
class _Member:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        last = len(self.weights) - 1
        for index, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if index != last:
                h = np.maximum(h, 0.0)  # ReLU
        return h[:, 0]


@dataclass(frozen=True)
class _TargetModel:
    name: str
    unit: str
    members: tuple[_Member, ...]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    conformal_q: float
    sigma_floor: float
    interval_level: float
    feature_low: np.ndarray
    feature_high: np.ndarray

    def predict_log10(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (features - self.x_mean) / self.x_std
        outputs = np.stack([member.forward(x) for member in self.members])
        outputs = outputs * self.y_std + self.y_mean
        return outputs.mean(axis=0), outputs.std(axis=0)


class WaterQualityModel:
    def __init__(self, metadata: dict, targets: dict[str, _TargetModel]):
        self.metadata = metadata
        self._targets = targets

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(self._targets)

    @property
    def satellite_correction(self) -> dict | None:
        """Residual Sentinel-2 L2A correction learned from match-ups, if validated as helpful."""
        correction = self.metadata.get("satellite_correction")
        return correction if correction and correction.get("applied") else None

    def correct_satellite_rrs(self, rrs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the match-up correction to L2A-derived Rrs (needs B11). No-op when not applied."""
        correction = self.satellite_correction
        if correction is None:
            return dict(rrs)
        corrected = dict(rrs)
        for band, c in correction["coefficients"].items():
            corrected[band] = c["intercept"] + c["gain"] * np.asarray(rrs[band]) + c["swir_b11"] * np.asarray(rrs["B11"])
        return corrected

    def target_spec(self, target: str) -> dict:
        return self.metadata["targets"][target]

    def satellite_input(self, target: str) -> str:
        """'corrected' or 'raw': which L2A reflectance validated better for this target."""
        if self.satellite_correction is None:
            return "raw"
        return self.metadata["targets"][target].get("satellite_input", "raw")

    def predict_satellite(self, rrs_l2a: Mapping[str, np.ndarray]) -> dict[str, TargetPrediction]:
        """Predict from Sentinel-2 L2A Rrs (rho / pi, including B11), using for each target the
        input (raw or match-up-corrected) that performed better on satellite validation."""
        raw = self.predict(rrs_l2a)
        if self.satellite_correction is None:
            return raw
        corrected = self.predict(self.correct_satellite_rrs(rrs_l2a))
        return {t: corrected[t] if self.satellite_input(t) == "corrected" else raw[t] for t in raw}

    @classmethod
    def load(cls, name: str = DEFAULT_MODEL_NAME, directory: str | Path = MODELS_DIR) -> "WaterQualityModel":
        directory = Path(directory)
        metadata = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Model expects features '{metadata.get('feature_version')}', code builds '{FEATURE_VERSION}'"
            )
        if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Model feature names do not match the feature builder")
        with np.load(directory / f"{name}.npz", allow_pickle=False) as arrays:
            targets = {}
            for target, spec in metadata["targets"].items():
                if not spec.get("deployed", False):
                    continue
                members = []
                for m in range(spec["n_members"]):
                    prefix = f"{target}/m{m}"
                    weights = tuple(arrays[f"{prefix}/W{layer}"] for layer in range(spec["n_layers"]))
                    biases = tuple(arrays[f"{prefix}/b{layer}"] for layer in range(spec["n_layers"]))
                    members.append(_Member(weights, biases))
                targets[target] = _TargetModel(
                    name=target,
                    unit=spec["unit"],
                    members=tuple(members),
                    x_mean=np.asarray(spec["x_mean"]),
                    x_std=np.asarray(spec["x_std"]),
                    y_mean=float(spec["y_mean"]),
                    y_std=float(spec["y_std"]),
                    conformal_q=float(spec["conformal_q"]),
                    sigma_floor=float(spec["sigma_floor"]),
                    interval_level=float(spec["interval_level"]),
                    feature_low=np.asarray(spec["feature_p01"]),
                    feature_high=np.asarray(spec["feature_p99"]),
                )
        return cls(metadata, targets)

    def predict(self, rrs: Mapping[str, np.ndarray]) -> dict[str, TargetPrediction]:
        """Predict every deployed target from per-band Rrs arrays (sr^-1)."""
        shape = np.shape(next(iter(rrs.values())))
        features = build_features(rrs)
        valid = np.all(np.isfinite(features), axis=1)
        safe = np.where(valid[:, None], features, 0.0)

        results = {}
        for name, model in self._targets.items():
            mu, sigma = model.predict_log10(safe)
            half_width = model.conformal_q * (sigma + model.sigma_floor)
            in_domain = valid & np.all(
                (safe >= model.feature_low) & (safe <= model.feature_high), axis=1
            )
            nan = np.where(valid, 1.0, np.nan)
            results[name] = TargetPrediction(
                target=name,
                unit=model.unit,
                median=(10.0 ** mu * nan).reshape(shape),
                low=(10.0 ** (mu - half_width) * nan).reshape(shape),
                high=(10.0 ** (mu + half_width) * nan).reshape(shape),
                in_domain=in_domain.reshape(shape),
                interval_level=model.interval_level,
            )
        return results

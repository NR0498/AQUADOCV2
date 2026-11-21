"""Validate the model on real Sentinel-2 pixels and learn a residual reflectance correction.

    python -m training.matchups              # first: extract match-ups
    python -m training.validate_satellite    # then: report, correction, recalibrated intervals

Steps
1. Radiometry: compare Sentinel-2 L2A Rrs (rho / pi) with GLORIA in-situ Rrs at the same place
   and time. Sen2Cor is designed for land and typically leaves a positive offset over water,
   often correlated with the residual SWIR signal (aerosol, adjacency, glint).
2. Correction: per band, Rrs_insitu ~ a + g * Rrs_sat + c * Rrs_sat(B11), fitted with a robust
   (Huber) regression and scored with grouped cross-validation (whole sites held out).
3. Honest model evaluation: retrain the ensemble *without* any match-up site, then predict from
   raw and corrected satellite Rrs and compare with the lab values.
4. Satellite-calibrated intervals: conformal prediction on satellite residuals, so the 80%
   ranges reflect real satellite error rather than simulated noise.
Results are written into the model metadata (``satellite_correction`` and, per target,
``satellite_validation`` and ``conformal_q``) and used automatically by the pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GroupKFold

from aquadoc_core import wq_algorithms
from aquadoc_core.wq_features import FEATURE_NAMES, FEATURE_VERSION, MODEL_BANDS
from aquadoc_core.wq_model import DEFAULT_MODEL_NAME, MODELS_DIR, WaterQualityModel

from .gloria import TARGETS
from .matchups import OUTPUT_PATH as MATCHUP_PATH, TABLE_PATH
from .train import TrainConfig, log_metrics, train_target

CORRECTION_VERSION = "s2-l2a-residual-correction-v1"
MAX_HOURS = 24.0
MIN_MATCHUPS_FOR_TARGET = 25


def load_matchups(max_hours: float = MAX_HOURS) -> pd.DataFrame:
    matchups = pd.read_csv(MATCHUP_PATH)
    matchups = matchups[(matchups["status"] == "ok") & (matchups["hours_apart"] <= max_hours)]
    table = pd.read_csv(TABLE_PATH)
    merged = matchups.merge(table, on="gloria_id", how="inner")
    for band in (*MODEL_BANDS, "B8A", "B11"):
        merged[f"sat_{band}"] = merged[f"rho_{band}"] / math.pi
    return merged.reset_index(drop=True)


def band_agreement(frame: pd.DataFrame, prefix: str) -> dict:
    out = {}
    for band in MODEL_BANDS:
        insitu, sat = frame[f"rrs_{band}"].to_numpy(), frame[f"{prefix}{band}"].to_numpy()
        diff = sat - insitu
        metrics = log_metrics(np.clip(insitu, 1e-5, None), np.clip(sat, 1e-5, None))
        out[band] = {"median_difference_sr": float(f"{np.median(diff):.3g}"),
                     "mdsa_pct": metrics.get("mdsa_pct"),
                     "r": round(float(np.corrcoef(insitu, sat)[0, 1]), 3)}
    return out


def fit_correction(frame: pd.DataFrame) -> dict[str, dict]:
    coefficients = {}
    for band in MODEL_BANDS:
        x = frame[[f"sat_{band}", "sat_B11"]].to_numpy()
        model = HuberRegressor(epsilon=1.35, alpha=1e-6, max_iter=1000).fit(x * 1000, frame[f"rrs_{band}"].to_numpy() * 1000)
        coefficients[band] = {"intercept": float(model.intercept_ / 1000), "gain": float(model.coef_[0]),
                              "swir_b11": float(model.coef_[1])}
    return coefficients


def apply_correction(frame: pd.DataFrame, coefficients: dict, prefix: str = "sat_") -> dict[str, np.ndarray]:
    return {band: c["intercept"] + c["gain"] * frame[f"{prefix}{band}"].to_numpy()
            + c["swir_b11"] * frame[f"{prefix}B11"].to_numpy() for band, c in coefficients.items()}


def cross_validated_correction(frame: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    """Out-of-fold corrected Rrs: each site corrected by coefficients fitted without it."""
    corrected = frame.copy()
    splitter = GroupKFold(n_splits=min(folds, frame["group"].nunique()))
    for train_idx, test_idx in splitter.split(frame, groups=frame["group"]):
        coefficients = fit_correction(frame.iloc[train_idx])
        values = apply_correction(frame.iloc[test_idx], coefficients)
        for band, array in values.items():
            corrected.loc[corrected.index[test_idx], f"cor_{band}"] = array
    return corrected


def validation_model(table: pd.DataFrame, exclude_groups: set[str], config: TrainConfig) -> WaterQualityModel:
    """Retrain every target without the match-up sites, returning an in-memory model."""
    import zlib

    holdout_free = table[~table["group"].isin(exclude_groups)].reset_index(drop=True)
    metadata = {"feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES), "targets": {}}
    arrays = {}
    for target in TARGETS:
        rng = np.random.default_rng([config.seed, zlib.crc32(target.encode())])
        spec, weights, _ = train_target(target, holdout_free, config, rng)
        spec["deployed"] = True
        metadata["targets"][target] = spec
        arrays.update(weights)
        print(f"  validation model trained: {target}")
    with tempfile.TemporaryDirectory() as directory:
        np.savez(Path(directory) / "v.npz", **arrays)
        (Path(directory) / "v.json").write_text(json.dumps(metadata), encoding="utf-8")
        return WaterQualityModel.load("v", directory)


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    n = len(scores)
    return float(np.quantile(scores, min(1.0, math.ceil((n + 1) * level) / n)))


def evaluate_targets(frame: pd.DataFrame, model: WaterQualityModel, level: float, sigma_floor: float) -> dict:
    sources = {
        "in_situ_rrs": {band: frame[f"rrs_{band}"].to_numpy() for band in MODEL_BANDS},
        "satellite_raw": {band: frame[f"sat_{band}"].to_numpy() for band in MODEL_BANDS},
        "satellite_corrected": {band: frame[f"cor_{band}"].to_numpy() for band in MODEL_BANDS},
    }
    predictions = {name: model.predict(rrs) for name, rrs in sources.items()}
    from aquadoc_core.wq_features import build_features
    features = {name: build_features(sources[name]) for name in ("satellite_raw", "satellite_corrected")}

    results = {}
    for target in TARGETS:
        truth = frame[target].to_numpy()
        has = np.isfinite(truth)
        if has.sum() < MIN_MATCHUPS_FOR_TARGET:
            results[target] = {"n": int(has.sum()), "note": "too few satellite match-ups to validate"}
            continue
        entry = {"n": int(has.sum()), "sites": int(frame.loc[has, "group"].nunique())}
        for name, prediction in predictions.items():
            entry[name] = log_metrics(truth[has], prediction[target].median[has])
        b4, b5 = sources["satellite_corrected"]["B4"], sources["satellite_corrected"]["B5"]
        published = {"chlorophyll_a": wq_algorithms.chlorophyll_ndci_mishra(b4, b5),
                     "tss": wq_algorithms.tss_nechad_665(b4),
                     "turbidity": wq_algorithms.turbidity_dogliotti(b4, frame["sat_B8A"].to_numpy())}.get(target)
        if published is not None:
            entry["published_algorithm_on_corrected_satellite"] = log_metrics(truth[has], published[has])

        target_model = model._targets[target]
        groups = frame.loc[has, "group"].to_numpy()
        for name, matrix in features.items():
            mu, sigma = target_model.predict_log10(matrix[has])
            scores = np.abs(np.log10(truth[has]) - mu) / (sigma + sigma_floor)
            # Grouped cross-validation of the conformal quantile: is coverage held on unseen sites?
            covered = np.zeros(len(scores), dtype=bool)
            splitter = GroupKFold(n_splits=min(5, len(set(groups))))
            for train_idx, test_idx in splitter.split(scores, groups=groups):
                covered[test_idx] = scores[test_idx] <= conformal_quantile(scores[train_idx], level)
            suffix = name.replace("satellite_", "")
            entry[f"conformal_q_{suffix}"] = conformal_quantile(scores, level)
            entry[f"interval_coverage_cv_{suffix}"] = round(float(covered.mean()), 3)
        results[target] = entry
    return results


def run(max_hours: float, members: int) -> dict:
    frame = load_matchups(max_hours)
    if len(frame) < 30:
        raise SystemExit(f"Only {len(frame)} usable match-ups; need at least 30.")
    print(f"{len(frame)} match-ups from {frame['group'].nunique()} sites (<= {max_hours} h)")
    report = {"version": CORRECTION_VERSION, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "matchups": int(len(frame)), "sites": int(frame["group"].nunique()), "max_hours_apart": max_hours,
              "by_water_body_type": frame["water_body_type"].value_counts().to_dict()}

    report["radiometry_raw"] = band_agreement(frame, "sat_")
    frame = cross_validated_correction(frame)
    report["radiometry_corrected_cv"] = band_agreement(frame, "cor_")
    coefficients = fit_correction(frame)
    report["coefficients"] = coefficients

    config = TrainConfig(n_members=members)
    table = pd.read_csv(TABLE_PATH)
    model = validation_model(table, set(frame["group"]), config)
    report["targets"] = evaluate_targets(frame, model, config.interval_level, config.sigma_floor)
    return report


def update_model_metadata(report: dict) -> Path:
    """Write correction, per-target input choice, validation and satellite-calibrated intervals.

    Each target uses whichever input (raw or corrected L2A reflectance) gave the lower median
    error on held-out satellite match-ups; intervals are calibrated on that same input.
    """
    path = MODELS_DIR / f"{DEFAULT_MODEL_NAME}.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    chosen = {}
    for target, spec in metadata["targets"].items():
        validation = report["targets"].get(target) or {}
        spec["satellite_validation"] = validation or None
        spec.setdefault("conformal_q_in_situ", spec["conformal_q"])
        if "satellite_raw" in validation:
            better = validation["satellite_corrected"]["mdsa_pct"] < validation["satellite_raw"]["mdsa_pct"]
            use = "corrected" if better else "raw"
            chosen[target] = use
            spec["satellite_input"] = use
            spec["conformal_q"] = validation[f"conformal_q_{use}"]
            spec["interval_basis"] = f"satellite match-ups, {use} reflectance (conformal)"
        else:
            spec["satellite_input"] = "raw"
            spec["conformal_q"] = spec["conformal_q_in_situ"]
            spec["interval_basis"] = "simulated in-situ spectra (not validated on satellite data)"
    metadata["satellite_correction"] = {
        "version": report["version"], "applied": any(v == "corrected" for v in chosen.values()),
        "targets_using_correction": sorted(t for t, v in chosen.items() if v == "corrected"),
        "formula": "Rrs = intercept + gain * Rrs_L2A + swir_b11 * Rrs_L2A(B11), Rrs_L2A = rho / pi",
        "coefficients": report["coefficients"], "matchups": report["matchups"], "sites": report["sites"],
        "max_hours_apart": report["max_hours_apart"], "by_water_body_type": report["by_water_body_type"],
        "radiometry_raw": report["radiometry_raw"], "radiometry_corrected_cv": report["radiometry_corrected_cv"],
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-hours", type=float, default=MAX_HOURS)
    parser.add_argument("--members", type=int, default=5, help="Ensemble size of the validation model")
    parser.add_argument("--report", type=Path, default=Path("docs/models/satellite_validation.json"))
    parser.add_argument("--write-model", action="store_true",
                        help="Store correction, validation and satellite intervals in the deployed model metadata")
    args = parser.parse_args(argv)
    report = run(args.max_hours, args.members)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for target, entry in report["targets"].items():
        if "satellite_raw" in entry:
            print(f"{target:14s} n={entry['n']:4d}  MdSA in-situ={entry['in_situ_rrs']['mdsa_pct']}%  "
                  f"raw={entry['satellite_raw']['mdsa_pct']}%  corrected={entry['satellite_corrected']['mdsa_pct']}%  "
                  f"coverage(cv) raw={entry['interval_coverage_cv_raw']} corrected={entry['interval_coverage_cv_corrected']}")
    if args.write_model:
        print("model metadata updated:", update_model_metadata(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

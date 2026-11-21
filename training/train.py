"""Train, evaluate and export the water-quality model ensemble.

    python -m training.download      # once
    python -m training.train         # builds table, trains, writes aquadoc_core/models/

For every target: split by spatial group into train / calibration / test, train an
ensemble of small neural networks on log10(target) from Sentinel-2-simulated Rrs
features, calibrate prediction intervals with split conformal prediction, then compare
against a dummy predictor and the published algorithm on the untouched test groups.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor

from aquadoc_core import wq_algorithms
from aquadoc_core.wq_features import FEATURE_NAMES, FEATURE_VERSION, MODEL_BANDS, build_features
from aquadoc_core.wq_model import DEFAULT_MODEL_NAME, MODELS_DIR

from .gloria import TARGETS, build_training_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLORIA_DIR = PROJECT_ROOT / "data" / "external" / "gloria" / "extracted" / "GLORIA_2022"
SRF_DIR = PROJECT_ROOT / "data" / "external" / "s2_srf"
TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "gloria" / "s2_training_table.csv"

MODEL_VERSION = "water-quality-mlp-ensemble-v1"


@dataclass(frozen=True)
class TrainConfig:
    hidden_layers: tuple[int, ...] = (64, 64)
    alpha: float = 1e-3            # L2 weight penalty
    max_iter: int = 300
    n_members: int = 10            # ensemble size; spread feeds the uncertainty estimate
    augment_copies: int = 3        # noisy copies per training spectrum
    multiplicative_noise: float = 0.05   # ~5% radiometric calibration / gain error
    additive_noise: float = 0.0005       # sr^-1; residual atmospheric-correction offset
    test_fraction: float = 0.2
    calibration_fraction: float = 0.15
    interval_level: float = 0.8
    sigma_floor: float = 0.05      # log10 units; keeps intervals from collapsing to zero
    seed: int = 42


# ---------------------------------------------------------------------------- metrics

def log_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Standard metrics for log-normally distributed water-quality variables.

    MdSA (median symmetric accuracy, %) and SSPB (signed symmetric percentage bias, %)
    follow Morley et al. (2018), Space Weather 16; they are standard in aquatic remote
    sensing because errors scale with concentration.
    """
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if ok.sum() == 0:
        return {"n": 0}
    t, p = y_true[ok], y_pred[ok]
    log_ratio = np.log(p / t)
    median_log_ratio = float(np.median(log_ratio))
    log_t, log_p = np.log10(t), np.log10(p)
    ss_res = float(np.sum((log_t - log_p) ** 2))
    ss_tot = float(np.sum((log_t - log_t.mean()) ** 2))
    return {
        "n": int(ok.sum()),
        "mdsa_pct": round(100 * (np.exp(np.median(np.abs(log_ratio))) - 1), 2),
        "sspb_pct": round(100 * np.sign(median_log_ratio) * (np.exp(abs(median_log_ratio)) - 1), 2),
        "rmse_log10": round(float(np.sqrt(np.mean((log_t - log_p) ** 2))), 4),
        "r2_log10": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "coverage_of_valid_rows": round(float(ok.mean()), 4),
    }


# ----------------------------------------------------------------------- augmentation

def augment(rrs: dict[str, np.ndarray], copies: int, config: TrainConfig, rng: np.random.Generator):
    """Noisy copies that mimic satellite radiometric and atmospheric-correction error."""
    out = {band: [values] for band, values in rrs.items()}
    n = len(next(iter(rrs.values())))
    for _ in range(copies):
        gain = 1.0 + rng.normal(0.0, config.multiplicative_noise, size=n)  # spectrally flat gain
        for band, values in rrs.items():
            offset = rng.normal(0.0, config.additive_noise, size=n)
            out[band].append(values * gain + offset)
    return {band: np.concatenate(parts) for band, parts in out.items()}


def rrs_from(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {band: frame[f"rrs_{band}"].to_numpy() for band in MODEL_BANDS}


def published_baseline(target: str, frame: pd.DataFrame) -> tuple[str, np.ndarray] | None:
    b4, b5 = frame["rrs_B4"].to_numpy(), frame["rrs_B5"].to_numpy()
    if target == "chlorophyll_a":
        return "NDCI, Mishra & Mishra 2012", wq_algorithms.chlorophyll_ndci_mishra(b4, b5)
    if target == "tss":
        return "Nechad et al. 2010 (665 nm)", wq_algorithms.tss_nechad_665(b4)
    if target == "turbidity":
        return "Dogliotti et al. 2015", wq_algorithms.turbidity_dogliotti(b4, frame["rrs_B8A"].to_numpy())
    return None


# --------------------------------------------------------------------------- training

def split_groups(frame: pd.DataFrame, config: TrainConfig):
    outer = GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=config.seed)
    trainval_idx, test_idx = next(outer.split(frame, groups=frame["group"]))
    trainval = frame.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=config.calibration_fraction, random_state=config.seed + 1)
    train_idx, calib_idx = next(inner.split(trainval, groups=trainval["group"]))
    return trainval.iloc[train_idx], trainval.iloc[calib_idx], frame.iloc[test_idx]


def ensemble_predict_log10(members, x_scaled, y_mean, y_std):
    outputs = np.stack([member.predict(x_scaled) for member in members]) * y_std + y_mean
    return outputs.mean(axis=0), outputs.std(axis=0)


def train_target(target: str, table: pd.DataFrame, config: TrainConfig, rng: np.random.Generator):
    unit = TARGETS[target][1]
    frame = table[np.isfinite(table[target])].reset_index(drop=True)
    train, calib, test = split_groups(frame, config)

    x_train_clean = build_features(rrs_from(train))
    x_mean = x_train_clean.mean(axis=0)
    x_std = np.where(x_train_clean.std(axis=0) > 1e-9, x_train_clean.std(axis=0), 1.0)
    y_train_log = np.log10(train[target].to_numpy())
    y_mean, y_std = float(y_train_log.mean()), float(y_train_log.std())

    members = []
    for m in range(config.n_members):
        bootstrap = rng.integers(0, len(train), size=len(train))
        sample = train.iloc[bootstrap]
        rrs_aug = augment(rrs_from(sample), config.augment_copies, config, rng)
        x = (build_features(rrs_aug) - x_mean) / x_std
        y = np.tile(np.log10(sample[target].to_numpy()), config.augment_copies + 1)
        net = MLPRegressor(
            hidden_layer_sizes=config.hidden_layers, activation="relu", alpha=config.alpha,
            max_iter=config.max_iter, batch_size=256, random_state=config.seed + m,
        )
        net.fit(x, (y - y_mean) / y_std)
        members.append(net)

    def predict(frame_part: pd.DataFrame, noisy: bool = False):
        rrs = rrs_from(frame_part)
        if noisy:
            noise_rng = np.random.default_rng(config.seed + 999)
            augmented = augment(rrs, 1, config, noise_rng)
            rrs = {band: values[len(frame_part):] for band, values in augmented.items()}
        x = (build_features(rrs) - x_mean) / x_std
        return ensemble_predict_log10(members, x, y_mean, y_std)

    # Split conformal calibration on held-out groups (normalized by ensemble spread).
    mu_cal, sigma_cal = predict(calib)
    y_cal = np.log10(calib[target].to_numpy())
    scores = np.abs(y_cal - mu_cal) / (sigma_cal + config.sigma_floor)
    n_cal = len(scores)
    rank = min(1.0, np.ceil((n_cal + 1) * config.interval_level) / n_cal)
    conformal_q = float(np.quantile(scores, rank))

    y_test = test[target].to_numpy()
    mu_test, sigma_test = predict(test)
    half = conformal_q * (sigma_test + config.sigma_floor)
    inside = (np.log10(y_test) >= mu_test - half) & (np.log10(y_test) <= mu_test + half)
    mu_noisy, _ = predict(test, noisy=True)

    evaluation = {
        "model": log_metrics(y_test, 10 ** mu_test),
        "model_with_simulated_satellite_noise": log_metrics(y_test, 10 ** mu_noisy),
        "dummy_train_median": log_metrics(y_test, np.full(len(y_test), 10 ** np.median(y_train_log))),
        "interval_coverage": round(float(inside.mean()), 4),
        "median_interval_factor": round(float(np.median(10 ** half)), 3),
    }

    # Reference model, not deployed: gradient boosting on the same augmented features.
    rrs_aug = augment(rrs_from(train), config.augment_copies, config, rng)
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=config.seed)
    hgb.fit(build_features(rrs_aug), np.tile(y_train_log, config.augment_copies + 1))
    evaluation["reference_gradient_boosting"] = log_metrics(y_test, 10 ** hgb.predict(build_features(rrs_from(test))))

    baseline = published_baseline(target, test)
    if baseline:
        name, values = baseline
        evaluation["published_algorithm"] = {"name": name, **log_metrics(y_test, values)}

    by_type = {}
    for water_type, part in test.assign(pred=10 ** mu_test).groupby("water_body_type"):
        if len(part) >= 20:
            by_type[water_type] = log_metrics(part[target].to_numpy(), part["pred"].to_numpy())
    evaluation["model_by_water_body_type"] = by_type

    # Deployment rule, fixed before training: beat the dummy and any published algorithm.
    model_mdsa = evaluation["model"]["mdsa_pct"]
    competitors = [evaluation["dummy_train_median"]["mdsa_pct"]]
    if baseline and evaluation["published_algorithm"]["n"] >= 0.5 * len(test):
        competitors.append(evaluation["published_algorithm"]["mdsa_pct"])
    deployed = all(model_mdsa < other for other in competitors)

    spec = {
        "unit": unit,
        "gloria_column": TARGETS[target][0],
        "deployed": bool(deployed),
        "n_members": config.n_members,
        "n_layers": len(config.hidden_layers) + 1,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
        "conformal_q": conformal_q,
        "sigma_floor": config.sigma_floor,
        "interval_level": config.interval_level,
        "feature_p01": np.percentile(x_train_clean, 1, axis=0).tolist(),
        "feature_p99": np.percentile(x_train_clean, 99, axis=0).tolist(),
        "target_range_p01_p99": [float(np.percentile(train[target], 1)), float(np.percentile(train[target], 99))],
        "split_sizes": {
            "train": len(train), "calibration": len(calib), "test": len(test),
            "train_groups": int(train["group"].nunique()), "test_groups": int(test["group"].nunique()),
        },
        "evaluation": evaluation,
    }
    weights = {}
    for m, net in enumerate(members):
        for layer, (w, b) in enumerate(zip(net.coefs_, net.intercepts_)):
            weights[f"{target}/m{m}/W{layer}"] = w.astype(np.float32)
            weights[f"{target}/m{m}/b{layer}"] = b.astype(np.float32)
    return spec, weights, (test, mu_test)


def table_sha256(table: pd.DataFrame) -> str:
    buffer = io.StringIO()
    table.to_csv(buffer, index=False, float_format="%.8g")
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def run(output_dir: Path, name: str, config: TrainConfig, targets: list[str]) -> dict:
    table, data_report = build_training_table(GLORIA_DIR, SRF_DIR)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False, float_format="%.8g")

    metadata = {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "input": "Sentinel-2 remote-sensing reflectance Rrs (sr^-1) = L2A surface reflectance / pi",
        "bands": list(MODEL_BANDS),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_data": {
            "source": "GLORIA (Lehmann et al. 2023, Sci. Data 10:100), doi:10.1594/PANGAEA.948492, CC-BY-4.0",
            "table_sha256": table_sha256(table),
            **data_report,
        },
        "config": asdict(config),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "targets": {},
    }
    arrays = {}
    for target in targets:
        # Independent, reproducible stream per target (results don't depend on target order).
        rng = np.random.default_rng([config.seed, zlib.crc32(target.encode())])
        spec, weights, _ = train_target(target, table, config, rng)
        metadata["targets"][target] = spec
        arrays.update(weights)
        ev = spec["evaluation"]
        print(f"{target:14s} deployed={spec['deployed']!s:5s} MdSA model={ev['model']['mdsa_pct']}% "
              f"dummy={ev['dummy_train_median']['mdsa_pct']}% "
              f"published={ev.get('published_algorithm', {}).get('mdsa_pct', '-')}% "
              f"coverage={ev['interval_coverage']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{name}.npz", **arrays)
    (output_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--members", type=int, default=TrainConfig.n_members)
    args = parser.parse_args(argv)
    run(args.output_dir, args.name, TrainConfig(n_members=args.members), args.targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

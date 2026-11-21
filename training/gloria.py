"""Build a Sentinel-2-band training table from the GLORIA dataset.

GLORIA (Lehmann et al., 2023, Sci. Data 10:100, CC-BY-4.0) pairs ~7,500 in-situ
hyperspectral Rrs spectra (350-900 nm) with laboratory water-quality measurements from
lakes, rivers, estuaries and coasts in 30 countries.

Steps: load -> drop QC-flagged spectra -> simulate Sentinel-2 bands -> attach targets
and spatial groups (for leakage-free evaluation splits).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aquadoc_core.wq_features import MODEL_BANDS

from .srf import convolve_sentinel2, load_sentinel2_srfs

# target -> (GLORIA column, unit, lower bound kept). GLORIA units: Chla mg m^-3 (= ug/L),
# TSS g m^-3 (= mg/L), aCDOM440 m^-1, Turbidity NTU, Secchi depth m.
TARGETS: dict[str, tuple[str, str, float]] = {
    "chlorophyll_a": ("Chla", "ug/L", 0.01),
    "tss": ("TSS", "mg/L", 0.01),
    "cdom_a440": ("aCDOM440", "1/m", 0.001),
    "turbidity": ("Turbidity", "NTU", 0.01),
    "secchi_depth": ("Secchi_depth", "m", 0.01),
}

# Spectra with these flags are removed. Kept on purpose: Oxygen_signal (an O2 absorption
# feature near 760 nm common to above-water radiometry) and Negative_uv_slope (UV is
# not used by the model).
EXCLUDED_QC_FLAGS = ("Noisy_red", "Noisy_blue", "Baseline_shift", "QWIP_fail", "Suspect")

# Bands simulated for the table: model bands plus B8A for the NIR turbidity algorithm.
SIMULATED_BANDS = (*MODEL_BANDS, "B8A")

WATER_BODY_TYPES = {1: "lake", 2: "estuary", 3: "coastal", 4: "river", 5: "other"}

SPATIAL_GROUP_DEGREES = 0.25  # ~28 km cells: nearby stations share a group


def spatial_group(latitude: pd.Series, longitude: pd.Series, fallback: pd.Series) -> pd.Series:
    """Grid-cell identifier so stations from the same water body stay in one split."""
    lat_cell = np.floor(latitude / SPATIAL_GROUP_DEGREES)
    lon_cell = np.floor(longitude / SPATIAL_GROUP_DEGREES)
    cell = "cell_" + lat_cell.astype("Int64").astype(str) + "_" + lon_cell.astype("Int64").astype(str)
    has_position = latitude.notna() & longitude.notna()
    return cell.where(has_position, "dataset_" + fallback.astype(str))


def build_training_table(gloria_dir: str | Path, srf_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    gloria_dir = Path(gloria_dir)
    meta = pd.read_csv(gloria_dir / "GLORIA_meta_and_lab.csv", low_memory=False)
    flags = pd.read_csv(gloria_dir / "GLORIA_qc_flags.csv")
    rrs = pd.read_csv(gloria_dir / "GLORIA_Rrs.csv")

    report: dict = {"input_spectra": int(len(rrs))}
    flagged_ids = set(flags.loc[flags[list(EXCLUDED_QC_FLAGS)].fillna(0).sum(axis=1) > 0, "GLORIA_ID"])
    report["excluded_qc_flagged"] = len(flagged_ids)
    rrs = rrs[~rrs["GLORIA_ID"].isin(flagged_ids)].reset_index(drop=True)

    wavelength_columns = [column for column in rrs.columns if column.startswith("Rrs_")]
    wavelengths = np.array([float(column.split("_")[1]) for column in wavelength_columns])
    spectra = rrs[wavelength_columns].to_numpy(dtype=np.float64)

    bands = convolve_sentinel2(spectra, wavelengths, load_sentinel2_srfs(srf_dir), SIMULATED_BANDS)
    table = pd.DataFrame({"gloria_id": rrs["GLORIA_ID"]})
    for band, values in bands.items():
        table[f"rrs_{band}"] = values

    model_columns = [f"rrs_{band}" for band in MODEL_BANDS]
    complete = table[model_columns].notna().all(axis=1)
    report["excluded_incomplete_b2_b7"] = int((~complete).sum())
    table = table[complete]

    meta = meta.set_index("GLORIA_ID")
    joined = meta.loc[table["gloria_id"]]
    table["dataset_id"] = joined["Dataset_ID"].to_numpy()
    table["water_body_type"] = (
        pd.to_numeric(joined["Water_body_type"], errors="coerce").map(WATER_BODY_TYPES).fillna("unknown").to_numpy()
    )
    latitude = pd.to_numeric(joined["Latitude"], errors="coerce").reset_index(drop=True)
    longitude = pd.to_numeric(joined["Longitude"], errors="coerce").reset_index(drop=True)
    table = table.reset_index(drop=True)
    table["latitude"], table["longitude"] = latitude, longitude
    table["group"] = spatial_group(latitude, longitude, table["dataset_id"])

    for target, (column, _unit, lower) in TARGETS.items():
        values = pd.to_numeric(joined[column], errors="coerce").to_numpy()
        # Non-positive values cannot be log-transformed and usually mean "below detection";
        # they are dropped rather than replaced with an invented small value.
        values = np.where(np.isfinite(values) & (values >= lower), values, np.nan)
        table[target] = values
        report[f"rows_with_{target}"] = int(np.isfinite(values).sum())

    report["training_spectra"] = int(len(table))
    report["spatial_groups"] = int(table["group"].nunique())
    report["datasets"] = int(table["dataset_id"].nunique())
    return table, report

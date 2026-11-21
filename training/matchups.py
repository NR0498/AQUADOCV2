"""Extract Sentinel-2 L2A pixels coincident with GLORIA in-situ samples (satellite match-ups).

    python -m training.matchups            # ~30-60 min, resumable; writes data/processed/gloria/s2_matchups.csv

Purpose: quantify what the deployed pipeline actually sees. For every GLORIA sample from
2017 onward, search for a Sentinel-2 acquisition within +/- MAX_HOURS, read a small window
centred on the sample, and keep it only if the centre 3x3 pixels are homogeneous, cloud-free
open water. The output pairs satellite reflectance with in-situ reflectance and lab values
for validation (training.validate_satellite). The product itself never needs lab data.

Protocol follows common ocean-colour match-up practice (e.g. Bailey & Werdell 2006, RSE 102):
small window, homogeneity test, time window, no clouds in the surrounding box.
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from aquadoc_core.s2_reader import REFLECTANCE_ASSETS, SCL_ASSET, CogBandReader, make_grid
from aquadoc_core.scene_search import search_scenes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "gloria" / "s2_training_table.csv"
META_PATH = PROJECT_ROOT / "data" / "external" / "gloria" / "extracted" / "GLORIA_2022" / "GLORIA_meta_and_lab.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gloria" / "s2_matchups.csv"

MAX_HOURS = 24.0
WINDOW_PX = 7          # 140 m box read around the sample
CORE_PX = 3            # 60 m centre used for the value
MAX_CORE_CV = 0.2      # coefficient of variation of B3/B4 in the core (homogeneity)
BANDS = list(REFLECTANCE_ASSETS.values())
FIELDS = ["gloria_id", "item_id", "platform", "acquired", "sample_time", "hours_apart", "cloud_cover",
          "status", *[f"rho_{b}" for b in BANDS], "core_cv_b3", "core_cv_b4"]


def load_samples() -> pd.DataFrame:
    table = pd.read_csv(TABLE_PATH)
    meta = pd.read_csv(META_PATH, low_memory=False).set_index("GLORIA_ID")
    table["sample_time"] = pd.to_datetime(meta.loc[table["gloria_id"], "Date_Time_UTC"].to_numpy(),
                                          errors="coerce", utc=True)
    keep = (table["sample_time"] >= "2017-01-01") & table["latitude"].notna() & table["longitude"].notna()
    return table[keep].reset_index(drop=True)


def evaluate_window(arrays: dict) -> tuple[str, dict]:
    scl = arrays[SCL_ASSET]
    c0 = (WINDOW_PX - CORE_PX) // 2
    core = (slice(c0, c0 + CORE_PX), slice(c0, c0 + CORE_PX))
    if not np.all(np.isfinite(scl)):
        return "no_data", {}
    if np.isin(scl, (3, 8, 9, 10, 11)).any():
        return "cloud_or_shadow_in_window", {}
    if not np.all(scl[core] == 6):
        return "core_not_water", {}
    rho = {REFLECTANCE_ASSETS[k]: arrays[k][core].astype(float) for k in REFLECTANCE_ASSETS}
    if any(not np.all(np.isfinite(v)) for v in rho.values()):
        return "no_data", {}
    if np.median(rho["B11"]) > 0.05 or np.median(rho["B3"]) <= np.median(rho["B11"]):
        return "glint_or_land", {}
    cv = {b: float(np.std(rho[b]) / np.mean(rho[b])) if np.mean(rho[b]) > 0 else np.inf for b in ("B3", "B4")}
    if max(cv.values()) > MAX_CORE_CV:
        return "inhomogeneous", {}
    values = {f"rho_{b}": float(np.median(rho[b])) for b in BANDS}
    values.update(core_cv_b3=round(cv["B3"], 4), core_cv_b4=round(cv["B4"], 4))
    return "ok", values


def process_group(rows: pd.DataFrame, reader: CogBandReader) -> list[dict]:
    first = rows.iloc[0]
    when = first["sample_time"]
    lat, lon = rows["latitude"].mean(), rows["longitude"].mean()
    d = 0.01
    start = (when - timedelta(hours=MAX_HOURS)).date().isoformat()
    end = (when + timedelta(hours=MAX_HOURS)).date().isoformat()
    try:
        candidates = search_scenes((lon - d, lat - d, lon + d, lat + d), start, end, max_cloud_cover=90, limit=20)
    except Exception as exc:
        return [{"gloria_id": g, "status": f"search_error:{type(exc).__name__}"} for g in rows["gloria_id"]]
    output = []
    for _, row in rows.iterrows():
        base = {"gloria_id": row["gloria_id"], "sample_time": row["sample_time"].isoformat()}
        timed = []
        for candidate in candidates:
            acquired = datetime.fromisoformat(candidate.datetime.replace("Z", "+00:00"))
            hours = abs((acquired - row["sample_time"].to_pydatetime()).total_seconds()) / 3600
            if hours <= MAX_HOURS:
                timed.append((hours, candidate))
        if not timed:
            output.append({**base, "status": "no_overpass"})
            continue
        result = None
        for hours, candidate in sorted(timed, key=lambda t: t[0]):
            grid = make_grid(row["latitude"], row["longitude"], WINDOW_PX * 20)
            try:
                arrays, _ = reader.read(candidate.item, grid, [*REFLECTANCE_ASSETS, SCL_ASSET])
            except Exception as exc:
                result = {**base, "status": f"read_error:{type(exc).__name__}"}
                continue
            status, values = evaluate_window(arrays)
            result = {**base, "item_id": candidate.item_id, "platform": candidate.platform,
                      "acquired": candidate.datetime, "hours_apart": round(hours, 2),
                      "cloud_cover": candidate.cloud_cover, "status": status, **values}
            if status == "ok":
                break
        output.append(result)
    return output


def run(workers: int = 4) -> Path:
    samples = load_samples()
    samples["group_key"] = samples["sample_time"].dt.date.astype(str) + "|" + samples["group"]
    done = set()
    if OUTPUT_PATH.exists():
        done = set(pd.read_csv(OUTPUT_PATH)["gloria_id"])
    todo = samples[~samples["gloria_id"].isin(done)]
    groups = [g for _, g in todo.groupby("group_key")]
    print(f"{len(samples)} S2-era samples, {len(done)} done, {len(groups)} groups to process")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUTPUT_PATH.exists()
    reader = CogBandReader(max_workers=4)
    with OUTPUT_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_group, g, reader) for g in groups]
            for index, future in enumerate(as_completed(futures), 1):
                for record in future.result():
                    writer.writerow(record)
                handle.flush()
                if index % 25 == 0:
                    print(f"{index}/{len(groups)} groups")
    summary = pd.read_csv(OUTPUT_PATH)["status"].value_counts()
    print(summary.to_string())
    return OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args().workers)

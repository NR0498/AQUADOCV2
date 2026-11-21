"""Reproducible, conservative staging of NLA 2017 chlorophyll observations.

A candidate passes these source checks; it is NOT a verified training example.
Date-only observations deliberately do not satisfy the timestamped DB importer.
"""

import csv
import hashlib
import json
import math
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import rasterio
from rasterio.warp import transform

VERSION = "nla2017-chla-v1"
STEMS = {
    "chemistry": "nla_2017_water_chemistry_chla",
    "sites": "nla_2017_site_information",
    "profiles": "nla_2017_profile",
}
REQUIRED = {
    "chemistry": {"UID", "SITE_ID", "VISIT_NO", "DATE_COL", "ANALYTE", "RESULT",
                  "RESULT_UNITS", "MDL", "RL", "NARS_FLAG", "LAB", "DATE_ANALYZED",
                  "MATRIX", "SAMPLE_TYPE", "SAM_CODE", "LAB_SAMPLE_ID"},
    "sites": {"UID", "SITE_ID", "VISIT_NO", "DATE_COL", "UNIQUE_ID", "SITESAMP",
              "LAT_DD83", "LON_DD83"},
    "profiles": {"UID", "SITE_ID", "VISIT_NO", "DATE_COL", "INDEX_LAT_DD",
                 "INDEX_LON_DD", "INDEX_GPS_FIX", "FLAG"},
}
FIELDS = [
    "observation_id", "source_row", "visit_uid", "site_id", "visit_no", "lake_id",
    "location_group", "parameter", "value", "unit", "mdl", "rl", "collected_date",
    "collection_time_precision", "collection_timezone", "analyzed_date", "lab_code",
    "lab_sample_id", "analytical_method", "sample_depth_m", "nars_flag",
    "latitude", "longitude", "coordinate_crs", "coordinate_source", "source_crs",
    "source_crs_basis", "index_latitude_raw", "index_longitude_raw", "gps_fix",
    "site_latitude_nad83", "site_longitude_nad83", "profile_rows", "profile_source_rows",
    "status", "review_reasons", "warnings", "transformations", "quality_status",
    "satellite_match_status", "source_record_json",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_date(value):
    """Source date formats only; no timezone or time-of-day imputation."""
    for fmt in ("%d-%b-%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Invalid source date: {value!r}")


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite numeric value")
    return result


def coordinate(value, latitude=False):
    text = value.strip()
    west = not latitude and re.fullmatch(r"[wW]\s*\d+(?:\.\d+)?", text)
    if west:
        text = "-" + text[1:].strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        raise ValueError("Ambiguous coordinate")
    result = number(text)
    limit = 90 if latitude else 180
    if not -limit <= result <= limit:
        raise ValueError("Coordinate outside geographic bounds")
    return result, bool(west)


def load_inputs(raw_dir, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    inventory = {entry["filename"]: entry for entry in manifest["files"]}
    tables, inputs = {}, []
    for role, stem in STEMS.items():
        for suffix in ("-data.csv", "-metadata.txt"):
            name = stem + suffix
            path = raw_dir / name
            if name not in inventory or digest(path) != inventory[name]["sha256"]:
                raise ValueError(f"Raw input checksum mismatch or missing inventory: {name}")
            inputs.append({"filename": name, "sha256": digest(path), "bytes": path.stat().st_size})
        encoding = "cp1252" if role == "profiles" else "utf-8-sig"
        with (raw_dir / (stem + "-data.csv")).open(encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED[role] - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing {role} columns: {sorted(missing)}")
            tables[role] = []
            for line, row in enumerate(reader, 2):
                if None in row or None in row.values():
                    raise ValueError(f"Malformed CSV in {role}, record {line}")
                tables[role].append((line, row))
    return tables, inputs


def preprocess(raw_dir="data/raw/nla2017", output_dir="data/processed/nla2017",
               manifest_path="data/manifests/nla2017_raw.json"):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.resolve() == raw_dir.resolve() or raw_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Processed output must be outside the raw directory")
    tables, inputs = load_inputs(raw_dir, manifest_path)
    selected = [(line, row) for line, row in tables["chemistry"] if row["ANALYTE"].strip().upper() == "CHLA"]
    if not selected:
        raise ValueError("No CHLA observations found")
    indexes = {}
    for role in ("sites", "profiles"):
        index = defaultdict(list)
        for line, row in tables[role]:
            index[row["UID"].strip()].append((line, row))
        indexes[role] = index
    uid_counts = Counter(row["UID"].strip() for _, row in selected)
    observations = []
    for line, source in selected:
        row = {key: value.strip() for key, value in source.items()}
        uid = row["UID"]
        reasons, warnings, changes = [], ["collection_time_unknown", "analytical_method_unknown", "sample_depth_unknown"], []
        out = dict.fromkeys(FIELDS, "")
        out.update(observation_id=f"nla2017:{uid}:{line}", source_row=line, visit_uid=uid,
                   site_id=row["SITE_ID"], visit_no=row["VISIT_NO"], parameter="chlorophyll_a",
                   unit="ug/L", collection_time_precision="date", lab_code=row["LAB"],
                   lab_sample_id=row["LAB_SAMPLE_ID"], nars_flag=row["NARS_FLAG"],
                   quality_status="provisional", satellite_match_status="unpaired",
                   source_record_json=json.dumps(source, sort_keys=True, ensure_ascii=False))
        if not uid or not row["SITE_ID"]:
            reasons.append("missing_identity")
        if uid_counts[uid] != 1:
            reasons.append("duplicate_chlorophyll_visit")
        if row["NARS_FLAG"]:
            reasons.append("nars_flag_requires_review")
        if (row["MATRIX"], row["SAMPLE_TYPE"], row["SAM_CODE"]) != ("WATER", "CHLX", "REGULAR"):
            reasons.append("unexpected_sample_context")
        unit = row["RESULT_UNITS"].upper().replace("µ", "Μ").replace("μ", "Μ")
        factor = {"UG/L": 1, "ΜG/L": 1, "MG/L": 1000}.get(unit)
        if factor is None:
            reasons.append("unsupported_unit")
            out["unit"] = ""
        else:
            changes.append("units_normalized_to_ug_L")
            for key, target in (("RESULT", "value"), ("MDL", "mdl"), ("RL", "rl")):
                if not row[key] and key != "RESULT":
                    warnings.append(f"missing_{target}")
                    continue
                try:
                    value = number(row[key]) * factor
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("Invalid concentration")
                    out[target] = value
                except ValueError:
                    reasons.append(f"invalid_{target}")
            if out["value"] != "":
                for limit in ("mdl", "rl"):
                    if out[limit] != "" and out["value"] < out[limit]:
                        reasons.append(f"below_{limit}")
        for source_key, target in (("DATE_COL", "collected_date"), ("DATE_ANALYZED", "analyzed_date")):
            if not row[source_key] and source_key == "DATE_ANALYZED":
                warnings.append("missing_analyzed_date")
                continue
            try:
                out[target] = parse_date(row[source_key])
            except ValueError:
                reasons.append(f"invalid_{target}")
        if out["collected_date"] and not out["collected_date"].startswith("2017-"):
            reasons.append("collection_outside_2017")
        if out["analyzed_date"] and out["collected_date"] and out["analyzed_date"] < out["collected_date"]:
            reasons.append("analysis_before_collection")
        sites = indexes["sites"].get(uid, [])
        profiles = indexes["profiles"].get(uid, [])
        if len(sites) != 1:
            reasons.append("site_join_not_one_to_one")
        else:
            site = sites[0][1]
            out.update(lake_id=site["UNIQUE_ID"].strip(),
                       site_latitude_nad83=site["LAT_DD83"], site_longitude_nad83=site["LON_DD83"])
            if not out["lake_id"]:
                reasons.append("missing_lake_id")
            else:
                out["location_group"] = "nla2017:" + out["lake_id"]
            if site["SITESAMP"].strip() != "Y":
                reasons.append("site_not_sampled")
        # Validate all joined rows before reducing the many-depth profile table.
        for role, joined in (("site", sites), ("profile", profiles)):
            for _, joined_row in joined:
                try:
                    agrees = (joined_row["SITE_ID"].strip() == row["SITE_ID"]
                              and joined_row["VISIT_NO"].strip() == row["VISIT_NO"]
                              and parse_date(joined_row["DATE_COL"]) == out["collected_date"])
                except ValueError:
                    agrees = False
                if not agrees:
                    reasons.append(f"{role}_identity_or_date_mismatch")
        out["profile_rows"] = len(profiles)
        coords = [(n, p) for n, p in profiles if p["INDEX_LAT_DD"].strip() or p["INDEX_LON_DD"].strip()]
        out["profile_source_rows"] = ";".join(str(n) for n, _ in coords)
        pairs = {(p["INDEX_LAT_DD"].strip(), p["INDEX_LON_DD"].strip()) for _, p in coords}
        if not pairs:
            reasons.append("missing_index_coordinates")
        elif len(pairs) != 1:
            reasons.append("conflicting_index_coordinates")
        else:
            lat_raw, lon_raw = next(iter(pairs))
            out.update(index_latitude_raw=lat_raw, index_longitude_raw=lon_raw,
                       gps_fix=";".join(sorted({p["INDEX_GPS_FIX"].strip() for _, p in coords})),
                       coordinate_source="profile_index_gps", source_crs="EPSG:4269",
                       source_crs_basis="NLA 2017 field protocol; per-record datum not supplied")
            if any(p["FLAG"].strip() for _, p in coords):
                reasons.append("profile_coordinate_row_flagged")
            try:
                lat, _ = coordinate(lat_raw, latitude=True)
                lon, west = coordinate(lon_raw)
                if west:
                    changes.append("explicit_west_prefix_to_negative_longitude")
                # NLA 2017 covers the conterminous United States: never guess a missing sign.
                if lat <= 0 or lon >= 0:
                    reasons.append("index_coordinate_hemisphere_mismatch")
                xs, ys = transform("EPSG:4269", "EPSG:4326", [lon], [lat])
                out.update(latitude=number(str(ys[0])), longitude=number(str(xs[0])), coordinate_crs="EPSG:4326")
                changes.append("NAD83_to_WGS84_via_GDAL")
                warnings.append("gps_datum_assumed_from_survey_protocol")
            except ValueError:
                reasons.append("invalid_index_coordinates")
        out.update(status="review" if reasons else "candidate", review_reasons=";".join(sorted(set(reasons))),
                   warnings=";".join(sorted(set(warnings))), transformations=";".join(sorted(set(changes))))
        observations.append(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for filename, rows in (("chlorophyll_observations.csv", observations),
                           ("chlorophyll_candidates.csv", [r for r in observations if r["status"] == "candidate"]),
                           ("chlorophyll_review.csv", [r for r in observations if r["status"] == "review"])):
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        outputs.append({"filename": filename, "rows": len(rows), "sha256": digest(path)})
    report = {
        "pipeline_version": VERSION, "implementation_sha256": digest(__file__),
        "input_inventory_sha256": digest(manifest_path), "inputs": inputs, "outputs": outputs,
        "runtime": {"python": platform.python_version(), "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__, "proj": rasterio.__proj_version__},
        "encoding": {"chemistry": "utf-8-sig", "sites": "utf-8-sig", "profiles": "cp1252"},
        "counts": {"chemistry_rows": len(tables["chemistry"]), "non_chla_rows_filtered": len(tables["chemistry"]) - len(selected),
                   "chlorophyll_rows": len(observations), "candidate_rows": sum(r["status"] == "candidate" for r in observations),
                   "review_rows": sum(r["status"] == "review" for r in observations),
                   "unique_lakes": len({r["lake_id"] for r in observations if r["lake_id"]})},
        "review_reason_counts": dict(sorted(Counter(reason for r in observations for reason in r["review_reasons"].split(";") if reason).items())),
        "warning_counts": dict(sorted(Counter(reason for r in observations for reason in r["warnings"].split(";") if reason).items())),
        "transformation_counts": dict(sorted(Counter(reason for r in observations for reason in r["transformations"].split(";") if reason).items())),
        "limitations": ["Candidates remain provisional and unpaired, not training-ready or bloom classifications.",
                        "Collection dates lack exact time and timezone; do not import as fabricated UTC timestamps.",
                        "Nonblank NARS flags require review; the meaning of S has not been resolved.",
                        "GPS datum follows survey protocol, not per-record confirmation; no precision claim.",
                        "Lake polygon containment and satellite footprint/cloud/shoreline checks remain pending.",
                        "No target imputation, clipping, log transform, statistical outlier removal, or split performed.",
                        "Review reason counts overlap; all CHLA source records are retained."],
    }
    (output_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

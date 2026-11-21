# NLA 2017 chlorophyll-a preprocessing

Implemented and run on 2026-10-22 using `aquadoc_core/nla_preprocessing.py`, pipeline version `nla2017-chla-v1`. This stage creates normalized, auditable observations for subsequent satellite pairing. Candidates remain provisional; they are not approved training examples or algal-bloom classifications.

## Run it

From the repository root, with the project's Python dependencies installed:

```sh
python -m aquadoc_core.cli preprocess-nla2017
```

Optional arguments: `--raw-dir`, `--output-dir`, and `--manifest`. Defaults are `data/raw/nla2017`, `data/processed/nla2017`, and `data/manifests/nla2017_raw.json`. Output filenames are replaced on rerun. The pipeline rejects writing inside its raw input directory. Missing required columns, malformed CSV structure, checksum mismatches, or an empty chlorophyll selection stop the run.

## Results

| Stage | Records |
| --- | ---: |
| Original chemistry table | 22,873 |
| Other analytes filtered out | 21,663 |
| Chlorophyll-a observations retained in audit table | 1,210 |
| Candidates passing implemented source checks | 1,145 |
| Records held for review | 65 |
| Lakes across all observations | 1,113 |
| Lakes represented among candidates | 1,055 |

Collection dates span 2017-05-02 through 2017-10-10. Both the full observation table and candidate table retain concentrations spanning 0.1–3,299.18 µg/L. High values were not clipped or removed just because they are extreme.

| Review reason | Records |
| --- | ---: |
| Nonblank NARS quality flag (`S`) | 38 |
| Result below its reported method detection limit (MDL) | 14 |
| Invalid or ambiguous index-site coordinates | 10 |
| Missing index-site coordinates | 5 |

Reasons overlap: these counts describe **65 distinct records**, not 67. The malformed-coordinate group includes one record containing only a dot in latitude and no longitude. No observations fell below a supplied reporting limit (RL). The meaning of `S` has not been established from the supplied metadata; it is retained verbatim and held for review, not reinterpreted as a specific failure.

## What preprocessing does

1. **Verify provenance.** Check six consumed input files (chemistry, sites, profiles and their metadata) against the existing SHA-256 inventory before processing. Leave all raw files unchanged. Record input/output hashes, implementation hash, pipeline version and Python/Rasterio/GDAL/PROJ versions in the report.
2. **Decode explicitly.** Read chemistry and sites as UTF-8 with optional BOM, and profiles as Windows-1252. Write normalized outputs as UTF-8. Raw source bytes remain intact.
3. **Select the exact target.** Trim and uppercase the analyte code, then select exactly `CHLA`. Chloride (`CHL`) and other analytes do not enter the target table. Preserve the original chemistry record as JSON alongside its source CSV record number.
4. **Join without multiplying observations.** Join by visit `UID`; require exactly one site record. Validate site ID, visit number and collection date across joined records. Reduce multiple depth-profile rows to one distinct index GPS pair. Missing/conflicting joins and duplicate chlorophyll visits go to review; they are not silently overwritten or averaged. The real inputs had no such join conflicts.
5. **Normalize units and numeric values.** Convert the source `UG/L` label to `ug/L`. The parser also supports mg/L, converting the result and both limits together. All actual records were already µg/L, so their concentrations did not change. Reject non-finite or negative values from candidates. Keep missing limits missing: 63 records lack MDL and 1,156 lack RL. A missing limit does not prove the observation is above that limit.
6. **Preserve censoring and quality flags.** Keep measured values below MDL/RL in the audit/review files; do not replace them with zero or half the limit. Preserve NARS flags and inspect flags on coordinate-bearing profile rows. Require the expected water/CHLX/regular sample context.
7. **Normalize dates without inventing time.** Parse source date formats to ISO `YYYY-MM-DD`. Preserve `collection_time_precision=date` and leave timezone blank. Keep the 32 missing analysis dates blank and report them as warnings. Do not use profile arrival time as the exact chlorophyll collection time. Invalid dates, collection outside 2017 and analysis before collection require review.
8. **Use index-site GPS rather than lake centroids.** Preserve raw GPS strings, GPS fix information and profile record numbers. Convert two unambiguous west prefixes (`w94.03035`, `w91.76919`) into negative longitude. Do not guess values with doubled decimal points, question marks, slashes or embedded spaces. Preserve site-selection coordinates separately; never silently substitute them for missing sampling coordinates.
9. **Record coordinate reference assumptions.** Interpret index GPS as NAD83 under the NLA field protocol and transform EPSG:4269 to WGS84/EPSG:4326 with Rasterio/GDAL. Per-record datum confirmation and positional accuracy are unavailable, so every successfully parsed GPS pair carries an explicit assumption warning. Depending on the available coordinate operation, values may remain numerically unchanged; this is not a claim of sub-metre precision. Check geographic bounds and the survey's northern/western hemisphere context. Polygon containment remains pending.
10. **Retain lake grouping and uncertainty.** Use NLA `UNIQUE_ID` for `location_group`, so repeat visits can stay together in future evaluation splits. Leave exact analytical method and chlorophyll sample depth unknown. A profile depth or total lake depth is not automatically the laboratory sample's integration depth. All records remain provisional and unpaired.

## Output files

Generated files in `data/processed/nla2017/` are ignored by Git:

| File | Contents |
| --- | --- |
| `chlorophyll_observations.csv` | All 1,210 CHLA observations, normalized fields, original chemistry record, provenance, warnings, transformations and row-level decisions |
| `chlorophyll_candidates.csv` | 1,145 observations passing the implemented checks |
| `chlorophyll_review.csv` | 65 observations with explicit review reasons; no source observations are deleted |
| `preprocessing_report.json` | Counts, overlapping reason counts, warnings, transformations, hashes, versions and limitations |

A reviewable copy of the report is saved at [data/manifests/nla2017_preprocessing.json](../../data/manifests/nla2017_preprocessing.json). That copy records this run; the CLI writes future reports into its selected output directory. Observation IDs combine dataset, visit UID and source record number; source hashes identify the exact input revision.

The original bundle and selected raw directory also remain ignored. Source code, tests, documentation and small manifests are available for version control.

## Verification and next boundary

26 focused tests passed: ten new preprocessing tests plus existing CLI, CSV importer and observation-schema tests. Tests cover exact analyte selection, join multiplication, date precision, duplicate/conflicting joins, quality flags, censoring, malformed GPS, units/limits, invalid values, checksum failure, deterministic output and the CLI. Two runs on the real data produced byte-identical CSVs and JSON reports. All 15 organized raw files still match their original hashes. Every CHLA record is accounted for exactly once in the candidate/review partition.

The existing database importer requires exact timezone-aware collection and analysis timestamps and a laboratory method. These exports intentionally do not fabricate those fields and must not be fed directly to that importer. The next integration needs date-precision-aware observation storage and satellite date-window matching, followed by lake-boundary, cloud, shoreline and spatial-footprint checks. Resolve flagged records separately using source documentation or corrected source data.

No satellite data were downloaded, model fitted, database changed, or dashboard built in this step. No imputation, scaling, target log transform or train/test split was applied. Fit learned preprocessing only on training folds when modeling begins. Chlorophyll-a concentration alone is not a verified harmful-bloom or toxin label. Supporting Secchi, toxin, taxonomy and lake-polygon files remain available for later steps.

## Source documentation

- [EPA NLA data catalogue](https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys).
- [NLA 2017 Field Operations Manual](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100YX4L.TXT): the survey specifies NAD83 GPS setup and decimal-degree coordinates; exceptions require documentation.
- [NLA 2017 Laboratory Operations Manual](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100YX9O.TXT): laboratory protocols provide context but do not identify the exact method used for each exported row.
- Companion metadata in the raw directory identifies fields and NAD83 site-selection coordinates, which may be lake centroids.

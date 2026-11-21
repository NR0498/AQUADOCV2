# AquaDoc project audit and pipeline roadmap

Audit date: 2026-10-22. Scope: README, `learning.md`, all current backend modules and tests, the legacy Streamlit/model/helpers, dependency/deployment files, the training CSV and manifest, and read-only inspection of all bundled SQLite databases.

## Assessment

AquaDoc already has a useful research backend: satellite ingestion, explainable features, measurement import, persisted runs, human match review, reproducible export, and grouped baseline training. Keep that structure and improve its correctness. The main constraint is trustworthy matched observations, followed by image quality and evaluation. Replacing Streamlit alone will not resolve those constraints.

The first selected target is **chlorophyll-a concentration (`chlorophyll_a`, `ug/L`)**. Bloom classification needs separate, documented labels. Do not convert the current NDCI heuristic into supervised ground truth. Do not equate chlorophyll estimates with harmful species, toxins, or drinking-water safety.

`learning.md` is a chronological journal: early statements that the project lacks ingestion/API/storage were superseded by later sections. The current code and the latest journal sections are the appropriate starting point.

## Dataset inspection

| Artifact | Actual contents | Implication |
| --- | --- | --- |
| `data/measurement_import_template.csv` | Two example rows: field turbidity and laboratory chlorophyll-a | A format demonstration, not field evidence; both rows nevertheless carry `quality=verified` |
| `data/training/turbidity.csv` | One row, one location, 48 features, target 18.2 NTU, `example-field-001` | No usable training corpus, and no chlorophyll training export |
| CSV manifest | Row count and SHA-256 match the file | File integrity is valid; scientific suitability is not established by a hash |
| `data/aquadoc.sqlite3` | One analysis run; zero sampling events, field/lab measurements, or matches | Main database contains no training labels |
| `data/experiment.sqlite3` | Empty analysis table | Experimental artifact |
| `data/import-test.sqlite3` | Two example events, one field result, one lab result, one turbidity match | Demonstration data only; its match table predates review columns |

The one exported match pairs a sample at `2026-06-15T05:00:00Z` with scene `S2B_MSIL2A_20260512T050649_R019_T43PGQ_20260512T085408`, captured on May 12. The recorded gap is **815.886 hours, about 34 days**, far beyond the usual configured ±3-day search. The existing record does not establish how that historical match was produced; the previous matcher relied on the provider's date filtering without independently enforcing the tolerance.

Its `valid_fraction` is **0.0036 (0.36%)**. That is only the current NDWI/finite-pixel support fraction, not a cloud-free water-quality assessment. A tiny set of selected pixels cannot justify treating a large chip as a representative sample. The geometry was not checked against a water-body polygon, so this audit makes no claim that the example coordinate identifies suitable open water.

Existing databases and dataset artifacts were preserved. They must not be treated as a newly validated corpus after code changes.

## Findings and priorities

| Priority | Finding | Required improvement | Status |
| --- | --- | --- | --- |
| P0 | Zero selected water pixels produced all-zero statistics and could return `likely_safe` | Reject unusable scenes; never interpret missing evidence as low concern | Empty/invalid-scene guard implemented in this pass |
| P0 | Scene matching trusted remote search dates | Enforce elapsed-time tolerance locally before reading rasters | Implemented in this pass |
| P0 | Median imputation used the whole dataset before grouped splitting | Fit preprocessing only on training groups and save it with the estimator | Implemented in this pass |
| P0 | Numeric observations could contain NaN/infinity | Reject non-finite values at observation schemas and CSV import | Implemented in this pass |
| P0 | Training recomputed a hash without checking the supplied manifest | Require matching hash, row count, feature columns, parameter, and feature version | Implemented in this pass |
| P0 | No real reviewed chlorophyll dataset | Acquire source observations, retain raw data and metadata, normalize with source-specific QA | Pending |
| P0 | No pixel cloud/shadow/nodata quality mask | Ingest Sentinel-2 SCL and raster validity masks before extracting features | Pending |
| P0 | Labels `likely_safe`/`unsafe` imply more than heuristic optics establish | Replace with screening concern levels plus explicit data sufficiency and estimate type | Pending API schema revision |
| P1 | Bands are resized to equal array shapes using independently rounded windows | Reproject every band onto one explicit CRS, transform, bounds, and pixel grid | Pending |
| P1 | Reflectance scaling is inferred from maximum value and division by 10,000 | Apply provider/asset scale and offset exactly once, mask nodata first, retain processing baseline | Pending |
| P1 | The scene query limits results to ten sorted by cloudiness | Paginate or query/rank candidates by temporal proximity and point coverage; handle duplicates and missing assets | Pending |
| P1 | A whole 1 km chip supplies summaries for a point measurement | Define a small open-water sampling window, shoreline buffer, valid-pixel support, and representativeness checks | Pending |
| P1 | Export identifies location by rounded coordinates | Preserve source station and lake IDs; prevent nearby/overlapping chips and shared scenes leaking across evaluation splits | Pending |
| P1 | One train/test split and no reference predictor or uncertainty | Add grouped validation, temporal holdout, dummy baseline, concentration-stratified errors and interval coverage | Pending |
| P1 | Review decision overwrites previous decision and permits an empty note | Preserve reviewer identity, decision history, note, pixel-quality evidence, and source version | Pending |
| P1 | One match per measurement cannot coexist across feature/config versions | Store matching policy and extraction versions; permit explicit rematching without destroying older evidence | Pending |
| P1 | SQLite stores only summaries and partial scene provenance | Retain asset IDs, geometry, scale/offset, masks, configuration hash, code version, source QA and artifact pointers | Pending |
| P1 | API performs satellite work synchronously and exposes only analysis routes | Introduce durable job states, retry/idempotency, inference/model metadata and controlled review/import routes | Pending |
| P2 | Legacy UI trains on synthetic image patches and computes illustrative accuracy-like values | Keep it a demo; build the eventual dashboard from real API results and held-out metrics | Deferred until dashboard phase |
| P2 | Broad dependencies mix UI, training, ingestion and API; project metadata is placeholder text | Separate dependency groups, pin a supported Python version, refresh lockfile, test a clean install | Pending |
| P2 | `.venv` has about 23,000 tracked files; no project `.gitignore` | Add ignore rules, then separately remove tracked generated files from the index without deleting the local environment | Pending repository cleanup |

The scale/offset requirement follows [Copernicus Sentinel-2 product definitions](https://sentiwiki.copernicus.eu/web/s2-products). Confirm how the chosen provider represents these values before applying any correction. Equal array dimensions alone do not prove geographic alignment.

## Pipeline to build before a dashboard

1. **Define the chlorophyll dataset contract.** Preserve station/lake ID, coordinates and uncertainty, collection timestamp and precision, sample depth/integration method, chlorophyll analytical method, corrected versus uncorrected result, unit, detection/quantification limits, qualifiers, source QA, source ID and licence. Never invent sample times, laboratory names or analysis times when absent. The current importer requires a laboratory analysis timestamp; adapt the schema with explicit missingness before importing a source that does not publish it.
2. **Acquire a small real pilot.** Inspect [EPA National Lakes Assessment data](https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys) for 2017/2022 observations and [Water Quality Portal](https://www.waterqualitydata.us/) for chlorophyll records overlapping Sentinel-2 availability. These are candidates, not adopted datasets. Check source methods and flags, timestamp precision, geography, and sampling representativeness before selecting one. Preserve raw files and checksums. The first geography remains to be chosen from suitable coverage.
3. **Make imagery trustworthy.** Select a common grid (a 20 m analysis grid is a candidate for the red-edge/SCL bands), preserve transform/CRS, apply metadata-defined radiometry, and mask cloud, shadow, snow, saturated and nodata pixels. Resample categorical SCL with nearest neighbour. Test with local GeoTIFF fixtures containing different grids and nodata boundaries. Add a configurable, documented sample-window and valid-water support policy; do not use a universal fraction as scientific validation.
4. **Build auditable sample/scene pairs.** Apply an explicit chlorophyll time window (start conservatively and validate its sensitivity), real extraction distances, sample-depth/method rules, QA metrics and shoreline checks. Save rejection reasons and candidate alternatives. Human review should see the actual chip, masks, support and source evidence. Bump the feature version when masking or feature definitions change.
5. **Export and validate a versioned dataset.** Exclude examples and rejected/provisional evidence under an explicit source policy. Include reviewer decisions, source/method/units, station/lake/scene identities, feature/config versions, missingness, target range and concentration distribution. Validate the manifest and report coverage by lake and season. Reject mixed feature versions and incompatible methods rather than silently merging them.
6. **Evaluate a chlorophyll baseline.** Compare a dummy predictor and Random Forest, consider a documented log target transform for skewed positive concentrations, fit every transform within training folds, and reserve unseen lakes and future periods. Report MAE/RMSE in `ug/L`, per-lake and concentration-range errors, and interval coverage. Pickle artifacts now include fitted imputation; add environment versions and model cards before model promotion. Set acceptance criteria after reviewing real sample coverage rather than inventing an accuracy target.
7. **Make the backend deployable.** Introduce shared storage, explicit migrations, durable jobs, model/version metadata, request limits, access control for writes/review, structured failure responses and an end-to-end pilot. Training remains an explicit research job; API startup and dashboard requests must not train a model.
8. **Design and build the dashboard last.** Establish the API contract first. Show maps, chlorophyll estimates and units, acquisition dates, quality/support, uncertainty, provenance, job progress and trends using real results. Add bloom views only when a separate validated label definition exists.

## Vercel architecture

Retain Python/FastAPI for the scientific backend and plan Next.js/React/TypeScript for the final dashboard. This avoids rewriting NumPy/Rasterio/scikit-learn work in JavaScript. [Vercel supports FastAPI directly](https://vercel.com/docs/frameworks/backend/fastapi), including an explicit module entrypoint; its documentation also describes deploying a frontend with a Python backend.

Target flow:

```text
Future dashboard on Vercel
  → FastAPI: validate request and create job
  → shared PostgreSQL: job state, source records, matches, reviews, results
  → durable worker/workflow: catalogue search → chip read → quality gate → features → inference
  → object storage: raster chips, masks, manifests, approved model artifacts
  → API: return job progress and reproducible completed result
```

This is a proposed architecture; PostgreSQL, object storage, job infrastructure, Next.js and deployment configuration are not implemented yet. Provider selection should follow measurements of dependency bundle size, chip memory, job duration, retry behavior and project budget. Slow work can use a Vercel-compatible durable workflow or a separate worker; that decision does not require returning to Streamlit.

Current deployment blockers:

- The root `app.py` is Streamlit; `aquadoc_core.api:app` is the actual FastAPI object. Configure an explicit backend entrypoint during deployment work.
- A local SQLite file is not shared durable state across function instances. Moving it to temporary storage would not solve persistence; use an external store. Separate preview and production databases/artifact prefixes.
- Split API/runtime dependencies from legacy UI and offline training dependencies. Verify Rasterio/GDAL wheels against the selected Linux Python runtime and measure the built bundle.
- Keep raster arrays out of normal JSON responses. Return summaries and object pointers; serve large chips/exports through object storage.
- Account for cold starts, request termination, retries and concurrency. In-process background tasks alone do not make jobs durable.

At audit time the [official function limits](https://vercel.com/docs/functions/limitations) list a 500 MB standard Python bundle, a 4.5 MB request/response payload limit, and a 300-second Hobby duration limit with Fluid compute. Eligibility for larger bundles or longer execution varies; verify the selected plan and current limits when deploying. No deployment was attempted during this audit.

## Compatibility notes for this implementation pass

- Empty/no-water/fully invalid scenes now raise `ValueError`; the existing API translates these to HTTP 400. They no longer create an apparent low-risk analysis.
- `train-baseline` now requires the adjacent `.csv.manifest.json`. Re-export datasets that were manually edited. A matching manifest proves integrity, not observation quality.
- `artifact['model']` is now a fitted scikit-learn Pipeline containing an imputer and estimator. It can process missing feature values after reload; consumers that previously expected direct forest-specific attributes must access the estimator step.
- Matching now enforces elapsed-time tolerance even if a catalogue returns out-of-window scenes. This does not repair historical matches automatically.
- New matches use `spectral-statistics-v2` because jointly filtering non-finite pixels changes feature support. Historical exports retain their actual version, and exports reject mixed versions. The planned SCL/grid/radiometry revision will require another feature version and fresh extraction.

See `study.md` for the technology journal and learning resources. Verification results for this pass are recorded there after running the backend suite.

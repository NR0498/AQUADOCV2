# AquaDoc study and implementation journal

This journal records technologies as we use them, why they are needed, and where to learn them. `learning.md` preserves the earlier build history; this file tracks the current work. Update it in each implementation phase.

## Project direction

- First scientific target: **chlorophyll-a concentration**, stored as `chlorophyll_a` in `ug/L`.
- Algal-bloom classification is a separate task requiring a documented label definition and suitable observations. A chlorophyll estimate alone does not establish harmful species or toxin presence.
- Order: reliable data pipeline → validated model → deployment-ready API → dashboard design and implementation → Vercel deployment.
- Dashboard work has not started.

## 2026-10-22 — Repository audit and pipeline correctness

| Technology / concept | Where we use it | What to learn | Official resource |
| --- | --- | --- | --- |
| Python and dataclasses | `aquadoc_core/schemas.py` | Typed structures, array validation, reusable modules | [Python dataclasses](https://docs.python.org/3/library/dataclasses.html) |
| NumPy | `pipeline.py`, `spectral_indices.py` | Arrays, finite values, boolean masks, percentiles | [NumPy beginner guide](https://numpy.org/doc/stable/user/absolute_beginners.html) |
| Sentinel-2 Level-2A | Satellite inputs | Surface reflectance, band resolutions, processing baselines, scene classification | [Copernicus product documentation](https://sentiwiki.copernicus.eu/web/s2-products) |
| STAC and Cloud Optimized GeoTIFF | `ingestion.py` | Search metadata first; read only a raster window | [STAC specification](https://stacspec.org/en), [Rasterio](https://rasterio.readthedocs.io/en/stable/) |
| UTC timestamps | Measurement/scene matching | Compare elapsed time explicitly, even when a search provider returns unsuitable results | [Python datetime](https://docs.python.org/3/library/datetime.html) |
| SQLite and read-only inspection | Existing local databases | Tables, joins, transactions, separating example data from real observations | [Python sqlite3](https://docs.python.org/3/library/sqlite3.html) |
| Pydantic | Observation and API schemas | Reject non-finite measurements and invalid input before persistence | [Pydantic field constraints](https://docs.pydantic.dev/latest/concepts/fields/) |
| scikit-learn Pipeline and SimpleImputer | `training_pipeline.py` | Split by location before fitting preprocessing; save preprocessing with the estimator | [Common pitfalls and leakage](https://scikit-learn.org/stable/common_pitfalls.html), [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) |
| SHA-256 and JSON manifests | Dataset export and training | Check exact input bytes before trusting a dataset manifest | [Python hashlib](https://docs.python.org/3/library/hashlib.html) |
| unittest and temporary fixtures | `tests/` | Regression tests for missing imagery, date-window violations, and held-out-data leakage | [Python unittest](https://docs.python.org/3/library/unittest.html) |
| uv and virtual environments | Isolated backend verification | Keep test dependencies outside the existing Windows `.venv` | [uv environments](https://docs.astral.sh/uv/pip/environments/) |

### Changes in this phase

- Audit the existing CSV, its manifest, all three SQLite databases, and current/legacy code paths.
- Reject scenes without usable finite water pixels instead of generating a low-risk result from zero-filled summaries.
- Enforce the actual measurement-to-scene time window locally.
- Fit missing-value preprocessing on training locations only and save it with the trained model.
- Reject NaN/infinite observation values and check exported dataset manifests before training.
- Version the changed pixel support as `spectral-statistics-v2`; reject exports that mix feature versions and preserve the identity of historical exports.
- Record the remaining scientific and deployment gaps in `docs/project-audit.md`.

Tests use generated fixtures to verify software behavior. Their scores are not evidence of real chlorophyll prediction performance.

### Verification

`python -m unittest discover -s tests -v`: **42 tests passed** after the changes, including API/CLI persistence flows and the new correctness regressions. All 23 backend/test Python files also parsed successfully. Verification used a separate `/tmp` environment because the repository's `.venv` is a Windows environment. The HTTP test client required running outside the workspace sandbox.

Tested versions: Python 3.12.15, NumPy 1.26.4, pandas 3.0.6, scikit-learn 1.9.1, FastAPI 0.159.1, Pydantic 2.14.1, HTTPX 0.28.1, Rasterio 1.4.4. This records the test environment; it does not replace the pending dependency/lockfile cleanup. FastAPI's test-client dependency emitted a deprecation warning that should be handled in that cleanup.

Satellite calls were injected in tests. Live imagery quality, a real chlorophyll model, and a Vercel deployment remain unverified. Existing example datasets/databases were inspected read-only and retained.

## 2026-10-22 — First raw NLA 2017 dataset

Selected 15 files from the locally downloaded `NLA 2017 Report Data Files/` bundle and moved them to `data/raw/nla2017/`, preserving original filenames and bytes. The selection includes chemistry, site information, profile, lake polygons, Secchi depth, phytoplankton counts/taxa, algal toxins, and their companion metadata. The remaining 30 files stay in the original folder. Both directories are excluded by the new root `.gitignore`.

The raw chemistry table contains 1,210 exact `CHLA` rows across 1,113 site IDs; all have raw units `UG/L`. There are 38 `S` quality flags requiring interpretation. The profile file needs an explicit legacy-text encoding policy, and its GPS coordinates are more relevant to sample matching than assuming site-selection lake centroids are sampling coordinates. No cleaning, database import, or satellite matching was performed in this organization step.

| Technology / concept | How it was used | Learning resource |
| --- | --- | --- |
| Python pathlib | Move a selected file list after checking sources and refusing destination overwrites | [pathlib](https://docs.python.org/3/library/pathlib.html) |
| CSV and text encoding | Inspect headers, exact analyte codes, units, row counts, and UTF-8 versus Windows-1252 decoding without changing the source files | [csv](https://docs.python.org/3/library/csv.html), [Python Unicode guide](https://docs.python.org/3/howto/unicode.html) |
| SHA-256 and a JSON inventory | Compare hashes before/after each move; record original names, sizes, roles, and source provenance | [hashlib](https://docs.python.org/3/library/hashlib.html), [json](https://docs.python.org/3/library/json.html) |
| Git ignore patterns | Keep local raw files out of Git while retaining a small reviewable inventory and dataset guide | [gitignore](https://git-scm.com/docs/gitignore), [git check-ignore](https://git-scm.com/docs/git-check-ignore) |

See `docs/datasets/nla2017.md` for the selected files and preprocessing considerations, and `data/manifests/nla2017_raw.json` for the file-level inventory. Verification for this step consists of checksum, move, and Git-ignore checks; no application code changed.

## Planned technologies — not implemented in this phase

| Phase | Technology / concept | Reason | Learning resource |
| --- | --- | --- | --- |
| Imagery quality | Rasterio reprojection and Sentinel-2 SCL | One geographic grid; nodata/cloud/shadow rejection; nearest-neighbour resampling for categorical masks | [Rasterio reprojection](https://rasterio.readthedocs.io/en/stable/topics/reproject.html), [Copernicus processing](https://sentiwiki.copernicus.eu/web/s2-processing) |
| Real-label preprocessing | EPA National Lakes Assessment / Water Quality Portal | NLA 2017 raw files are now organized locally; implement source-specific normalization and quality checks before adopting observations for training | [EPA NLA data](https://www.epa.gov/national-aquatic-resource-surveys/data-national-aquatic-resource-surveys), [WQP](https://www.waterqualitydata.us/) |
| Evaluation | Grouped cross-validation and temporal holdout | Test unseen lakes and future observations; compare against a dummy predictor | [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html) |
| Persistent deployment storage | PostgreSQL, optional PostGIS, migrations | Share metadata across API instances; retain local SQLite for development | [PostgreSQL tutorial](https://www.postgresql.org/docs/current/tutorial.html), [PostGIS workshop](https://postgis.net/workshops/postgis-intro/) |
| Raster and artifact storage | Object storage | Store chips, manifests, and versioned models outside a function filesystem | [Vercel Blob](https://vercel.com/docs/vercel-blob) |
| API deployment | FastAPI on Vercel | Reuse the Python backend behind a stable HTTP contract | [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/), [FastAPI on Vercel](https://vercel.com/docs/frameworks/backend/fastapi) |
| Durable processing | Job records and a queue/workflow, provider to be selected | Satellite reads, retries, and training must survive request termination | [Vercel Functions limits](https://vercel.com/docs/functions/limitations), [Vercel Workflow](https://vercel.com/docs/workflows) |
| Dashboard, last | Next.js, React, TypeScript; map library to be selected | Display real job state, maps, chlorophyll estimates, evidence, and uncertainty | [Next.js learning](https://nextjs.org/learn), [React learning](https://react.dev/learn), [TypeScript handbook](https://www.typescriptlang.org/docs/handbook/intro.html) |

Suggested study order: NumPy and masks → satellite bands and grids → observation quality and SQL → leakage-free ML → FastAPI → persistent jobs/storage → React and Next.js.

## 2026-10-22 — Chlorophyll preprocessing implemented

Built `aquadoc_core/nla_preprocessing.py` and the `preprocess-nla2017` CLI command. See [the preprocessing report](docs/datasets/nla2017-preprocessing.md) for the complete decisions and output schema. The run preserved all 1,210 chlorophyll observations: 1,145 candidates across 1,055 lakes and 65 records for review. These are provisional observations, not trained-model inputs yet.

| Technology / concept | Where and why used | Learn it |
| --- | --- | --- |
| Python `csv`, `pathlib`, Unicode encodings | Read source tables with explicit UTF-8/Windows-1252 decoding; write portable CSV outputs | [CSV](https://docs.python.org/3/library/csv.html), [Unicode guide](https://docs.python.org/3/howto/unicode.html) |
| Dictionary joins, `defaultdict`, `Counter` | Validate one-to-one site joins, reduce many depth rows, detect duplicate visits and count review reasons | [Collections](https://docs.python.org/3/library/collections.html) |
| `datetime.strptime` and date precision | Normalize dates without inventing UTC timestamps; keep missing analysis dates missing | [Datetime](https://docs.python.org/3/library/datetime.html) |
| Regular expressions and numeric validation | Normalize explicit west prefixes while refusing ambiguous GPS strings and non-finite concentrations | [Regular expressions](https://docs.python.org/3/howto/regex.html), [math.isfinite](https://docs.python.org/3/library/math.html#math.isfinite) |
| Rasterio/GDAL/PROJ coordinate transformation | Convert protocol-assumed NAD83 index GPS to WGS84 and preserve the datum assumption | [Rasterio coordinate transformation](https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html#rasterio.warp.transform), [EPA field manual](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100YX4L.TXT) |
| Detection limits and source quality flags | Keep below-limit observations and unresolved flags in a review file; avoid making up target values | [EPA laboratory manual](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100YX9O.TXT) |
| SHA-256 and deterministic manifests | Refuse changed raw inputs, record code/input/output hashes, and reproduce identical outputs | [hashlib](https://docs.python.org/3/library/hashlib.html), [JSON](https://docs.python.org/3/library/json.html) |
| `unittest` and temporary CSV fixtures | Verify scientific data-handling contracts without committing the downloaded dataset | [unittest](https://docs.python.org/3/library/unittest.html), [tempfile](https://docs.python.org/3/library/tempfile.html) |

Verification: 26 focused tests passed, including ten new preprocessing tests. Repeated real-data runs produced identical outputs; all 15 raw checksums remained unchanged. Generated data are ignored by Git; a small run manifest and documentation remain reviewable. No new Python dependency was needed.

Next study topic: represent date-only measurements honestly in the database, then match satellite observations using explicit date windows and lake geometry. Missing features must later be imputed using training folds only; repeat visits must stay grouped by lake during evaluation. Dashboard design and deployment remain later phases.

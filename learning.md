# AquaDoc V2 Learning And Build Guide

This file is a beginner-friendly build journal for turning the current AquaDoc demo into a real geospatial machine-learning project.

> **Reading order.** Sections 1–22 are the original journal (Streamlit demo → lab-measurement matching). In November 2026 the project changed direction: estimate optically detectable water quality from Sentinel-2 alone, using a model trained on a public reflectance dataset, and show it in a Vercel dashboard. Sections 23 onward follow the new plan, one section per phase. Each one has what was built, the concepts, a step-by-step way to redo it yourself, exercises and free resources. The lab-matching code from sections 14–22 now lives in `research/`.

## 1. What The Project Wants To Become

Goal: given coordinates, fetch or prepare multispectral satellite imagery, estimate water safety, and explain likely pollution signals such as algae bloom, turbidity, oil/surface film, floating vegetation, or debris.

Important truth: satellite images can strongly suggest some water-quality conditions, but they cannot directly prove every pollutant. Chemical pollutants, bacteria, heavy metals, and many toxins usually need field or lab measurements. A serious project should combine satellite imagery with labels from trusted water-quality datasets.

## 2. What The Current Repo Already Has

- A Streamlit app in `app.py`.
- A small database of famous water bodies in `data/water_bodies.py`.
- Synthetic demo image generation in `data/sample_images.py`.
- A Google Earth Engine thumbnail client in `utils/gee_client.py`.
- Image preprocessing utilities in `utils/image_processor.py`.
- A contour-based detector in `utils/waste_detector.py`.
- A synthetic Random Forest / Gradient Boosting model in `models/waste_detection_model.py`.

## 3. What It Lacks Right Now

- Real training data: the current model trains on synthetic patches, so its accuracy does not measure real pollution detection.
- Real labels: we need ground-truth measurements like chlorophyll-a, turbidity, harmful algal bloom records, water advisories, oil-spill polygons, or plastic/debris annotations.
- True multispectral workflow: RGB thumbnail images are not enough. We need raw bands such as blue, green, red, red-edge, near-infrared, and shortwave infrared.
- Geospatial correctness: pixel boxes currently map to latitude/longitude using rough approximations. Real work needs raster geotransforms, coordinate reference systems, and geometry operations.
- Backend API: the app is mostly UI/session code. A real product needs a separate service for ingestion, inference, storage, and reproducible analysis.
- Model evaluation: no validation set, metrics, calibration, uncertainty, or geographic/time-based split exists yet.
- Data storage: no database for scenes, coordinates, predictions, source metadata, or labels.
- Safety language: predictions must clearly say whether they are verified measurements, model estimates, or weak proxies.

## 4. What I Added In This Pass

I added a small backend core package:

- `aquadoc_core/schemas.py`: typed data structures for a multispectral scene and analysis result.
- `aquadoc_core/spectral_indices.py`: common remote-sensing index calculations.
- `aquadoc_core/risk.py`: an explainable baseline risk estimator.
- `aquadoc_core/pipeline.py`: one function that validates bands, computes indices, summarizes statistics, and returns a risk result.
- `tests/test_core_pipeline.py`: simple tests for clean water and bloom-like water.

This is not the final ML model. It is the first real foundation: a reproducible statistics pipeline that can later feed a trained model.

## 5. Concepts Used So Far

### Python Packages

A package is a folder of Python files that can be imported. `aquadoc_core` is now a package because it has an `__init__.py` file.

Learn:
- Python modules and packages: https://docs.python.org/3/tutorial/modules.html

### Dataclasses

Dataclasses are a clean way to store structured data. We use them for things like `MultispectralScene` and `AnalysisResult`.

Learn:
- Dataclasses: https://docs.python.org/3/library/dataclasses.html

### NumPy Arrays

Satellite bands are grids of numbers. In Python, these are usually handled as NumPy arrays. Each pixel has reflectance values for different wavelengths.

Learn:
- NumPy beginner guide: https://numpy.org/doc/stable/user/absolute_beginners.html

### Multispectral Satellite Imagery

A normal image has red, green, and blue channels. A multispectral satellite scene has more bands, often including:

- Blue: useful for water and atmosphere effects.
- Green: useful for water detection.
- Red: useful for sediment/turbidity and vegetation.
- Red-edge: useful for chlorophyll and algae.
- NIR: near-infrared, useful for water/land separation and floating material.
- SWIR: shortwave infrared, useful for water, moisture, burn scars, oil/surface-film clues, and clouds.

Learn:
- Sentinel-2 bands overview: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/
- USGS remote sensing basics: https://www.usgs.gov/faqs/what-remote-sensing-and-what-it-used

### Spectral Indices

A spectral index combines bands mathematically to highlight a property.

Examples added:

- `NDWI = (green - nir) / (green + nir)`: often used to identify water.
- `MNDWI = (green - swir1) / (green + swir1)`: another water index that can better separate built-up land.
- `NDCI = (red_edge - red) / (red_edge + red)`: useful as a chlorophyll/algae proxy.
- `Turbidity proxy = (red - green) / (red + green)`: a simple suspended sediment signal.

Learn:
- NDWI paper overview: https://en.wikipedia.org/wiki/Normalized_difference_water_index
- Chlorophyll indices background: https://www.mdpi.com/2072-4292/4/11/3514

### Baseline Model

Before training a big neural network, build a simple baseline. A baseline helps answer: “Can the data signal even solve the problem?”

The current baseline uses rules like:

- High NDCI may mean algae bloom risk.
- High red-vs-green response may mean turbidity.
- High SWIR-blue contrast may mean oil, surface film, clouds, or contamination.

Learn:
- Machine learning baseline concept: https://developers.google.com/machine-learning/crash-course

## 6. The Real Backend We Should Build Next

### Step 1: Data Ingestion

Build code that downloads real Sentinel-2 or Landsat scenes for a coordinate and date range.

Recommended sources:

- Google Earth Engine for querying scenes.
- Microsoft Planetary Computer STAC API.
- AWS Open Data Sentinel-2.

Tech:
- `earthengine-api`
- `pystac-client`
- `rasterio`
- `geopandas`
- `shapely`

Learn:
- Google Earth Engine Python: https://developers.google.com/earth-engine/guides/python_install
- STAC basics: https://stacspec.org/en
- Rasterio docs: https://rasterio.readthedocs.io/

### Step 2: Label Collection

A model needs pairs:

`satellite image at place/time -> known water-quality label at place/time`

Possible labels:

- Chlorophyll-a measurements.
- Turbidity measurements.
- Cyanobacteria or harmful algal bloom advisories.
- Oil-spill maps.
- Plastic/debris annotations from research datasets.
- In-situ station readings from government agencies.

Learn:
- Train/validation/test split: https://scikit-learn.org/stable/modules/cross_validation.html
- Data leakage: https://developers.google.com/machine-learning/crash-course/overfitting/data-leakage

### Step 3: Feature Store

For every coordinate/date, store:

- Raw scene metadata.
- Cloud cover.
- Sensor name.
- Bands used.
- Spectral indices.
- Statistical summaries.
- Ground-truth labels if available.

Tech:
- PostgreSQL + PostGIS for geospatial metadata.
- Object storage for raster chips.
- Parquet files for ML feature tables.

Learn:
- PostGIS intro: https://postgis.net/workshops/postgis-intro/
- Apache Parquet: https://parquet.apache.org/docs/

### Step 4: First Real ML Model

Start with tabular ML before a huge deep-learning model.

Input:
- Spectral index statistics.
- Seasonal features.
- Sensor metadata.
- Local history.

Models:
- Logistic Regression.
- Random Forest.
- XGBoost or LightGBM.

Why first? They are easier to debug, need less data, and reveal whether labels are useful.

Learn:
- Scikit-learn supervised learning: https://scikit-learn.org/stable/supervised_learning.html

### Step 5: Deep Learning Model

After a strong baseline, train a model on image chips.

Good architecture options:

- U-Net for segmentation: “which pixels look like algae/debris?”
- ResNet/EfficientNet for classification: “is this scene risky?”
- Vision Transformer or Swin Transformer for larger models.
- Temporal model if using many dates per location.

Tech:
- PyTorch
- PyTorch Lightning
- TorchGeo

Learn:
- PyTorch tutorials: https://pytorch.org/tutorials/
- TorchGeo docs: https://torchgeo.readthedocs.io/
- U-Net paper: https://arxiv.org/abs/1505.04597

### Step 6: Inference API

Build a backend API endpoint:

`POST /analyze`

Input:

```json
{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "date_range": {
    "start": "2026-01-01",
    "end": "2026-07-01"
  }
}
```

Output:

```json
{
  "risk_score": 0.64,
  "safety_label": "caution",
  "likely_signals": ["algae_bloom", "turbidity"],
  "limitations": ["Estimate only; not a lab-certified water-quality result."]
}
```

Tech:
- FastAPI
- Pydantic
- Uvicorn

Learn:
- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/

## 7. Why We Should Not Train A Huge Model Immediately

Training from scratch sounds powerful, but for this problem the hard part is not model size. The hard part is trusted data.

A huge model trained on weak or fake labels will learn fake patterns. Better path:

1. Build real data ingestion.
2. Build labels.
3. Build spectral baselines.
4. Train small models.
5. Measure them honestly.
6. Only then train a large image model.

## 8. Immediate Next Build Task

The next practical backend task should be:

Build a real Sentinel-2 data ingestion module that returns a small aligned multispectral chip for any coordinate.

Target function:

```python
fetch_sentinel2_chip(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    size_m: int = 3000,
) -> MultispectralScene
```

After that, the new `aquadoc_core.analyze_multispectral_scene()` function can analyze the real chip.

## 9. Commands Used To Verify Current Work

```powershell
python -m unittest tests.test_core_pipeline
```

Result: tests passed.

## 10. Ingestion Layer Build

The ingestion layer is the part of the backend that turns this request:

```text
latitude + longitude + date range
```

into this object:

```text
MultispectralScene
```

That `MultispectralScene` contains aligned satellite bands like blue, green, red, NIR, red-edge, and SWIR. The analysis pipeline can then compute spectral indices from those bands.

### What Was Added

New file:

- `aquadoc_core/ingestion.py`

New tests:

- `tests/test_ingestion.py`

Updated dependencies:

- `pystac`
- `pystac-client`
- `planetary-computer`
- `rasterio`

Main function:

```python
fetch_sentinel2_chip(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    size_m: int = 3000,
) -> MultispectralScene
```

### Why We Built It This Way

The ingestion layer has three jobs:

1. Search for a real satellite scene near the coordinate.
2. Choose the clearest available scene.
3. Read the required spectral bands into NumPy arrays.

These jobs are separated because each one can fail for different reasons:

- Search can fail because no satellite scene exists in the date range.
- Search can return cloudy imagery.
- Asset URLs may need signing before reading.
- Raster reading can fail if geospatial libraries are missing.
- Different bands may have different native resolutions.

By separating these pieces, we can test the logic without downloading huge real satellite files every time.

### STAC

STAC means SpatioTemporal Asset Catalog.

Plain meaning:

STAC is a standard JSON format for searching Earth observation data. It lets us ask questions like:

```text
Find Sentinel-2 images near this coordinate between these dates with cloud cover below 20%.
```

Why this matters:

Without STAC, every satellite provider would have its own custom search format. With STAC, our code can later support multiple data providers more easily.

Learn:

- STAC overview: https://stacspec.org/en
- PySTAC Client usage: https://pystac-client.readthedocs.io/en/stable/usage.html

### Sentinel-2 Level-2A

We are starting with Sentinel-2 Level-2A.

Sentinel-2 is a European satellite mission that captures multispectral images of Earth. Level-2A means the data is processed into surface reflectance, which is much more useful for environmental analysis than raw sensor brightness.

Why it is a good first source:

- It has useful bands for water analysis.
- It is global.
- It revisits locations often.
- It has 10 m, 20 m, and 60 m bands.
- It is available through cloud catalogs such as Microsoft Planetary Computer.

Learn:

- Planetary Computer Sentinel-2 L2A dataset: https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a

### COG

COG means Cloud Optimized GeoTIFF.

Plain meaning:

A COG is a satellite raster file designed so software can read a small window from the cloud instead of downloading the full file.

Why this matters:

A full Sentinel-2 scene can be very large. For our app, the user gives one coordinate, so we only need a small chip around that point.

Tech used:

- `rasterio.open(...)` opens the remote band asset.
- A window is created around the coordinate.
- The chip is read into a NumPy array.

Learn:

- Rasterio docs: https://rasterio.readthedocs.io/
- Cloud Optimized GeoTIFF: https://www.cogeo.org/

### Planetary Computer Signing

The STAC metadata is public, but the actual imagery assets may require signed URLs.

The code tries to use:

```python
planetary_computer.sign_inplace(...)
```

If the optional package is installed, the asset URLs become readable. If it is not installed, the code keeps the original URLs, but live reads may fail.

Learn:

- Planetary Computer data access tokens: https://planetarycomputer.microsoft.com/docs/concepts/sas/

### Band Mapping

The project uses friendly names internally:

```python
{
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "red_edge": "B05",
    "nir": "B08",
    "swir1": "B11",
}
```

The left side is what our code understands. The right side is the Sentinel-2 asset name.

Why this helps:

If we later add Landsat or another satellite source, the analysis pipeline can still ask for `blue`, `green`, `red`, and `nir` without caring about provider-specific band names.

### Cloud Filtering

Clouds can ruin water analysis because clouds are bright and can look like pollution signals.

The STAC search currently asks for:

```text
eo:cloud_cover < 20
```

Then `choose_best_item(...)` selects the lowest-cloud scene from the returned items.

Important limitation:

Scene-level cloud cover does not guarantee the exact coordinate is cloud-free. Later we should use cloud masks like Sentinel-2 Scene Classification Layer.

### Band Alignment

Sentinel-2 bands do not all have the same resolution.

Examples:

- B02, B03, B04, B08 are 10 m.
- B05 and B11 are 20 m.

The ingestion layer reads every band into the same output shape. This lets our NumPy analysis use pixel-by-pixel formulas.

Beginner idea:

Imagine one grid has small squares and another grid has bigger squares. To combine them, we resize them to a common grid.

### Why Tests Use Fake Search And Fake Reader

Tests should be fast, deterministic, and not depend on internet access.

So `fetch_sentinel2_chip(...)` accepts:

- `search_client`
- `band_reader`

In production, these are real. In tests, they are fake.

This is called dependency injection.

Learn:

- Dependency injection idea: https://en.wikipedia.org/wiki/Dependency_injection

### Commands Used To Verify This Pass

```powershell
python -m unittest tests.test_core_pipeline tests.test_ingestion
```

Result: tests passed.

## 11. Full Backend Command-Line Path

We now have a small command-line interface (CLI) that runs the full backend path:

1. Fetch a Sentinel-2 chip.
2. Analyze it with `analyze_multispectral_scene`.
3. Print risk score, safety label, and signal explanations.

Run it with:

```powershell
python -m aquadoc_core.cli analyze --lat 12.9716 --lon 77.5946 --start 2026-01-01 --end 2026-07-01
```

This gives us a backend workflow before we connect it to the UI.

### What The Command Does

The data moves through the system in this order:

```text
terminal arguments
    -> Sentinel-2 STAC search
    -> COG band download
    -> MultispectralScene
    -> spectral indices
    -> risk estimation
    -> JSON output
```

The command accepts `--lat`, `--lon`, `--start`, and `--end`. It also accepts
`--size-m` to control the width of the image chip. The default is 3000 metres,
so the analyzed area is approximately 3 km by 3 km.

Use `--compact` when another program will consume the result and readable
indentation is unnecessary:

```powershell
python -m aquadoc_core.cli analyze --lat 12.9716 --lon 77.5946 --start 2026-01-01 --end 2026-07-01 --compact
```

### Why JSON Is The Output Format

JSON is structured text made from objects, arrays, strings, numbers, booleans,
and null values. It is useful here because the CLI, a future web API, the UI,
and data-processing scripts can all understand the same result shape.

`aquadoc_core/serialization.py` converts the nested analysis dataclasses into
plain JSON-compatible dictionaries. This is called a **serialization boundary**:
the internal Python representation is translated into a format that can leave
the program safely.

### Thin Interface, Reusable Backend

`aquadoc_core/cli.py` does not contain satellite or pollution algorithms. It
only reads terminal arguments, calls the existing ingestion and analysis
functions, and prints their result. This separation is important because a
future API can reuse the same backend without copying its logic.

The CLI returns exit code `0` after success and exit code `1` for an expected
data or validation error. Errors are also JSON and are written to **standard
error** (`stderr`), while successful results go to **standard output**
(`stdout`). Scripts can therefore separate results from errors reliably.

### How The CLI Is Tested Without Downloading Satellite Data

The test passes a `fake_fetcher` into the CLI. It creates a tiny in-memory
multispectral scene, but the real analysis function still computes indices and
risk. This tests the complete command orchestration while keeping the unit test
fast, deterministic, and available without internet access.

Concepts introduced in this step:

- Command-line argument parsing with Python's `argparse`
- Serialization with `dataclasses.asdict`
- JSON as a program-to-program data format
- Standard output, standard error, and process exit codes
- Function injection for deterministic tests

Learning resources:

- `argparse`: https://docs.python.org/3/library/argparse.html
- `json`: https://docs.python.org/3/library/json.html
- `dataclasses.asdict`: https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
- Standard streams: https://docs.python.org/3/library/sys.html#sys.stdin

### Commands Used To Verify This Pass

```powershell
python -m unittest tests.test_core_pipeline tests.test_ingestion tests.test_cli
python -m aquadoc_core.cli --help
python -m aquadoc_core.cli analyze --help
```

The real `analyze` command needs internet access and all packages from
`requirements.txt`. Its result is still a spectral screening estimate, not a
medical, environmental-regulatory, or laboratory safety verdict.

## 12. Persistence And Reproducible Runs

The backend now saves every successful analysis to `data/aquadoc.sqlite3`.
Persistence means data survives after the Python process exits. Previously, an
analysis existed only in memory and disappeared when the command finished.

Internally, the default path is resolved from the AquaDoc project directory,
not from the terminal's current directory. This means `analyze`, `list-runs`,
and `show-run` reach the same database even when the command is launched from a
different folder. A relative path supplied explicitly with `--db` is still
resolved from the terminal's current directory, following normal command-line
behavior.

### Why SQLite

SQLite is a relational database stored in one local file. It is included with
Python, supports transactions and indexes, and does not require us to run a
separate database server. It is a strong fit while developing locally. Later,
the repository interface can be implemented with PostgreSQL when concurrent
users and a deployed API require it.

The `analysis_runs` table stores two kinds of data:

- Searchable columns such as coordinates, dates, source, risk score, and label
- The complete analysis result as JSON so no index statistics are lost

This is a hybrid storage model. Columns make common database queries efficient,
while JSON preserves the complete output as the analysis schema develops.

### Run Identifiers And Reproducibility

Every saved analysis gets a UUID, such as
`6f9619ff-8b86-4f2d-8f5d-2e4fdc6ad391`. A UUID is designed to be unique without
asking a central counter for the next number.

The saved record includes:

- `run_id`: stable identity for this execution
- `created_at`: UTC time when AquaDoc performed the analysis
- `start_date` and `end_date`: satellite search window
- `size_m`: requested chip size
- Scene source, satellite capture time, resolution, coordinates, and risk data
- The complete spectral result and stated limitations

`created_at` and `captured_at` are different. The satellite may have captured
an image weeks before AquaDoc analyzed it. Keeping both prevents time-series
comparisons from accidentally using the processing time as the observation time.

### Database Commands

Run and automatically save an analysis:

```powershell
python -m aquadoc_core.cli analyze --lat 12.9716 --lon 77.5946 --start 2026-01-01 --end 2026-07-01
```

The JSON response now contains `run_id`, storage metadata, and the analysis
inside `result`.

List the 20 most recent runs:

```powershell
python -m aquadoc_core.cli list-runs
```

Retrieve one reproducible result:

```powershell
python -m aquadoc_core.cli show-run YOUR_RUN_ID
```

All commands accept `--db PATH` when a different database file is needed. This
is useful for tests, experiments, or separate datasets.

### Repository Pattern

`AnalysisRunRepository` owns SQL operations. The CLI asks it to `save`, `get`,
or `list` runs and does not need to know table details. This is the repository
pattern: database logic is placed behind a small interface so other callers,
including a future API, can reuse it.

The repository creates its table and indexes automatically. An index is an
extra database structure that speeds up reads. We index location plus creation
time and satellite capture time because those fields will matter for historical
comparisons.

Database connections are wrapped in a context manager that commits successful
transactions, rolls back failed transactions, and always closes the connection.
Closing matters especially on Windows, where an open connection keeps the
database file locked.

Concepts introduced in this step:

- Persistent storage and relational tables
- SQLite transactions and parameterized SQL
- Primary keys, UUIDs, and database indexes
- UTC timestamps and observation-time versus processing-time data
- Repository pattern
- Hybrid relational-column and JSON storage

Learning resources:

- Python SQLite module: https://docs.python.org/3/library/sqlite3.html
- SQL tutorial: https://www.sqlitetutorial.net/
- Python UUID module: https://docs.python.org/3/library/uuid.html
- Database indexes: https://www.sqlite.org/queryplanner.html
- UTC-aware datetimes: https://docs.python.org/3/library/datetime.html

### Commands Used To Verify This Pass

```powershell
python -m unittest tests.test_core_pipeline tests.test_ingestion tests.test_cli tests.test_persistence
python -m aquadoc_core.cli list-runs --help
python -m aquadoc_core.cli show-run --help
```

Tests use a temporary SQLite file, so verification never adds fake analysis
runs to the real project database.

## 13. HTTP API

The backend is now available through an HTTP API in `aquadoc_core/api.py`. An
API lets another program send requests to AquaDoc over HTTP and receive JSON
responses. The future UI will use this instead of importing Python functions or
running terminal commands.

### Starting The Server

Install the dependencies and start the development server:

```powershell
pip install -r requirements.txt
python -m uvicorn aquadoc_core.api:app --reload
```

The API is then available at `http://127.0.0.1:8000`. FastAPI automatically
creates interactive documentation at `http://127.0.0.1:8000/docs`. The docs
page lets you inspect and execute each endpoint in a browser.

`aquadoc_core.api:app` has two parts. `aquadoc_core.api` is the Python module,
and `app` is the FastAPI object inside it. Uvicorn is the server that imports
that object and handles network connections. `--reload` restarts the development
server after code changes; it should not be used as the production process.

### Available Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check whether the API process is responding |
| `POST` | `/analyses` | Fetch imagery, analyze it, and save the run |
| `GET` | `/analyses?limit=20` | List saved runs, newest first |
| `GET` | `/analyses/{run_id}` | Retrieve one saved run |

An endpoint combines an HTTP method and path. `GET` reads data. `POST` submits
data and creates a resource. The create endpoint responds with HTTP status
`201 Created`, and a missing run responds with `404 Not Found`.

### Creating An Analysis

Send this JSON body to `POST /analyses`:

```json
{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "start_date": "2026-01-01",
  "end_date": "2026-07-01",
  "size_m": 3000
}
```

From PowerShell, this can be tested without the browser documentation:

```powershell
$body = @{
    latitude = 12.9716
    longitude = 77.5946
    start_date = "2026-01-01"
    end_date = "2026-07-01"
    size_m = 3000
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/analyses -ContentType "application/json" -Body $body
```

This endpoint performs the same sequence as the CLI: ingestion, spectral
analysis, risk estimation, and SQLite persistence. It returns the generated
`run_id` and complete saved result.

### Validation With Pydantic

`AnalysisRequest` is a Pydantic model describing valid request data. It checks:

- Latitude is between -90 and 90
- Longitude is between -180 and 180
- Dates use ISO `YYYY-MM-DD` format and start is not after end
- Chip size is between 100 and 20,000 metres
- Unknown JSON fields are rejected instead of silently ignored

FastAPI runs this validation before ingestion. Invalid requests receive status
`422 Unprocessable Entity`, and no satellite download or database write occurs.
This protects both computing resources and data quality.

### Application Factory And Test Isolation

`create_app()` constructs the FastAPI application. It accepts a database path,
fetch function, and analysis function. Production uses the real implementations;
tests use a temporary SQLite database and an in-memory multispectral scene.
This is the same dependency-injection idea used by the CLI.

The API endpoints use ordinary `def` functions because satellite HTTP requests,
raster reads, and SQLite operations are blocking work. FastAPI runs these
handlers in a worker threadpool, allowing the main asynchronous server loop to
continue handling other connections.

### Automatic API Documentation

The request models and routes are converted into an OpenAPI schema at
`/openapi.json`. The `/docs` interface is generated from that schema. OpenAPI
is a standard machine-readable description of an HTTP API, so it can later
generate frontend clients and API tests.

Concepts introduced in this step:

- HTTP methods, paths, JSON bodies, and status codes
- FastAPI application and Uvicorn server
- Pydantic request validation
- REST-style resources
- OpenAPI and interactive API documentation
- Application factory and integration testing
- Blocking work and server threadpools

Learning resources:

- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
- Request bodies: https://fastapi.tiangolo.com/tutorial/body/
- HTTP status codes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
- OpenAPI: https://spec.openapis.org/oas/latest.html
- Uvicorn settings: https://www.uvicorn.org/settings/

### Commands Used To Verify This Pass

```powershell
python -m unittest tests.test_api tests.test_core_pipeline tests.test_ingestion tests.test_cli tests.test_persistence tests.test_persistence_path
python -m uvicorn aquadoc_core.api:app --reload
```

The API analysis endpoint currently waits for ingestion and analysis to finish
before responding. The next backend improvement should introduce a background
job model with run states such as `pending`, `running`, `completed`, and
`failed`. That will prevent long satellite downloads from tying up one HTTP
request and will let the UI display progress.

## 14. Ground-Truth Observation Schema

We now have a validated schema for laboratory and field observations in
`aquadoc_core/ground_truth.py`. This is the beginning of the real ML dataset.

### What Ground Truth Means

Ground truth is the best available real-world answer for the quantity a model
will learn to predict. For AquaDoc, examples include:

- A laboratory reports chlorophyll-a as 42 micrograms per litre
- A calibrated field meter reports turbidity as 18.2 NTU
- A trained survey team confirms a harmful algae bloom
- A verified annotation identifies floating waste in a water body

The satellite image is not ground truth. It is the **input** from which we want
the model to make a prediction. The field or laboratory observation is the
**target** against which that prediction is trained and evaluated.

In common ML notation:

```text
X = satellite bands and derived spectral features
y = verified field or laboratory observation

model learns: X -> y
```

If `y` is wrong, vague, or unrelated to the image, a large model will learn the
wrong relationship. More computing power cannot repair unreliable targets.

### What A Schema Is

A schema is a formal description of valid data. It names every field, defines
its type, and states the rules that records must follow. Without a schema, one
dataset might write turbidity as `18 NTU`, another as `18`, and another as
`high`. A computer cannot safely assume those mean the same thing.

Our `GroundTruthObservation` schema is implemented as a Pydantic model. It
validates data before SQLite accepts it.

### Fields In One Observation

| Field | Meaning | Why it matters |
| --- | --- | --- |
| `observation_id` | Unique UUID | Prevents records from being confused or overwritten |
| `latitude`, `longitude` | Sampling location | Allows spatial matching with satellite pixels |
| `observed_at` | Sampling time with timezone | Allows temporal matching with satellite capture time |
| `parameter` | What was measured | Separates turbidity, chlorophyll, algae labels, and other targets |
| `numeric_value` | Measured number | Used for regression targets such as turbidity |
| `categorical_value` | Named class | Used for classification targets such as bloom confirmed/not observed |
| `unit` | Measurement unit | Gives a number scientific meaning and enables conversion |
| `depth_m` | Sampling depth | Surface conditions may differ from deeper water |
| `spatial_uncertainty_m` | Location accuracy | Helps decide which satellite pixels are valid matches |
| `method` | Instrument or procedure | Different methods can have different bias and precision |
| `source` | Organisation or dataset | Provides provenance and accountability |
| `source_reference` | Report, file, or external ID | Lets us trace the original evidence |
| `quality` | Trust status | Lets training exclude suspect or rejected measurements |
| `notes` | Additional context | Preserves useful information not covered by fixed fields |
| `created_at` | Database insertion time | Supports auditing and dataset versioning |

### Numeric And Categorical Targets

Some ML problems predict a number. This is called **regression**:

```text
turbidity = 18.2 NTU
chlorophyll_a = 42 ug/L
dissolved_oxygen = 6.8 mg/L
```

Other problems predict a class. This is called **classification**:

```text
harmful_algae_bloom = confirmed
floating_waste = observed
oil_or_surface_film = not_observed
```

One observation must contain exactly one of `numeric_value` or
`categorical_value`. Supplying both would make the learning objective ambiguous;
supplying neither would give the model no target.

The initial supported parameters are:

- Numeric: `chlorophyll_a`, `turbidity`, `total_suspended_solids`,
  `dissolved_oxygen`, and `ph`
- Categorical: `harmful_algae_bloom`, `floating_waste`, and
  `oil_or_surface_film`

Numeric parameters require a unit. Categorical parameters reject units. This
is enforced in Python and the exactly-one-value rule is also enforced by a
SQLite `CHECK` constraint.

### Why Time And Location Must Be Precise

Water can change quickly after rain, discharge, wind, or temperature changes.
A laboratory sample collected a month after the satellite image may not
describe what the satellite saw.

Every `observed_at` value therefore requires a timezone:

```text
2026-06-15T05:00:00+00:00
```

Later, scene matching will calculate:

```text
spatial distance = distance between sample and satellite pixel
time difference  = satellite capture time - sample collection time
```

We will define acceptable limits for each parameter. A bloom label might need
a very small time gap, while a more stable measurement may permit a wider one.

### Why Method, Source, And Quality Matter

Two instruments can produce different values for the same water sample. A
visual social-media report and an accredited laboratory result should not have
equal trust. Provenance records where a value came from and how it was created.

The quality values are:

- `verified`: passed the source's quality checks
- `provisional`: usable but not fully reviewed
- `suspect`: likely affected by a known problem
- `rejected`: retained for auditing but excluded from training

Keeping rejected observations is useful because deletion hides history. Dataset
building can filter them out while preserving why they were rejected.

### Example Python Observation

```python
from aquadoc_core.ground_truth import GroundTruthObservation

observation = GroundTruthObservation(
    latitude=12.9716,
    longitude=77.5946,
    observed_at="2026-06-15T05:00:00Z",
    parameter="turbidity",
    numeric_value=18.2,
    unit="NTU",
    depth_m=0.2,
    spatial_uncertainty_m=5.0,
    method="ISO 7027 nephelometric field meter",
    source="Example monitoring programme",
    quality="verified",
)
```

Save it to the same AquaDoc SQLite database:

```python
from aquadoc_core.ground_truth import GroundTruthRepository

repository = GroundTruthRepository()
repository.save(observation)
```

The repository can retrieve an observation by ID or list observations filtered
by parameter. Parameterized SQL is used so values are never assembled directly
into SQL strings.

### SQLite Table And Indexes

The repository automatically creates `ground_truth_observations` inside the
existing `data/aquadoc.sqlite3` database. Analysis runs and observations live
in different tables because predictions and real measurements are different
kinds of evidence.

Indexes cover parameter plus observation time and location plus observation
time. These support the future operations we expect most often: finding all
turbidity labels in a period and matching observations to nearby scenes.

### What This Does Not Do Yet

Defining the schema does not create training data. It gives incoming data one
consistent, validated shape. We still need trusted source datasets and must
respect their licences, measurement standards, and quality-control rules.

The CSV import pipeline added in the next section now:

1. Reads source rows without modifying the original file.
2. Validates source values and supported units.
3. Converts units to agreed canonical units.
4. Reports rejected rows and exact validation reasons.
5. Detects duplicates before saving.

After importing enough observations, we can match them with Sentinel-2 scenes
by distance and time and export the first supervised training table.

Concepts introduced in this step:

- Ground truth, features (`X`), and targets (`y`)
- Regression versus classification
- Data schemas and validation
- Measurement units and sampling depth
- Spatial and temporal uncertainty
- Data provenance and quality flags
- Database constraints and indexes

Learning resources:

- Pydantic models: https://docs.pydantic.dev/latest/concepts/models/
- Supervised learning: https://scikit-learn.org/stable/tutorial/basic/tutorial.html
- Measurement uncertainty overview: https://www.nist.gov/pml/nist-technical-note-1297
- FAIR data principles: https://www.go-fair.org/fair-principles/
- SQLite constraints: https://www.sqlite.org/lang_createtable.html#ckconst

### Commands Used To Verify This Pass

```powershell
python -m unittest tests.test_ground_truth tests.test_api tests.test_core_pipeline tests.test_ingestion tests.test_cli tests.test_persistence tests.test_persistence_path
```

All tests use temporary databases, so examples never contaminate the real
ground-truth dataset.

## 15. Field/Laboratory Tables And CSV Import

The database now separates a sampling event from its measurement results.

```text
sampling_events
    |-- field_measurements
    `-- laboratory_measurements
```

This relationship matters because one water sample can produce several results.
For example, a team can measure temperature and turbidity beside the lake, then
send one bottle to a laboratory for chlorophyll-a and suspended-solids analysis.
Coordinates, collection time, source, depth, and uncertainty belong to the
shared sampling event and should not be independently rewritten for every test.

### The Three Tables

`sampling_events` stores:

- Sample ID
- Latitude and longitude
- Exact collection time
- Sampling depth
- Spatial uncertainty
- Source organisation and source reference

`field_measurements` stores results produced at the sampling location:

- Parameter and value
- Unit
- Field method and instrument
- Quality flag and notes

`laboratory_measurements` stores results produced after sample collection:

- Parameter and value
- Unit and analytical method
- Laboratory name
- Analysis time
- Detection limit
- Quality flag and notes

Both measurement tables use a foreign key to `sampling_events.sample_id`. A
foreign key prevents a measurement from referring to a sample that does not
exist. If the same sample ID appears with conflicting coordinates or collection
times, the importer rejects it rather than silently changing the event.

### CSV Template

The import template is `data/measurement_import_template.csv`. Its first row is
the required header, followed by one example field row and one example laboratory
row. Replace example records with real source data before building a dataset.

Important columns are:

```text
measurement_type     field or laboratory
sample_id            source sample identity
latitude             decimal degrees, WGS84
longitude            decimal degrees, WGS84
collected_at          ISO timestamp with timezone
parameter             one supported ground-truth parameter
numeric_value         regression target
categorical_value     classification target
unit                  scientific unit for numeric targets
method                collection or analysis procedure
source                data-producing organisation
```

Laboratory rows must also provide `laboratory_name` and `analyzed_at`. Field rows
can provide `instrument`. Blank optional cells remain empty, but the column must
still exist in the header.

### Import Commands

Import a CSV into the main AquaDoc database:

```powershell
python -m aquadoc_core.cli import-measurements path/to/measurements.csv
```

To experiment with the included example without adding fake rows to the real
database, use a separate SQLite path:

```powershell
python -m aquadoc_core.cli import-measurements data/measurement_import_template.csv --db data/import-test.sqlite3
```

Count stored records:

```powershell
python -m aquadoc_core.cli measurement-counts
```

The import result has this shape:

```json
{
  "total_rows": 10,
  "imported_rows": 7,
  "duplicate_rows": 1,
  "rejected_rows": 2,
  "errors": [
    {"row_number": 4, "message": "latitude: Input should be less than or equal to 90"}
  ]
}
```

Valid rows are not discarded because another row is bad. Every rejected row
keeps its CSV line number and validation message so the source file can be
corrected deliberately.

### Unit Normalization

Training cannot safely mix `0.042 mg/L` with `42 ug/L` as though they were
different concentrations. The importer converts supported values to canonical
units:

| Parameter | Canonical unit | Accepted examples |
| --- | --- | --- |
| Chlorophyll-a | `ug/L` | `ug/L`, `µg/L`, `mg/L` |
| Turbidity | `NTU` | `NTU` |
| Total suspended solids | `mg/L` | `mg/L`, `g/L` |
| Dissolved oxygen | `mg/L` | `mg/L`, `g/L` |
| pH | `pH` | `pH`, `unitless` |

Detection limits are converted by the same factor as laboratory values. An
unknown unit is rejected because guessing a conversion could corrupt every model
trained from the database.

### Duplicate Detection

Every measurement receives a SHA-256 fingerprint calculated from its scientific
content. SHA-256 converts the normalized record into a stable identifier. The
database has a unique constraint on this fingerprint, so importing the same row
twice produces `duplicate_rows: 1` instead of creating a second label.

This property is called **idempotence**: repeating an operation has the same
final effect as performing it once.

## 16. Where Laboratory Measurements Come From

Free satellite images and laboratory measurements are different resources.
Sentinel-2 gives broad and repeated Earth coverage, but it does not include a
laboratory result for every image or coordinate. Ground measurements require
people, instruments, sample handling, and chemical analysis, so coverage is
sparse and uneven.

We cannot obtain laboratory measurements for every place. The practical strategy
is to gather measurements from monitoring stations, match each one to a nearby
cloud-free satellite scene, and train a model that may generalize to unmeasured
locations. Whether it actually generalizes must be tested by withholding entire
locations during evaluation.

Useful official sources include:

- The [US EPA Water Quality Portal](https://www.epa.gov/waterdata/water-quality-data-download),
  which combines hundreds of millions of public results from USGS, EPA, and
  federal, state, Tribal, local, and research organisations.
- The [USGS Water Quality Portal API](https://api.waterdata.usgs.gov/docs/),
  which provides programmatic access to discrete public measurements.
- [UNEP GEMS/Water GEMStat](https://gemstat.org/data-gemstat/data-portal/), a
  global station database whose open subset is published for download; licences
  vary by provider, so each dataset's policy and attribution must be preserved.
- India's [CPCB National Water Quality Monitoring Programme](https://www.cpcb.nic.in/uploads/WQM_Objective.pdf),
  which operates monitoring stations and publishes reports, though machine-readable
  raw measurements and parameter coverage may be less consistent.
- NASA SeaBASS and related ocean-colour archives for coastal and marine optical
  measurements. NASA also provides training on combining in-situ observations
  with remote sensing through its [ARSET water-quality course](https://appliedsciences.nasa.gov/get-involved/training/english/arset-monitoring-coastal-and-estuarine-water-quality-using-remote).

Other routes are university collaborations, environmental agencies, citizen
science with documented instruments, and a targeted sampling campaign. Citizen
reports can be useful categorical evidence, but they should not be presented as
laboratory measurements and should receive an appropriate quality flag.

The key rule is: absence of a lab measurement means **unknown**, not safe. We
must never generate fake labels from the current spectral risk rules and then
claim the model learned real pollution.

## 17. How We Access Satellite Images

The current ingestion code uses the public Microsoft Planetary Computer STAC
API. Its documentation states that the STAC API is publicly accessible without
an account and that most data can be downloaded anonymously, subject to
throttling. AquaDoc performs these steps:

```text
coordinates + date range
        -> STAC search request
        -> Sentinel-2 Level-2A item metadata
        -> signed band asset URLs
        -> Rasterio reads a small COG window
        -> aligned NumPy band arrays
```

STAC is a catalogue, not the image itself. It returns metadata and links to band
assets. Planetary Computer signs those asset links with temporary access tokens.
Rasterio then reads only the requested window from each Cloud Optimized GeoTIFF.
This avoids downloading a complete Sentinel tile.

Our current endpoint is convenient for development. For resilience, a later
ingestion provider should also support the official
[Copernicus Data Space STAC API](https://documentation.dataspace.copernicus.eu/APIs/STAC.html),
whose current catalogue endpoint is `https://stac.dataspace.copernicus.eu/v1/`.
The provider abstraction should let us switch catalogues without changing the
analysis or training code.

Sentinel-2 Level-2A provides corrected surface reflectance. It is open Earth
observation input data, but “free imagery” does not remove the costs of network
transfer, storage, preprocessing, ground-truth collection, or model training.

## 18. How The Model Will Be Trained

We are not training the final ML model yet. We are building the data system that
makes honest training possible. The eventual process is:

```text
1. Import verified field/laboratory observations
2. Find Sentinel-2 scenes near each sample time and location
3. Reject clouds, shadows, invalid water pixels, and poor matches
4. Extract multispectral patches and spectral features
5. Pair each patch (X) with its measured target (y)
6. Split data by station or water body
7. Fit a baseline model
8. Evaluate only on held-out places and times
9. Calibrate uncertainty and document failure cases
10. Consider a larger image model only after the baseline is credible
```

For numeric targets such as turbidity, we will start with regression models and
report errors such as MAE and RMSE. For bloom or floating-waste classes, we will
start with classification and report precision, recall, F1, and confusion
matrices.

The first baseline should use tabular features such as band statistics and
spectral indices with Random Forest or gradient boosting. This tells us whether
the data contains a useful signal and is much easier to debug than a large neural
network. Then we can compare it with a multispectral CNN that receives image
patches shaped like:

```text
[bands, height, width]
```

The split must be grouped by monitoring station or water body. A random row split
could put almost identical images from the same lake in both training and test
sets, producing an unrealistically high score. A location-held-out test asks the
question we truly care about: does the model work somewhere it did not memorize?

Satellite predictions will be strongest for optically visible surface properties
such as chlorophyll, turbidity, suspended material, and some surface blooms.
Sentinel-2 alone generally cannot establish drinking safety, bacteria, dissolved
heavy metals, or arbitrary chemical pollution. Those require other sensors,
contextual data, or direct sampling.

### Next Implementation

The next code layer should match measurements to satellite scenes. It will store
distance in metres, time difference in hours, cloud information, selected scene
ID, and match-quality reasons. That pairing table will become the reproducible
index from which we export the first training dataset.

### Commands Used To Verify This Pass

```powershell
python -m unittest tests.test_measurement_import tests.test_ground_truth tests.test_api tests.test_core_pipeline tests.test_ingestion tests.test_cli tests.test_persistence tests.test_persistence_path
python -m aquadoc_core.cli import-measurements data/measurement_import_template.csv --db data/import-test.sqlite3
python -m aquadoc_core.cli measurement-counts --db data/import-test.sqlite3
```

## 19. Matching Measurements To Satellite Scenes

The first supervised-learning stage is implemented in
`aquadoc_core/training_pipeline.py`. It creates one auditable relationship between
a verified field/laboratory measurement and a Sentinel-2 scene.

Run matching for one parameter:

```powershell
.venv\Scripts\python.exe -m aquadoc_core.cli match-scenes --parameter turbidity --max-days 3 --chip-size-m 1000
```

The operation follows this sequence:

```text
verified measurement
    -> search from 3 days before to 3 days after collection
    -> discard scenes without a capture timestamp
    -> choose smallest absolute time difference
    -> use lower cloud cover as a tie-breaker
    -> read a chip centred on the sample coordinate
    -> calculate spectral statistics
    -> save match provenance and features
```

Time proximity is considered before scene-level cloud cover because water can
change quickly. Cloud cover is still imperfect: a tile-level percentage does
not prove that the sample pixel is cloud-free. A future quality-mask pass must
use Sentinel-2's pixel-level scene classification layer.

### What Is Saved

The `measurement_scene_matches` table stores:

- Deterministic match ID and original measurement ID
- Sample location, collection time, target value, unit, and quality
- Sentinel scene ID and capture time
- Absolute time difference in hours
- Scene cloud-cover metadata
- Chip size and extraction distance
- Feature-definition version
- Complete flattened spectral feature JSON

The extraction distance is currently zero because every chip is centred on the
measurement coordinate. We store it explicitly because future station-to-pixel
matching may select a nearby valid water pixel instead.

The feature version is `spectral-statistics-v1`. Versioning matters because a
future change to water masking or an index formula creates a different feature
definition. Old and new features must not be mixed invisibly.

Only `verified` measurements are matched by default. Provisional, suspect, and
rejected records remain in the database but do not become training examples.

Re-running matching is safe. Measurements that already have a persisted match
are reported as `already_matched` rather than silently selecting a newer scene.
That preserves reproducibility.

## 20. Reproducible Dataset Export

After matching, export one parameter at a time:

```powershell
.venv\Scripts\python.exe -m aquadoc_core.cli export-training --parameter turbidity --output data/training/turbidity.csv
```

One parameter per dataset keeps the target definition and unit consistent.
Turbidity and chlorophyll-a are different regression problems; combining them
as one target would have no scientific meaning.

The CSV contains three groups of columns:

```text
provenance: measurement_id, scene_id, collection/capture times
target:     target_numeric or target_category, parameter, unit
features:   feature__ndwi__median, feature__ndci__p90, ...
```

Rows are sorted by measurement ID and scene ID. Feature columns are sorted by
name. Given the same match table and feature version, the exporter produces the
same CSV bytes.

### Dataset Manifest And Hash

The exporter also writes:

```text
turbidity.csv.manifest.json
```

The manifest records row count, parameter, feature version, feature columns,
sort order, and a SHA-256 hash of the CSV. A hash acts like a content
fingerprint. If one byte changes, the hash changes.

This lets a model report prove exactly which dataset it used. Reproducibility
does not mean every future run must have the same score after data changes. It
means we can identify and recreate the precise inputs and settings behind one
reported result.

`location_group` is coordinates rounded to three decimal places, approximately
a neighbourhood-scale grouping. This is an initial station grouping, not a
perfect water-body identifier. Later we should assign stable station and
water-body IDs from source metadata or a geographic water-body dataset.

## 21. Baseline Model Training And Evaluation

Train the first baseline after the exported dataset contains enough real rows
from several locations:

```powershell
.venv\Scripts\python.exe -m aquadoc_core.cli train-baseline `
  --dataset data/training/turbidity.csv `
  --model models/turbidity_baseline.pkl `
  --report models/turbidity_baseline_report.json `
  --test-size 0.25 `
  --seed 42
```

Use the project virtual environment because the system Python may not contain
NumPy, pandas, or scikit-learn:

```powershell
.venv\Scripts\Activate.ps1
python -m aquadoc_core.cli --help
```

### Why Random Forest

The baseline uses a Random Forest: many decision trees trained on different
samples and feature subsets, whose outputs are combined. It works well as a
first model because it handles nonlinear feature relationships, needs little
feature scaling, supports regression and classification, and is easier to
diagnose than a large neural network.

It is a benchmark, not automatically the final model. A multispectral CNN must
eventually outperform this baseline on genuinely unseen locations to justify
its extra complexity.

### Grouped Train/Test Split

Rows are split by `location_group`, not randomly. Every row from one location
goes entirely into training or testing:

```text
training locations: A, B, C
test locations:     D
```

A random row split could place nearly identical observations from location A on
both sides. The model could memorize that water body and appear accurate without
generalizing. Grouped evaluation is harder and more honest.

The random seed fixes the pseudo-random split and Random Forest construction.
Using the same dataset, library versions, test size, and seed should reproduce
the same experiment.

### Metrics

Numeric datasets use regression and report:

- MAE: average absolute prediction error in the target's unit
- RMSE: error metric that penalizes large mistakes more strongly
- R-squared: fraction of variation explained, when the test set has enough rows

Categorical datasets use classification and report accuracy, weighted precision,
weighted recall, weighted F1, labels, and a confusion matrix. Accuracy alone can
be misleading when one class is much more common.

### Saved Artifacts

The model pickle contains the estimator, ordered feature names, task type,
dataset SHA-256, and random seed. The JSON report stores metrics, train/test row
counts, and the exact location groups on each side.

Python pickle files can execute code while loading. Only load model artifacts
created by this trusted project; never load an untrusted `.pkl` file.

### Minimum Data And Honest Interpretation

The command refuses to train with fewer than four rows or fewer than two
location groups. That is only a software sanity check, not scientific
sufficiency. A credible model will usually need far more observations covering
different seasons, water-body types, atmospheric conditions, concentrations,
and geographic regions.

Do not describe a score from the two example CSV rows as model performance.
Those rows are formatting examples, not measured evidence. If the real database
has no matched measurements, export and training should stop rather than create
synthetic labels.

### Current Limitations And Next Work

- Matching uses scene-level cloud cover, not pixel-level cloud/shadow masks.
- One measurement gets one closest scene; temporal composites are not supported.
- Features are summary statistics, not raw multispectral image patches.
- Rounded coordinates approximate a location group.
- No uncertainty weighting uses detection limits or measurement quality yet.
- Model artifacts do not yet record every installed package version.

The next scientific improvement should add Sentinel-2 pixel quality masks and a
match-review command before scaling data acquisition. After that, import a real
open monitoring dataset and run this pipeline without example records.

### Commands Used To Verify These Stages

```powershell
.venv\Scripts\python.exe -m unittest tests.test_training_pipeline
.venv\Scripts\python.exe -m unittest discover -s tests
```

## 22. Human Review Of Satellite Matches

The matcher can find a satellite scene near a measurement in time and location,
but it cannot yet prove that the exact sampled water is visible and usable. Each
new match is therefore stored with review status `pending`. Pending matches are
excluded from `export-training` until a person approves them.

This is a quality-control gate between automatic data collection and machine
learning. It is much cheaper to reject a bad pair here than to discover later
that a model learned clouds, land pixels, or unrelated dates.

### Review Commands

List pending matches:

```powershell
.venv\Scripts\python.exe -m aquadoc_core.cli review-matches `
  --parameter turbidity `
  --status pending `
  --db data\aquadoc.sqlite3
```

The output includes the `match_id`, measurement and scene IDs, sample and scene
times, time difference in hours, cloud metadata, chip size, target value, and
current review status.

Approve a pair after checking it:

```powershell
.venv\Scripts\python.exe -m aquadoc_core.cli review-match MATCH_ID `
  --status approved `
  --note "Sampling time is close and imagery shows valid open water." `
  --db data\aquadoc.sqlite3
```

Reject a pair when it is unsuitable:

```powershell
.venv\Scripts\python.exe -m aquadoc_core.cli review-match MATCH_ID `
  --status rejected `
  --note "Time gap is too large for this rapidly changing water body." `
  --db data\aquadoc.sqlite3
```

Rejected pairs remain in the database. This preserves the audit trail and lets
us study why a source, location, season, or matching rule produced poor data.

### What To Check Before Approval

Check the pair against these questions:

- Is the time difference within the scientifically agreed window for this
  parameter? Turbidity and blooms may change within hours or days.
- Is the coordinate actually on the intended water body, rather than shore,
  land, an urban area, or a mixed pixel?
- Does the selected satellite date fall in a cloudy, shadowed, flooded, or
  otherwise unusual period?
- Is the sample source and quality status appropriate for model training?
- Are the measurement unit, depth, and laboratory/field method plausible for
  the selected target?

The current command shows scene-level cloud cover, which averages a large tile.
It is a useful warning but not proof of a clear target pixel. The next imaging
improvement will read Sentinel-2's pixel-level Scene Classification Layer (SCL)
and reject cloud, shadow, snow, and invalid pixels automatically before review.

### Why Export Is Now Strict

If no approved match exists, export stops with an error like:

```text
No approved matches found for parameter 'turbidity'; review 3 pending match(es) first
```

This is intentional. A training CSV should represent a reviewed dataset, not
merely whatever remote scenes happened to be returned by a catalogue query.

The match table records `review_status`, `review_note`, and `reviewed_at` in
addition to the original provenance fields. Review decisions are part of the
dataset's history and can be traced when a model is retrained.

### Commands Used To Verify This Pass

```powershell
.venv\Scripts\python.exe -m unittest tests.test_training_pipeline tests.test_cli
.venv\Scripts\python.exe -m aquadoc_core.cli review-matches --help
.venv\Scripts\python.exe -m aquadoc_core.cli review-match --help
```

## 23. Phase 0 — Repository Cleanup

### What was done

- Removed the Streamlit demo (`app.py`, `utils/`, `models/`, demo data generators, Streamlit deploy files). It's still in Git history.
- Stopped tracking ~23,000 files of `.venv/` with `git rm -r --cached .venv`. This removes them from Git's index but leaves your local files untouched.
- Wrote a real `.gitignore` (environments, caches, secrets, databases, large data).
- Moved the lab-matching workflow into `research/lab_matching/` with its own CLI and tests. The deployed package `aquadoc_core` no longer imports it.
- Split dependencies. `[project.dependencies]` is only what the API needs. `training`, `research` and `dev` are optional extras.
- Pinned Python 3.12 and verified with a fresh virtual environment.
- Fixed a Windows-only test failure: `with sqlite3.connect(...)` commits but does **not** close the connection, so Windows refused to delete the temporary database. `contextlib.closing(...)` fixes it.

### Concepts to study

| Concept | Why it matters here | Resource |
| --- | --- | --- |
| `.gitignore` patterns | Keep environments, secrets and data out of the repository | https://git-scm.com/docs/gitignore |
| Index vs. working tree (`git rm --cached`) | Untrack files without deleting them | https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository |
| Virtual environments | Isolated, reproducible installs; never commit them | https://docs.python.org/3/library/venv.html |
| `pyproject.toml` and optional dependencies | One source of truth for what to install, per use case | https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ |
| Package layout and dependency direction | The deployed package must not depend on research code | https://docs.python.org/3/tutorial/modules.html#packages |
| Context managers | Why `with connect()` ≠ `with closing(connect())` | https://docs.python.org/3/library/sqlite3.html#sqlite3-connection-context-manager |

### Redo it yourself

1. `git ls-files | wc -l` to see how much is tracked. `git ls-files | grep -c '^.venv'` shows how much of that is the environment.
2. Write `.gitignore`, then `git rm -r --cached .venv` and `git status` to confirm the files still exist on disk.
3. Draw the import graph: `grep -rn "^from \.\|^import" aquadoc_core/`. Anything the API doesn't need should move out.
4. Create a throwaway venv (`py -3.12 -m venv /tmp/v`), `pip install -e ".[dev]"`, run the tests. If anything is missing from `pyproject.toml`, this step exposes it. Here it found that `shapely` and `pyshp` were used but never declared.

### Exercises

- Why is a stale lockfile worse than none for a deployment platform?
- Install only `pip install -e .` (no extras) and run `python -c "import aquadoc_core.api"`. Which packages had to be present?


## 24. Phase 1 — Training A Water-Quality Model On Public Spectral Data

### The idea in one paragraph

A satellite pixel gives you reflectance in a handful of bands. To turn reflectance into "12 µg/L chlorophyll" a model must have seen many examples of *reflectance → measured concentration*. Collecting those yourself means boats, bottles and labs. Instead we use **GLORIA**, a public dataset of 7,572 hyperspectral reflectance spectra measured just above the water surface in lakes, rivers, estuaries and coasts, each paired with lab results. We convert each 1-nm spectrum into what Sentinel-2 would see, train on that, and later apply the model to real satellite pixels.

### What was built

| File | Role |
| --- | --- |
| `training/download.py` | Downloads GLORIA (PANGAEA) and the ESA Sentinel-2 spectral response functions, verifying pinned SHA-256 checksums |
| `training/srf.py` | Parses spectral response functions; convolves hyperspectral spectra into Sentinel-2 bands |
| `training/gloria.py` | Loads GLORIA, removes QC-flagged spectra, simulates bands B2–B7 and B8A, attaches targets and spatial groups |
| `training/train.py` | Spatial train/calibration/test split, MLP ensembles, noise augmentation, conformal intervals, baselines, export |
| `training/model_card.py` | Writes `docs/models/water_quality_v1.md` from the exported metadata |
| `aquadoc_core/wq_features.py` | The feature builder, shared by training and inference so they can never drift apart |
| `aquadoc_core/wq_algorithms.py` | Published algorithms: Mishra NDCI chlorophyll, Nechad TSS, Dogliotti turbidity |
| `aquadoc_core/wq_model.py` | NumPy-only inference: loads `models/water_quality_v1.{npz,json}`, returns median, interval and an out-of-domain flag per pixel |

Results are in `docs/models/water_quality_v1.md`. On regions the model never saw, chlorophyll-a has a median error of about 47%, against 84% for the best-known published formula. The model is best in lakes and coasts and weakest in rivers and estuaries.

### Concepts to study, in order

**1. Remote-sensing reflectance (Rrs).** Rrs = water-leaving radiance ÷ downwelling irradiance (sr⁻¹). It describes the water itself, independent of how bright the day is. Satellite L2A "surface reflectance" ρ relates to it as Rrs ≈ ρ/π.
- Ocean Optics Web Book, *Reflectances*: https://www.oceanopticsbook.info/view/radiometry-and-photometry/reflectances
- Ocean Optics Web Book (overview): https://www.oceanopticsbook.info/

**2. Why each band sees a different substance.**
- Chlorophyll absorbs strongly at ~665 nm (B4). Algal cells scatter, so reflectance peaks near 705 nm (B5). The ratio of B5 to B4 is the NDCI.
- CDOM (dissolved organic matter) absorbs blue (B2) far more than green (B3).
- Suspended sediment raises reflectance across red and NIR (B4, B7, B8A). The more turbid the water, the further into the NIR the signal survives.
- Water itself absorbs NIR strongly, so clear water is almost black beyond 700 nm.
- Gholizadeh et al. (2016), *A Comprehensive Review on Water Quality Parameters Estimation Using Remote Sensing Techniques*, Sensors 16(8):1298 (open access): https://www.mdpi.com/1424-8220/16/8/1298
- IOCCG Report 3, *Remote Sensing of Ocean Colour in Coastal and Other Optically-Complex Waters* (free PDF): https://ioccg.org/reports/report3.pdf

**3. Spectral response functions and band convolution.** A band is a weighted average over a wavelength range. `training/srf.py` implements `sum(Rrs·S) / sum(S)`. Rows missing part of the band are rejected rather than averaged over the part they do have.
- Sentinel-2 mission and MSI bands: https://sentiwiki.copernicus.eu/web/s2-mission

**4. The GLORIA dataset.** Read its paper before using it. It explains the QC flags we filter on.
- Lehmann et al. (2023), Scientific Data 10:100: https://www.nature.com/articles/s41597-023-01973-y
- Data: https://doi.pangaea.de/10.1594/PANGAEA.948492

**5. Log-space modelling and the right metrics.** Concentrations span 4–5 orders of magnitude and errors scale with concentration. So we predict log10(value) and score with MdSA (median symmetric accuracy; 100% = typical factor-of-two error) and SSPB (signed bias).
- Morley, Brito & Welling (2018), Space Weather 16: https://doi.org/10.1002/2017SW001669
- Seegers et al. (2018), *Performance metrics for the assessment of satellite data products*, Optics Express 26(6): https://doi.org/10.1364/OE.26.007404

**6. Spatial (grouped) validation.** Nearby stations are almost duplicates. A random split would let the model "recognise" a lake instead of learning optics. We group by ~28 km cells and hold out entire groups.
- scikit-learn, grouped cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html#group-cv
- Roberts et al. (2017), Ecography 40: https://doi.org/10.1111/ecog.02881

**7. Neural networks and ensembles.** Each target has 10 small MLPs, each trained on a bootstrap sample. Their disagreement measures uncertainty.
- scikit-learn MLP guide: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
- Lakshminarayanan et al. (2017), *Deep Ensembles*: https://arxiv.org/abs/1612.01474
- Pahlevan et al. (2020), Mixture Density Networks for Sentinel-2/3 chlorophyll, RSE 240. This is the state-of-the-art version of what we built: https://doi.org/10.1016/j.rse.2019.111604

**8. Augmentation for the sensor gap.** Field radiometers are cleaner than satellites, so we add random gain (5%) and offset (0.0005 sr⁻¹) noise to training copies. This only approximates real atmospheric-correction error.
- Pahlevan et al. (2021), ACIX-Aqua atmospheric correction intercomparison, RSE 258: https://doi.org/10.1016/j.rse.2021.112366

**9. Conformal prediction intervals.** On held-out calibration groups we measure how wrong the model is relative to its own uncertainty, then take the 80th percentile. That makes an interval that should contain the truth about 80% of the time. Coverage is checked on the test groups.
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*: https://arxiv.org/abs/2107.07511

**10. Baselines and a pre-registered rule.** A model is only worth shipping if it beats (a) predicting the training median and (b) the published formula. That rule is in code and was fixed *before* looking at results.
- Mishra & Mishra (2012), NDCI, RSE 117: https://doi.org/10.1016/j.rse.2011.10.016
- Nechad, Ruddick & Park (2010), TSS, RSE 114: https://doi.org/10.1016/j.rse.2009.11.022
- Dogliotti et al. (2015), turbidity, RSE 156: https://doi.org/10.1016/j.rse.2014.09.020

**11. Exporting a model without its framework.** The trained MLP is just matrices. We save them with `np.savez` and do the forward pass with `x @ W + b` and ReLU. The API then doesn't need scikit-learn, which keeps the Vercel bundle small. A test confirms that NumPy and scikit-learn produce the same predictions.
- NumPy `savez`: https://numpy.org/doc/stable/reference/generated/numpy.savez.html

### Redo it yourself

```powershell
pip install -e ".[training,dev]"
python -m training.download          # ~60 MB GLORIA + SRFs, checksums verified
python -m training.train             # ~10-15 min on a laptop CPU
python -m training.model_card        # regenerate docs/models/water_quality_v1.md
python -m unittest discover -s tests
python -m unittest discover -s training/tests -t .
```

Then, step by step in a notebook:

1. Load `GLORIA_Rrs.csv`. Plot ten high-chlorophyll and ten low-chlorophyll spectra. Find the 665 nm trough and the 705 nm peak.
2. Load `S2A_MSI_V4.txt` with `training.srf.parse_acolite_rsr` and plot B4 and B5 over a spectrum.
3. Convolve one spectrum by hand with NumPy and compare with `training.srf.convolve`.
4. Compute NDCI for all spectra and scatter it against log10(Chla). This is the signal the model learns.
5. Train one `MLPRegressor` with a *random* split and one with the grouped split. Compare MdSA. The gap you see is spatial leakage.

### Exercises

1. Turbidity intervals cover only ~56% of test samples, not 80%. Why might calibration and test groups differ? Hint: count how few groups have turbidity data.
2. Rivers and estuaries score much worse than lakes. List three optical reasons.
3. Increase `additive_noise` to 0.001 and retrain. What happens to clean-test and noisy-test MdSA? Which matters more for satellites?
4. Add B8A as a model band. How many training spectra do you lose, and does TSS improve?
5. Gradient boosting beats the MLP on turbidity and CDOM. Why might we still deploy one model family for everything? What would it take to deploy the better one per target?

### Honest limitations

- Scores are for simulated Sentinel-2 bands built from field spectra. On real satellite imagery, expect worse results, especially for CDOM (blue bands) and small or dark lakes.
- "Deployed" means it beat the baselines, not that it is accurate enough for regulatory or drinking-water decisions.
- Some ensemble members stopped at the 300-iteration limit before formally converging. Held-out scores already account for this, but it's a tuning opportunity.
- Non-optical pollutants (metals, bacteria, nutrients, pesticides) can't be estimated this way at all.

## 25. Phase 2 — From Satellite Scene To Per-Pixel Pollutant Maps

### What happens when you analyse a lake

```text
locate   → OSM outline: Nominatim lookup by id (~1 s) or Overpass point query (20-50 s, run in parallel)
search   → Sentinel-2 L2A catalogue (STAC), newest first, one tile per acquisition
scene    → read 10 bands + SCL on one 20 m UTM grid; mask clouds, shadows, shore, scum, glint
           → if under 20% of the water is clear, try the next older acquisition
estimate → 3×3 smoothing → Rrs = ρ/π → model (5 targets) + published algorithms + FDI
explain  → median spectrum + band sensitivity (which band moved each estimate here)
render   → PNG layers: true colour, pixel classes, one heatmap per target, debris index
complete → JSON result (~15 KB) + PNGs
```

Try it:

```powershell
python -m aquadoc_core.cli water --lat 17.4239 --lon 78.4738 --osm relation/2833155 --out output/hussain
python -m aquadoc_core.cli water --lat 41.70 --lon -83.25 --out output/erie
```

### What was built

| File | Role |
| --- | --- |
| `aquadoc_core/waterbody.py` | Nominatim search/lookup and Overpass point lookup; assembles OSM multipolygon relations (outer/inner rings) with Shapely |
| `aquadoc_core/scene_search.py` | STAC search; drops tiles that barely cover the area; keeps one tile per acquisition |
| `aquadoc_core/s2_reader.py` | Explicit 20 m UTM grid; parallel windowed COG reads warped onto it; reflectance scale/offset from metadata |
| `aquadoc_core/water_masks.py` | Ordered pixel classification, including geometric cloud-shadow projection |
| `aquadoc_core/render.py` | A ~40-line PNG encoder (zlib + CRC) and colour ramps, with no imaging library needed |
| `aquadoc_core/band_guide.py` | What each band reveals, units, trophic scale, and what *cannot* be detected |
| `aquadoc_core/water_pipeline.py` | Orchestrates the stages as a generator of events (streamable to the dashboard in Phase 3) |
| `tests/test_water_pipeline.py` | 16 offline tests with fake services and synthetic scenes |

### The real-world lesson from this phase: always look at the pictures

The first Lake Erie run "worked": it returned plausible numbers in 38 s. The true-colour image showed scattered cumulus, but Sentinel-2's own cloud mask (SCL) had marked **zero** shadow pixels. Shadowed water just looks like dark water. The suspended-solids heatmap was visibly tracing the shadows. Fixes, in the order we discovered them:

1. **Project shadows geometrically.** A cloud at height *h* casts a shadow *h·tan(solar zenith)* away from the sun. We shift the cloud mask along that direction for heights 200–4000 m (sun angles come from the STAC item).
2. **Only call it shadow if it is dark.** Inside that zone, a pixel counts as shadow if its visible brightness is < 80% of the local mean of cloud-free water (1.2 km window). Using only pixels *outside* the zone as reference failed: with many clouds, the zone covered 69% of the window and everything got flagged.
3. **Clouds outside the window cast shadows inside it.** We read the cheap SCL band over a wider area (up to 6 km margin) just for this.
4. **Haze around clouds.** The cloud buffer was raised from 40 m to 100 m.
5. **Relative, not absolute, debris thresholds.** A fixed FDI threshold flagged 27% of turbid Lake Erie as "debris". Using "4 robust standard deviations above this lake's median" brought it to ~1.6%.
6. **Noise.** Chlorophyll depends on the small difference between B5 and B4, so per-pixel sensor noise was amplified ~10×. A 3×3 mean of valid neighbours before the model removed the speckle.

### Concepts to study

**Coordinate reference systems and grids.** Sentinel-2 tiles use UTM. We build one explicit grid (CRS + affine transform + width/height) and warp every band onto it. Equal array shapes are *not* equal geography.
- Rasterio: georeferencing and affine transforms: https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html
- Rasterio: virtual warping (`WarpedVRT`): https://rasterio.readthedocs.io/en/stable/topics/virtual-warping.html
- UTM explained: https://www.usgs.gov/faqs/how-are-utm-coordinates-measured-usgs-topographic-maps

**Cloud Optimized GeoTIFFs and STAC.** HTTP range requests let us read a 5 km window from a 110 km tile without downloading it.
- COG intro: https://cogeo.org/
- STAC spec and the item-search API: https://stacspec.org/en/tutorials/intro-to-stac/
- Planetary Computer Sentinel-2 L2A: https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a

**Sentinel-2 Level-2A products.** Reflectance scaling and the processing-baseline-04.00 offset (−0.1), and the Scene Classification Layer classes.
- Sentinel-2 L2A algorithm / SCL: https://sentiwiki.copernicus.eu/web/s2-processing
- Processing baseline 04.00 offset: https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-L2AProcessingBaseline

**Cloud and shadow masking.** The geometry and darkness ideas above come from Fmask.
- Zhu & Woodcock (2012), *Object-based cloud and cloud shadow detection in Landsat imagery* (Fmask), RSE 118: https://doi.org/10.1016/j.rse.2011.10.028
- Skakun et al. (2022), *Cloud mask intercomparison eXercise (CMIX)* for Sentinel-2, RSE 274: https://doi.org/10.1016/j.rse.2022.112990

**Water detection and adjacency.** MNDWI (green vs SWIR) separates water from land. Pixels near shore are contaminated by bright land light ("adjacency effect"), hence the 40 m inward shoreline buffer.
- Xu (2006), MNDWI, Int. J. Remote Sens. 27(14): https://doi.org/10.1080/01431160600589179
- Adjacency effects over inland water: Vanhellemont & Ruddick (2018), RSE 216 (ACOLITE): https://doi.org/10.1016/j.rse.2018.07.015

**Floating material.** FDI compares NIR with a baseline interpolated between red-edge and SWIR. Floating things are NIR-bright; water is not.
- Biermann et al. (2020), *Finding Plastic Patches in Coastal Waters using Optical Satellite Data*, Sci. Rep. 10: https://doi.org/10.1038/s41598-020-62298-z

**Robust statistics.** The median and the MAD (median absolute deviation, ×1.4826 ≈ standard deviation for normal data) aren't fooled by the outliers we are trying to find.
- NIST handbook, measures of scale: https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm

**Explaining a model locally (sensitivity / elasticity).** We nudge each band ±10% at this lake's median spectrum and see how much each estimate moves. That's a simple local explanation, related to gradients and SHAP.
- Molnar, *Interpretable Machine Learning* (free book): https://christophm.github.io/interpretable-ml-book/

**OpenStreetMap data model.** Ways vs relations, multipolygons with outer/inner roles, and ODbL attribution.
- OSM multipolygons: https://wiki.openstreetmap.org/wiki/Relation:multipolygon
- Overpass QL: https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL
- Nominatim API and usage policy: https://nominatim.org/release-docs/latest/api/Overview/ and https://operations.osmfoundation.org/policies/nominatim/

**Python concurrency.** A `ThreadPoolExecutor` overlaps slow network calls (Overpass, 10 COG reads). GDAL releases the GIL during I/O, so threads help.
- `concurrent.futures`: https://docs.python.org/3/library/concurrent.futures.html

**Generators as progress streams.** `run_water_analysis` *yields* events. The CLI prints them now; in Phase 3 the API will stream them to the browser with Server-Sent Events.
- Python generators: https://docs.python.org/3/howto/functional.html#generators

**The PNG format.** Signature, chunks, CRC32, zlib-compressed scanlines with a filter byte per row.
- PNG specification: https://www.w3.org/TR/png-3/

### Redo it yourself

1. Search a scene for a lake with `aquadoc_core.scene_search.search_scenes` and print the candidates.
2. Build a grid with `make_grid`, read `B04` and `SCL` with `CogBandReader`, and save a true-colour PNG with `render.true_color`. Open it and look for clouds.
3. Compute `project_shadow_zone` for the cloud mask and overlay it. Does the zone land on the dark patches? Try azimuth ±20° to see how sensitive it is.
4. Compare `counts` from `classify_pixels` with and without sun angles.
5. Run the full CLI on a lake you know. Then pass `--start/--end` for a summer month and compare chlorophyll.

### Exercises

1. Why is 20 m, not 10 m, the analysis resolution? Which bands would need upsampling at 10 m, and what would that imply?
2. Hussain Sagar had 0% of pixels in the turbidity training domain. Look at its spectrum in `result.json`. Which band is unusual compared with Lake Erie? What could cause it (think: small urban lake, adjacency)?
3. The shadow test uses a fixed 0.8 darkness ratio. Find a scene where it fails, either missing shadows over very dark water or flagging natural plumes. Propose a better test.
4. Why does the pipeline try *older* scenes rather than averaging several dates?
5. `debris_like_pct` uses a 4×MAD threshold. What false positives remain (hint: boats, bridges, wakes, small islands)?

### Honest limitations

- Sen2Cor atmospheric correction is land-oriented. Over water, ACOLITE or POLYMER would be better but are too heavy for a serverless function. This is the largest error source left.
- Shadow detection is heuristic; thin cirrus and haze are only partly caught.
- Estimates for small urban lakes are often out of the training domain. The result flags this per target (`in_training_domain_pct`, `confidence`).
- Confidence is at most "medium". "High" would require validating against satellite–in-situ match-ups, which we chose not to build.

## 26. Reducing The Limitations — Satellite Match-Up Validation

### Why

After Phase 2 the honest answer to "how accurate is this on real satellite images?" was "we don't know". The model was tested on *simulated* Sentinel-2 bands made from field spectra. Real images pass through Sen2Cor atmospheric correction, which was built for land. This section measures that gap and shrinks it.

### Method: satellite match-ups

A *match-up* is a satellite pixel and a field measurement taken at the same place at nearly the same time. GLORIA has 1,365 samples from 2017 onward, inside the Sentinel-2 era. For each one, `training/matchups.py`:

1. searches Sentinel-2 L2A scenes within ±24 h,
2. reads a 7×7 pixel (140 m) box centred on the sample,
3. rejects it if any pixel is cloud or shadow, if the centre 3×3 is not water, if SWIR shows glint or land, or if the centre is not homogeneous (coefficient of variation > 0.2),
4. keeps the median of the centre 3×3.

Result: **506 usable match-ups at 107 sites.** The rest were lost to no overpass (597), clouds (179) and the quality checks.

`training/validate_satellite.py` then:

- compares satellite reflectance with in-situ reflectance band by band;
- fits a **residual correction** per band: `Rrs = a + g·Rrs_L2A + k·Rrs_L2A(B11)`, using a robust Huber regression. Each site is scored with coefficients fitted *without* that site;
- **retrains the model without any match-up site**, so the accuracy numbers are genuinely out-of-sample;
- recalibrates the 80% intervals on *satellite* residuals (conformal prediction, coverage checked by grouped cross-validation);
- picks, per target, raw or corrected input, whichever validated better.

### What we learned

| | Chlorophyll-a | Suspended solids | CDOM | Turbidity | Secchi |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model, field-spectra input (best case) | 53% | 60% | 97% | 53% | 35% |
| Model, raw Sentinel-2 | 94% | 103% | 152% | 67% | **43%** |
| Model, corrected Sentinel-2 | **78%** | **76%** | **115%** | **58%** | 73% |
| Best published algorithm, corrected | 103% | 87% | – | 56% | – |

(Median symmetric accuracy; 100% = typical factor-of-two error.)

- Sen2Cor leaves a **positive offset of ~0.002–0.004 sr⁻¹ over water in every band**, which is large relative to dark-water reflectance (~0.002–0.01). The fitted correction is essentially "subtract ~0.8 × the SWIR signal", because water should be black in SWIR, so leftover SWIR measures the residual aerosol, adjacency and glint error. The offset falls ~10× after correction.
- The correction helps four targets and *hurts* Secchi depth. That's why the choice is made per target, not globally.
- CDOM stays worse than a factor of two, so the dashboard always marks it **low** confidence. Blue bands carry the largest atmospheric error.
- Intervals recalibrated on satellite data cover 76–80% of unseen sites, close to the 80% target. Before, turbidity covered only 56%.
- Hussain Sagar is the practical example. Raw reflectance put 72% of pixels *outside* the training domain. After correction 100% were inside, and the chlorophyll estimate moved from 17 to 81 µg/L, consistent with the lake's documented hypereutrophic state.

### Concepts to study

- **Match-up validation protocol** (window size, homogeneity, time window): Bailey & Werdell (2006), *A multi-sensor approach for the on-orbit validation of ocean color satellite data products*, RSE 102: https://doi.org/10.1016/j.rse.2006.01.015
- **Atmospheric correction over inland water and its typical errors**: Pahlevan et al. (2021), ACIX-Aqua, RSE 258: https://doi.org/10.1016/j.rse.2021.112366
- **Why SWIR helps**: water absorbs SWIR almost completely, so SWIR-based corrections (e.g. Wang & Shi 2007, *The NIR-SWIR combined atmospheric correction approach for MODIS ocean color data processing*, Optics Express 15): https://doi.org/10.1364/OE.15.015722
- **Robust regression (Huber loss)**: scikit-learn guide: https://scikit-learn.org/stable/modules/linear_model.html#huber-regression
- **Out-of-sample evaluation when correcting data**: fit the correction inside cross-validation folds, never on the points you score.
- **Conformal prediction under distribution shift**: recalibrating on data from the deployment domain: Angelopoulos & Bates (see section 24).

### Redo it yourself

```powershell
python -m training.matchups                         # ~30-60 min, resumable
python -m training.validate_satellite               # report -> docs/models/satellite_validation.json
python -m training.validate_satellite --write-model # also update aquadoc_core/models/water_quality_v1.json
python -m training.model_card
```

### Exercises

1. Restrict match-ups to ±3 h (`--max-hours 3`). How many remain, and do errors fall? What does that say about how fast lake water changes?
2. Plot satellite vs in-situ B2 reflectance before and after correction. Where does the correction fail?
3. Why might the correction hurt Secchi depth while helping chlorophyll? (Hint: look at which bands each target's sensitivity relies on.)
4. 440 of 506 match-ups are lakes. Design a check to warn users when an analysed water body is a type the validation barely covers.

## 27. Phase 3 — The HTTP API

### What was built

| File | Role |
| --- | --- |
| `aquadoc_core/api.py` | FastAPI app under `/api`: guide, search, streaming analyses, results, layers |
| `aquadoc_core/store.py` | Result + PNG storage: SQLite locally, PostgreSQL (Neon) when `DATABASE_URL` is set |
| `tests/test_api.py` | Streaming, caching, layer serving, validation errors, rate limit, all offline |

Key design decisions:

- **Server-Sent Events** for progress. `POST /api/analyses/stream` returns `text/event-stream`: one `stage` event per pipeline step, then `complete` (or `error`). SSE is plain HTTP, so it works through Vercel's functions and needs no WebSocket server. The pipeline runs in a worker thread, and the response sends a `: keep-alive` comment every 10 s so proxies don't close idle connections.
- **Stateless functions, durable store.** Serverless instances share no memory and disappear. Results, images and the cache live in the database, and nothing important is kept in process memory.
- **Images in the database**, served with `Cache-Control: immutable`. Layers never change after creation, so the CDN and browsers cache them forever. That avoids a separate object store.
- **Request cache**: same water body or point plus same dates (including the default "last 60 days ending today") within 24 h returns the stored result in ~0.1 s instead of ~10 s of satellite processing.
- **Validation at the edge of the system**: Pydantic rejects bad coordinates, future dates and malformed OSM references before any work starts.
- **Honest rate limiting**: the in-memory limiter is per instance and only a speed bump. Real limits belong in the Vercel Firewall.

Try it:

```powershell
python -m uvicorn aquadoc_core.api:app --reload
curl.exe "http://127.0.0.1:8000/api/waterbodies/search?q=Hussain%20Sagar"
curl.exe -N -X POST http://127.0.0.1:8000/api/analyses/stream -H "Content-Type: application/json" -d "{\"latitude\":17.4239,\"longitude\":78.4738,\"osm_ref\":\"relation/2833155\"}"
```

Then open http://127.0.0.1:8000/api/docs.

### Concepts to study

- **FastAPI**: path operations, Pydantic models, dependency injection, `StreamingResponse`: https://fastapi.tiangolo.com/tutorial/ and https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
- **Server-Sent Events** format (`event:`, `data:`, blank line; comments start with `:`): https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
- **HTTP caching** (`Cache-Control`, `immutable`, CDN caching): https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching
- **CORS**, and why the dashboard on another origin needs it in development: https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
- **Serverless constraints**: stateless instances, duration and payload limits: https://vercel.com/docs/functions/limitations and https://vercel.com/docs/fluid-compute
- **FastAPI on Vercel**: https://vercel.com/docs/frameworks/backend/fastapi
- **PostgreSQL basics and psycopg 3**: https://www.postgresql.org/docs/current/tutorial.html and https://www.psycopg.org/psycopg3/docs/basic/usage.html
- **Rate limiting** (sliding window) and why it belongs at the edge: https://vercel.com/docs/vercel-firewall/vercel-waf/rate-limiting
- **Testing APIs with `TestClient`** and fakes instead of the network: https://fastapi.tiangolo.com/tutorial/testing/

### Exercises

1. Write a 20-line Python client that reads the SSE stream with `requests` (`stream=True`, `iter_lines`) and prints each stage.
2. Why are cached results keyed by pipeline *and* model version? What would go wrong without the model version?
3. The rate limiter is in memory. Describe two ways a determined client could bypass it on Vercel, and how the Firewall avoids both.
4. Run two identical requests at the same moment. Both miss the cache and do the work twice. How would you prevent that?

# AquaDoc: Satellite Water-Quality Analysis

> Check the health of any lake, reservoir or river from space. AquaDoc reads the latest clear Sentinel-2 satellite image, removes clouds, shadows and shoreline, estimates algae, sediment, clarity and dissolved organic matter pixel by pixel, and shows every step of how it got there.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Dashboard-Next.js%2016-000000?logo=nextdotjs&logoColor=white)
![Sentinel-2](https://img.shields.io/badge/Imagery-Sentinel--2%20L2A-1B5E20)
![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?logo=vercel&logoColor=white)

**Live demo:** https://aquadoc-ivory.vercel.app

## What it is

- **Water quality without a lab visit.** Pick a featured lake, search by name, type coordinates, drop a pin or use "Lakes near me". AquaDoc estimates **chlorophyll-a, suspended solids, turbidity, water clarity (Secchi depth) and CDOM**, and flags floating vegetation and surface scum.
- **A transparent 7-stage pipeline.** The stages are locate, acquire, read, mask, correct, estimate and explain. Each one streams live to the dashboard with the image it produced, so you can see why a pixel was kept or removed and which spectral bands drove each estimate.
- **Honest about its limits.** Every estimate comes with an uncertainty range and a confidence level measured on real satellite data. When clouds, floating vegetation, glint or a too-small water body make an image unusable, AquaDoc stops and explains why instead of guessing.

> Satellites see only **optically active** constituents. Heavy metals, bacteria, dissolved nutrients and most chemicals cannot be estimated from imagery and need laboratory testing.

## How it works

```text
Lake outline (OpenStreetMap) → Sentinel-2 search (Planetary Computer STAC) → windowed COG band reads (20 m)
→ cloud / shadow / shoreline / scum masking → atmospheric residual correction → neural-network ensemble per pixel
→ band sensitivity + index maps → stored result with map layers and a replayable timeline
```

## Getting Started

These instructions get a copy of the project running on your local machine for development and testing. See [Deployment](#deployment) for deploying to Vercel.

### Prerequisites

- **Python 3.12** (the API and pipeline)
- **Node.js 20+** and npm (the dashboard)
- Internet access: imagery comes from Microsoft Planetary Computer and outlines from OpenStreetMap. Neither needs an API key.

### Installing

1. Clone the repository and create a virtual environment:

   ```powershell
   git clone https://github.com/NR0498/AQUADOCV2.git
   cd AQUADOCV2
   python -m venv .venv
   .venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
   ```

2. Install the API with its test client:

   ```powershell
   pip install -e ".[dev]"
   # Optional extras:
   # pip install -e ".[training]"     # offline model training
   # pip install -e ".[research]"     # shelved lab-matching workflow
   ```

3. Install the dashboard:

   ```powershell
   cd web
   npm install
   ```

4. Run the API and the dashboard together:

   ```powershell
   npm run dev:all
   ```

   Open http://localhost:3000. Use `localhost`, not `127.0.0.1`. Choose **Hussain Sagar** under featured lakes and press **Analyse**. You should see the seven stages complete one by one, followed by the estimates, maps and band explanations.

You can also run a single analysis from the command line. It prints each stage live, then writes `result.json` and PNG map layers:

```powershell
python -m aquadoc_core.cli water --lat 17.4239 --lon 78.4738 --osm relation/2833155 --out output/hussain
```

## Running the tests

All tests run offline with synthetic imagery, so they need no network access.

```powershell
python -m unittest discover -s tests                       # API, pipeline, masking, model, persistence
python -m unittest discover -s training/tests -t .         # model training and validation
python -m unittest discover -s research/tests -t .         # lab-matching workflow
```

### What the tests cover

- **Pipeline** (`tests/test_water_pipeline.py`): a full run on a synthetic lake. It also covers cloud and shadow rejection, fallback to older images, and that every failure (clouds, vegetation, no scenes, invalid input) is reported with a reason code.
- **API** (`tests/test_api.py`): streaming Server-Sent Events, caching, rate limiting and the layer endpoints.
- **Model** (`tests/test_wq_model.py`): spectral features, published algorithms (NDCI, Nechad, Dogliotti), model save and load, uncertainty intervals, and handling of invalid pixels.

### Coding style

The dashboard is type-checked and linted:

```powershell
cd web
npx tsc --noEmit
npm run lint
```

## API

Interactive docs are at http://127.0.0.1:8000/api/docs while the API is running.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health check, including geospatial library status |
| `GET` | `/api/guide` | Band explanations, parameters, model and satellite-validation metrics |
| `GET` | `/api/waterbodies/search?q=` | Find water bodies by name (OpenStreetMap) |
| `POST` | `/api/analyses/stream` | Run an analysis and stream each stage as Server-Sent Events |
| `POST` | `/api/analyses` | Same, returning only the final result |
| `GET` | `/api/analyses?limit=` | Recent analyses |
| `GET` | `/api/analyses/{id}` | A stored result |
| `GET` | `/api/analyses/{id}/layers/{name}.png` | Map layer image |

Example request body: `{"latitude": 17.4239, "longitude": 78.4738, "osm_ref": "relation/2833155"}`. Identical requests within 24 hours return the stored result. A failed run ends with a `failed` stage event whose `data.failure.reason` is one of `cloud`, `floating_vegetation`, `not_open_water`, `glint`, `too_small`, `unreadable`, `no_scenes`, `catalogue_error` or `invalid_request`.

## Deployment

The dashboard and the API deploy together as one Vercel project (`vercel.json`). `/api/*` routes to the FastAPI app (`main.py`) and everything else to Next.js. Results are stored in PostgreSQL (Neon) when `DATABASE_URL` is set, and in SQLite otherwise.

```powershell
vercel deploy          # preview
vercel deploy --prod   # production
```

Pushing to `main` also deploys production. After deploying, `/api/health` should return `"status": "ok"` and `"geospatial": "ok"`. Some Vercel instances lack the system library `libexpat.so.1` that GDAL needs, so `aquadoc_core/native.py` loads a vetted copy from `aquadoc_core/vendor/` when it is missing.

## Built With

| Area | Technology |
| --- | --- |
| Satellite data | [Sentinel-2 L2A](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC, Cloud Optimized GeoTIFFs |
| Geospatial | [rasterio](https://rasterio.readthedocs.io/) (GDAL), [Shapely](https://shapely.readthedocs.io/), [pystac](https://pystac.readthedocs.io/), OpenStreetMap (Nominatim and Overpass) |
| Science and ML | [NumPy](https://numpy.org/); a neural-network ensemble trained on the [GLORIA](https://doi.org/10.1594/PANGAEA.948492) in-situ reflectance dataset; published algorithms (NDCI, Nechad, Dogliotti, FDI) |
| Backend | [FastAPI](https://fastapi.tiangolo.com/), Pydantic, Uvicorn, Server-Sent Events |
| Storage | PostgreSQL ([Neon](https://neon.tech/)) in production, SQLite locally |
| Frontend | [Next.js 16](https://nextjs.org/) (App Router), React 19, TypeScript, [Tailwind CSS 4](https://tailwindcss.com/), [MapLibre GL](https://maplibre.org/) |
| Hosting | [Vercel](https://vercel.com/) (Python on Fluid Compute and Next.js in one project) |

## Repository structure

```text
AQUADOCV2/
├── aquadoc_core/   # Deployed backend: ingestion, masking, model, pipeline, API, CLI
├── web/            # Next.js dashboard
├── training/       # Offline model training and satellite validation
├── tests/          # Offline tests for aquadoc_core
├── research/       # Shelved lab-measurement matching workflow (not deployed)
├── data/           # Manifests and small training tables (raw data is git-ignored)
├── docs/           # Model card, dataset notes and project audit
├── scripts/        # One-off build helpers
└── main.py         # Vercel entrypoint for the API
```

## Accuracy and limitations

- Accuracy was measured on **506 real Sentinel-2 match-ups at 107 sites** the model never saw. The typical error is about 58% for turbidity, 76–78% for chlorophyll-a and suspended solids, and 43% for Secchi depth. CDOM error exceeds 100%, so CDOM is always marked low confidence. See the [model card](docs/models/water_quality_v1.md).
- Sen2Cor atmospheric correction is designed for land. A residual correction learned from the match-ups removes most of its offset over water, but not all of it.
- Validation is dominated by lakes. Accuracy for rivers and estuaries is largely unknown.
- Lakes that are cloudy for long periods, covered by floating vegetation, or smaller than a few 20 m pixels cannot be analysed. The dashboard explains which of these applied.
- AquaDoc is a research and educational tool, not a regulatory or drinking-water safety system.

## Contributing

Issues and pull requests are welcome. Please:

1. Fork the repository and create a feature branch (`git checkout -b feature/my-change`).
2. Keep the tests passing and add tests for new behaviour.
3. Open a pull request that describes what changed and why.

## Versioning

The project uses [SemVer](https://semver.org/). The package version is in `pyproject.toml` (currently `0.2.0`). The pipeline version (`water-pipeline-v4`) is reported by `/api/health` and stored with every result. See the [commit history](https://github.com/NR0498/AQUADOCV2/commits/main) for changes.

## Authors

- **NR0498**: design and development ([GitHub](https://github.com/NR0498))

## License

Developed for educational and research use. Check data-source licences and attribution requirements (Copernicus Sentinel data, OpenStreetMap contributors, GLORIA dataset) before redistribution.

## Acknowledgments

- ESA Copernicus programme for free Sentinel-2 imagery
- Microsoft Planetary Computer for STAC search and cloud-hosted imagery
- The GLORIA dataset authors for the global in-situ reflectance and water-quality data
- OpenStreetMap contributors for water-body outlines
- The [beginner's guide to writing a kickass README](https://meakaakka.medium.com/a-beginners-guide-to-writing-a-kickass-readme-7ac01da88ab3) that this README follows

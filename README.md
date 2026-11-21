# AquaDoc

> Estimate optically detectable water-quality parameters for any water body from Sentinel-2 multispectral imagery, and show how each band contributes to the estimate.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Sentinel--2](https://img.shields.io/badge/Imagery-Sentinel--2%20L2A-1B5E20)
![Status](https://img.shields.io/badge/Status-In%20development-F59E0B)

## What it does

1. Pick a water body or coordinate.
2. Find the most recent clear Sentinel-2 Level-2A scene (Microsoft Planetary Computer STAC, no API key needed).
3. Read only the bands and area needed (Cloud Optimized GeoTIFF windowed reads).
4. Mask cloud, shadow, land and shoreline pixels.
5. Run spectral indices and a model trained on public in-situ reflectance data, pixel by pixel.
6. Report estimates with uncertainty, and explain which bands drove each one.

Satellites see only **optically active** constituents. AquaDoc estimates chlorophyll-a, suspended solids, turbidity, CDOM and water clarity, and flags likely floating debris and surface scum. It cannot detect heavy metals, bacteria, dissolved nutrients or most chemicals, which need laboratory testing.

## Status

| Component | State |
| --- | --- |
| Sentinel-2 STAC search and COG band reads (`aquadoc_core/ingestion.py`) | Working |
| Aligned 20 m grid, SCL cloud masking, radiometric scaling (`aquadoc_core/imagery_quality.py`) | Implemented, not yet wired into the API |
| Spectral indices and heuristic screening (`spectral_indices.py`, `risk.py`, `pipeline.py`) | Working |
| FastAPI service (`aquadoc_core/api.py`) | Working, local SQLite persistence |
| Model trained on public spectral data | Planned |
| Next.js dashboard and Vercel deployment | Planned |

## Repository structure

```text
AQUADOCV2/
├── aquadoc_core/        # Deployed backend: ingestion, imagery QA, indices, API, CLI
├── tests/               # Offline tests for aquadoc_core
├── research/            # Shelved lab-measurement matching workflow (not deployed)
│   ├── lab_matching/
│   └── tests/
├── data/                # Local data, manifests and SQLite (databases are git-ignored)
├── docs/                # Project audit and dataset notes
├── learning.md          # Implementation learning journal
└── study.md             # Technology log and learning resources
```

## Quick start

Requires Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"            # API runtime + test client
# pip install -e ".[training]"     # offline model training
# pip install -e ".[research]"     # shelved lab-matching workflow
```

Run the tests:

```powershell
python -m unittest discover -s tests
python -m unittest discover -s research/tests -t .
```

Analyze a coordinate from the command line:

```powershell
python -m aquadoc_core.cli analyze --lat 41.68 --lon -83.30 --start 2026-08-01 --end 2026-09-01
```

Start the API and open the interactive docs at http://127.0.0.1:8000/docs:

```powershell
python -m uvicorn aquadoc_core.api:app --reload
```

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/analyses` | Fetch, analyze and persist a coordinate request |
| `GET` | `/analyses` | List saved runs |
| `GET` | `/analyses/{run_id}` | Retrieve one run |

## Dependencies

`requirements.txt` and `[project.dependencies]` in `pyproject.toml` contain only the API runtime, which is what gets deployed. Training and research dependencies are optional extras, so the deployed bundle stays small.

## Limitations

- Research and educational tool, not a regulatory or drinking-water safety system.
- Sentinel-2 L2A atmospheric correction is designed for land and is imperfect over water. Estimates carry real uncertainty.
- The current risk score is a transparent heuristic. Its labels will be replaced with screening levels once the trained model is in place.

## License

Developed for educational and research use. Check data-source licences and attribution requirements before redistribution.

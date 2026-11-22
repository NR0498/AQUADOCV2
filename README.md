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
| Water-body outline (OpenStreetMap), 20 m aligned reads, cloud/shadow/shore/scum masking (`waterbody.py`, `s2_reader.py`, `water_masks.py`) | Working |
| Per-pixel estimates, band sensitivity, PNG map layers, staged events (`water_pipeline.py`) | Working via `python -m aquadoc_core.cli water`; API in Phase 3 |
| Spectral indices and heuristic screening (`spectral_indices.py`, `risk.py`, `pipeline.py`) | Working |
| FastAPI service with streaming analyses, caching, layer serving (`api.py`, `store.py`) | Working; SQLite locally, PostgreSQL via `DATABASE_URL` |
| Water-quality model trained on GLORIA (`aquadoc_core/wq_model.py`, [model card](docs/models/water_quality_v1.md)) | Trained; used by the water pipeline |
| Next.js dashboard (`web/`): live processing, map layers, band explanations, model accuracy | Working locally |
| Vercel deployment | Planned (Phase 5) |

## Repository structure

```text
AQUADOCV2/
├── aquadoc_core/        # Deployed backend: ingestion, imagery QA, indices, API, CLI
├── web/                 # Next.js dashboard
├── training/            # Offline model training and satellite validation
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
python -m unittest discover -s training/tests -t .
```

Estimate water quality for a water body. It prints each stage live, then writes `result.json` and PNG map layers:

```powershell
# By OpenStreetMap id (fast outline lookup), e.g. Hussain Sagar, Hyderabad
python -m aquadoc_core.cli water --lat 17.4239 --lon 78.4738 --osm relation/2833155 --out output/hussain
# By coordinate only (outline found via Overpass in parallel; slower)
python -m aquadoc_core.cli water --lat 41.70 --lon -83.25 --out output/erie
```

Run the dashboard: start the API on port 8000, then `cd web && npm install && npm run dev`, and open http://localhost:3000 (see `web/README.md`).

Start the API and open the interactive docs at http://127.0.0.1:8000/docs:

```powershell
python -m uvicorn aquadoc_core.api:app --reload
```

## API

Start it with `python -m uvicorn aquadoc_core.api:app --reload`. Interactive docs are at http://127.0.0.1:8000/api/docs.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/guide` | Band explanations, parameters, non-detectable pollutants, model and satellite-validation metrics |
| `GET` | `/api/waterbodies/search?q=` | Find water bodies by name (OpenStreetMap); returns `osm` references |
| `POST` | `/api/analyses/stream` | Run an analysis and stream each stage as Server-Sent Events; the final `complete` event has the result |
| `POST` | `/api/analyses` | Same, returning only the final result |
| `GET` | `/api/analyses?limit=` | Recent analyses |
| `GET` | `/api/analyses/{id}` | A stored result |
| `GET` | `/api/analyses/{id}/layers/{name}.png` | Map layer image (immutable, cacheable) |

Request body: `{"latitude": 17.4239, "longitude": 78.4738, "osm_ref": "relation/2833155"}`, plus optional `start_date`, `end_date` and `max_size_m`. Identical requests within 24 hours return the stored result.

Storage is SQLite locally and PostgreSQL when `DATABASE_URL` is set. CORS origins come from `AQUADOC_CORS_ORIGINS` (default `http://localhost:3000`).

## Dependencies

`requirements.txt` and `[project.dependencies]` in `pyproject.toml` contain only the API runtime, which is what gets deployed. Training and research dependencies are optional extras, so the deployed bundle stays small.

## Limitations

- Research and educational tool, not a regulatory or drinking-water safety system.
- Accuracy is measured on 506 real Sentinel-2 match-ups at 107 sites the model never saw. The typical error is about 58% for turbidity, 76-78% for chlorophyll-a and suspended solids, and 43% for Secchi depth; CDOM exceeds 100% and is always marked low confidence. See the [model card](docs/models/water_quality_v1.md).
- Sen2Cor atmospheric correction is designed for land. A residual correction learned from those match-ups removes most of its offset over water, but not all of it.
- Satellite validation is dominated by lakes; accuracy for rivers and estuaries is largely unknown.
- Non-optical pollutants (metals, bacteria, nutrients, pesticides, toxins) cannot be estimated from imagery.

## License

Developed for educational and research use. Check data-source licences and attribution requirements before redistribution.

"""AquaDoc HTTP API, served under /api for the dashboard.

Run locally:   python -m uvicorn aquadoc_core.api:app --reload    (docs at /api/docs)

Endpoints
    GET  /api/health
    GET  /api/guide                         bands, parameters, non-detectable list, model metrics
    GET  /api/waterbodies/search?q=...      name search (OpenStreetMap / Nominatim)
    POST /api/analyses/stream               run an analysis, streaming stage events (Server-Sent Events)
    POST /api/analyses                      run an analysis and return only the final result
    GET  /api/analyses?limit=20             recent analyses
    GET  /api/analyses/{id}                 stored result
    GET  /api/analyses/{id}/layers/{name}.png   map layer image

Identical requests (same water body or point, same dates) within CACHE_HOURS return the
stored result instead of re-processing imagery.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Callable, Iterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import LIBEXPAT, band_guide
from .store import AnalysisStore, store_from_env
from .water_pipeline import PIPELINE_VERSION, PipelineEvent, Services, WaterAnalysisRequest, run_water_analysis
from .waterbody import nearby_water_bodies, search_water_bodies
from .wq_model import WaterQualityModel

log = logging.getLogger("aquadoc.api")

# Import the heavy geospatial libraries once, at start-up, on the main thread. The pipeline imports
# them lazily inside a worker thread; on a cold serverless instance a first-time import racing with
# other requests can fail half-way, and the broken module then stays cached for the instance's life.
_PRELOAD = ("rasterio", "rasterio.enums", "rasterio.features", "rasterio.transform", "rasterio.vrt",
            "rasterio.warp", "rasterio.windows", "shapely.geometry", "planetary_computer", "pystac")


_PRELOAD_ERRORS: dict[str, str] = {}  # reported by /api/health so a broken deployment is diagnosable


def _preload_geospatial() -> None:
    import importlib

    for name in _PRELOAD:
        try:
            importlib.import_module(name)
        except Exception as exc:  # surfaced again, with context, when an analysis needs the module
            log.exception("Could not preload %s", name)
            _PRELOAD_ERRORS[name] = f"{type(exc).__name__}: {str(exc)[:200]}"


_preload_geospatial()

API_VERSION = "1.0.0"
CACHE_HOURS = 24
KEEPALIVE_SECONDS = 10
RATE_LIMIT = (8, 600)  # analyses per client per window (seconds); best effort, per instance
_LAYER_NAME = r"^[a-z0-9_]{1,40}$"


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    osm_ref: str | None = Field(default=None, pattern=r"^(node|way|relation)/\d+$",
                                description="OpenStreetMap object from /api/waterbodies/search, e.g. relation/2833155")
    start_date: date | None = None
    end_date: date | None = None
    max_size_m: int = Field(default=5000, ge=600, le=10000)

    @field_validator("end_date")
    @classmethod
    def not_in_future(cls, value):
        if value and value > datetime.now(timezone.utc).date() + timedelta(days=1):
            raise ValueError("end_date cannot be in the future")
        return value

    def to_pipeline(self) -> WaterAnalysisRequest:
        return WaterAnalysisRequest(
            latitude=self.latitude, longitude=self.longitude, osm_ref=self.osm_ref,
            start_date=self.start_date.isoformat() if self.start_date else None,
            end_date=self.end_date.isoformat() if self.end_date else None,
            max_size_m=self.max_size_m,
        )


def request_key(request: WaterAnalysisRequest, model_version: str | None) -> str:
    start, end = request.resolved_dates()
    where = request.osm_ref or f"{request.latitude:.4f},{request.longitude:.4f}"
    return f"{where}|{start}|{end}|{request.max_size_m}|{PIPELINE_VERSION}|{model_version}"


class RateLimiter:
    """Sliding-window limit per client. Per-instance memory only: a speed bump, not security.
    Use Vercel Firewall rate-limiting rules for real protection."""

    def __init__(self, limit: int, window_s: int):
        self.limit, self.window_s = limit, window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self.lock:
            hits = self.hits[client]
            while hits and now - hits[0] > self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


def _sse(kind: str, payload: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _client_id(http_request: Request) -> str:
    forwarded = http_request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (http_request.client.host if http_request.client else "unknown")


def create_app(
    store: AnalysisStore | None = None,
    services_factory: Callable[[], Services] = Services,
    model_loader: Callable[[], WaterQualityModel] | None = None,
    rate_limit: tuple[int, int] = RATE_LIMIT,
) -> FastAPI:
    app = FastAPI(title="AquaDoc API", version=API_VERSION, docs_url="/api/docs", openapi_url="/api/openapi.json",
                  description="Satellite (Sentinel-2) estimates of optically detectable water-quality parameters.")
    default_origins = "http://localhost:3000,http://127.0.0.1:3000"  # local dashboard dev server
    origins = [o.strip() for o in os.environ.get("AQUADOC_CORS_ORIGINS", default_origins).split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    state: dict = {"store": store}
    limiter = RateLimiter(*rate_limit)

    def get_store() -> AnalysisStore:
        if state["store"] is None:
            state["store"] = store_from_env()  # lazily, so importing the app never touches the database
        return state["store"]

    def get_model() -> WaterQualityModel:
        if "model" not in state:
            state["model"] = (model_loader or WaterQualityModel.load)()
        return state["model"]

    def services() -> Services:
        built = services_factory()
        if built.load_model is None:
            built.load_model = get_model
        return built

    def finalize(result: dict, images: dict[str, bytes], key: str) -> dict:
        analysis_id = uuid.uuid4().hex[:16]
        result = {**result, "id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "cached": False}
        result["layers"] = [{**layer, "url": f"/api/analyses/{analysis_id}/layers/{layer['id']}.png"}
                            for layer in result.get("layers", [])]
        get_store().save(analysis_id, key, result, images)
        return result

    def check_rate(http_request: Request) -> None:
        if not limiter.allow(_client_id(http_request)):
            raise HTTPException(429, "Too many analyses from this client; please wait a few minutes.")

    def prepare(body: AnalyzeRequest) -> tuple[WaterAnalysisRequest, str, dict | None]:
        pipeline_request = body.to_pipeline()
        try:
            pipeline_request.validate()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        key = request_key(pipeline_request, get_model().metadata.get("model_version"))
        cached = get_store().find_recent(key, timedelta(hours=CACHE_HOURS))
        if cached and pipeline_request.osm_ref and not cached.get("water_body"):
            cached = None  # the outline lookup failed on that run (busy OSM servers): do not reuse a degraded result
        return pipeline_request, key, ({**cached, "cached": True} if cached else None)

    @app.exception_handler(Exception)
    async def unexpected(_, exc: Exception):  # keep internal details out of responses
        return JSONResponse(status_code=500, content={"detail": f"Internal error ({type(exc).__name__})"})

    @app.get("/api/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok", "api_version": API_VERSION, "pipeline_version": PIPELINE_VERSION,
                "geospatial": "ok" if not _PRELOAD_ERRORS else _PRELOAD_ERRORS, "libexpat": LIBEXPAT}

    @app.get("/api/guide", tags=["reference"])
    def guide(response: Response) -> dict:
        model = get_model()
        targets = {}
        for name, spec in model.metadata["targets"].items():
            evaluation = spec["evaluation"]
            targets[name] = {
                "unit": spec["unit"], "deployed": spec["deployed"],
                "test_mdsa_pct": evaluation["model"]["mdsa_pct"],
                "test_mdsa_with_noise_pct": evaluation["model_with_simulated_satellite_noise"]["mdsa_pct"],
                "published_algorithm": evaluation.get("published_algorithm"),
                "interval_coverage": evaluation["interval_coverage"],
                "training_range": spec["target_range_p01_p99"],
                "satellite_validation": spec.get("satellite_validation"),
            }
        response.headers["Cache-Control"] = "public, max-age=3600"
        return {
            "bands": band_guide.BANDS,
            "parameters": band_guide.PARAMETERS,
            "not_detectable": band_guide.NOT_DETECTABLE,
            "model": {"version": model.metadata.get("model_version"),
                      "training_data": model.metadata.get("training_data", {}).get("source"),
                      "targets": targets,
                      "satellite_correction": model.metadata.get("satellite_correction")},
        }

    @app.get("/api/waterbodies/search", tags=["reference"])
    def search(response: Response, q: str = Query(min_length=2, max_length=120)) -> dict:
        try:
            results = search_water_bodies(q)
        except Exception as exc:
            raise HTTPException(502, "Place search is temporarily unavailable") from exc
        response.headers["Cache-Control"] = "public, max-age=86400"
        return {"results": results, "attribution": "© OpenStreetMap contributors"}

    @lru_cache(maxsize=512)
    def cached_nearby(latitude: float, longitude: float, radius_km: int) -> tuple:
        return tuple(nearby_water_bodies(latitude, longitude, radius_km))

    def latest_analysis_for(lake: dict, recent: list[dict]) -> dict | None:
        """The newest saved analysis whose point lies on this lake (inside its bounding box)."""
        south, north, west, east = lake["bbox"]
        for row in recent:  # newest first
            lat, lon = row.get("latitude"), row.get("longitude")
            if lat is None or lon is None or not (south - 0.002 <= lat <= north + 0.002 and west - 0.002 <= lon <= east + 0.002):
                continue
            result = get_store().get(row["id"])
            if not result:
                continue
            estimates = result.get("estimates", {})
            pick = lambda key: (estimates.get(key) or {}).get("value")  # noqa: E731
            return {"id": row["id"], "scene_datetime": row.get("scene_datetime"), "created_at": row.get("created_at"),
                    "chlorophyll_a": pick("chlorophyll_a"), "turbidity": pick("turbidity"),
                    "secchi_depth": pick("secchi_depth"),
                    "trophic_state": (estimates.get("chlorophyll_a") or {}).get("trophic_state"),
                    "clear_fraction": (result.get("pixels") or {}).get("clear_fraction")}
        return None

    @app.get("/api/waterbodies/nearby", tags=["reference"])
    def nearby(latitude: float = Query(ge=-90, le=90), longitude: float = Query(ge=-180, le=180),
               radius_km: int = Query(default=25, ge=5, le=50)) -> dict:
        # Rounded to ~1 km before use: enough to find lakes, without passing on a precise location.
        lat, lon = round(latitude, 2), round(longitude, 2)
        try:
            lakes = [dict(lake) for lake in cached_nearby(lat, lon, radius_km)]
        except Exception as exc:
            raise HTTPException(502, "The lake directory (OpenStreetMap) is temporarily unavailable") from exc
        recent = get_store().list_recent(200)
        for lake in lakes:
            lake["latest_analysis"] = latest_analysis_for(lake, recent)
        return {"origin": {"latitude": lat, "longitude": lon}, "radius_km": radius_km, "lakes": lakes,
                "attribution": "© OpenStreetMap contributors"}

    @app.post("/api/analyses/stream", tags=["analyses"])
    def analyze_stream(body: AnalyzeRequest, http_request: Request) -> StreamingResponse:
        pipeline_request, key, cached = prepare(body)
        if cached is None:
            check_rate(http_request)

        def events() -> Iterator[str]:
            if cached is not None:
                yield _sse("stage", {"stage": "cache", "status": "completed",
                                     "message": "Returning an analysis made earlier today", "data": {}, "elapsed_ms": 0})
                yield _sse("complete", cached)
                return
            # Run the pipeline in a thread so we can send keep-alive comments while it waits on
            # slow services; idle HTTP/1.1 connections may otherwise be closed by proxies.
            channel: queue.Queue = queue.Queue()
            done = object()

            def worker():
                try:
                    for item in run_water_analysis(pipeline_request, services()):
                        channel.put(item)
                except Exception as exc:  # report instead of silently dropping the stream
                    log.exception("Analysis failed")  # full traceback in server logs; clients get the type only
                    channel.put(exc)
                finally:
                    channel.put(done)

            threading.Thread(target=worker, daemon=True).start()
            while True:
                try:
                    item = channel.get(timeout=KEEPALIVE_SECONDS)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if item is done:
                    return
                if isinstance(item, Exception):
                    detail = type(item).__name__
                    if isinstance(item, ImportError):  # module names and paths are safe and make deploys diagnosable
                        detail += f": {str(item)[:200]}"
                    yield _sse("error", {"message": f"Analysis failed unexpectedly ({detail})"})
                    continue
                event: PipelineEvent = item
                if event.stage == "complete":
                    try:
                        yield _sse("complete", finalize(event.data, event.images, key))
                    except Exception as exc:
                        yield _sse("error", {"message": f"Could not save the result ({type(exc).__name__})"})
                    continue
                yield _sse("stage", event.to_dict())
                if event.status == "failed":
                    yield _sse("error", {"message": event.message, "stage": event.stage})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 # no-transform stops proxies (including the Next.js dev server) from gzipping the
                                 # stream, which would buffer events and make the live pipeline jump.
                                 headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})

    @app.post("/api/analyses", tags=["analyses"])
    def analyze(body: AnalyzeRequest, http_request: Request) -> dict:
        pipeline_request, key, cached = prepare(body)
        if cached is not None:
            return cached
        check_rate(http_request)
        stages = []
        for event in run_water_analysis(pipeline_request, services()):
            if event.stage == "complete":
                return finalize(event.data, event.images, key)
            stages.append(event.to_dict())
            if event.status == "failed":
                raise HTTPException(422, {"message": event.message, "stage": event.stage, "stages": stages})
        raise HTTPException(500, "Analysis ended without a result")

    @app.get("/api/analyses", tags=["analyses"])
    def list_analyses(limit: int = Query(default=20, ge=1, le=100)) -> dict:
        return {"analyses": get_store().list_recent(limit)}

    @app.get("/api/analyses/{analysis_id}", tags=["analyses"])
    def get_analysis(analysis_id: str) -> dict:
        result = get_store().get(analysis_id)
        if result is None:
            raise HTTPException(404, "Analysis not found")
        return result

    @app.get("/api/analyses/{analysis_id}/layers/{name}.png", tags=["analyses"])
    def get_layer(analysis_id: str, name: str) -> Response:
        import re

        if not re.match(_LAYER_NAME, name):
            raise HTTPException(404, "Layer not found")
        png = get_store().get_layer(analysis_id, name)
        if png is None:
            raise HTTPException(404, "Layer not found")
        # Layers never change once written, so browsers and the CDN can cache them forever.
        return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=31536000, immutable"})

    return app


app = create_app()

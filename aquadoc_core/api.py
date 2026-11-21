"""HTTP API for AquaDoc ingestion, analysis, and persisted runs."""

import sqlite3
from datetime import date
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ingestion import fetch_sentinel2_chip
from .persistence import AnalysisRunRepository, DEFAULT_DATABASE_PATH
from .pipeline import analyze_multispectral_scene
from .schemas import AnalysisResult, MultispectralScene


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    start_date: date
    end_date: date
    size_m: int = Field(default=3000, ge=100, le=20000)

    @model_validator(mode="after")
    def validate_date_range(self) -> "AnalysisRequest":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")
        return self


def create_app(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    fetcher: Callable[..., MultispectralScene] = fetch_sentinel2_chip,
    analyzer: Callable[[MultispectralScene], AnalysisResult] = analyze_multispectral_scene,
) -> FastAPI:
    repository = AnalysisRunRepository(database_path)
    app = FastAPI(
        title="AquaDoc API",
        version="0.1.0",
        description="Multispectral satellite screening for water-quality signals.",
    )

    @app.get("/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/analyses", status_code=status.HTTP_201_CREATED, tags=["analyses"])
    def create_analysis(request: AnalysisRequest) -> dict:
        try:
            scene = fetcher(
                latitude=request.latitude,
                longitude=request.longitude,
                start_date=request.start_date.isoformat(),
                end_date=request.end_date.isoformat(),
                size_m=request.size_m,
            )
            result = analyzer(scene)
            return repository.save(
                result=result,
                start_date=request.start_date.isoformat(),
                end_date=request.end_date.isoformat(),
                size_m=request.size_m,
            ).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (RuntimeError, OSError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail="Could not save the analysis run") from exc

    @app.get("/analyses", tags=["analyses"])
    def list_analyses(limit: int = Query(default=20, ge=1, le=1000)) -> dict:
        try:
            return {"runs": [run.to_dict() for run in repository.list(limit)]}
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail="Could not read analysis runs") from exc

    @app.get("/analyses/{run_id}", tags=["analyses"])
    def get_analysis(run_id: str) -> dict:
        try:
            run = repository.get(run_id)
        except sqlite3.Error as exc:
            raise HTTPException(status_code=500, detail="Could not read the analysis run") from exc
        if run is None:
            raise HTTPException(status_code=404, detail=f"Analysis run '{run_id}' was not found")
        return run.to_dict()

    return app


app = create_app()

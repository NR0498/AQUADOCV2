"""Vercel entrypoint for the AquaDoc API (FastAPI). Locally: uvicorn aquadoc_core.api:app --port 8000"""
from aquadoc_core.api import app

__all__ = ["app"]

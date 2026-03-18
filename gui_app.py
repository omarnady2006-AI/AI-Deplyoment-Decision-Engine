"""
gui_app.py — Root-level entrypoint shim.

Exposes the FastAPI `app` object at the project root so the server can be
started with:

    PYTHONPATH=. uvicorn gui_app:app --host 0.0.0.0 --port 8000

All implementation lives in src/api/gui_app.py; this file simply re-exports it.
"""
from src.api.gui_app import app  # noqa: F401 — re-export for uvicorn

__all__ = ["app"]

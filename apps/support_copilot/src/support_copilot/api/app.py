"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from rag_core.service import RagService
from sqlalchemy import Engine

from support_copilot.api import kb, tickets
from support_copilot.api.deps import RagDep
from support_copilot.api.ratelimit import RateLimiter
from support_copilot.config import CopilotSettings
from support_copilot.db import init_db, make_engine
from support_copilot.schemas import Health

STATIC_DIR = Path(__file__).parent / "static"
# The page itself is never cached, so a new deploy is picked up on reload.
NO_STORE = {"Cache-Control": "no-store"}


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser re-checks on every load (a cheap 304 via ETag), so an update
    of app.js is picked up without a hard refresh."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(settings: CopilotSettings, rag: RagService, engine: Engine | None = None) -> FastAPI:
    """Build the app around a ready RagService, so tests can pass fakes."""
    engine = engine or make_engine(settings.resolved_database_url)
    init_db(engine)

    app = FastAPI(
        title="AI Customer Support Copilot",
        version="0.1.0",
        description="Drafts support replies from the knowledge base and escalates the rest.",
    )
    app.state.settings = settings
    app.state.rag = rag
    app.state.engine = engine
    app.state.rate_limiter = (
        RateLimiter(settings.rate_limit_per_hour) if settings.rate_limit_per_hour > 0 else None
    )
    app.include_router(kb.router)
    app.include_router(tickets.router)

    @app.get("/health", response_model=Health, tags=["service"])
    def health(rag: RagDep) -> Health:
        stats = rag.stats()
        return Health(status="ok", documents=stats["documents"], llm_enabled=stats["llm_enabled"])

    # Operator web UI: plain static files on top of the same API (Swagger stays at /docs).
    app.mount("/static", RevalidatedStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers=NO_STORE)

    return app

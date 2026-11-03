"""SQLite engine setup."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, event
from sqlmodel import SQLModel, create_engine

from support_copilot import models  # noqa: F401  (registers the tables on SQLModel.metadata)


def make_engine(url: str) -> Engine:
    if not url.startswith("sqlite"):
        return create_engine(url)
    database = url.split(":///", 1)[-1]
    if database and database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)
    # FastAPI runs sync endpoints in a thread pool, so a connection may change threads.
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)

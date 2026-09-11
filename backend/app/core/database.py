"""SQLAlchemy engine, session factory, and development table initialization."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


class Base(DeclarativeBase):
    """Declarative base for ORM models."""


def normalize_database_url(database_url: str) -> str:
    """Select PyMySQL when Railway supplies SQLAlchemy's bare MySQL URL.

    Railway MySQL commonly exposes ``mysql://...``. SQLAlchemy maps that
    legacy scheme to the optional MySQLdb driver, while this project declares
    and supports PyMySQL. Explicit ``mysql+pymysql://...`` URLs pass through
    unchanged; no credentials are added, altered, or logged.
    """
    url = database_url.strip()
    if url.lower().startswith("mysql://"):
        return f"mysql+pymysql://{url[len('mysql://'):]}"
    return url


def _engine_kwargs(database_url: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_recycle"] = 3600
    return kwargs


def get_engine(settings: Settings | None = None) -> Engine:
    global _engine
    if _engine is None:
        cfg = settings or get_settings()
        url = normalize_database_url(cfg.database_url or "")
        if not url:
            raise RuntimeError("DATABASE_URL is not configured.")
        _engine = create_engine(url, **_engine_kwargs(url))
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(settings),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory


def SessionLocal() -> Session:
    """Create a new Session (SQLAlchemy conventional name)."""
    return get_session_factory()()


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped session. Routes should prefer the repository."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db(settings: Settings | None = None) -> None:
    """
    Create tables if they do not exist (development-safe, no Alembic).

    Safe to call on API startup. Does not drop or migrate existing data.
    """
    from app.models import document as _document_model  # noqa: F401

    engine = get_engine(settings)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")


def reset_engine() -> None:
    """Dispose the process-wide engine (used by tests)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None

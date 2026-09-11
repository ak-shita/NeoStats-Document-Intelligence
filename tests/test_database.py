"""Offline database URL/engine tests; no MySQL connection is opened."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.database import get_engine, normalize_database_url, reset_engine


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "mysql://user:password@mysql.example.internal:3306/neostats",
            "mysql+pymysql://user:password@mysql.example.internal:3306/neostats",
        ),
        (
            "mysql+pymysql://user:password@mysql.example.internal:3306/neostats",
            "mysql+pymysql://user:password@mysql.example.internal:3306/neostats",
        ),
    ],
)
def test_normalize_database_url_uses_pymysql_for_mysql_urls(source: str, expected: str):
    assert normalize_database_url(source) == expected


def test_normalize_database_url_leaves_non_mysql_urls_unchanged():
    assert normalize_database_url("sqlite+pysqlite:///:memory:") == "sqlite+pysqlite:///:memory:"


@pytest.mark.parametrize(
    "database_url",
    [
        "mysql://user:password@mysql.example.internal:3306/neostats",
        "mysql+pymysql://user:password@mysql.example.internal:3306/neostats",
    ],
)
def test_get_engine_uses_pymysql_without_connecting(database_url: str):
    reset_engine()
    try:
        engine = get_engine(Settings(database_url=database_url))
        assert engine.url.drivername == "mysql+pymysql"
        assert engine.url.database == "neostats"
    finally:
        reset_engine()

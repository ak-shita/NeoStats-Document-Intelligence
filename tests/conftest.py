"""Pytest configuration.

Normal suite uses in-memory SQLite so MySQL is not required.
Real MySQL tests run only when RUN_MYSQL_INTEGRATION=1.
"""

from __future__ import annotations

import os

import pytest

# Do not override DATABASE_URL when the operator explicitly wants MySQL tests.
if os.environ.get("RUN_MYSQL_INTEGRATION") != "1":
    os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "mysql_integration: requires real local MySQL (RUN_MYSQL_INTEGRATION=1)",
    )
    config.addinivalue_line(
        "markers",
        "dataset_smoke: real assessment dataset files; Gemini/OCR mocked unless noted",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_MYSQL_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(
        reason="Set RUN_MYSQL_INTEGRATION=1 to run real MySQL integration tests"
    )
    for item in items:
        if "mysql_integration" in item.keywords:
            item.add_marker(skip)

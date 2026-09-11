"""API persistence tests against the SQLAlchemy repository (SQLite, mocked orchestration)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_document_repository, get_process_document_fn
from app.core.config import get_settings
from app.core.database import Base
from app.main import app
from app.models.document import Document  # noqa: F401
from app.repositories.base import DocumentRepositoryError
from app.repositories.sqlalchemy_repository import SqlAlchemyDocumentRepository
from app.schemas.document import DocumentProcessingResult
from test_api import RESPONSE_KEYS, _assert_no_secrets, _post_process, _sample_result


@pytest.fixture
def sqlite_repo() -> SqlAlchemyDocumentRepository:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    try:
        yield SqlAlchemyDocumentRepository(factory)
    finally:
        engine.dispose()


@pytest.fixture
def process_fn() -> AsyncMock:
    return AsyncMock(return_value=_sample_result())


@pytest.fixture
def client(sqlite_repo: SqlAlchemyDocumentRepository, process_fn: AsyncMock) -> TestClient:
    app.dependency_overrides[get_document_repository] = lambda: sqlite_repo
    app.dependency_overrides[get_process_document_fn] = lambda: process_fn
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_api_get_after_sqlalchemy_save(client: TestClient):
    created = _post_process(client)
    assert created.status_code == 200
    response = client.get("/api/v1/documents/invoice.pdf")
    assert response.status_code == 200
    body = response.json()
    assert RESPONSE_KEYS.issubset(body.keys())
    parsed = DocumentProcessingResult.model_validate(body)
    assert parsed.document_name == "invoice.pdf"
    _assert_no_secrets(response)


def test_api_list_after_sqlalchemy_save(client: TestClient, process_fn: AsyncMock):
    empty = client.get("/api/v1/documents")
    assert empty.status_code == 200
    assert empty.json() == {"count": 0, "documents": []}

    _post_process(client)
    process_fn.return_value = _sample_result(document_name="second.pdf")
    _post_process(client, filename="second.pdf")

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [doc["document_name"] for doc in body["documents"]] == ["invoice.pdf", "second.pdf"]
    _assert_no_secrets(response)


def test_api_reprocess_same_name_overwrites(client: TestClient, process_fn: AsyncMock):
    _post_process(client)
    process_fn.return_value = _sample_result(
        overall_confidence=0.42,
        extracted_data={"invoice_number": {"value": "NEW", "confidence": 0.8, "page_number": 1}},
    )
    updated = _post_process(client)
    assert updated.status_code == 200

    listed = client.get("/api/v1/documents")
    assert listed.json()["count"] == 1
    fetched = client.get("/api/v1/documents/invoice.pdf")
    assert fetched.json()["overall_confidence"] == 0.42
    assert fetched.json()["extracted_data"]["invoice_number"]["value"] == "NEW"


def test_fresh_balance_sheet_validation_is_persisted_and_returned(
    client: TestClient, process_fn: AsyncMock
):
    """Regression: populated fresh checks must not be replaced by stale NA JSON."""
    checks = [
        {
            "name": f"balance_sheet_check_{index}",
            "formula": "a ≈ b",
            "operands": {"a": 100.0, "b": 100.0},
            "calculated_value": 100.0,
            "reported_value": 100.0,
            "variance": 0.0,
            "status": "PASS",
            "reported_raw": "100.00",
        }
        for index in range(1, 7)
    ]
    process_fn.return_value = _sample_result(
        document_name="balance-sheet.pdf",
        document_type="balance_sheet",
        validation={"checks": checks, "overall_status": "PASS", "issues": []},
    )

    created = _post_process(
        client, filename="balance-sheet.pdf", document_type="balance_sheet"
    )
    assert created.status_code == 200
    fetched = client.get("/api/v1/documents/balance-sheet.pdf")
    assert fetched.status_code == 200
    persisted_checks = fetched.json()["validation"]["checks"]
    assert len(persisted_checks) == 6
    assert all(check["status"] == "PASS" for check in persisted_checks)
    assert all(check["calculated_value"] == 100.0 for check in persisted_checks)
    assert all(check["reported_value"] == 100.0 for check in persisted_checks)
    assert all(check["variance"] == 0.0 for check in persisted_checks)
    assert fetched.headers["cache-control"] == "no-store"


def test_document_list_is_not_cacheable(client: TestClient):
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_api_repository_error_is_controlled(
    client: TestClient, sqlite_repo: SqlAlchemyDocumentRepository
):
    def boom(_name: str):
        raise DocumentRepositoryError(
            code="DATABASE_ERROR",
            message="Failed to retrieve the processed document.",
        )

    sqlite_repo.get = boom  # type: ignore[method-assign]
    response = client.get("/api/v1/documents/invoice.pdf")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "DATABASE_ERROR"
    assert "traceback" not in response.text.lower()
    _assert_no_secrets(response)


def test_api_persistence_does_not_expose_secrets(client: TestClient):
    response = _post_process(client)
    settings = get_settings()
    text = response.text + client.get("/api/v1/documents").text
    assert "DATABASE_URL" not in text
    assert "mysql+pymysql" not in text
    if settings.database_url:
        assert settings.database_url not in text
    _assert_no_secrets(response)

"""API + repository integration against REAL local MySQL.

Gated by RUN_MYSQL_INTEGRATION=1.
Processing is mocked — no OCR or Gemini calls.
Temporary rows use the prefix __pytest_mysql__ and are deleted after each test.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.api.deps import get_document_repository, get_process_document_fn
from app.core.config import Settings, get_settings
from app.core.database import get_session_factory, init_db, reset_engine
from app.main import app
from app.models.document import Document
from app.repositories.base import DocumentRepositoryError
from app.repositories.sqlalchemy_repository import SqlAlchemyDocumentRepository
from app.schemas.document import DocumentProcessingResult
from app.services.document_service import DocumentProcessingError
from test_api import PROCESS_URL, RESPONSE_KEYS, _assert_no_secrets, _post_process

pytestmark = pytest.mark.mysql_integration

TEST_PREFIX = "__pytest_mysql__"


def _nested_result(**overrides: Any) -> dict[str, Any]:
    payload = {
        "document_name": f"{TEST_PREFIX}invoice.pdf",
        "document_type": "invoice",
        "processing_status": "PASS",
        "overall_confidence": 0.91,
        "file_validation": {
            "file_type": "application/pdf",
            "is_supported": True,
            "is_readable": True,
            "page_count": 1,
            "status": "PASS",
        },
        "extracted_data": {
            "invoice_number": {"value": "94404257", "confidence": 0.99, "page_number": 1},
            "line_items": [
                {
                    "description": {"value": "Widget", "confidence": 0.88, "page_number": 1},
                    "amount": {"value": "12.50", "confidence": 0.87, "page_number": 1},
                }
            ],
            "nested": {"evidence": {"source_text": "Invoice no: 94404257", "page_number": 1}},
        },
        "validation": {
            "checks": [
                {
                    "name": "invoice_taxable_plus_tax_vs_total",
                    "formula": "a ≈ b",
                    "status": "PASS",
                    "variance": 0.0,
                }
            ],
            "overall_status": "PASS",
            "issues": [],
        },
        "processing_metadata": {
            "ocr_used": True,
            "processed_at": "2026-09-11T00:00:00Z",
            "processing_time_ms": 15,
            "ocr_page_count": 1,
            "ocr_engine": 2,
            "extraction_document_type": "Invoice",
        },
    }
    payload.update(overrides)
    return payload


def _cleanup(session_factory) -> None:
    with session_factory() as session:
        session.execute(
            delete(Document).where(Document.document_name.startswith(TEST_PREFIX))
        )
        session.commit()


def _assert_no_db_secrets(response) -> None:
    _assert_no_secrets(response)
    text = response.text
    assert "DATABASE_URL" not in text
    assert "mysql+pymysql://" not in text


@pytest.fixture
def mysql_settings() -> Settings:
    get_settings.cache_clear()
    reset_engine()
    settings = Settings()
    url = (settings.database_url or "").strip()
    if not url.lower().startswith("mysql"):
        pytest.skip("DATABASE_URL is not a MySQL URL")
    init_db(settings)
    yield settings
    reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def mysql_repo(mysql_settings: Settings) -> SqlAlchemyDocumentRepository:
    factory = get_session_factory(mysql_settings)
    _cleanup(factory)
    repo = SqlAlchemyDocumentRepository(factory)
    yield repo
    _cleanup(factory)


@pytest.fixture
def process_fn() -> AsyncMock:
    return AsyncMock(return_value=_nested_result())


@pytest.fixture
def client(mysql_repo: SqlAlchemyDocumentRepository, process_fn: AsyncMock) -> TestClient:
    app.dependency_overrides[get_document_repository] = lambda: mysql_repo
    app.dependency_overrides[get_process_document_fn] = lambda: process_fn
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_mysql_repo_save_and_get_json_roundtrip(mysql_repo: SqlAlchemyDocumentRepository):
    original = _nested_result()
    mysql_repo.save(original)
    stored = mysql_repo.get(f"{TEST_PREFIX}invoice.pdf")
    assert stored is not None
    assert stored["document_name"] == original["document_name"]
    assert stored["document_type"] == "invoice"
    assert stored["processing_status"] == "PASS"
    assert stored["overall_confidence"] == 0.91
    assert stored["extracted_data"]["line_items"][0]["description"]["value"] == "Widget"
    assert stored["extracted_data"]["nested"]["evidence"]["page_number"] == 1
    assert stored["validation"]["overall_status"] == "PASS"


def test_mysql_repo_upsert_same_name_does_not_duplicate(
    mysql_repo: SqlAlchemyDocumentRepository, mysql_settings: Settings
):
    name = f"{TEST_PREFIX}invoice.pdf"
    mysql_repo.save(_nested_result(overall_confidence=0.4))
    factory = get_session_factory(mysql_settings)
    with factory() as session:
        first = session.scalar(select(Document).where(Document.document_name == name))
        assert first is not None
        created_at = first.created_at
        first_id = first.id

    mysql_repo.save(
        _nested_result(
            overall_confidence=0.77,
            extracted_data={
                "invoice_number": {"value": "UPDATED", "confidence": 1.0, "page_number": 2}
            },
        )
    )
    with factory() as session:
        rows = session.scalars(select(Document).where(Document.document_name == name)).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.id == first_id
        assert row.created_at == created_at
        assert row.overall_confidence == pytest.approx(0.77)

    stored = mysql_repo.get(name)
    assert stored is not None
    assert stored["extracted_data"]["invoice_number"]["value"] == "UPDATED"


def test_api_post_persists_to_mysql_and_get_returns_it(client: TestClient):
    created = _post_process(client, filename=f"{TEST_PREFIX}invoice.pdf")
    assert created.status_code == 200
    body = created.json()
    assert RESPONSE_KEYS.issubset(body.keys())
    parsed = DocumentProcessingResult.model_validate(body)
    assert parsed.document_name == f"{TEST_PREFIX}invoice.pdf"
    _assert_no_db_secrets(created)

    fetched = client.get(f"/api/v1/documents/{TEST_PREFIX}invoice.pdf")
    assert fetched.status_code == 200
    assert fetched.json()["extracted_data"]["line_items"][0]["amount"]["value"] == "12.50"
    DocumentProcessingResult.model_validate(fetched.json())
    _assert_no_db_secrets(fetched)


def test_api_list_includes_mysql_row(client: TestClient):
    _post_process(client, filename=f"{TEST_PREFIX}invoice.pdf")
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    names = [doc["document_name"] for doc in response.json()["documents"]]
    assert f"{TEST_PREFIX}invoice.pdf" in names
    _assert_no_db_secrets(response)


def test_api_reprocess_upserts_mysql_row(
    client: TestClient, process_fn: AsyncMock, mysql_settings: Settings
):
    _post_process(client, filename=f"{TEST_PREFIX}invoice.pdf")
    process_fn.return_value = _nested_result(
        overall_confidence=0.33,
        extracted_data={"invoice_number": {"value": "SECOND", "confidence": 0.5, "page_number": 1}},
    )
    updated = _post_process(client, filename=f"{TEST_PREFIX}invoice.pdf")
    assert updated.status_code == 200
    assert updated.json()["overall_confidence"] == 0.33

    listed = client.get("/api/v1/documents")
    matches = [
        doc
        for doc in listed.json()["documents"]
        if doc["document_name"] == f"{TEST_PREFIX}invoice.pdf"
    ]
    assert len(matches) == 1
    assert matches[0]["extracted_data"]["invoice_number"]["value"] == "SECOND"

    factory = get_session_factory(mysql_settings)
    with factory() as session:
        rows = session.scalars(
            select(Document).where(Document.document_name == f"{TEST_PREFIX}invoice.pdf")
        ).all()
        assert len(rows) == 1


def test_api_unknown_document_404_from_mysql(client: TestClient):
    response = client.get(f"/api/v1/documents/{TEST_PREFIX}missing.pdf")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
    _assert_no_db_secrets(response)


def test_api_process_failure_does_not_expose_secrets(client: TestClient, process_fn: AsyncMock):
    process_fn.side_effect = DocumentProcessingError(
        code="UNSUPPORTED_FILE_TYPE",
        message="File type is not supported.",
        stage="file_validation",
    )
    response = _post_process(client, filename=f"{TEST_PREFIX}notes.txt")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
    assert "traceback" not in response.text.lower()
    _assert_no_db_secrets(response)


def test_api_repository_error_is_safe(client: TestClient, mysql_repo: SqlAlchemyDocumentRepository):
    def boom(_name: str):
        raise DocumentRepositoryError(
            code="DATABASE_ERROR",
            message="Failed to retrieve the processed document.",
        )

    mysql_repo.get = boom  # type: ignore[method-assign]
    response = client.get(f"/api/v1/documents/{TEST_PREFIX}invoice.pdf")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "DATABASE_ERROR"
    assert "traceback" not in response.text.lower()
    _assert_no_db_secrets(response)


def test_mysql_flag_is_enabled():
    assert os.environ.get("RUN_MYSQL_INTEGRATION") == "1"

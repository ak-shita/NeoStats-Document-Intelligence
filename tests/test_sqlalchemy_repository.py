"""SQLAlchemy repository tests using isolated SQLite (no MySQL server)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.document import Document
from app.repositories.base import DocumentRepositoryError
from app.repositories.sqlalchemy_repository import SqlAlchemyDocumentRepository


def _sample_result(**overrides: Any) -> dict[str, Any]:
    payload = {
        "document_name": "invoice.pdf",
        "document_type": "invoice",
        "processing_status": "PASS",
        "overall_confidence": 0.9,
        "file_validation": {
            "file_type": "application/pdf",
            "is_supported": True,
            "is_readable": True,
            "page_count": 1,
            "status": "PASS",
        },
        "extracted_data": {"invoice_number": {"value": "94404257", "confidence": 0.99, "page_number": 1}},
        "validation": {"checks": [], "overall_status": "PASS", "issues": []},
        "processing_metadata": {
            "ocr_used": True,
            "processed_at": "2026-09-10T18:00:00Z",
            "processing_time_ms": 12,
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def repo() -> SqlAlchemyDocumentRepository:
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


def test_save_and_retrieve_document(repo: SqlAlchemyDocumentRepository):
    repo.save(_sample_result())
    stored = repo.get("invoice.pdf")
    assert stored is not None
    assert stored["document_name"] == "invoice.pdf"
    assert stored["document_type"] == "invoice"
    assert stored["extracted_data"]["invoice_number"]["value"] == "94404257"


def test_list_documents_in_insertion_order(repo: SqlAlchemyDocumentRepository):
    assert repo.list_all() == []
    repo.save(_sample_result())
    repo.save(_sample_result(document_name="second.pdf"))
    names = [item["document_name"] for item in repo.list_all()]
    assert names == ["invoice.pdf", "second.pdf"]


def test_same_document_name_updates_latest_result(repo: SqlAlchemyDocumentRepository):
    repo.save(_sample_result(overall_confidence=0.4, processing_status="PASS"))
    repo.save(
        _sample_result(
            overall_confidence=0.95,
            processing_status="PASS",
            extracted_data={"invoice_number": {"value": "UPDATED", "confidence": 1.0, "page_number": 1}},
        )
    )
    stored = repo.get("invoice.pdf")
    assert stored is not None
    assert stored["overall_confidence"] == 0.95
    assert stored["extracted_data"]["invoice_number"]["value"] == "UPDATED"
    listed = repo.list_all()
    assert len(listed) == 1
    assert listed[0]["overall_confidence"] == 0.95


def test_save_without_document_name_raises(repo: SqlAlchemyDocumentRepository):
    with pytest.raises(DocumentRepositoryError) as exc_info:
        repo.save(_sample_result(document_name=""))
    assert exc_info.value.code == "INVALID_DOCUMENT_NAME"


def test_get_unknown_document_returns_none(repo: SqlAlchemyDocumentRepository):
    assert repo.get("missing.pdf") is None


def test_database_error_is_controlled():
    factory = MagicMock(
        side_effect=OperationalError("SELECT 1", {}, Exception("connection failed"))
    )
    repository = SqlAlchemyDocumentRepository(factory)

    with pytest.raises(DocumentRepositoryError) as save_err:
        repository.save(_sample_result())
    assert save_err.value.code == "DATABASE_ERROR"
    assert "connection failed" not in save_err.value.message

    with pytest.raises(DocumentRepositoryError) as get_err:
        repository.get("invoice.pdf")
    assert get_err.value.code == "DATABASE_ERROR"

    with pytest.raises(DocumentRepositoryError) as list_err:
        repository.list_all()
    assert list_err.value.code == "DATABASE_ERROR"


def test_upsert_does_not_insert_duplicate_rows(repo: SqlAlchemyDocumentRepository):
    repo.save(_sample_result())
    repo.save(_sample_result(overall_confidence=0.1))
    # Access the same session factory used by the repository.
    with repo._session_factory() as session:
        count = len(session.scalars(select(Document)).all())
    assert count == 1

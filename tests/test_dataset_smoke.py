"""Selected real-dataset smoke tests.

Labels:
- dataset_smoke: uses actual files from New Dataset/
- mocked OCR/Gemini unless a saved real extraction fixture is reused
- mysql_integration: only the Balance Sheet 2017 downstream persist/GET path

No live Gemini or OCR API calls.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.api.deps import get_document_repository, get_process_document_fn
from app.core.config import Settings, get_settings
from app.core.database import get_session_factory, init_db, reset_engine
from app.main import app
from app.models.document import Document
from app.repositories.sqlalchemy_repository import SqlAlchemyDocumentRepository
from app.schemas.document import DocumentProcessingResult
from app.services.document_service import process_document
from app.services.file_validation_service import validate_file
from app.services.financial_validation_service import validate_financials
from sqlalchemy import delete
from test_api import RESPONSE_KEYS, _assert_no_secrets, _post_process

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "New Dataset"
FIXTURE_EXTRACTION = (
    ROOT / "sample_outputs" / "extraction_smoke" / "Consolidated_Balance_Sheet_2017.extraction.json"
)
FIXTURE_OCR = (
    ROOT / "sample_outputs" / "extraction_smoke" / "Consolidated_Balance_Sheet_2017.ocr.json"
)

SELECTED_FILES = [
    ("Balance Sheet/Consolidated Balance Sheet 2017.pdf", "balance_sheet", "application/pdf", 1),
    ("Profit & Loss/Consolidated Profit & Loss 2022.pdf", "profit_and_loss", "application/pdf", 1),
    ("Cash Flows/Consolidated Cash Flow Statement 2017.pdf", "cash_flow", "application/pdf", 2),
    ("Invoices/batch1-1109.jpg", "invoice", "image/jpeg", 1),
    ("Invoices/20251118_000612.jpg", "invoice", "image/jpeg", 1),
]

MYSQL_PREFIX = "__pytest_mysql__bs2017.pdf"


def _upload(path: Path) -> UploadFile:
    return UploadFile(file=BytesIO(path.read_bytes()), filename=path.name)


@pytest.mark.dataset_smoke
@pytest.mark.parametrize(
    ("relative_path", "document_type", "expected_mime", "expected_pages"),
    SELECTED_FILES,
)
@pytest.mark.asyncio
async def test_real_dataset_file_validation(
    relative_path: str, document_type: str, expected_mime: str, expected_pages: int
):
    """REAL FILE. No OCR/Gemini. File validation only."""
    path = DATASET / relative_path
    if not path.is_file():
        pytest.skip(f"Dataset file missing: {path}")

    result = await validate_file(_upload(path))
    assert result["file_name"] == path.name
    assert result["file_validation"]["status"] == "PASS"
    assert result["file_validation"]["file_type"] == expected_mime
    assert result["file_validation"]["page_count"] == expected_pages
    assert document_type in {
        "invoice",
        "balance_sheet",
        "profit_and_loss",
        "cash_flow",
    }


@pytest.mark.dataset_smoke
def test_balance_sheet_2017_saved_extraction_through_financial_validation():
    """Saved REAL Gemini extraction fixture → REAL financial validation. No live Gemini/OCR."""
    if not FIXTURE_EXTRACTION.is_file():
        pytest.skip("Balance Sheet 2017 extraction fixture is missing")
    extraction = json.loads(FIXTURE_EXTRACTION.read_text(encoding="utf-8"))
    result = validate_financials(extraction)
    assert result["document_type"] == "Balance Sheet"
    assert result["check"]
    assert any(c["status"] == "PASS" for c in result["check"])


@pytest.mark.dataset_smoke
@pytest.mark.asyncio
async def test_balance_sheet_2017_orchestration_with_saved_ai_fixtures():
    """REAL FILE + saved OCR/Gemini fixtures + REAL file validation + REAL financial validation.

    Not a live OCR→Gemini E2E run.
    """
    path = DATASET / "Balance Sheet" / "Consolidated Balance Sheet 2017.pdf"
    if not path.is_file() or not FIXTURE_EXTRACTION.is_file() or not FIXTURE_OCR.is_file():
        pytest.skip("Balance Sheet 2017 file or AI fixtures missing")

    ocr = json.loads(FIXTURE_OCR.read_text(encoding="utf-8"))
    extraction = json.loads(FIXTURE_EXTRACTION.read_text(encoding="utf-8"))

    async def fake_ocr(_file, settings=None):
        return ocr

    async def fake_extract(_ocr_result, document_type, settings=None):
        assert document_type == "balance_sheet"
        return extraction

    result = await process_document(
        _upload(path),
        "balance_sheet",
        extract_text_fn=fake_ocr,
        extract_structured_data_fn=fake_extract,
    )
    assert result["document_type"] == "balance_sheet"
    assert result["processing_status"] == "PASS"
    assert result["document_name"] == path.name
    assert result["file_validation"]["page_count"] == 1
    assert result["validation"]["checks"]
    DocumentProcessingResult.model_validate(result)


@pytest.mark.dataset_smoke
@pytest.mark.mysql_integration
def test_balance_sheet_2017_fixture_persisted_to_mysql_and_retrieved_via_api():
    """Saved REAL extraction → REAL financial validation → REAL MySQL → GET API.

    Gemini/OCR are not called live.
    """
    path = DATASET / "Balance Sheet" / "Consolidated Balance Sheet 2017.pdf"
    if not path.is_file() or not FIXTURE_EXTRACTION.is_file():
        pytest.skip("Balance Sheet 2017 file or extraction fixture missing")

    get_settings.cache_clear()
    reset_engine()
    settings = Settings()
    if not (settings.database_url or "").strip().lower().startswith("mysql"):
        pytest.skip("DATABASE_URL is not a MySQL URL")
    init_db(settings)
    factory = get_session_factory(settings)
    repo = SqlAlchemyDocumentRepository(factory)

    extraction = json.loads(FIXTURE_EXTRACTION.read_text(encoding="utf-8"))
    financial = validate_financials(extraction)
    result: dict[str, Any] = {
        "document_name": MYSQL_PREFIX,
        "document_type": "balance_sheet",
        "processing_status": "PASS",
        "overall_confidence": 0.95,
        "file_validation": {
            "file_type": "application/pdf",
            "is_supported": True,
            "is_readable": True,
            "page_count": 1,
            "status": "PASS",
        },
        "extracted_data": extraction.get("document") or {},
        "validation": {
            "checks": financial.get("check") or [],
            "overall_status": "PASS"
            if all(c.get("status") != "FAIL" for c in (financial.get("check") or []))
            else "FAIL",
            "issues": [c for c in (financial.get("check") or []) if c.get("status") == "FAIL"],
        },
        "processing_metadata": {
            "ocr_used": True,
            "processed_at": "2026-09-11T00:00:00Z",
            "processing_time_ms": 1,
            "ocr_page_count": 1,
            "extraction_document_type": "Balance Sheet",
        },
    }

    async def fake_process(_file, document_type, **_kwargs):
        assert document_type == "balance_sheet"
        return dict(result)

    try:
        with factory() as session:
            session.execute(delete(Document).where(Document.document_name == MYSQL_PREFIX))
            session.commit()

        app.dependency_overrides[get_document_repository] = lambda: repo
        app.dependency_overrides[get_process_document_fn] = lambda: AsyncMock(
            side_effect=fake_process
        )
        with TestClient(app) as client:
            created = _post_process(
                client,
                filename=MYSQL_PREFIX,
                document_type="balance_sheet",
            )
            assert created.status_code == 200
            assert RESPONSE_KEYS.issubset(created.json().keys())
            DocumentProcessingResult.model_validate(created.json())
            _assert_no_secrets(created)

            fetched = client.get(f"/api/v1/documents/{MYSQL_PREFIX}")
            assert fetched.status_code == 200
            body = fetched.json()
            assert body["document_type"] == "balance_sheet"
            assert body["extracted_data"]["entity_name"]["value"] == "HDFC Bank Limited"
            assert body["validation"]["checks"]
            listed = client.get("/api/v1/documents")
            names = [d["document_name"] for d in listed.json()["documents"]]
            assert MYSQL_PREFIX in names
    finally:
        with factory() as session:
            session.execute(delete(Document).where(Document.document_name == MYSQL_PREFIX))
            session.commit()
        app.dependency_overrides.clear()
        reset_engine()
        get_settings.cache_clear()


@pytest.mark.dataset_smoke
def test_full_live_ocr_gemini_e2e_skipped_due_to_quota():
    pytest.skip(
        "True OCR→Gemini→validation E2E skipped: Gemini free-tier quota exhausted "
        "(429 RESOURCE_EXHAUSTED). Balance Sheet 2017 uses the saved extraction fixture instead."
    )

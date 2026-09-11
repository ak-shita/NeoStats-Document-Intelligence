"""API tests with mocked orchestration (no OCR or Gemini calls)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_document_api_service
from app.core.config import get_settings
from app.main import app
from app.repositories.memory import InMemoryDocumentRepository
from app.schemas.document import DocumentProcessingResult
from app.services.document_api_service import DocumentApiService
from app.services.document_service import DocumentProcessingError

PROCESS_URL = "/api/v1/documents/process"
RESPONSE_KEYS = {
    "document_name",
    "document_type",
    "processing_status",
    "overall_confidence",
    "file_validation",
    "extracted_data",
    "validation",
    "processing_metadata",
}


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
        "extracted_data": {
            "invoice_number": {"value": "94404257", "confidence": 0.99, "page_number": 1},
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
            "processed_at": "2026-09-10T18:00:00Z",
            "processing_time_ms": 12,
            "ocr_page_count": 1,
            "ocr_engine": 2,
            "extraction_document_type": "Invoice",
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def repo() -> InMemoryDocumentRepository:
    return InMemoryDocumentRepository()


@pytest.fixture
def process_fn() -> AsyncMock:
    return AsyncMock(return_value=_sample_result())


@pytest.fixture
def client(repo: InMemoryDocumentRepository, process_fn: AsyncMock) -> TestClient:
    service = DocumentApiService(repository=repo, process_fn=process_fn)
    app.dependency_overrides[get_document_api_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _post_process(
    client: TestClient,
    *,
    filename: str = "invoice.pdf",
    document_type: str | None = "invoice",
    include_file: bool = True,
) -> Any:
    data = {}
    if document_type is not None:
        data["document_type"] = document_type
    files = None
    if include_file:
        files = {"file": (filename, b"%PDF-1.4 mock-bytes", "application/pdf")}
    return client.post(PROCESS_URL, data=data, files=files)


def _assert_no_secrets(response) -> None:
    body = response.text
    settings = get_settings()
    assert "gemini_api_key" not in body
    assert "ocr_space_api_key" not in body
    if settings.gemini_api_key:
        assert settings.gemini_api_key not in body
    if settings.ocr_space_api_key:
        assert settings.ocr_space_api_key not in body


def test_health_returns_200(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "document-intelligence-api",
    }
    _assert_no_secrets(response)


def test_process_document_success_returns_200(client: TestClient, process_fn: AsyncMock):
    response = _post_process(client)
    assert response.status_code == 200
    body = response.json()
    assert RESPONSE_KEYS.issubset(body.keys())
    parsed = DocumentProcessingResult.model_validate(body)
    assert parsed.processing_status == "PASS"
    assert parsed.document_type == "invoice"
    assert parsed.document_name == "invoice.pdf"
    process_fn.assert_awaited_once()
    called_type = process_fn.await_args.args[1]
    assert called_type == "invoice"
    _assert_no_secrets(response)


def test_process_missing_file_returns_controlled_4xx(client: TestClient, process_fn: AsyncMock):
    response = _post_process(client, include_file=False)
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "MISSING_FILE"
    assert "file" in body["error"]["message"].lower()
    process_fn.assert_not_awaited()
    _assert_no_secrets(response)


def test_process_invalid_document_type_returns_controlled_4xx(
    client: TestClient, process_fn: AsyncMock
):
    response = _post_process(client, document_type="bank_statement")
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "INVALID_DOCUMENT_TYPE"
    process_fn.assert_not_awaited()
    _assert_no_secrets(response)


def test_process_orchestration_file_validation_failure(client: TestClient, process_fn: AsyncMock):
    process_fn.side_effect = DocumentProcessingError(
        code="UNSUPPORTED_FILE_TYPE",
        message="File type is not supported.",
        stage="file_validation",
    )
    response = _post_process(client, filename="notes.txt")
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == {
        "code": "UNSUPPORTED_FILE_TYPE",
        "message": "File type is not supported.",
    }
    assert "traceback" not in response.text.lower()
    _assert_no_secrets(response)


def test_get_existing_document_returns_200(client: TestClient):
    created = _post_process(client)
    assert created.status_code == 200
    response = client.get("/api/v1/documents/invoice.pdf")
    assert response.status_code == 200
    assert response.json()["document_name"] == "invoice.pdf"
    DocumentProcessingResult.model_validate(response.json())
    _assert_no_secrets(response)


def test_get_unknown_document_returns_404(client: TestClient):
    response = client.get("/api/v1/documents/missing.pdf")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "DOCUMENT_NOT_FOUND"
    _assert_no_secrets(response)


def test_list_documents_returns_200(client: TestClient, process_fn: AsyncMock):
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
    assert [doc["document_name"] for doc in body["documents"]] == [
        "invoice.pdf",
        "second.pdf",
    ]
    _assert_no_secrets(response)


def test_response_structure_matches_pydantic_schema(client: TestClient):
    response = _post_process(client)
    parsed = DocumentProcessingResult.model_validate(response.json())
    assert parsed.file_validation.status == "PASS"
    assert parsed.validation.overall_status == "PASS"
    assert parsed.processing_metadata.ocr_used is True


def test_secrets_are_never_exposed_on_unexpected_failure(client: TestClient, process_fn: AsyncMock):
    process_fn.side_effect = RuntimeError("gemini_api_key=should-never-leak")
    response = _post_process(client)
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert "should-never-leak" not in response.text
    assert "RuntimeError" not in response.text
    _assert_no_secrets(response)

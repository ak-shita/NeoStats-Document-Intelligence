"""Mocked unit tests for document orchestration (no OCR or Gemini calls)."""

from __future__ import annotations

from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile

from app.core.config import Settings
from app.schemas.document import DocumentProcessingResult
from app.services.document_service import DocumentProcessingError, process_document
from app.services.extraction_service import ExtractionError
from app.services.file_validation_service import FileValidationError
from app.services.financial_validation_service import FinancialValidationError
from app.services.ocr_service import OCRError

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


def _upload(filename: str = "invoice.pdf") -> UploadFile:
    return UploadFile(file=BytesIO(b"%PDF-1.4 mock"), filename=filename)


def _settings() -> Settings:
    return Settings(
        gemini_api_key="test-gemini-key",
        ocr_space_api_key="test-ocr-key",
    )


def _field(name: str, value: Any, confidence: float = 0.9, page: int = 1) -> dict[str, Any]:
    return {
        "field": name,
        "value": value,
        "confidence": confidence,
        "evidence": {"source_text": f"{name}: {value}", "page_number": page},
    }


def _file_validation_result(file_name: str = "invoice.pdf") -> dict[str, Any]:
    return {
        "file_name": file_name,
        "file_validation": {
            "file_type": "application/pdf",
            "is_supported": True,
            "is_readable": True,
            "page_count": 1,
            "status": "PASS",
        },
    }


def _ocr_result(file_name: str = "invoice.pdf") -> dict[str, Any]:
    return {
        "file_name": file_name,
        "success": True,
        "page_count": 1,
        "pages": [{"page_number": 1, "text": "Invoice 94404257 Total 100"}],
        "ocr_engine": 2,
    }


def _extraction_result(file_name: str = "invoice.pdf") -> dict[str, Any]:
    return {
        "file_name": file_name,
        "document_type": "Invoice",
        "document": {
            "document_type": "Invoice",
            "invoice_number": _field("invoice_number", "94404257", 0.99, 1),
            "currency": _field("currency", "USD", 0.81, 1),
            "total": _field("total", "100.00", 0.90, 1),
        },
    }


def _check(*, name: str, status: str, variance: float | None = 0.0) -> dict[str, Any]:
    return {
        "name": name,
        "formula": "a ≈ b",
        "operands": {"a": 100.0, "b": 100.0},
        "calculated_value": 100.0,
        "reported_value": 100.0,
        "variance": variance,
        "status": status,
        "reported_raw": "100.00",
    }


def _financial_result(checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file_name": "invoice.pdf",
        "document_type": "Invoice",
        "tolerance": 0.05,
        "check": checks,
    }


def _pipeline_mocks(
    *,
    file_name: str = "invoice.pdf",
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if checks is None:
        checks = [_check(name="invoice_taxable_plus_tax_vs_total", status="PASS")]
    return {
        "validate_file_fn": AsyncMock(return_value=_file_validation_result(file_name)),
        "extract_text_fn": AsyncMock(return_value=_ocr_result(file_name)),
        "extract_structured_data_fn": AsyncMock(return_value=_extraction_result(file_name)),
        "validate_financials_fn": MagicMock(return_value=_financial_result(checks)),
    }


async def _process(**overrides: Any) -> dict[str, Any]:
    kwargs = _pipeline_mocks()
    kwargs.update(overrides)
    return await process_document(
        _upload(kwargs.pop("filename", "invoice.pdf")),
        kwargs.pop("document_type", "Invoice"),
        settings=_settings(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_successful_end_to_end_orchestration():
    mocks = _pipeline_mocks()
    result = await process_document(
        _upload("invoice.pdf"),
        "Invoice",
        settings=_settings(),
        **mocks,
    )

    mocks["validate_file_fn"].assert_awaited_once()
    mocks["extract_text_fn"].assert_awaited_once()
    mocks["extract_structured_data_fn"].assert_awaited_once()
    mocks["validate_financials_fn"].assert_called_once()

    extract_kwargs = mocks["extract_structured_data_fn"].await_args
    assert extract_kwargs.args[1] == "invoice"

    assert result["processing_status"] == "PASS"
    assert result["document_name"] == "invoice.pdf"
    assert result["document_type"] == "invoice"
    assert result["overall_confidence"] == 0.9
    assert result["extracted_data"]["invoice_number"] == {
        "value": "94404257",
        "confidence": 0.99,
        "page_number": 1,
    }
    assert result["validation"]["overall_status"] == "PASS"
    assert result["validation"]["issues"] == []
    assert result["file_validation"]["status"] == "PASS"
    assert result["processing_metadata"]["ocr_used"] is True
    assert result["processing_metadata"]["ocr_engine"] == 2
    assert result["processing_metadata"]["extraction_document_type"] == "Invoice"


@pytest.mark.asyncio
async def test_final_response_structure_matches_assessment_envelope():
    result = await _process()

    assert set(result.keys()) == RESPONSE_KEYS
    assert set(result["file_validation"].keys()) == {
        "file_type",
        "is_supported",
        "is_readable",
        "page_count",
        "status",
    }
    assert set(result["validation"].keys()) == {"checks", "overall_status", "issues"}
    assert "ocr_used" in result["processing_metadata"]
    assert "processed_at" in result["processing_metadata"]
    assert "processing_time_ms" in result["processing_metadata"]

    parsed = DocumentProcessingResult.model_validate(result)
    assert parsed.processing_status == "PASS"
    assert parsed.document_type == "invoice"


@pytest.mark.asyncio
async def test_processing_status_is_pass_when_pipeline_completes():
    result = await _process()
    assert result["processing_status"] == "PASS"


@pytest.mark.asyncio
async def test_unsupported_document_type_fails_before_pipeline():
    mocks = _pipeline_mocks()

    with pytest.raises(DocumentProcessingError) as exc_info:
        await process_document(
            _upload("invoice.pdf"),
            "Bank Statement",
            settings=_settings(),
            **mocks,
        )

    err = exc_info.value
    assert err.code == "UNSUPPORTED_DOCUMENT_TYPE"
    assert err.stage == "document_type"
    assert "Unsupported document_type" in err.message
    mocks["validate_file_fn"].assert_not_awaited()
    mocks["extract_text_fn"].assert_not_awaited()
    mocks["extract_structured_data_fn"].assert_not_awaited()
    mocks["validate_financials_fn"].assert_not_called()


@pytest.mark.asyncio
async def test_file_validation_failure_raises_failed_processing():
    mocks = _pipeline_mocks()
    mocks["validate_file_fn"] = AsyncMock(
        side_effect=FileValidationError(
            code="UNSUPPORTED_FILE_TYPE",
            message="File type is not supported.",
        )
    )

    with pytest.raises(DocumentProcessingError) as exc_info:
        await process_document(
            _upload("notes.txt"),
            "Invoice",
            settings=_settings(),
            **mocks,
        )

    err = exc_info.value
    assert err.code == "UNSUPPORTED_FILE_TYPE"
    assert err.stage == "file_validation"
    mocks["extract_text_fn"].assert_not_awaited()
    mocks["extract_structured_data_fn"].assert_not_awaited()
    mocks["validate_financials_fn"].assert_not_called()


@pytest.mark.asyncio
async def test_ocr_failure_raises_failed_processing():
    mocks = _pipeline_mocks()
    mocks["extract_text_fn"] = AsyncMock(
        side_effect=OCRError(
            code="OCR_UNREADABLE",
            message="The document could not be read.",
        )
    )

    with pytest.raises(DocumentProcessingError) as exc_info:
        await process_document(
            _upload("invoice.pdf"),
            "Invoice",
            settings=_settings(),
            **mocks,
        )

    err = exc_info.value
    assert err.code == "OCR_UNREADABLE"
    assert err.stage == "ocr"
    mocks["extract_structured_data_fn"].assert_not_awaited()
    mocks["validate_financials_fn"].assert_not_called()


@pytest.mark.asyncio
async def test_extraction_failure_raises_failed_processing():
    mocks = _pipeline_mocks()
    mocks["extract_structured_data_fn"] = AsyncMock(
        side_effect=ExtractionError(
            code="EXTRACTION_FAILED",
            message="Structured extraction failed.",
        )
    )

    with pytest.raises(DocumentProcessingError) as exc_info:
        await process_document(
            _upload("invoice.pdf"),
            "Invoice",
            settings=_settings(),
            **mocks,
        )

    err = exc_info.value
    assert err.code == "EXTRACTION_FAILED"
    assert err.stage == "extraction"
    mocks["validate_financials_fn"].assert_not_called()


@pytest.mark.asyncio
async def test_financial_validation_fail_does_not_set_processing_status_failed():
    fail_check = _check(
        name="invoice_taxable_plus_tax_vs_total",
        status="FAIL",
        variance=12.5,
    )
    mocks = _pipeline_mocks(checks=[fail_check])

    result = await process_document(
        _upload("invoice.pdf"),
        "Invoice",
        settings=_settings(),
        **mocks,
    )

    assert result["processing_status"] == "PASS"
    assert result["validation"]["overall_status"] == "FAIL"
    assert result["validation"]["issues"] == [
        {
            "name": "invoice_taxable_plus_tax_vs_total",
            "status": "FAIL",
            "variance": 12.5,
            "formula": "a ≈ b",
        }
    ]


@pytest.mark.asyncio
async def test_not_applicable_validation_keeps_processing_status_pass():
    na_checks = [
        _check(name="invoice_line_1_quantity_times_unit_price", status="NOT_APPLICABLE", variance=None),
        _check(name="invoice_sum_line_totals_vs_subtotal", status="NOT_APPLICABLE", variance=None),
    ]
    mocks = _pipeline_mocks(checks=na_checks)

    result = await process_document(
        _upload("invoice.pdf"),
        "Invoice",
        settings=_settings(),
        **mocks,
    )

    assert result["processing_status"] == "PASS"
    assert result["validation"]["overall_status"] == "NOT_APPLICABLE"
    assert result["validation"]["issues"] == []
    assert {c["status"] for c in result["validation"]["checks"]} == {"NOT_APPLICABLE"}


@pytest.mark.asyncio
async def test_unrecoverable_financial_validation_error_is_failed_processing():
    mocks = _pipeline_mocks()
    mocks["validate_financials_fn"] = MagicMock(
        side_effect=FinancialValidationError(
            code="INVALID_EXTRACTION_PAYLOAD",
            message="extraction_result must be a mapping/dict.",
        )
    )

    with pytest.raises(DocumentProcessingError) as exc_info:
        await process_document(
            _upload("invoice.pdf"),
            "Invoice",
            settings=_settings(),
            **mocks,
        )

    err = exc_info.value
    assert err.code == "INVALID_EXTRACTION_PAYLOAD"
    assert err.stage == "financial_validation"

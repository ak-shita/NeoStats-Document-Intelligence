"""
Document orchestration service.

End-to-end flow:
    Upload → File Validation → OCR → Gemini extraction →
    Pydantic validation (inside extraction) → financial validation →
    final structured assessment result

Does not implement API routes, persistence, or frontend.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping

from fastapi import UploadFile

from app.core.config import Settings, get_settings
from app.schemas.extraction import DocumentType
from app.services.extraction_service import ExtractionError, extract_structured_data, normalize_document_type
from app.services.file_validation_service import FileValidationError, validate_file
from app.services.financial_validation_service import (
    DEFAULT_TOLERANCE,
    FinancialValidationError,
    validate_financials,
)
from app.services.ocr_service import OCRError, extract_text

logger = logging.getLogger(__name__)

PUBLIC_DOCUMENT_TYPE: dict[DocumentType, str] = {
    DocumentType.INVOICE: "invoice",
    DocumentType.BALANCE_SHEET: "balance_sheet",
    DocumentType.PROFIT_AND_LOSS: "profit_and_loss",
    DocumentType.CASH_FLOW: "cash_flow",
}

ValidateFileFn = Callable[[UploadFile], Awaitable[dict[str, Any]]]
ExtractTextFn = Callable[..., Awaitable[dict[str, Any]]]
ExtractStructuredFn = Callable[..., Awaitable[dict[str, Any]]]
ValidateFinancialsFn = Callable[..., dict[str, Any]]


class DocumentProcessingError(Exception):
    """
    Controlled orchestration failure for the API layer to map into:

        {"error": {"code": "...", "message": "..."}}
    """

    def __init__(self, code: str, message: str, *, stage: str | None = None) -> None:
        self.code = code
        self.message = message
        self.stage = stage
        super().__init__(message)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_extracted_field(node: Any) -> bool:
    return (
        isinstance(node, Mapping)
        and "value" in node
        and "field" in node
        and not any(k in node for k in ("line_items", "document_type"))
    )


def _public_extracted_field(node: Mapping[str, Any]) -> dict[str, Any]:
    evidence = node.get("evidence")
    page_number = None
    if isinstance(evidence, Mapping):
        page_number = evidence.get("page_number")
    return {
        "value": node.get("value"),
        "confidence": node.get("confidence"),
        "page_number": page_number,
    }


def _to_public_extracted_data(node: Any) -> Any:
    """
    Convert internal ExtractedField-heavy extraction document into the
    assessment-facing extracted_data shape (value/confidence/page_number).
    """
    if node is None:
        return None

    if isinstance(node, list):
        return [_to_public_extracted_data(item) for item in node]

    if not isinstance(node, Mapping):
        return node

    if _is_extracted_field(node):
        return _public_extracted_field(node)

    # Statement / invoice nested objects and top-level document.
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "document_type":
            # Public document_type lives on the envelope; keep nested type only
            # when useful for consumers of statement payloads.
            out[key] = value
            continue
        out[key] = _to_public_extracted_data(value)
    return out


def _collect_confidences(node: Any, bucket: list[float]) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_confidences(item, bucket)
        return
    if not isinstance(node, Mapping):
        return
    if "confidence" in node and "value" in node and "page_number" in node:
        conf = node.get("confidence")
        if isinstance(conf, (int, float)):
            bucket.append(float(conf))
        return
    for value in node.values():
        _collect_confidences(value, bucket)


def _overall_confidence(extracted_data: Mapping[str, Any]) -> float | None:
    scores: list[float] = []
    _collect_confidences(extracted_data, scores)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def _validation_overall_status(checks: list[Mapping[str, Any]]) -> str:
    if not checks:
        return "NOT_APPLICABLE"
    statuses = [str(c.get("status")) for c in checks]
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status == "PASS" for status in statuses):
        return "PASS"
    if all(status == "NOT_APPLICABLE" for status in statuses):
        return "NOT_APPLICABLE"
    return "PASS"


def _validation_issues(checks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for check in checks:
        if check.get("status") != "FAIL":
            continue
        issues.append(
            {
                "name": check.get("name"),
                "status": check.get("status"),
                "variance": check.get("variance"),
                "formula": check.get("formula"),
            }
        )
    return issues


def _wrap_error(exc: Exception, *, stage: str) -> DocumentProcessingError:
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None)
    if isinstance(code, str) and isinstance(message, str) and code and message:
        return DocumentProcessingError(code=code, message=message, stage=stage)
    return DocumentProcessingError(
        code="DOCUMENT_PROCESSING_ERROR",
        message="Document processing failed due to an unexpected error.",
        stage=stage,
    )


async def process_document(
    file: UploadFile,
    document_type: str,
    *,
    settings: Settings | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    validate_file_fn: ValidateFileFn | None = None,
    extract_text_fn: ExtractTextFn | None = None,
    extract_structured_data_fn: ExtractStructuredFn | None = None,
    validate_financials_fn: ValidateFinancialsFn | None = None,
) -> dict[str, Any]:
    """
    Orchestrate the full document intelligence pipeline.

    Returns the assessment structured response on successful completion of
    extraction + financial validation. Financial check FAIL values are reported
    inside ``validation``; ``processing_status`` remains ``PASS`` in that case.

    Raises ``DocumentProcessingError`` for unsupported types and unrecoverable
    file / OCR / extraction / processing failures (no secrets in messages).
    """
    started = time.perf_counter()
    cfg = settings or get_settings()

    validate_file_fn = validate_file_fn or validate_file
    extract_text_fn = extract_text_fn or extract_text
    extract_structured_data_fn = extract_structured_data_fn or extract_structured_data
    validate_financials_fn = validate_financials_fn or validate_financials

    # 1) document_type gate (before any I/O / provider calls)
    try:
        resolved_type = normalize_document_type(document_type)
    except ExtractionError as exc:
        raise _wrap_error(exc, stage="document_type") from exc

    public_type = PUBLIC_DOCUMENT_TYPE[resolved_type]
    document_name = file.filename or "upload"

    # 2) File validation
    try:
        validation_result = await validate_file_fn(file)
    except FileValidationError as exc:
        raise _wrap_error(exc, stage="file_validation") from exc
    except Exception as exc:
        logger.exception("Unexpected file validation failure")
        raise DocumentProcessingError(
            code="FILE_VALIDATION_UNEXPECTED",
            message="File validation failed due to an unexpected error.",
            stage="file_validation",
        ) from exc

    file_validation = validation_result.get("file_validation") or {}
    document_name = validation_result.get("file_name") or document_name

    # 3) OCR
    try:
        ocr_result = await extract_text_fn(file, settings=cfg)
    except OCRError as exc:
        raise _wrap_error(exc, stage="ocr") from exc
    except Exception as exc:
        logger.exception("Unexpected OCR failure")
        raise DocumentProcessingError(
            code="OCR_UNEXPECTED",
            message="OCR failed due to an unexpected error.",
            stage="ocr",
        ) from exc

    # 4) Gemini structured extraction (+ internal Pydantic validation)
    try:
        extraction_result = await extract_structured_data_fn(
            ocr_result,
            public_type,
            settings=cfg,
        )
    except ExtractionError as exc:
        raise _wrap_error(exc, stage="extraction") from exc
    except Exception as exc:
        logger.exception("Unexpected extraction failure")
        raise DocumentProcessingError(
            code="EXTRACTION_UNEXPECTED",
            message="Extraction failed due to an unexpected error.",
            stage="extraction",
        ) from exc

    # 5) Financial validation (deterministic; FAIL checks are not processing failures)
    try:
        financial_result = validate_financials_fn(
            extraction_result,
            tolerance=tolerance,
        )
    except FinancialValidationError as exc:
        raise _wrap_error(exc, stage="financial_validation") from exc
    except Exception as exc:
        logger.exception("Unexpected financial validation failure")
        raise DocumentProcessingError(
            code="FINANCIAL_VALIDATION_UNEXPECTED",
            message="Financial validation failed due to an unexpected error.",
            stage="financial_validation",
        ) from exc

    checks = list(financial_result.get("check") or [])
    extracted_document = extraction_result.get("document") or {}
    extracted_data = _to_public_extracted_data(extracted_document)
    if not isinstance(extracted_data, dict):
        extracted_data = {}

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response = {
        "document_name": document_name,
        "document_type": public_type,
        "processing_status": "PASS",
        "overall_confidence": _overall_confidence(extracted_data),
        "file_validation": {
            "file_type": file_validation.get("file_type"),
            "is_supported": bool(file_validation.get("is_supported", True)),
            "is_readable": bool(file_validation.get("is_readable", True)),
            "page_count": file_validation.get("page_count"),
            "status": file_validation.get("status", "PASS"),
        },
        "extracted_data": extracted_data,
        "validation": {
            "checks": checks,
            "overall_status": _validation_overall_status(checks),
            "issues": _validation_issues(checks),
        },
        "processing_metadata": {
            "ocr_used": True,
            "processed_at": _utc_now_iso(),
            "processing_time_ms": elapsed_ms,
            "ocr_page_count": ocr_result.get("page_count"),
            "ocr_engine": ocr_result.get("ocr_engine"),
            "extraction_document_type": extraction_result.get("document_type"),
        },
    }
    return response

"""Map controlled service exceptions to HTTP error envelopes."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.document_service import DocumentProcessingError

logger = logging.getLogger(__name__)

CLIENT_ERROR_CODES = frozenset(
    {
        "UNSUPPORTED_DOCUMENT_TYPE",
        "EMPTY_FILE",
        "UNSUPPORTED_FILE_TYPE",
        "INVALID_FILE_CONTENT",
        "INVALID_EXTRACTION_PAYLOAD",
        "INVALID_TOLERANCE",
        "EXTRACTION_EMPTY_OCR_INPUT",
        "OCR_UNSUPPORTED_FILE",
        "OCR_EMPTY_FILE",
        "MISSING_FILE",
        "INVALID_DOCUMENT_TYPE",
        "INVALID_FILENAME",
        "INVALID_DOCUMENT_NAME",
        "VALIDATION_ERROR",
    }
)

UNPROCESSABLE_CODES = frozenset(
    {
        "CORRUPTED_OR_UNREADABLE_FILE",
        "PAGE_LIMIT_EXCEEDED",
        "OCR_EMPTY_TEXT",
        "OCR_UNREADABLE",
        "OCR_PARTIAL_FAILURE",
        "EXTRACTION_MALFORMED_RESPONSE",
        "EXTRACTION_EMPTY_RESPONSE",
    }
)

BAD_GATEWAY_CODES = frozenset(
    {
        "OCR_TIMEOUT",
        "OCR_NETWORK_ERROR",
        "OCR_HTTP_ERROR",
        "OCR_PROVIDER_ERROR",
        "OCR_MALFORMED_RESPONSE",
        "OCR_UNEXPECTED",
        "EXTRACTION_TIMEOUT",
        "EXTRACTION_MODEL_ERROR",
        "EXTRACTION_FAILED",
        "EXTRACTION_UNEXPECTED",
    }
)

SERVICE_UNAVAILABLE_CODES = frozenset(
    {
        "MISSING_GEMINI_API_KEY",
        "MISSING_OCR_API_KEY",
    }
)


def error_payload(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def status_for_code(code: str) -> int:
    if code == "DOCUMENT_NOT_FOUND":
        return 404
    if code in CLIENT_ERROR_CODES:
        return 400
    if code in UNPROCESSABLE_CODES:
        return 422
    if code in BAD_GATEWAY_CODES:
        return 502
    if code in SERVICE_UNAVAILABLE_CODES:
        return 503
    return 500


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocumentProcessingError)
    async def handle_document_processing_error(
        request: Request, exc: DocumentProcessingError
    ) -> JSONResponse:
        status = status_for_code(exc.code)
        logger.warning(
            "Document processing failed method=%s path=%s code=%s stage=%s status=%s",
            request.method,
            request.url.path,
            exc.code,
            exc.stage,
            status,
        )
        return JSONResponse(
            status_code=status,
            content=error_payload(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        code, message = _request_validation_error(exc)
        logger.warning(
            "Request validation failed method=%s path=%s code=%s",
            request.method,
            request.url.path,
            code,
        )
        return JSONResponse(status_code=400, content=error_payload(code, message))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
            code = str(exc.detail["code"])
            message = str(exc.detail["message"])
        elif exc.status_code == 404:
            code = "NOT_FOUND"
            message = "The requested resource was not found."
        else:
            code = "HTTP_ERROR"
            message = str(exc.detail) if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(code, message),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unexpected API failure method=%s path=%s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "INTERNAL_SERVER_ERROR",
                "An unexpected error occurred.",
            ),
        )


def _request_validation_error(exc: RequestValidationError) -> tuple[str, str]:
    errors = exc.errors()

    def _field(err: dict) -> str | None:
        loc = err.get("loc") or ()
        return str(loc[-1]) if loc else None

    file_missing = any(_field(err) == "file" and err.get("type") == "missing" for err in errors)
    type_missing = any(
        _field(err) == "document_type" and err.get("type") == "missing" for err in errors
    )
    type_invalid = any(_field(err) == "document_type" for err in errors)

    if file_missing:
        return "MISSING_FILE", "A file upload is required."
    if type_missing:
        return (
            "INVALID_DOCUMENT_TYPE",
            "document_type is required and must be one of: "
            "invoice, balance_sheet, profit_and_loss, cash_flow.",
        )
    if type_invalid:
        return (
            "INVALID_DOCUMENT_TYPE",
            "document_type must be one of: invoice, balance_sheet, profit_and_loss, cash_flow.",
        )
    return "VALIDATION_ERROR", "Request validation failed."

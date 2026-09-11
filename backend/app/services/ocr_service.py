"""
OCR layer — OCR.Space text extraction only.

Pipeline position:
    Upload → File Validation → OCR/Text Extraction → Gemini → ...

This service does not perform semantic field extraction, Gemini calls,
financial validation, persistence, or FastAPI routing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from fastapi import UploadFile

from app.core.config import Settings, get_settings
from app.services.image_preprocessing_service import (
    ImagePreprocessingError,
    prepare_image_for_ocr,
)

logger = logging.getLogger(__name__)

DEFAULT_OCR_ENGINE = 2
MAX_OCR_ATTEMPTS = 3
RETRYABLE_HTTP_STATUSES = {500, 502, 503, 504}


class OCRError(Exception):
    """
    Controlled OCR failure for the API layer to map into:

        {"error": {"code": "...", "message": "..."}}
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _detect_payload_type(content: bytes, filename: str) -> tuple[str, str]:
    """
    Return ``(content_type, filetype)`` for the OCR.Space multipart upload.

    ``filetype`` is the OCR.Space override (PDF / JPG / PNG).
    """

    suffix = Path(filename).suffix.lower()

    if content.startswith(b"%PDF") or suffix == ".pdf":
        return "application/pdf", "PDF"
    if content.startswith(b"\x89PNG\r\n\x1a\n") or suffix == ".png":
        return "image/png", "PNG"
    if content.startswith(b"\xff\xd8\xff") or suffix in {".jpg", ".jpeg"}:
        return "image/jpeg", "JPG"

    raise OCRError(
        code="OCR_UNSUPPORTED_FILE",
        message="OCR supports only PDF / JPG / PNG documents.",
    )


def _normalize_error_message(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def _looks_like_size_limit_error(message: str | None) -> bool:
    if not message:
        return False
    lowered = message.lower()
    needles = (
        "file size",
        "filesize",
        "too large",
        "1 mb",
        "1mb",
        "maximum size",
        "size limit",
    )
    return any(needle in lowered for needle in needles)


def _parse_orientation(page: dict[str, Any]) -> int | None:
    raw = page.get("TextOrientation", page.get("Orientation"))
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _build_pages(parsed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []

    for index, page in enumerate(parsed_results, start=1):
        if not isinstance(page, dict):
            raise OCRError(
                code="OCR_MALFORMED_RESPONSE",
                message="OCR provider returned a malformed page result.",
            )

        exit_code = page.get("FileParseExitCode")
        try:
            exit_code_int = int(exit_code) if exit_code is not None else None
        except (TypeError, ValueError):
            exit_code_int = None

        page_error = _normalize_error_message(
            page.get("ErrorMessage") or page.get("ErrorDetails")
        )

        if exit_code_int is not None and exit_code_int != 1:
            raise OCRError(
                code="OCR_PARTIAL_FAILURE",
                message=(
                    page_error
                    or f"OCR failed while processing page {index}."
                ),
            )

        text = page.get("ParsedText")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise OCRError(
                code="OCR_MALFORMED_RESPONSE",
                message="OCR provider returned non-text ParsedText.",
            )

        pages.append(
            {
                "page_number": index,
                "text": text,
                "orientation": _parse_orientation(page),
                "file_parse_exit_code": exit_code_int,
            }
        )

    return pages


def _map_provider_payload(payload: dict[str, Any], *, engine: int) -> dict[str, Any]:
    if "ParsedResults" not in payload and payload.get("IsErroredOnProcessing"):
        message = _normalize_error_message(
            payload.get("ErrorMessage") or payload.get("ErrorDetails")
        ) or "OCR provider reported a processing error."
        code = (
            "OCR_FILE_TOO_LARGE"
            if _looks_like_size_limit_error(message)
            else "OCR_PROVIDER_ERROR"
        )
        raise OCRError(code=code, message=message)

    parsed_results = payload.get("ParsedResults")
    if parsed_results is None:
        raise OCRError(
            code="OCR_MALFORMED_RESPONSE",
            message="OCR provider response is missing ParsedResults.",
        )
    if not isinstance(parsed_results, list):
        raise OCRError(
            code="OCR_MALFORMED_RESPONSE",
            message="OCR provider returned an unexpected ParsedResults type.",
        )

    provider_error = _normalize_error_message(
        payload.get("ErrorMessage") or payload.get("ErrorDetails")
    )

    if payload.get("IsErroredOnProcessing"):
        code = (
            "OCR_FILE_TOO_LARGE"
            if _looks_like_size_limit_error(provider_error)
            else "OCR_PROVIDER_ERROR"
        )
        raise OCRError(
            code=code,
            message=provider_error or "OCR provider reported a processing error.",
        )

    try:
        exit_code = int(payload.get("OCRExitCode"))
    except (TypeError, ValueError) as exc:
        raise OCRError(
            code="OCR_MALFORMED_RESPONSE",
            message="OCR provider returned an invalid OCRExitCode.",
        ) from exc

    # 1 = all pages OK, 2 = partial, 3/4 = failure.
    if exit_code == 2:
        raise OCRError(
            code="OCR_PARTIAL_FAILURE",
            message=provider_error
            or "OCR completed only partially for a multi-page document.",
        )
    if exit_code in {3, 4}:
        code = (
            "OCR_FILE_TOO_LARGE"
            if _looks_like_size_limit_error(provider_error)
            else "OCR_PROVIDER_ERROR"
        )
        raise OCRError(
            code=code,
            message=provider_error or "OCR provider failed to parse the document.",
        )
    if exit_code != 1:
        raise OCRError(
            code="OCR_PROVIDER_ERROR",
            message=provider_error or f"Unexpected OCRExitCode: {exit_code}.",
        )

    if not parsed_results:
        raise OCRError(
            code="OCR_EMPTY_TEXT",
            message="OCR returned no page results.",
        )

    pages = _build_pages(parsed_results)

    combined = "\n".join(page["text"] for page in pages).strip()
    if not combined:
        raise OCRError(
            code="OCR_EMPTY_TEXT",
            message="OCR returned no usable text.",
        )

    processing_time_raw = payload.get("ProcessingTimeInMilliseconds")
    try:
        processing_time_ms = (
            int(processing_time_raw) if processing_time_raw is not None else None
        )
    except (TypeError, ValueError):
        processing_time_ms = None

    return {
        "success": True,
        "page_count": len(pages),
        "pages": [
            {
                "page_number": page["page_number"],
                "text": page["text"],
                "orientation": page["orientation"],
            }
            for page in pages
        ],
        "ocr_engine": engine,
        "processing_time_ms": processing_time_ms,
        "provider_error": None,
    }


async def _post_ocr_space(
    *,
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    filename: str,
    content: bytes,
    content_type: str,
    filetype: str,
    engine: int,
) -> dict[str, Any]:
    # Never log api_key or raw credentials.
    files = {
        "file": (filename, content, content_type),
    }
    data = {
        "apikey": api_key,
        "language": "eng",
        "OCREngine": str(engine),
        "isOverlayRequired": "true",
        "detectOrientation": "true",
        "isTable": "true",
        "scale": "true",
        "filetype": filetype,
    }

    try:
        response = await client.post(endpoint, data=data, files=files)
    except httpx.TimeoutException as exc:
        raise OCRError(
            code="OCR_TIMEOUT",
            message="OCR provider request timed out.",
        ) from exc
    except httpx.RequestError as exc:
        raise OCRError(
            code="OCR_NETWORK_ERROR",
            message="Unable to reach the OCR provider.",
        ) from exc

    if response.status_code >= 400:
        raise OCRError(
            code="OCR_HTTP_ERROR",
            message=f"OCR provider returned HTTP {response.status_code}.",
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise OCRError(
            code="OCR_MALFORMED_RESPONSE",
            message="OCR provider returned non-JSON content.",
        ) from exc

    if not isinstance(payload, dict):
        raise OCRError(
            code="OCR_MALFORMED_RESPONSE",
            message="OCR provider returned an unexpected JSON payload.",
        )

    return payload


def _is_retryable_ocr_error(error: OCRError) -> bool:
    """Return whether a provider failure is safe and useful to retry."""
    if error.code in {"OCR_TIMEOUT", "OCR_NETWORK_ERROR"}:
        return True
    if error.code != "OCR_HTTP_ERROR":
        return False
    return any(
        error.message.endswith(f"HTTP {status}.")
        for status in RETRYABLE_HTTP_STATUSES
    )


async def extract_text(
    file: UploadFile,
    *,
    settings: Settings | None = None,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """
    Run OCR.Space against a validated upload and return an internal result.

    The returned structure is for downstream Gemini extraction — not the
    final public API response.
    """

    cfg = settings or get_settings()
    resolved_key = (api_key if api_key is not None else cfg.ocr_space_api_key).strip()
    if not resolved_key:
        raise OCRError(
            code="MISSING_OCR_API_KEY",
            message="OCR_SPACE_API_KEY is not configured.",
        )

    filename = file.filename or "upload"
    content = await file.read()
    await file.seek(0)

    if not content:
        raise OCRError(
            code="OCR_EMPTY_FILE",
            message="Cannot run OCR on an empty file.",
        )

    content_type, filetype = _detect_payload_type(content, filename)
    upload_name = filename
    upload_bytes = content
    upload_content_type = content_type

    # PDFs are sent as PDFs. Only oversized images get an OCR-only copy.
    if filetype in {"JPG", "PNG"} and len(content) > cfg.ocr_space_max_upload_bytes:
        try:
            upload_bytes, upload_name, upload_content_type = prepare_image_for_ocr(
                content,
                filename=filename,
                max_bytes=cfg.ocr_space_max_upload_bytes,
            )
            filetype = "JPG" if upload_content_type == "image/jpeg" else "PNG"
            logger.info(
                "Prepared OCR image copy for %s (%s bytes -> %s bytes)",
                filename,
                len(content),
                len(upload_bytes),
            )
        except ImagePreprocessingError as exc:
            raise OCRError(code=exc.code, message=exc.message) from exc

    engine = cfg.ocr_space_engine or DEFAULT_OCR_ENGINE
    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.ocr_space_timeout_seconds)
    )

    started = time.perf_counter()
    try:
        for attempt in range(1, MAX_OCR_ATTEMPTS + 1):
            try:
                # upload_bytes is an immutable in-memory payload, so every retry
                # gets a complete multipart file rather than a consumed stream.
                payload = await _post_ocr_space(
                    client=http_client,
                    endpoint=cfg.ocr_space_endpoint,
                    api_key=resolved_key,
                    filename=upload_name,
                    content=upload_bytes,
                    content_type=upload_content_type,
                    filetype=filetype,
                    engine=engine,
                )
                break
            except OCRError as exc:
                if attempt == MAX_OCR_ATTEMPTS or not _is_retryable_ocr_error(exc):
                    raise
                delay_seconds = 2 ** (attempt - 1)
                logger.warning(
                    "OCR request failed (%s); retrying attempt %s/%s in %s second(s)",
                    exc.code,
                    attempt + 1,
                    MAX_OCR_ATTEMPTS,
                    delay_seconds,
                )
                await asyncio.sleep(delay_seconds)
        mapped = _map_provider_payload(payload, engine=engine)
    finally:
        if owns_client:
            await http_client.aclose()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if mapped.get("processing_time_ms") is None:
        mapped["processing_time_ms"] = elapsed_ms

    return {
        "file_name": filename,
        **mapped,
    }

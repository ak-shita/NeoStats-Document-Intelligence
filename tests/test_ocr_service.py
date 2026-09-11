"""Unit tests for the OCR.Space service (mocked — no live API calls)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import UploadFile
from PIL import Image

from app.core.config import Settings
from app.services.image_preprocessing_service import prepare_image_for_ocr
from app.services.ocr_service import OCRError, extract_text


def _upload(content: bytes, filename: str) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


def _tiny_jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(200, 200, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _large_jpeg(min_bytes: int = 1_200_000) -> bytes:
    """Build an in-memory JPEG larger than the OCR.Space free-tier limit."""

    import os

    # Random RGB noise compresses poorly → reliably exceeds 1 MB.
    width, height = 2200, 2200
    image = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded = buffer.getvalue()
    if len(encoded) < min_bytes:
        width, height = 3200, 3200
        image = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        encoded = buffer.getvalue()
    assert len(encoded) >= min_bytes
    return encoded


def _settings(**overrides) -> Settings:
    base = {
        "ocr_space_api_key": "test-key",
        "ocr_space_endpoint": "https://api.ocr.space/parse/image",
        "ocr_space_max_upload_bytes": 1_000_000,
        "ocr_space_timeout_seconds": 5.0,
        "ocr_space_engine": 2,
    }
    base.update(overrides)
    return Settings(**base)


def _success_payload_single() -> dict:
    return {
        "ParsedResults": [
            {
                "ParsedText": "Invoice Total 100.00",
                "FileParseExitCode": 1,
                "TextOrientation": "0",
                "ErrorMessage": None,
                "ErrorDetails": None,
            }
        ],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
        "ErrorMessage": None,
        "ErrorDetails": None,
        "ProcessingTimeInMilliseconds": "321",
    }


def _success_payload_multi() -> dict:
    return {
        "ParsedResults": [
            {
                "ParsedText": "Page one cash flow",
                "FileParseExitCode": 1,
                "TextOrientation": "0",
            },
            {
                "ParsedText": "Page two notes",
                "FileParseExitCode": 1,
                "TextOrientation": "270",
            },
        ],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
        "ProcessingTimeInMilliseconds": "900",
    }


@pytest.mark.asyncio
async def test_successful_single_page_ocr():
    payload = _success_payload_single()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://api.ocr.space/parse/image")
        assert b"name=\"OCREngine\"\r\n\r\n2" in request.content or b"OCREngine" in request.content
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await extract_text(
            _upload(_tiny_jpeg(), "invoice.jpg"),
            settings=_settings(),
            client=client,
        )

    assert result["file_name"] == "invoice.jpg"
    assert result["success"] is True
    assert result["page_count"] == 1
    assert result["ocr_engine"] == 2
    assert result["processing_time_ms"] == 321
    assert result["pages"][0]["text"] == "Invoice Total 100.00"
    assert result["pages"][0]["orientation"] == 0


@pytest.mark.asyncio
async def test_successful_multi_page_ocr():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_payload_multi())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await extract_text(
            _upload(b"%PDF-1.4 fake-but-typed", "statement.pdf"),
            settings=_settings(),
            client=client,
        )

    assert result["page_count"] == 2
    assert result["pages"][0]["page_number"] == 1
    assert result["pages"][1]["page_number"] == 2
    assert result["pages"][0]["text"] == "Page one cash flow"
    assert result["pages"][1]["orientation"] == 270


@pytest.mark.asyncio
async def test_provider_error_response():
    payload = {
        "ParsedResults": [],
        "OCRExitCode": 4,
        "IsErroredOnProcessing": True,
        "ErrorMessage": ["OCR engine failed"],
        "ErrorDetails": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(_tiny_jpeg(), "invoice.jpg"),
                settings=_settings(),
                client=client,
            )

    assert exc_info.value.code == "OCR_PROVIDER_ERROR"
    assert "OCR engine failed" in exc_info.value.message


@pytest.mark.asyncio
async def test_missing_api_key():
    with pytest.raises(OCRError) as exc_info:
        await extract_text(
            _upload(_tiny_jpeg(), "invoice.jpg"),
            settings=_settings(ocr_space_api_key=""),
        )

    assert exc_info.value.code == "MISSING_OCR_API_KEY"


@pytest.mark.asyncio
async def test_timeout_failure(monkeypatch: pytest.MonkeyPatch):
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.ocr_service.asyncio.sleep", sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(_tiny_jpeg(), "invoice.jpg"),
                settings=_settings(),
                client=client,
            )

    assert exc_info.value.code == "OCR_TIMEOUT"
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_network_failure(monkeypatch: pytest.MonkeyPatch):
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.ocr_service.asyncio.sleep", sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(_tiny_jpeg(), "invoice.jpg"),
                settings=_settings(),
                client=client,
            )

    assert exc_info.value.code == "OCR_NETWORK_ERROR"
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_http_error_status(monkeypatch: pytest.MonkeyPatch):
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.ocr_service.asyncio.sleep", sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(_tiny_jpeg(), "invoice.jpg"),
                settings=_settings(),
                client=client,
            )

    assert exc_info.value.code == "OCR_HTTP_ERROR"
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_503_retries_with_reusable_upload_bytes_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.ocr_service.asyncio.sleep", sleep)
    attempts = 0
    request_bodies: list[bytes] = []
    image = _tiny_jpeg()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        request_bodies.append(request.content)
        if attempts == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=_success_payload_single())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await extract_text(
            _upload(image, "invoice.jpg"), settings=_settings(), client=client
        )

    assert result["success"] is True
    assert attempts == 2
    sleep.assert_awaited_once_with(1)
    assert all(image in body for body in request_bodies)


@pytest.mark.asyncio
async def test_repeated_503_returns_existing_http_error_after_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
):
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.ocr_service.asyncio.sleep", sleep)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="unavailable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(_tiny_jpeg(), "invoice.jpg"), settings=_settings(), client=client
            )

    assert exc_info.value.code == "OCR_HTTP_ERROR"
    assert attempts == 3
    assert sleep.await_args_list[0].args == (1,)
    assert sleep.await_args_list[1].args == (2,)


@pytest.mark.asyncio
async def test_timeout_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.ocr_service.asyncio.sleep", sleep)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json=_success_payload_single())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await extract_text(
            _upload(_tiny_jpeg(), "invoice.jpg"), settings=_settings(), client=client
        )

    assert result["success"] is True
    assert attempts == 2
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_400_is_not_retried(monkeypatch: pytest.MonkeyPatch):
    sleep = AsyncMock()
    monkeypatch.setattr("app.services.ocr_service.asyncio.sleep", sleep)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(_tiny_jpeg(), "invoice.jpg"), settings=_settings(), client=client
            )

    assert exc_info.value.code == "OCR_HTTP_ERROR"
    assert attempts == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_ocr_text():
    payload = {
        "ParsedResults": [
            {
                "ParsedText": "   \n",
                "FileParseExitCode": 1,
                "TextOrientation": "0",
            }
        ],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(_tiny_jpeg(), "blank.jpg"),
                settings=_settings(),
                client=client,
            )

    assert exc_info.value.code == "OCR_EMPTY_TEXT"


@pytest.mark.asyncio
async def test_malformed_provider_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(_tiny_jpeg(), "invoice.jpg"),
                settings=_settings(),
                client=client,
            )

    assert exc_info.value.code == "OCR_MALFORMED_RESPONSE"


@pytest.mark.asyncio
async def test_partial_multi_page_failure():
    payload = {
        "ParsedResults": [
            {"ParsedText": "ok", "FileParseExitCode": 1},
            {
                "ParsedText": None,
                "FileParseExitCode": -10,
                "ErrorMessage": "page failed",
            },
        ],
        "OCRExitCode": 2,
        "IsErroredOnProcessing": False,
        "ErrorMessage": "Parsed partially",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(b"%PDF-1.4 multi", "multi.pdf"),
                settings=_settings(),
                client=client,
            )

    assert exc_info.value.code == "OCR_PARTIAL_FAILURE"


@pytest.mark.asyncio
async def test_provider_file_size_error_for_pdf():
    payload = {
        "OCRExitCode": 4,
        "IsErroredOnProcessing": True,
        "ErrorMessage": "File size exceeds the maximum size limit of 1 MB",
        "ParsedResults": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OCRError) as exc_info:
            await extract_text(
                _upload(b"%PDF-1.4 " + b"x" * 100, "big.pdf"),
                settings=_settings(),
                client=client,
            )

    assert exc_info.value.code == "OCR_FILE_TOO_LARGE"


@pytest.mark.asyncio
async def test_large_image_preprocessing_path():
    large = _large_jpeg()
    assert len(large) > 1_000_000

    prepared_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # Multipart body should contain the compressed OCR copy, not the original.
        prepared_sizes.append(len(request.content))
        return httpx.Response(200, json=_success_payload_single())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await extract_text(
            _upload(large, "huge_invoice.jpg"),
            settings=_settings(),
            client=client,
        )

    assert result["success"] is True
    assert result["file_name"] == "huge_invoice.jpg"
    # Request framing adds multipart overhead, but the file part must be reduced.
    assert prepared_sizes
    assert prepared_sizes[0] < len(large)


def test_prepare_image_for_ocr_reduces_below_limit():
    large = _large_jpeg()
    encoded, name, content_type = prepare_image_for_ocr(
        large,
        filename="scan.png",
        max_bytes=1_000_000,
    )
    assert len(encoded) <= 1_000_000
    assert content_type == "image/jpeg"
    assert name.endswith(".jpg")
    # Original bytes object is untouched.
    assert len(large) > 1_000_000

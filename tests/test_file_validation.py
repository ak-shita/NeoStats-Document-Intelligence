"""Tests for the file validation service (input control only)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from PIL import Image
from pypdf import PdfWriter

from app.services.file_validation_service import FileValidationError, validate_file

DATASET_ROOT = Path(__file__).resolve().parents[1] / "New Dataset"


def _upload(
    content: bytes,
    filename: str,
    content_type: str | None = None,
) -> UploadFile:
    headers = {"content-type": content_type} if content_type else None
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=headers,
    )


def _pdf_bytes(page_count: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(0, 128, 255)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_valid_pdf_passes():
    content = _pdf_bytes(1)
    result = await validate_file(
        _upload(content, "sample_invoice.pdf", "application/pdf")
    )

    assert result == {
        "file_name": "sample_invoice.pdf",
        "file_validation": {
            "file_type": "application/pdf",
            "is_supported": True,
            "is_readable": True,
            "page_count": 1,
            "status": "PASS",
        },
    }


@pytest.mark.asyncio
async def test_valid_three_page_pdf_passes():
    result = await validate_file(
        _upload(_pdf_bytes(3), "three.pdf", "application/pdf")
    )
    assert result["file_validation"]["status"] == "PASS"
    assert result["file_validation"]["page_count"] == 3


@pytest.mark.asyncio
async def test_valid_png_passes():
    result = await validate_file(_upload(_png_bytes(), "scan.png", "image/png"))

    assert result["file_validation"]["file_type"] == "image/png"
    assert result["file_validation"]["page_count"] == 1
    assert result["file_validation"]["status"] == "PASS"


@pytest.mark.asyncio
async def test_valid_jpeg_passes():
    result = await validate_file(
        _upload(_jpeg_bytes(), "photo.jpg", "image/jpeg")
    )

    assert result["file_validation"]["file_type"] == "image/jpeg"
    assert result["file_validation"]["is_supported"] is True
    assert result["file_validation"]["status"] == "PASS"


@pytest.mark.asyncio
async def test_jpeg_with_jpeg_extension_passes():
    result = await validate_file(
        _upload(_jpeg_bytes(), "photo.jpeg", "image/jpeg")
    )
    assert result["file_validation"]["file_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_empty_file_fails():
    with pytest.raises(FileValidationError) as exc_info:
        await validate_file(_upload(b"", "empty.pdf", "application/pdf"))

    assert exc_info.value.code == "EMPTY_FILE"
    assert "empty" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_unsupported_extension_fails():
    with pytest.raises(FileValidationError) as exc_info:
        await validate_file(
            _upload(b"not-a-real-doc", "notes.docx", "application/octet-stream")
        )

    assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"
    assert "PDF / JPG / PNG" in exc_info.value.message


@pytest.mark.asyncio
async def test_unsupported_signature_with_pdf_name_fails():
    with pytest.raises(FileValidationError) as exc_info:
        await validate_file(
            _upload(b"plain text pretending to be pdf", "fake.pdf", "application/pdf")
        )

    assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"


@pytest.mark.asyncio
async def test_extension_content_mismatch_fails():
    with pytest.raises(FileValidationError) as exc_info:
        await validate_file(
            _upload(_png_bytes(), "looks_like.pdf", "application/pdf")
        )

    assert exc_info.value.code == "INVALID_FILE_CONTENT"
    assert "does not match" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_page_limit_exceeded_fails():
    with pytest.raises(FileValidationError) as exc_info:
        await validate_file(
            _upload(_pdf_bytes(4), "too_long.pdf", "application/pdf")
        )

    assert exc_info.value.code == "PAGE_LIMIT_EXCEEDED"
    assert "3 pages" in exc_info.value.message


@pytest.mark.asyncio
async def test_corrupted_pdf_fails():
    # Valid PDF header, truncated / invalid body.
    corrupted = b"%PDF-1.4\n% corrupted garbage"

    with pytest.raises(FileValidationError) as exc_info:
        await validate_file(_upload(corrupted, "broken.pdf", "application/pdf"))

    assert exc_info.value.code == "CORRUPTED_OR_UNREADABLE_FILE"


@pytest.mark.asyncio
async def test_corrupted_png_fails():
    corrupted = b"\x89PNG\r\n\x1a\n" + b"not-a-real-png-body"

    with pytest.raises(FileValidationError) as exc_info:
        await validate_file(_upload(corrupted, "broken.png", "image/png"))

    assert exc_info.value.code == "CORRUPTED_OR_UNREADABLE_FILE"


@pytest.mark.asyncio
async def test_corrupted_jpeg_fails():
    # Valid JPEG SOI marker, invalid body.
    corrupted = b"\xff\xd8\xff\xe0" + b"not-a-real-jpeg-body"

    with pytest.raises(FileValidationError) as exc_info:
        await validate_file(_upload(corrupted, "broken.jpg", "image/jpeg"))

    assert exc_info.value.code == "CORRUPTED_OR_UNREADABLE_FILE"


@pytest.mark.asyncio
async def test_upload_pointer_is_reset_after_validation():
    content = _pdf_bytes(1)
    upload = _upload(content, "reset.pdf", "application/pdf")

    await validate_file(upload)

    # Downstream OCR must still be able to read the full file.
    assert await upload.read() == content


@pytest.mark.asyncio
async def test_upload_pointer_is_reset_after_validation_failure():
    content = b"plain text pretending to be pdf"
    upload = _upload(content, "fake.pdf", "application/pdf")

    with pytest.raises(FileValidationError):
        await validate_file(upload)

    assert await upload.read() == content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relative_path", "expected_mime", "expected_pages"),
    [
        (
            "Balance Sheet/Consolidated Balance Sheet 2017.pdf",
            "application/pdf",
            1,
        ),
        (
            "Profit & Loss/Consolidated Profit & Loss 2022.pdf",
            "application/pdf",
            1,
        ),
        (
            "Cash Flows/Consolidated Cash Flow Statement 2017.pdf",
            "application/pdf",
            2,
        ),
        (
            "Invoices/batch1-1109.jpg",
            "image/jpeg",
            1,
        ),
        (
            "Invoices/20251118_000612.jpg",
            "image/jpeg",
            1,
        ),
    ],
)
async def test_real_dataset_smoke(relative_path, expected_mime, expected_pages):
    path = DATASET_ROOT / relative_path
    assert path.is_file(), f"Missing dataset file: {path}"

    content = path.read_bytes()
    result = await validate_file(_upload(content, path.name))

    assert result["file_name"] == path.name
    assert result["file_validation"] == {
        "file_type": expected_mime,
        "is_supported": True,
        "is_readable": True,
        "page_count": expected_pages,
        "status": "PASS",
    }

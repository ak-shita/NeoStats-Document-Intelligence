"""
File validation service — input control before OCR / AI extraction.

Validates PDF / JPG / PNG uploads for type, emptiness, readability,
integrity, and page count (max 3). Does not classify document type
(invoice vs receipt, etc.).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError


SUPPORTED_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "image/jpeg",
        "image/png",
    }
)

ALLOWED_EXTENSIONS = frozenset({".pdf", ".jpg", ".jpeg", ".png"})

EXTENSION_TO_MIME = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

MAX_PAGES = 3


class FileValidationError(Exception):
    """
    Controlled validation failure for the API layer to map into:

        {"error": {"code": "...", "message": "..."}}
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _detect_mime_from_signature(content: bytes) -> str | None:
    """Identify PDF / JPEG / PNG from magic bytes (not the filename)."""

    if content.startswith(b"%PDF"):
        return "application/pdf"

    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    # JPEG SOI marker; subsequent byte is typically 0xE0–0xEF (APPn) or 0xDB, etc.
    if len(content) >= 3 and content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    return None


def _validate_pdf(content: bytes) -> int:
    """Ensure the PDF is readable and return its page count (1–3)."""

    try:
        reader = PdfReader(BytesIO(content))
    except (PdfReadError, OSError, ValueError) as exc:
        raise FileValidationError(
            code="CORRUPTED_OR_UNREADABLE_FILE",
            message="The PDF is corrupted or unreadable.",
        ) from exc

    if getattr(reader, "is_encrypted", False):
        # Encrypted PDFs cannot be processed reliably without a password.
        raise FileValidationError(
            code="CORRUPTED_OR_UNREADABLE_FILE",
            message="The PDF is encrypted or unreadable.",
        )

    try:
        page_count = len(reader.pages)
        # Touch each page to surface structural corruption early.
        for page in reader.pages:
            _ = page.mediabox
    except Exception as exc:
        raise FileValidationError(
            code="CORRUPTED_OR_UNREADABLE_FILE",
            message="The PDF is corrupted or unreadable.",
        ) from exc

    if page_count < 1:
        raise FileValidationError(
            code="CORRUPTED_OR_UNREADABLE_FILE",
            message="The PDF contains no readable pages.",
        )

    if page_count > MAX_PAGES:
        raise FileValidationError(
            code="PAGE_LIMIT_EXCEEDED",
            message="Documents must contain no more than 3 pages.",
        )

    return page_count


def _validate_image(content: bytes) -> int:
    """Ensure the image can be opened and decoded. Images count as 1 page."""

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()

        # verify() leaves the image unusable; reopen and fully load pixels.
        with Image.open(BytesIO(content)) as image:
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise FileValidationError(
            code="CORRUPTED_OR_UNREADABLE_FILE",
            message="The image is corrupted or unreadable.",
        ) from exc

    return 1


async def validate_file(file: UploadFile) -> dict:
    """
    Validate an uploaded document before OCR or AI extraction.

    Returns the assessment success structure:

        {
          "file_name": "...",
          "file_validation": {
            "file_type": "...",
            "is_supported": true,
            "is_readable": true,
            "page_count": N,
            "status": "PASS"
          }
        }

    Raises FileValidationError on controlled failures. Always resets the
    UploadFile pointer so downstream services can re-read the same stream.
    """

    filename = file.filename or "upload"
    extension = Path(filename).suffix.lower()

    try:
        content = await file.read()
    finally:
        # Downstream OCR must be able to read the same UploadFile again.
        await file.seek(0)

    if not content:
        raise FileValidationError(
            code="EMPTY_FILE",
            message="The uploaded file is empty.",
        )

    # Extension is a helpful hint, but signatures are authoritative.
    if extension and extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            code="UNSUPPORTED_FILE_TYPE",
            message="Only PDF / JPG / PNG documents are supported.",
        )

    mime_type = _detect_mime_from_signature(content)

    if mime_type is None or mime_type not in SUPPORTED_MIME_TYPES:
        raise FileValidationError(
            code="UNSUPPORTED_FILE_TYPE",
            message="Only PDF / JPG / PNG documents are supported.",
        )

    if extension in EXTENSION_TO_MIME and EXTENSION_TO_MIME[extension] != mime_type:
        raise FileValidationError(
            code="INVALID_FILE_CONTENT",
            message="File content does not match the file extension.",
        )

    if mime_type == "application/pdf":
        page_count = _validate_pdf(content)
    else:
        page_count = _validate_image(content)

    return {
        "file_name": filename,
        "file_validation": {
            "file_type": mime_type,
            "is_supported": True,
            "is_readable": True,
            "page_count": page_count,
            "status": "PASS",
        },
    }

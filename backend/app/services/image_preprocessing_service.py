"""
Image preprocessing for OCR.Space uploads only.

IMPORTANT
---------
This module exists solely because OCR.Space's free tier rejects uploads
larger than ~1 MB. The assessment does NOT define an application-level
file-size limit, so:

- The original user upload is never modified or overwritten.
- Only an in-memory OCR copy of JPEG/PNG bytes may be resized/compressed.
- PDFs are not preprocessed here; the OCR service sends validated PDFs
  as-is and surfaces provider size errors as controlled OCR failures.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


class ImagePreprocessingError(Exception):
    """Raised when an OCR-safe image copy cannot be produced."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _normalize_rgb(image: Image.Image) -> Image.Image:
    """Convert to RGB so JPEG compression is always available."""

    if image.mode in ("RGB", "L"):
        return image.convert("RGB") if image.mode != "RGB" else image

    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background

    return image.convert("RGB")


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def prepare_image_for_ocr(
    content: bytes,
    *,
    filename: str,
    max_bytes: int,
) -> tuple[bytes, str, str]:
    """
    Return an OCR upload triple ``(bytes, filename, content_type)``.

    If ``content`` is already within ``max_bytes``, the original bytes are
    returned unchanged (still a separate reference used only for OCR).

    Oversized images are iteratively downscaled and JPEG-compressed in
    memory until they fit, or until quality/dimensions cannot be reduced
    further — in which case ``ImagePreprocessingError`` is raised.
    """

    if max_bytes <= 0:
        raise ImagePreprocessingError(
            code="OCR_PREPROCESSING_FAILED",
            message="Invalid OCR provider size limit configuration.",
        )

    suffix = Path(filename).suffix.lower()
    original_content_type = (
        "image/png" if suffix == ".png" else "image/jpeg"
    )

    if len(content) <= max_bytes:
        return content, filename, original_content_type

    try:
        with Image.open(BytesIO(content)) as opened:
            image = _normalize_rgb(opened.copy())
    except Exception as exc:
        raise ImagePreprocessingError(
            code="OCR_PREPROCESSING_FAILED",
            message="Unable to open the image for OCR preprocessing.",
        ) from exc

    # Prefer a .jpg OCR copy for compression efficiency; original is untouched.
    ocr_filename = f"{Path(filename).stem}_ocr.jpg"
    qualities = (85, 75, 65, 55, 45, 35)
    scale = 1.0
    min_dimension = 640

    for _ in range(12):
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        working = image if scale == 1.0 else image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

        for quality in qualities:
            encoded = _encode_jpeg(working, quality)
            if len(encoded) <= max_bytes:
                return encoded, ocr_filename, "image/jpeg"

        # Shrink further while keeping a readable minimum dimension.
        next_scale = scale * 0.75
        next_width = int(image.width * next_scale)
        next_height = int(image.height * next_scale)
        if min(next_width, next_height) < min_dimension and scale < 1.0:
            break
        scale = next_scale

    raise ImagePreprocessingError(
        code="OCR_PREPROCESSING_FAILED",
        message=(
            "Unable to produce an OCR-ready image under the provider "
            "upload size limit while preserving readability."
        ),
    )

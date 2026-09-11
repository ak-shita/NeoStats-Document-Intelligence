"""One-shot OCR → Gemini extraction smoke test. Do not commit secrets."""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from google.genai import errors as genai_errors
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.extraction import ExtractionResult
from app.services.extraction_service import ExtractionError, extract_structured_data
from app.services.ocr_service import OCRError, extract_text

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "sample_outputs" / "extraction_smoke"

CASES = [
    (
        ROOT / "New Dataset" / "Balance Sheet" / "Consolidated Balance Sheet 2017.pdf",
        "balance_sheet",
        "Consolidated_Balance_Sheet_2017",
    ),
    (
        ROOT / "New Dataset" / "Profit & Loss" / "Consolidated Profit & Loss 2020.pdf",
        "profit_and_loss",
        "Consolidated_Profit_Loss_2020",
    ),
    (
        ROOT / "New Dataset" / "Cash Flows" / "Consolidated Cash Flow Statement 2025.pdf",
        "cash_flow",
        "Consolidated_Cash_Flow_Statement_2025",
    ),
    (
        ROOT / "New Dataset" / "Invoices" / "20251118_000612.jpg",
        "invoice",
        "20251118_000612",
    ),
    (
        ROOT / "New Dataset" / "Invoices" / "X51008142038.jpg",
        "invoice",
        "X51008142038",
    ),
]


def _safe_api_error_detail(exc: BaseException) -> str:
    """Capture Gemini API error identity without secrets."""
    parts: list[str] = [type(exc).__name__]
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    message = getattr(exc, "message", None)
    if code is not None:
        parts.append(f"code={code}")
    if status is not None:
        parts.append(f"status={status}")
    if message:
        text = str(message)
        # Strip anything that looks like a key just in case.
        if "AIza" in text:
            text = "[redacted]"
        parts.append(f"message={text[:400]}")
    else:
        text = str(exc)
        if "AIza" in text:
            text = "[redacted]"
        parts.append(f"str={text[:400]}")
    return " | ".join(parts)


async def run_one(path: Path, document_type: str, stem: str, settings: Settings) -> dict:
    report: dict = {
        "file_name": path.name,
        "relative_path": str(path.relative_to(ROOT)),
        "document_type_requested": document_type,
        "ocr_success": False,
        "gemini_success": False,
        "extraction_time_ms": None,
        "page_count_processed": None,
        "pydantic_validation_success": False,
        "error_stage": None,
        "error_code": None,
        "error_message": None,
        "api_error_detail": None,
        "output_json": None,
        "ocr_json": None,
    }

    if not path.exists():
        report["error_stage"] = "input"
        report["error_code"] = "FILE_NOT_FOUND"
        report["error_message"] = f"Missing file: {path}"
        return report

    upload = UploadFile(file=BytesIO(path.read_bytes()), filename=path.name)

    # --- OCR ---
    ocr_t0 = time.perf_counter()
    try:
        ocr_result = await extract_text(upload, settings=settings)
    except OCRError as exc:
        report["error_stage"] = "ocr"
        report["error_code"] = exc.code
        report["error_message"] = exc.message
        report["ocr_time_ms"] = int((time.perf_counter() - ocr_t0) * 1000)
        err_path = OUT_DIR / f"{stem}.error.txt"
        err_path.write_text(
            f"stage=ocr\ncode={exc.code}\nmessage={exc.message}\n",
            encoding="utf-8",
        )
        return report
    except Exception as exc:
        report["error_stage"] = "ocr"
        report["error_code"] = "OCR_UNEXPECTED"
        report["error_message"] = str(exc)[:400]
        report["ocr_time_ms"] = int((time.perf_counter() - ocr_t0) * 1000)
        return report

    ocr_ms = int((time.perf_counter() - ocr_t0) * 1000)
    report["ocr_success"] = True
    report["ocr_time_ms"] = ocr_ms
    report["page_count_processed"] = ocr_result.get("page_count")

    ocr_path = OUT_DIR / f"{stem}.ocr.json"
    ocr_path.write_text(json.dumps(ocr_result, indent=2, ensure_ascii=False), encoding="utf-8")
    report["ocr_json"] = str(ocr_path.relative_to(ROOT))

    # --- Gemini extraction ---
    gem_t0 = time.perf_counter()
    try:
        extraction = await extract_structured_data(
            ocr_result, document_type, settings=settings
        )
    except ExtractionError as exc:
        report["extraction_time_ms"] = int((time.perf_counter() - gem_t0) * 1000)
        report["error_stage"] = "extraction"
        report["error_code"] = exc.code
        report["error_message"] = exc.message
        # Surface underlying Gemini API identity once (no secrets).
        cause = exc.__cause__
        if isinstance(cause, genai_errors.APIError):
            report["api_error_detail"] = _safe_api_error_detail(cause)
        elif cause is not None:
            report["api_error_detail"] = _safe_api_error_detail(cause)
        err_path = OUT_DIR / f"{stem}.error.txt"
        detail = report["api_error_detail"] or ""
        err_path.write_text(
            f"stage=extraction\ncode={exc.code}\nmessage={exc.message}\n"
            f"api_error_detail={detail}\n",
            encoding="utf-8",
        )
        return report
    except Exception as exc:
        report["extraction_time_ms"] = int((time.perf_counter() - gem_t0) * 1000)
        report["error_stage"] = "extraction"
        report["error_code"] = "EXTRACTION_UNEXPECTED"
        report["error_message"] = str(exc)[:400]
        report["api_error_detail"] = _safe_api_error_detail(exc)
        return report

    report["extraction_time_ms"] = int((time.perf_counter() - gem_t0) * 1000)
    report["gemini_success"] = True

    # --- Pydantic re-validation of final envelope ---
    try:
        ExtractionResult.model_validate(extraction)
        report["pydantic_validation_success"] = True
    except ValidationError as exc:
        report["error_stage"] = "pydantic"
        report["error_code"] = "PYDANTIC_VALIDATION_FAILED"
        report["error_message"] = str(exc)[:500]
        err_path = OUT_DIR / f"{stem}.error.txt"
        err_path.write_text(
            f"stage=pydantic\ncode=PYDANTIC_VALIDATION_FAILED\nmessage={exc}\n",
            encoding="utf-8",
        )
        # Still save raw extraction for inspection.
        out_path = OUT_DIR / f"{stem}.extraction.json"
        out_path.write_text(json.dumps(extraction, indent=2, ensure_ascii=False), encoding="utf-8")
        report["output_json"] = str(out_path.relative_to(ROOT))
        return report

    out_path = OUT_DIR / f"{stem}.extraction.json"
    out_path.write_text(json.dumps(extraction, indent=2, ensure_ascii=False), encoding="utf-8")
    report["output_json"] = str(out_path.relative_to(ROOT))

    # Remove prior error artifact if present.
    err_path = OUT_DIR / f"{stem}.error.txt"
    if err_path.exists():
        err_path.unlink()

    return report


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    print("Gemini key loaded:", bool(settings.gemini_api_key.strip()))
    print("OCR key loaded:", bool(settings.ocr_space_api_key.strip()))
    print("Gemini model:", settings.gemini_model)
    print("---")

    summary: list[dict] = []
    for path, document_type, stem in CASES:
        print(f"Processing: {path.name} ({document_type}) ...")
        try:
            report = await run_one(path, document_type, stem, settings)
        except Exception as exc:
            report = {
                "file_name": path.name,
                "document_type_requested": document_type,
                "ocr_success": False,
                "gemini_success": False,
                "pydantic_validation_success": False,
                "error_stage": "runner",
                "error_code": "SMOKE_RUNNER_ERROR",
                "error_message": str(exc)[:400],
                "traceback": traceback.format_exc()[-800:],
            }
        summary.append(report)
        print(
            json.dumps(
                {
                    "file": report.get("file_name"),
                    "ocr_success": report.get("ocr_success"),
                    "gemini_success": report.get("gemini_success"),
                    "extraction_time_ms": report.get("extraction_time_ms"),
                    "page_count_processed": report.get("page_count_processed"),
                    "pydantic_validation_success": report.get("pydantic_validation_success"),
                    "error_code": report.get("error_code"),
                    "error_message": report.get("error_message"),
                    "api_error_detail": report.get("api_error_detail"),
                    "output_json": report.get("output_json"),
                },
                indent=2,
            )
        )
        print("---")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", summary_path)


if __name__ == "__main__":
    asyncio.run(main())

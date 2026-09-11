"""
Final assessment response schemas for document orchestration.

These models describe the public structured result returned after:
    file validation → OCR → Gemini extraction → financial validation
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ProcessingStatus = Literal["PASS", "FAILED"]
ValidationOverallStatus = Literal["PASS", "FAIL", "NOT_APPLICABLE"]
CheckStatus = Literal["PASS", "FAIL", "NOT_APPLICABLE"]


class FileValidationResult(BaseModel):
    file_type: str
    is_supported: bool
    is_readable: bool
    page_count: int
    status: Literal["PASS"]


class PublicExtractedValue(BaseModel):
    value: Any = None
    confidence: float | None = None
    page_number: int | None = None


class ValidationCheck(BaseModel):
    name: str
    formula: str
    operands: dict[str, Any] = Field(default_factory=dict)
    calculated_value: float | None = None
    reported_value: float | None = None
    variance: float | None = None
    status: CheckStatus
    reported_raw: Any = None


class ValidationIssue(BaseModel):
    name: str
    status: CheckStatus
    variance: float | None = None
    formula: str | None = None


class ValidationSection(BaseModel):
    checks: list[dict[str, Any]] = Field(default_factory=list)
    overall_status: ValidationOverallStatus
    issues: list[dict[str, Any]] = Field(default_factory=list)


class ProcessingMetadata(BaseModel):
    ocr_used: bool = True
    processed_at: str
    processing_time_ms: int
    ocr_page_count: int | None = None
    ocr_engine: int | None = None
    extraction_document_type: str | None = None


class DocumentProcessingResult(BaseModel):
    """Assessment-facing end-to-end processing response."""

    document_name: str
    document_type: str
    processing_status: ProcessingStatus
    overall_confidence: float | None = None
    file_validation: FileValidationResult
    extracted_data: dict[str, Any]
    validation: ValidationSection
    processing_metadata: ProcessingMetadata
    error: dict[str, str] | None = None

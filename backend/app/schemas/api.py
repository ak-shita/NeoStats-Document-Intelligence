"""Pydantic models for the public REST API layer."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.document import DocumentProcessingResult


class ApiDocumentType(str, Enum):
    """Caller-supplied document types accepted by POST /documents/process."""

    invoice = "invoice"
    balance_sheet = "balance_sheet"
    profit_and_loss = "profit_and_loss"
    cash_flow = "cash_flow"


class HealthResponse(BaseModel):
    status: str = Field(examples=["healthy"])
    service: str = Field(examples=["document-intelligence-api"])


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class DocumentListResponse(BaseModel):
    count: int
    documents: list[DocumentProcessingResult]

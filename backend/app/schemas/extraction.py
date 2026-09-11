"""
Pydantic schemas for Gemini structured document extraction.

Each extracted scalar / line-item value uses:

    {
      "field": "...",
      "value": ...,
      "confidence": 0.97,
      "evidence": {"source_text": "...", "page_number": 1}
    }

Missing or unreadable values must be null — never inferred.
Financial calculations are intentionally excluded from these schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    INVOICE = "Invoice"
    BALANCE_SHEET = "Balance Sheet"
    PROFIT_AND_LOSS = "Profit & Loss"
    CASH_FLOW = "Cash Flow"


class Evidence(BaseModel):
    source_text: str | None = None
    page_number: int | None = None


class ExtractedField(BaseModel):
    """Evidence-backed extracted value used for every field and line-item cell."""

    field: str
    value: Any = None
    confidence: float | None = Field(
        default=None,
        description="Model confidence that the value is supported by OCR text (0–1).",
        ge=0.0,
        le=1.0,
    )
    evidence: Evidence | None = None


class InvoiceLineItem(BaseModel):
    line_number: ExtractedField | None = None
    description: ExtractedField | None = None
    quantity: ExtractedField | None = None
    unit: ExtractedField | None = None
    unit_price: ExtractedField | None = None
    net_amount: ExtractedField | None = None
    vat_percent: ExtractedField | None = None
    vat_amount: ExtractedField | None = None
    gross_amount: ExtractedField | None = None
    hsn_sac: ExtractedField | None = None
    additional_values: list[ExtractedField] = Field(default_factory=list)


class InvoiceExtraction(BaseModel):
    document_type: Literal["Invoice"] = "Invoice"
    invoice_number: ExtractedField | None = None
    invoice_date: ExtractedField | None = None
    due_date: ExtractedField | None = None
    currency: ExtractedField | None = None
    seller_name: ExtractedField | None = None
    seller_address: ExtractedField | None = None
    seller_tax_id: ExtractedField | None = None
    buyer_name: ExtractedField | None = None
    buyer_address: ExtractedField | None = None
    buyer_tax_id: ExtractedField | None = None
    payment_details: ExtractedField | None = None
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    subtotal_net: ExtractedField | None = None
    total_vat: ExtractedField | None = None
    total_gross: ExtractedField | None = None
    amount_in_words: ExtractedField | None = None
    additional_fields: list[ExtractedField] = Field(default_factory=list)


class StatementLineItem(BaseModel):
    """One financial statement row with comparative period support."""

    description: ExtractedField | None = None
    schedule: ExtractedField | None = None
    current_period_value: ExtractedField | None = None
    comparative_period_value: ExtractedField | None = None
    additional_values: list[ExtractedField] = Field(default_factory=list)


class BalanceSheetExtraction(BaseModel):
    document_type: Literal["Balance Sheet"] = "Balance Sheet"
    entity_name: ExtractedField | None = None
    statement_title: ExtractedField | None = None
    as_of_date: ExtractedField | None = None
    currency_unit: ExtractedField | None = None
    current_period_label: ExtractedField | None = None
    comparative_period_label: ExtractedField | None = None
    capital_and_liabilities: list[StatementLineItem] = Field(default_factory=list)
    assets: list[StatementLineItem] = Field(default_factory=list)
    contingent_items: list[StatementLineItem] = Field(default_factory=list)
    additional_fields: list[ExtractedField] = Field(default_factory=list)


class ProfitAndLossExtraction(BaseModel):
    document_type: Literal["Profit & Loss"] = "Profit & Loss"
    entity_name: ExtractedField | None = None
    statement_title: ExtractedField | None = None
    period_ended: ExtractedField | None = None
    currency_unit: ExtractedField | None = None
    current_period_label: ExtractedField | None = None
    comparative_period_label: ExtractedField | None = None
    income: list[StatementLineItem] = Field(default_factory=list)
    expenditure: list[StatementLineItem] = Field(default_factory=list)
    profit: list[StatementLineItem] = Field(default_factory=list)
    appropriations: list[StatementLineItem] = Field(default_factory=list)
    earnings_per_share: list[StatementLineItem] = Field(default_factory=list)
    additional_fields: list[ExtractedField] = Field(default_factory=list)


class CashFlowExtraction(BaseModel):
    document_type: Literal["Cash Flow"] = "Cash Flow"
    entity_name: ExtractedField | None = None
    statement_title: ExtractedField | None = None
    period_ended: ExtractedField | None = None
    currency_unit: ExtractedField | None = None
    current_period_label: ExtractedField | None = None
    comparative_period_label: ExtractedField | None = None
    operating_activities: list[StatementLineItem] = Field(default_factory=list)
    investing_activities: list[StatementLineItem] = Field(default_factory=list)
    financing_activities: list[StatementLineItem] = Field(default_factory=list)
    net_change_and_cash_balances: list[StatementLineItem] = Field(default_factory=list)
    additional_fields: list[ExtractedField] = Field(default_factory=list)


ExtractionDocument = (
    InvoiceExtraction
    | BalanceSheetExtraction
    | ProfitAndLossExtraction
    | CashFlowExtraction
)


class ExtractionResult(BaseModel):
    """Internal extraction envelope (not the final public API response)."""

    file_name: str | None = None
    document_type: DocumentType
    document: ExtractionDocument

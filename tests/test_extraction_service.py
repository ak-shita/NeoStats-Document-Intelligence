"""Mocked unit tests for Gemini structured extraction (no live API calls)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors as genai_errors

from app.core.config import Settings
from app.services.extraction_service import (
    ExtractionError,
    extract_structured_data,
    normalize_document_type,
)
from app.schemas.extraction import DocumentType


def _settings(**overrides: Any) -> Settings:
    base = {
        "gemini_api_key": "test-gemini-key",
        "gemini_model": "gemini-2.0-flash",
        "gemini_timeout_seconds": 30.0,
    }
    base.update(overrides)
    return Settings(**base)


def _ocr(*, file_name: str, pages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "success": True,
        "page_count": len(pages),
        "pages": pages,
        "ocr_engine": 2,
    }


def _field(name: str, value: Any, confidence: float = 0.95, page: int = 1, source: str | None = None) -> dict:
    return {
        "field": name,
        "value": value,
        "confidence": confidence,
        "evidence": {
            "source_text": source if source is not None else (None if value is None else f"{name}: {value}"),
            "page_number": None if value is None else page,
        },
    }


def _fake_client(response_text: str | None = None, *, side_effect: Exception | None = None) -> MagicMock:
    mock_response = SimpleNamespace(text=response_text)
    generate = AsyncMock(return_value=mock_response)
    if side_effect is not None:
        generate = AsyncMock(side_effect=side_effect)

    models = MagicMock()
    models.generate_content = generate

    aio = MagicMock()
    aio.models = models
    aio.aclose = AsyncMock()

    client = MagicMock()
    client.aio = aio
    return client


@pytest.mark.asyncio
async def test_extract_invoice_with_evidence_and_confidence():
    payload = {
        "document_type": "Invoice",
        "invoice_number": _field("invoice_number", "94404257", 0.99, 1, "Invoice no: 94404257"),
        "invoice_date": _field("invoice_date", "07/03/2013", 0.96, 1, "Date of issue: 07/03/2013"),
        "due_date": _field("due_date", None, None, 1, None),
        "currency": _field("currency", "USD", 0.8, 1, "Total $ 138,90"),
        "seller_name": _field("seller_name", "Cruz PLC", 0.94, 1, "Seller: Cruz PLC"),
        "seller_address": None,
        "seller_tax_id": _field("seller_tax_id", "964-99-8203", 0.93, 1, "Tax Id: 964-99-8203"),
        "buyer_name": _field("buyer_name", "Sandoval-Phillips", 0.9, 1, "Client: Sandoval-Phillips"),
        "buyer_address": None,
        "buyer_tax_id": _field("buyer_tax_id", "976-94-5245", 0.9, 1, "Tax Id: 976-94-5245"),
        "payment_details": _field("payment_details", "GB31XOVK34958853133204", 0.88, 1, "IBAN: GB31XOVK34958853133204"),
        "line_items": [
            {
                "line_number": _field("line_number", "1", 0.9, 1, "1."),
                "description": _field("description", "B0028NA06C Shoeless Joe", 0.91, 1, "B0028NA06C Shoeless Joe"),
                "quantity": _field("quantity", "5,00", 0.92, 1, "5,00"),
                "unit": _field("unit", "each", 0.9, 1, "each"),
                "unit_price": _field("unit_price", "3,49", 0.9, 1, "3,49"),
                "net_amount": _field("net_amount", "17,45", 0.9, 1, "17,45"),
                "vat_percent": _field("vat_percent", "10%", 0.9, 1, "10%"),
                "vat_amount": None,
                "gross_amount": _field("gross_amount", "19,20", 0.9, 1, "19,20"),
                "hsn_sac": None,
                "additional_values": [],
            }
        ],
        "subtotal_net": _field("subtotal_net", "126,27", 0.95, 1, "Total $ 126,27"),
        "total_vat": _field("total_vat", "12,63", 0.95, 1, "$ 12,63"),
        "total_gross": _field("total_gross", "138,90", 0.97, 1, "$ 138,90"),
        "amount_in_words": None,
        "additional_fields": [],
    }
    import json

    client = _fake_client(json.dumps(payload))
    result = await extract_structured_data(
        _ocr(
            file_name="batch1-1109.jpg",
            pages=[{"page_number": 1, "text": "Invoice no: 94404257\nDate of issue: 07/03/2013\nTotal $ 138,90"}],
        ),
        "Invoice",
        settings=_settings(),
        client=client,
    )

    assert result["document_type"] == "Invoice"
    assert result["file_name"] == "batch1-1109.jpg"
    invoice = result["document"]
    assert invoice["invoice_number"]["value"] == "94404257"
    assert invoice["invoice_number"]["confidence"] == 0.99
    assert invoice["invoice_number"]["evidence"]["page_number"] == 1
    assert "94404257" in invoice["invoice_number"]["evidence"]["source_text"]
    assert invoice["due_date"]["value"] is None
    assert invoice["line_items"][0]["gross_amount"]["value"] == "19,20"
    client.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_balance_sheet_comparative_periods():
    import json

    payload = {
        "document_type": "Balance Sheet",
        "entity_name": _field("entity_name", "HDFC Bank Limited", 0.85, 1, "HDFC Bank Limited"),
        "statement_title": _field("statement_title", "Consolidated Balance Sheet", 0.99, 1, "Consolidated Balance Sheet"),
        "as_of_date": _field("as_of_date", "March 31, 2017", 0.97, 1, "As at March 31, 2017"),
        "currency_unit": _field("currency_unit", "in '000", 0.9, 1, "& in '000"),
        "current_period_label": _field("current_period_label", "31-Mar-17", 0.98, 1, "31-Mar-17"),
        "comparative_period_label": _field("comparative_period_label", "31-Mar-16", 0.98, 1, "31-Mar-16"),
        "capital_and_liabilities": [
            {
                "description": _field("description", "Capital", 0.99, 1, "Capital"),
                "schedule": _field("schedule", "1", 0.95, 1, "1"),
                "current_period_value": _field("current_period_value", "5,125,091", 0.97, 1, "5,125,091"),
                "comparative_period_value": _field("comparative_period_value", "5,056,373", 0.97, 1, "5,056,373"),
                "additional_values": [],
            }
        ],
        "assets": [],
        "contingent_items": [],
        "additional_fields": [],
    }
    client = _fake_client(json.dumps(payload))
    result = await extract_structured_data(
        _ocr(
            file_name="Consolidated Balance Sheet 2017.pdf",
            pages=[{"page_number": 1, "text": "Consolidated Balance Sheet\nCapital 1 5,125,091 5,056,373"}],
        ),
        "Balance Sheet",
        settings=_settings(),
        client=client,
    )

    doc = result["document"]
    assert result["document_type"] == "Balance Sheet"
    row = doc["capital_and_liabilities"][0]
    assert row["current_period_value"]["value"] == "5,125,091"
    assert row["comparative_period_value"]["value"] == "5,056,373"
    assert row["current_period_value"]["evidence"]["page_number"] == 1


@pytest.mark.asyncio
async def test_extract_profit_and_loss_bracketed_and_nulls():
    import json

    payload = {
        "document_type": "Profit & Loss",
        "entity_name": None,
        "statement_title": _field("statement_title", "Consolidated Profit and Loss Account", 0.98, 1),
        "period_ended": _field("period_ended", "March 31, 2022", 0.96, 1),
        "currency_unit": _field("currency_unit", "in crore", 0.9, 1),
        "current_period_label": _field("current_period_label", "March 31, 2022", 0.95, 1),
        "comparative_period_label": _field("comparative_period_label", "March 31, 2021", 0.95, 1),
        "income": [
            {
                "description": _field("description", "Interest earned", 0.98, 1),
                "schedule": None,
                "current_period_value": _field("current_period_value", "135,936.41", 0.97, 1),
                "comparative_period_value": _field("comparative_period_value", "128,552.40", 0.97, 1),
                "additional_values": [],
            }
        ],
        "expenditure": [],
        "profit": [],
        "appropriations": [
            {
                "description": _field(
                    "description",
                    "Transfer to / (from) Minority Interest (opening adjustment)",
                    0.9,
                    1,
                ),
                "schedule": None,
                "current_period_value": _field("current_period_value", "(48.34)", 0.94, 1, "(48.34)"),
                "comparative_period_value": _field("comparative_period_value", None),
                "additional_values": [],
            }
        ],
        "earnings_per_share": [],
        "additional_fields": [],
    }
    client = _fake_client(json.dumps(payload))
    result = await extract_structured_data(
        _ocr(
            file_name="pnl.pdf",
            pages=[{"page_number": 1, "text": "Interest earned 135,936.41 128,552.40\n(48.34)"}],
        ),
        "Profit & Loss",
        settings=_settings(),
        client=client,
    )

    appropriation = result["document"]["appropriations"][0]
    assert appropriation["current_period_value"]["value"] == "(48.34)"
    assert appropriation["comparative_period_value"]["value"] is None
    assert result["document"]["entity_name"] is None


@pytest.mark.asyncio
async def test_extract_cash_flow_multi_page_evidence():
    import json

    payload = {
        "document_type": "Cash Flow",
        "entity_name": _field("entity_name", "HDFC Bank Limited", 0.8, 1),
        "statement_title": _field("statement_title", "Consolidated Cash Flow Statement", 0.99, 1),
        "period_ended": _field("period_ended", "March 31, 2017", 0.95, 1),
        "currency_unit": _field("currency_unit", "in 000", 0.85, 1),
        "current_period_label": _field("current_period_label", "31-Mar-17", 0.97, 1),
        "comparative_period_label": _field("comparative_period_label", "31-Mar-16", 0.97, 1),
        "operating_activities": [
            {
                "description": _field("description", "Consolidated profit before income tax", 0.98, 1),
                "schedule": None,
                "current_period_value": _field("current_period_value", "233,311,478", 0.96, 1),
                "comparative_period_value": _field("comparative_period_value", "194,949,948", 0.96, 1),
                "additional_values": [],
            },
            {
                "description": _field(
                    "description",
                    "(Profit) / loss on revaluation of investments",
                    0.93,
                    1,
                    "(Profit) / loss on revaluation of investments (87,543)",
                ),
                "schedule": None,
                "current_period_value": _field("current_period_value", "(87,543)", 0.95, 1, "(87,543)"),
                "comparative_period_value": _field("comparative_period_value", "173,689", 0.95, 1),
                "additional_values": [],
            },
        ],
        "investing_activities": [],
        "financing_activities": [
            {
                "description": _field("description", "Increase in minority interest", 0.94, 2),
                "schedule": None,
                "current_period_value": _field("current_period_value", "818,605", 0.93, 2, "818,605"),
                "comparative_period_value": _field("comparative_period_value", "189,954", 0.93, 2),
                "additional_values": [],
            }
        ],
        "net_change_and_cash_balances": [],
        "additional_fields": [],
    }
    client = _fake_client(json.dumps(payload))
    result = await extract_structured_data(
        _ocr(
            file_name="cashflow.pdf",
            pages=[
                {"page_number": 1, "text": "Consolidated profit before income tax 233,311,478"},
                {"page_number": 2, "text": "Increase in minority interest 818,605 189,954"},
            ],
        ),
        "Cash Flow",
        settings=_settings(),
        client=client,
    )

    financing = result["document"]["financing_activities"][0]
    assert financing["current_period_value"]["evidence"]["page_number"] == 2
    assert result["document"]["operating_activities"][1]["current_period_value"]["value"] == "(87,543)"


@pytest.mark.asyncio
async def test_missing_gemini_api_key():
    with pytest.raises(ExtractionError) as exc_info:
        await extract_structured_data(
            _ocr(file_name="x.pdf", pages=[{"page_number": 1, "text": "hello"}]),
            "Invoice",
            settings=_settings(gemini_api_key=""),
        )
    assert exc_info.value.code == "MISSING_GEMINI_API_KEY"


@pytest.mark.asyncio
async def test_empty_gemini_response():
    client = _fake_client("   ")
    with pytest.raises(ExtractionError) as exc_info:
        await extract_structured_data(
            _ocr(file_name="x.pdf", pages=[{"page_number": 1, "text": "Invoice no: 1"}]),
            "Invoice",
            settings=_settings(),
            client=client,
        )
    assert exc_info.value.code == "EXTRACTION_EMPTY_RESPONSE"


@pytest.mark.asyncio
async def test_malformed_gemini_response():
    client = _fake_client('{"document_type": "Invoice", "invoice_number": "not-an-object"}')
    with pytest.raises(ExtractionError) as exc_info:
        await extract_structured_data(
            _ocr(file_name="x.pdf", pages=[{"page_number": 1, "text": "Invoice no: 1"}]),
            "Invoice",
            settings=_settings(),
            client=client,
        )
    assert exc_info.value.code == "EXTRACTION_MALFORMED_RESPONSE"


@pytest.mark.asyncio
async def test_model_api_error():
    side_effect = genai_errors.APIError(500, {"error": {"message": "internal"}})
    client = _fake_client(side_effect=side_effect)
    with pytest.raises(ExtractionError) as exc_info:
        await extract_structured_data(
            _ocr(file_name="x.pdf", pages=[{"page_number": 1, "text": "Invoice no: 1"}]),
            "Invoice",
            settings=_settings(),
            client=client,
        )
    assert exc_info.value.code == "EXTRACTION_MODEL_ERROR"


@pytest.mark.asyncio
async def test_timeout_maps_to_controlled_error():
    client = _fake_client(side_effect=TimeoutError("deadline exceeded"))
    with pytest.raises(ExtractionError) as exc_info:
        await extract_structured_data(
            _ocr(file_name="x.pdf", pages=[{"page_number": 1, "text": "Invoice no: 1"}]),
            "Invoice",
            settings=_settings(),
            client=client,
        )
    assert exc_info.value.code == "EXTRACTION_TIMEOUT"


def test_normalize_document_type_aliases():
    assert normalize_document_type("Invoice") is DocumentType.INVOICE
    assert normalize_document_type("profit and loss") is DocumentType.PROFIT_AND_LOSS
    with pytest.raises(ExtractionError) as exc_info:
        normalize_document_type("Receipt")
    assert exc_info.value.code == "UNSUPPORTED_DOCUMENT_TYPE"

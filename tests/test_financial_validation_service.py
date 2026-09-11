"""Unit tests for deterministic financial validation (no LLM / live APIs)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from app.services.financial_validation_service import (
    DEFAULT_TOLERANCE,
    FinancialValidationError,
    parse_amount,
    validate_financials,
)
from app.schemas.extraction import BalanceSheetExtraction


BALANCE_SHEET_2017_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "sample_outputs"
    / "extraction_smoke"
    / "Consolidated_Balance_Sheet_2017.extraction.json"
)


def _field(name: str, value: Any) -> dict[str, Any]:
    return {
        "field": name,
        "value": value,
        "confidence": 0.9 if value is not None else None,
        "evidence": {
            "source_text": None if value is None else str(value),
            "page_number": None if value is None else 1,
        },
    }


def _row(
    description: str,
    current: Any,
    comparative: Any = None,
    *,
    schedule: Any = None,
) -> dict[str, Any]:
    return {
        "description": _field("description", description),
        "schedule": None if schedule is None else _field("schedule", schedule),
        "current_period_value": _field("current_period_value", current),
        "comparative_period_value": _field("comparative_period_value", comparative),
        "additional_values": [],
    }


def _by_name(result: dict[str, Any], name: str) -> dict[str, Any]:
    for item in result["check"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"check not found: {name}")


def _names(result: dict[str, Any]) -> list[str]:
    return [c["name"] for c in result["check"]]


# ---------------------------------------------------------------------------
# parse_amount
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,234.56", 1234.56),
        ("5,125,091", 5_125_091.0),
        ("(87,543)", -87_543.0),
        ("(48.34)", -48.34),
        ("-48.34", -48.34),
        ("138,90", 138.90),
        ("5,00", 5.0),
        ("3,49", 3.49),
        ("$ 12,63", 12.63),
        (100, 100.0),
        (100.5, 100.5),
        (None, None),
        ("", None),
        ("n/a", None),
        ("—", None),
    ],
)
def test_parse_amount_formats(raw: Any, expected: float | None):
    got = parse_amount(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------


def test_invoice_pass_all_core_checks():
    extraction = {
        "file_name": "inv.jpg",
        "document_type": "Invoice",
        "document": {
            "document_type": "Invoice",
            "line_items": [
                {
                    "line_number": _field("line_number", "1"),
                    "description": _field("description", "Item A"),
                    "quantity": _field("quantity", "5,00"),
                    "unit_price": _field("unit_price", "3,49"),
                    "net_amount": _field("net_amount", "17,45"),
                    "gross_amount": None,
                    "additional_values": [],
                },
                {
                    "line_number": _field("line_number", "2"),
                    "description": _field("description", "Item B"),
                    "quantity": _field("quantity", "2"),
                    "unit_price": _field("unit_price", "10.00"),
                    "net_amount": _field("net_amount", "20.00"),
                    "gross_amount": None,
                    "additional_values": [],
                },
            ],
            "subtotal_net": _field("subtotal_net", "37,45"),
            "total_vat": _field("total_vat", "3,75"),
            "total_gross": _field("total_gross", "41,20"),
            "additional_fields": [
                _field("cash_paid", "50,00"),
                _field("change", "8,80"),
            ],
        },
    }
    result = validate_financials(extraction, tolerance=0.01)
    assert result["document_type"] == "Invoice"
    assert "check" in result
    assert _by_name(result, "invoice_line_1_quantity_times_unit_price")["status"] == "PASS"
    assert _by_name(result, "invoice_line_2_quantity_times_unit_price")["status"] == "PASS"
    assert _by_name(result, "invoice_sum_line_totals_vs_subtotal")["status"] == "PASS"
    assert _by_name(result, "invoice_taxable_plus_tax_vs_total")["status"] == "PASS"
    assert _by_name(result, "invoice_cash_paid_minus_total_vs_change")["status"] == "PASS"
    # Original reported values preserved in operands
    line1 = _by_name(result, "invoice_line_1_quantity_times_unit_price")
    assert line1["operands"]["quantity"]["raw"] == "5,00"
    assert line1["reported_raw"] == "17,45"


def test_invoice_fail_quantity_times_price():
    extraction = {
        "file_name": "bad.jpg",
        "document_type": "Invoice",
        "document": {
            "document_type": "Invoice",
            "line_items": [
                {
                    "quantity": _field("quantity", "2"),
                    "unit_price": _field("unit_price", "10"),
                    "net_amount": _field("net_amount", "25"),  # should be 20
                    "additional_values": [],
                }
            ],
            "subtotal_net": _field("subtotal_net", "25"),
            "total_vat": _field("total_vat", "0"),
            "total_gross": _field("total_gross", "25"),
            "additional_fields": [],
        },
    }
    result = validate_financials(extraction)
    check = _by_name(result, "invoice_line_1_quantity_times_unit_price")
    assert check["status"] == "FAIL"
    assert check["calculated_value"] == pytest.approx(20.0)
    assert check["reported_value"] == pytest.approx(25.0)
    assert check["variance"] == pytest.approx(-5.0)


def test_invoice_not_applicable_missing_required_and_optional_change():
    extraction = {
        "file_name": "partial.jpg",
        "document_type": "invoice",
        "document": {
            "document_type": "Invoice",
            "line_items": [
                {
                    "quantity": _field("quantity", "2"),
                    "unit_price": _field("unit_price", None),  # missing
                    "net_amount": _field("net_amount", "20"),
                    "additional_values": [],
                }
            ],
            "subtotal_net": None,
            "total_vat": _field("total_vat", "2"),
            "total_gross": _field("total_gross", "22"),
            "additional_fields": [],  # no cash/change
        },
    }
    result = validate_financials(extraction)
    assert _by_name(result, "invoice_line_1_quantity_times_unit_price")["status"] == "NOT_APPLICABLE"
    assert _by_name(result, "invoice_sum_line_totals_vs_subtotal")["status"] == "NOT_APPLICABLE"
    assert _by_name(result, "invoice_taxable_plus_tax_vs_total")["status"] == "NOT_APPLICABLE"
    assert _by_name(result, "invoice_cash_paid_minus_total_vs_change")["status"] == "NOT_APPLICABLE"
    # Never invent zeros
    line = _by_name(result, "invoice_line_1_quantity_times_unit_price")
    assert line["calculated_value"] is None
    assert line["operands"]["unit_price"]["parsed"] is None


def test_invoice_tolerance_boundary():
    extraction = {
        "file_name": "tol.jpg",
        "document_type": "Invoice",
        "document": {
            "document_type": "Invoice",
            "line_items": [
                {
                    "quantity": _field("quantity", "3"),
                    "unit_price": _field("unit_price", "1.005"),
                    "net_amount": _field("net_amount", "3.02"),  # 3.015 vs 3.02
                    "additional_values": [],
                }
            ],
            "subtotal_net": _field("subtotal_net", "3.02"),
            "total_vat": _field("total_vat", "0"),
            "total_gross": _field("total_gross", "3.02"),
            "additional_fields": [],
        },
    }
    strict = validate_financials(extraction, tolerance=0.001)
    assert _by_name(strict, "invoice_line_1_quantity_times_unit_price")["status"] == "FAIL"

    loose = validate_financials(extraction, tolerance=0.01)
    assert _by_name(loose, "invoice_line_1_quantity_times_unit_price")["status"] == "PASS"
    assert loose["tolerance"] == 0.01


# ---------------------------------------------------------------------------
# Balance sheet
# ---------------------------------------------------------------------------


def test_balance_sheet_pass_both_periods():
    extraction = {
        "file_name": "bs.pdf",
        "document_type": "Balance Sheet",
        "document": {
            "document_type": "Balance Sheet",
            "capital_and_liabilities": [
                _row("Capital", "100", "90"),
                _row("Reserves and Surplus", "50", "40"),
                _row("Deposits", "50", "70"),
                _row("Total Capital and Liabilities", "200", "200"),
            ],
            "assets": [
                _row("Cash", "80", "100"),
                _row("Investments", "120", "100"),
                _row("Total Assets", "200", "200"),
            ],
            "contingent_items": [],
            "additional_fields": [],
        },
    }
    result = validate_financials(extraction)
    assert _by_name(result, "balance_sheet_current_period_capital_liabilities_vs_assets")["status"] == "PASS"
    assert _by_name(result, "balance_sheet_comparative_period_capital_liabilities_vs_assets")["status"] == "PASS"
    assert _by_name(result, "balance_sheet_current_period_sum_capital_liability_components")["status"] == "PASS"
    assert _by_name(result, "balance_sheet_current_period_sum_asset_components")["status"] == "PASS"
    assert _by_name(result, "balance_sheet_comparative_period_sum_asset_components")["status"] == "PASS"


def test_balance_sheet_fail_and_bracketed_negative_components():
    extraction = {
        "file_name": "bs_fail.pdf",
        "document_type": "Balance Sheet",
        "document": {
            "document_type": "Balance Sheet",
            "capital_and_liabilities": [
                _row("Capital", "100", "100"),
                _row("Reserves", "(10)", "(10)"),  # -10
                _row("Total Capital and Liabilities", "90", "90"),
            ],
            "assets": [
                _row("Cash", "50", "50"),
                _row("Investments", "50", "50"),
                _row("Total Assets", "100", "100"),  # CL 90 != Assets 100
            ],
        },
    }
    result = validate_financials(extraction)
    eq = _by_name(result, "balance_sheet_current_period_capital_liabilities_vs_assets")
    assert eq["status"] == "FAIL"
    assert eq["calculated_value"] == pytest.approx(90.0)
    assert eq["reported_value"] == pytest.approx(100.0)

    cl_sum = _by_name(result, "balance_sheet_current_period_sum_capital_liability_components")
    assert cl_sum["status"] == "PASS"
    assert cl_sum["calculated_value"] == pytest.approx(90.0)
    # Bracketed raw preserved
    assert any(
        c["amount"]["raw"] == "(10)" for c in cl_sum["operands"]["components"]
    )


def test_balance_sheet_not_applicable_insufficient_components_and_missing_total():
    extraction = {
        "file_name": "bs_na.pdf",
        "document_type": "balance_sheet",
        "document": {
            "document_type": "Balance Sheet",
            "capital_and_liabilities": [
                _row("Capital", "100", None),  # only one component + missing comparative
                # no total row
            ],
            "assets": [
                _row("Cash", None, "10"),  # current missing
                _row("Total Assets", None, "10"),
            ],
        },
    }
    result = validate_financials(extraction)
    assert (
        _by_name(result, "balance_sheet_current_period_capital_liabilities_vs_assets")["status"]
        == "NOT_APPLICABLE"
    )
    assert (
        _by_name(result, "balance_sheet_current_period_sum_capital_liability_components")["status"]
        == "NOT_APPLICABLE"
    )
    assert (
        _by_name(result, "balance_sheet_comparative_period_capital_liabilities_vs_assets")["status"]
        == "NOT_APPLICABLE"
    )


def test_saved_balance_sheet_fixture_has_six_populated_pass_checks():
    """Known-good extraction fixture validates without any OCR/Gemini call."""
    extraction = json.loads(BALANCE_SHEET_2017_FIXTURE.read_text(encoding="utf-8"))

    result = validate_financials(extraction)

    assert len(result["check"]) == 6
    assert all(check["status"] == "PASS" for check in result["check"])
    assert all(check["calculated_value"] is not None for check in result["check"])
    assert all(check["reported_value"] is not None for check in result["check"])
    assert all(check["variance"] == 0.0 for check in result["check"])


def test_balance_sheet_generic_total_rows_match_live_extraction_shape():
    """Regression for live Gemini output that abbreviates both section totals."""
    extraction = json.loads(BALANCE_SHEET_2017_FIXTURE.read_text(encoding="utf-8"))
    live_document = copy.deepcopy(extraction["document"])
    live_document["capital_and_liabilities"][-1]["description"]["value"] = "Total"
    live_document["assets"][-1]["description"]["value"] = "Total"

    # This is the same Pydantic model_dump boundary used after Gemini output.
    normalized_document = BalanceSheetExtraction.model_validate(live_document).model_dump(
        mode="json"
    )
    result = validate_financials(
        {
            "file_name": extraction["file_name"],
            "document_type": "Balance Sheet",
            "document": normalized_document,
        }
    )

    assert len(result["check"]) == 6
    assert all(check["status"] == "PASS" for check in result["check"])
    assert all(check["calculated_value"] is not None for check in result["check"])
    assert all(check["reported_value"] is not None for check in result["check"])
    assert all(check["variance"] == 0.0 for check in result["check"])


# ---------------------------------------------------------------------------
# Profit & Loss
# ---------------------------------------------------------------------------


def test_pnl_pass_with_appropriations_and_comparative():
    extraction = {
        "file_name": "pnl.pdf",
        "document_type": "Profit & Loss",
        "document": {
            "document_type": "Profit & Loss",
            "income": [
                _row("Interest earned", "100.00", "80.00"),
                _row("Other income", "20.00", "10.00"),
                _row("Total income", "120.00", "90.00"),
            ],
            "expenditure": [
                _row("Interest expended", "30.00", "20.00"),
                _row("Operating expenses", "40.00", "30.00"),
                _row("Provisions and contingencies", "10.00", "5.00"),
                _row("Total expenditure", "80.00", "55.00"),
            ],
            "profit": [
                _row("Consolidated net profit before minority interest", "40.00", "35.00"),
                _row("Minority interest", "5.00", "2.00"),
                _row("Net profit attributable to Group", "35.00", "33.00"),
            ],
            "appropriations": [
                _row("Profit for the year", "35.00", "33.00"),
                _row("Balance brought forward", "15.00", "10.00"),
                _row("Total available for appropriation", "50.00", "43.00"),
            ],
            "earnings_per_share": [],
            "additional_fields": [],
        },
    }
    result = validate_financials(extraction)
    for period in ("current_period", "comparative_period"):
        assert _by_name(result, f"pnl_{period}_interest_plus_other_income")["status"] == "PASS"
        assert _by_name(result, f"pnl_{period}_expenditure_components")["status"] == "PASS"
        assert _by_name(result, f"pnl_{period}_income_minus_expenditure")["status"] == "PASS"
        assert _by_name(result, f"pnl_{period}_profit_after_minority")["status"] == "PASS"
        assert _by_name(result, f"pnl_{period}_appropriations_available")["status"] == "PASS"


def test_pnl_fail_and_bracketed_minority():
    extraction = {
        "file_name": "pnl_fail.pdf",
        "document_type": "Profit & Loss",
        "document": {
            "document_type": "Profit & Loss",
            "income": [
                _row("Interest earned", "100", "100"),
                _row("Other income", "20", "20"),
                _row("Total income", "130", "120"),  # current wrong (should be 120)
            ],
            "expenditure": [
                _row("Interest expended", "30", "30"),
                _row("Operating expenses", "40", "40"),
                _row("Provisions & Contingencies", "10", "10"),
                _row("Total expenditure", "80", "80"),
            ],
            "profit": [
                _row("Profit before minority interest", "40", "40"),
                _row("Minority interest", "(5.00)", "(5.00)"),  # -5
                _row("Net profit attributable to Group", "45", "45"),  # 40 - (-5) = 45
            ],
            "appropriations": [],
        },
    }
    result = validate_financials(extraction)
    income_check = _by_name(result, "pnl_current_period_interest_plus_other_income")
    assert income_check["status"] == "FAIL"
    assert income_check["calculated_value"] == pytest.approx(120.0)
    assert income_check["reported_value"] == pytest.approx(130.0)

    minority = _by_name(result, "pnl_current_period_profit_after_minority")
    assert minority["status"] == "PASS"
    assert minority["operands"]["minority_interest"]["raw"] == "(5.00)"
    assert minority["operands"]["minority_interest"]["parsed"] == pytest.approx(-5.0)

    # No appropriations section → NOT_APPLICABLE
    assert _by_name(result, "pnl_current_period_appropriations_available")["status"] == "NOT_APPLICABLE"


def test_pnl_not_applicable_missing_fields():
    extraction = {
        "file_name": "pnl_na.pdf",
        "document_type": "profit_and_loss",
        "document": {
            "document_type": "Profit & Loss",
            "income": [
                _row("Interest earned", "100", None),
                # other income missing entirely
                _row("Total income", "100", None),
            ],
            "expenditure": [],
            "profit": [],
            "appropriations": [
                _row("Profit for the year", "10", "10"),
                # brought forward missing
                _row("Total available for appropriation", "10", "10"),
            ],
        },
    }
    result = validate_financials(extraction)
    assert _by_name(result, "pnl_current_period_interest_plus_other_income")["status"] == "NOT_APPLICABLE"
    assert _by_name(result, "pnl_comparative_period_interest_plus_other_income")["status"] == "NOT_APPLICABLE"
    assert _by_name(result, "pnl_current_period_expenditure_components")["status"] == "NOT_APPLICABLE"
    assert _by_name(result, "pnl_current_period_appropriations_available")["status"] == "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Cash flow
# ---------------------------------------------------------------------------


def test_cash_flow_pass_both_periods_with_brackets():
    extraction = {
        "file_name": "cf.pdf",
        "document_type": "Cash Flow",
        "document": {
            "document_type": "Cash Flow",
            "operating_activities": [
                _row("Consolidated profit before income tax", "50", "40"),
                _row("Net cash from operating activities", "30", "20"),
            ],
            "investing_activities": [
                _row("Purchase of investments", "(10)", "(5)"),
                _row("Net cash used in investing activities", "(10)", "(5)"),
            ],
            "financing_activities": [
                _row("Increase in minority interest", "5", "2"),
                _row("Net cash from financing activities", "5", "2"),
            ],
            "net_change_and_cash_balances": [
                _row("Effect of exchange rate changes", "1", "0"),
                _row("Net increase in cash and cash equivalents", "26", "17"),
                _row("Cash and cash equivalents at the beginning of the year", "100", "83"),
                _row("Cash and cash equivalents at the end of the year", "126", "100"),
            ],
        },
    }
    result = validate_financials(extraction)
    cur_net = _by_name(result, "cash_flow_current_period_activities_vs_net_increase")
    assert cur_net["status"] == "PASS"
    assert cur_net["calculated_value"] == pytest.approx(26.0)  # 30 - 10 + 5 + 1
    assert cur_net["operands"]["investing"]["raw"] == "(10)"
    assert cur_net["operands"]["investing"]["parsed"] == pytest.approx(-10.0)

    cur_bridge = _by_name(result, "cash_flow_current_period_opening_plus_net_vs_closing")
    assert cur_bridge["status"] == "PASS"
    assert cur_bridge["calculated_value"] == pytest.approx(126.0)

    assert _by_name(result, "cash_flow_comparative_period_activities_vs_net_increase")["status"] == "PASS"
    assert _by_name(result, "cash_flow_comparative_period_opening_plus_net_vs_closing")["status"] == "PASS"


def test_cash_flow_fail_and_not_applicable_missing():
    extraction = {
        "file_name": "cf_bad.pdf",
        "document_type": "Cash Flow",
        "document": {
            "document_type": "Cash Flow",
            "operating_activities": [
                _row("Net cash from operating activities", "10", None),
            ],
            "investing_activities": [
                _row("Net cash from investing activities", "5", None),
            ],
            "financing_activities": [
                _row("Net cash from financing activities", "5", None),
            ],
            "net_change_and_cash_balances": [
                # No FX line → O+I+F ≈ net increase
                _row("Net increase in cash and cash equivalents", "30", None),  # should be 20 → FAIL
                _row("Cash and cash equivalents at the beginning of the year", None, "10"),
                _row("Cash and cash equivalents at the end of the year", "50", "10"),
            ],
        },
    }
    result = validate_financials(extraction)
    cur = _by_name(result, "cash_flow_current_period_activities_vs_net_increase")
    assert cur["status"] == "FAIL"
    assert cur["calculated_value"] == pytest.approx(20.0)
    assert cur["reported_value"] == pytest.approx(30.0)

    # Opening missing for current → bridge NOT_APPLICABLE
    assert (
        _by_name(result, "cash_flow_current_period_opening_plus_net_vs_closing")["status"]
        == "NOT_APPLICABLE"
    )
    # Comparative period missing activity values → NOT_APPLICABLE
    assert (
        _by_name(result, "cash_flow_comparative_period_activities_vs_net_increase")["status"]
        == "NOT_APPLICABLE"
    )


def test_cash_flow_fx_row_present_but_null_is_not_applicable():
    extraction = {
        "file_name": "cf_fx_null.pdf",
        "document_type": "cash_flow",
        "document": {
            "document_type": "Cash Flow",
            "operating_activities": [_row("Net cash from operating activities", "10", "10")],
            "investing_activities": [_row("Net cash from investing activities", "0", "0")],
            "financing_activities": [_row("Net cash from financing activities", "0", "0")],
            "net_change_and_cash_balances": [
                _row("Translation adjustment", None, None),  # present but missing
                _row("Net increase in cash and cash equivalents", "10", "10"),
                _row("Opening cash and cash equivalents", "1", "1"),
                _row("Closing cash and cash equivalents", "11", "11"),
            ],
        },
    }
    result = validate_financials(extraction)
    assert (
        _by_name(result, "cash_flow_current_period_activities_vs_net_increase")["status"]
        == "NOT_APPLICABLE"
    )


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


def test_result_shape_and_default_tolerance():
    extraction = {
        "file_name": "x.jpg",
        "document_type": "Invoice",
        "document": {
            "document_type": "Invoice",
            "line_items": [],
            "subtotal_net": None,
            "total_vat": None,
            "total_gross": None,
            "additional_fields": [],
        },
    }
    result = validate_financials(extraction)
    assert result["tolerance"] == DEFAULT_TOLERANCE
    assert isinstance(result["check"], list)
    for item in result["check"]:
        assert set(item.keys()) >= {
            "name",
            "formula",
            "operands",
            "calculated_value",
            "reported_value",
            "variance",
            "status",
        }
        assert item["status"] in {"PASS", "FAIL", "NOT_APPLICABLE"}


def test_unsupported_document_type_and_invalid_tolerance():
    with pytest.raises(FinancialValidationError) as exc:
        validate_financials(
            {
                "document_type": "Receipt",
                "document": {"document_type": "Receipt"},
            }
        )
    assert exc.value.code == "UNSUPPORTED_DOCUMENT_TYPE"

    with pytest.raises(FinancialValidationError) as exc2:
        validate_financials(
            {
                "document_type": "Invoice",
                "document": {"document_type": "Invoice", "line_items": []},
            },
            tolerance=-1,
        )
    assert exc2.value.code == "INVALID_TOLERANCE"


def test_comma_formatted_large_balance_sheet_values():
    extraction = {
        "file_name": "hdfc.pdf",
        "document_type": "Balance Sheet",
        "document": {
            "document_type": "Balance Sheet",
            "capital_and_liabilities": [
                _row("Capital", "5,125,091", "5,056,373"),
                _row("Reserves and Surplus", "4,000,000", "3,000,000"),
                _row("Total Capital and Liabilities", "9,125,091", "8,056,373"),
            ],
            "assets": [
                _row("Cash and balances with Reserve Bank of India", "4,000,000", "3,500,000"),
                _row("Investments", "5,125,091", "4,556,373"),
                _row("Total Assets", "9,125,091", "8,056,373"),
            ],
        },
    }
    result = validate_financials(extraction)
    eq = _by_name(result, "balance_sheet_current_period_capital_liabilities_vs_assets")
    assert eq["status"] == "PASS"
    assert eq["operands"]["total_assets"]["raw"] == "9,125,091"
    assert eq["operands"]["total_assets"]["parsed"] == pytest.approx(9_125_091.0)

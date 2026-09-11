"""
Deterministic financial validation layer.

Pipeline position:
    Upload → File Validation → OCR → Gemini extraction → financial validation → ...

Consumes extraction output only. Performs arithmetic checks with a configurable
tolerance. Never calls Gemini/LLMs and never invents missing values.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, Sequence

Status = Literal["PASS", "FAIL", "NOT_APPLICABLE"]

DEFAULT_TOLERANCE = 0.01

PeriodKey = Literal["current", "comparative"]
PERIOD_VALUE_FIELDS: dict[PeriodKey, str] = {
    "current": "current_period_value",
    "comparative": "comparative_period_value",
}


class FinancialValidationError(Exception):
    """
    Controlled validation failure for the API layer to map into:

        {"error": {"code": "...", "message": "..."}}
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def parse_amount(raw: Any) -> float | None:
    """
    Parse a reported financial amount.

    Supports commas as thousands or decimal separators, bracketed negatives,
    leading minus, and common currency decoration. Returns None when the value
    is missing or unparseable — never coerces missing input to zero.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        if raw != raw:  # NaN
            return None
        return float(raw)

    text = str(raw).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"n/a", "na", "nil", "-", "—", "–", "."}:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()

    # Keep digits, separators, and minus; drop currency letters/symbols.
    text = re.sub(r"[^\d,.\-]", "", text)
    if not text or text in {".", ",", "-", "-.", ".-"}:
        return None

    if text.startswith("-"):
        negative = True
        text = text[1:].strip()
    if not text:
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            # European: 1.234,56
            text = text.replace(".", "").replace(",", ".")
        else:
            # US/Indian: 1,234.56
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2 and parts[0] != "":
            # Decimal comma: 138,90 / 5,00
            text = parts[0] + "." + parts[1]
        else:
            # Thousands separators: 5,125,091
            text = text.replace(",", "")
    elif text.count(".") > 1:
        # Dotted thousands: 1.234.567
        text = text.replace(".", "")

    try:
        value = float(text)
    except ValueError:
        return None

    return -value if negative else value


def _approx_equal(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= abs(tolerance)


def _normalize_label(text: str) -> str:
    cleaned = text.lower().replace("&", " and ")
    cleaned = cleaned.replace("/", " ")
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _label_matches(description: str | None, patterns: Sequence[str]) -> bool:
    if description is None:
        return False
    label = _normalize_label(description)
    if not label:
        return False
    for pattern in patterns:
        needle = _normalize_label(pattern)
        if needle and needle in label:
            return True
    return False


def _field_raw(node: Any) -> Any:
    if node is None:
        return None
    if isinstance(node, Mapping):
        return node.get("value")
    return None


def _field_parsed(node: Any) -> float | None:
    return parse_amount(_field_raw(node))


def _operand(raw: Any, parsed: float | None = None) -> dict[str, Any]:
    if parsed is None and raw is not None:
        parsed = parse_amount(raw)
    return {"raw": raw, "parsed": parsed}


def _check(
    *,
    name: str,
    formula: str,
    operands: Mapping[str, Any],
    calculated_value: float | None,
    reported_value: float | None,
    status: Status,
    reported_raw: Any = None,
) -> dict[str, Any]:
    variance: float | None
    if status == "NOT_APPLICABLE" or calculated_value is None or reported_value is None:
        variance = None
    else:
        variance = calculated_value - reported_value

    result: dict[str, Any] = {
        "name": name,
        "formula": formula,
        "operands": dict(operands),
        "calculated_value": calculated_value,
        "reported_value": reported_value,
        "variance": variance,
        "status": status,
    }
    if reported_raw is not None:
        result["reported_raw"] = reported_raw
    return result


def _na_check(name: str, formula: str, operands: Mapping[str, Any]) -> dict[str, Any]:
    return _check(
        name=name,
        formula=formula,
        operands=operands,
        calculated_value=None,
        reported_value=None,
        status="NOT_APPLICABLE",
    )


def _compare_check(
    *,
    name: str,
    formula: str,
    operands: Mapping[str, Any],
    calculated: float | None,
    reported: float | None,
    reported_raw: Any,
    tolerance: float,
    required_present: bool,
) -> dict[str, Any]:
    if not required_present or calculated is None or reported is None:
        return _na_check(name, formula, operands)

    status: Status = "PASS" if _approx_equal(calculated, reported, tolerance) else "FAIL"
    return _check(
        name=name,
        formula=formula,
        operands=operands,
        calculated_value=calculated,
        reported_value=reported,
        status=status,
        reported_raw=reported_raw,
    )


def _row_description(row: Mapping[str, Any]) -> str | None:
    raw = _field_raw(row.get("description"))
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _row_period_raw(row: Mapping[str, Any], period: PeriodKey) -> Any:
    return _field_raw(row.get(PERIOD_VALUE_FIELDS[period]))


def _row_period_parsed(row: Mapping[str, Any], period: PeriodKey) -> float | None:
    return parse_amount(_row_period_raw(row, period))


def _is_total_like(description: str | None) -> bool:
    if description is None:
        return False
    label = _normalize_label(description)
    return bool(
        re.search(r"\btotal\b", label)
        or re.search(r"\bsub[\s-]?total\b", label)
        or label.startswith("net ")
        or " grand total" in f" {label}"
    )


def _find_row(
    rows: Sequence[Mapping[str, Any]],
    patterns: Sequence[str],
    *,
    exclude_patterns: Sequence[str] = (),
) -> Mapping[str, Any] | None:
    """Return the best-matching row (longest pattern wins)."""
    best: Mapping[str, Any] | None = None
    best_score = -1
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        desc = _row_description(row)
        if exclude_patterns and _label_matches(desc, exclude_patterns):
            continue
        if desc is None:
            continue
        label = _normalize_label(desc)
        for pattern in patterns:
            needle = _normalize_label(pattern)
            if not needle:
                continue
            if needle == label or needle in label:
                score = len(needle)
                # Prefer exact label equality slightly.
                if needle == label:
                    score += 1000
                if score > best_score:
                    best = row
                    best_score = score
    return best


def _find_balance_sheet_total_row(
    rows: Sequence[Mapping[str, Any]],
    patterns: Sequence[str],
) -> Mapping[str, Any] | None:
    """Find a labelled balance-sheet total, then an exact generic ``Total``.

    Gemini can preserve the section rows while abbreviating the last label to
    ``Total``. Within the already-separated capital/liabilities or assets
    sections, that exact row is the reported section total. Other total-like
    labels are deliberately not accepted as a fallback.
    """
    labelled_total = _find_row(rows, patterns)
    if labelled_total is not None:
        return labelled_total

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if _normalize_label(_row_description(row) or "") == "total":
            return row
    return None


def _find_additional_amount(
    fields: Sequence[Mapping[str, Any]] | None,
    patterns: Sequence[str],
) -> tuple[Any, float | None]:
    if not fields:
        return None, None
    for item in fields:
        if not isinstance(item, Mapping):
            continue
        name = item.get("field")
        name_text = str(name).strip() if name is not None else ""
        # Prefer explicit field names; also allow value labels via evidence/source.
        candidates = [name_text]
        evidence = item.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("source_text"):
            candidates.append(str(evidence.get("source_text")))
        raw = item.get("value")
        if any(_label_matches(c, patterns) for c in candidates if c):
            return raw, parse_amount(raw)
    return None, None


# ---------------------------------------------------------------------------
# Invoice
# ---------------------------------------------------------------------------

def _invoice_line_total_nodes(line: Mapping[str, Any]) -> tuple[Any, Any]:
    """Prefer net line total; fall back to gross when net is absent."""
    net = line.get("net_amount")
    if _field_raw(net) is not None:
        return net, "net_amount"
    gross = line.get("gross_amount")
    if _field_raw(gross) is not None:
        return gross, "gross_amount"
    return None, "net_amount"


def _validate_invoice(document: Mapping[str, Any], tolerance: float) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    lines = document.get("line_items") or []
    if not isinstance(lines, list):
        lines = []

    line_totals_parsed: list[float] = []
    line_totals_raw: list[Any] = []
    all_line_totals_present = True if lines else False

    for index, line in enumerate(lines):
        if not isinstance(line, Mapping):
            all_line_totals_present = False
            continue

        qty_raw = _field_raw(line.get("quantity"))
        price_raw = _field_raw(line.get("unit_price"))
        total_node, total_field = _invoice_line_total_nodes(line)
        total_raw = _field_raw(total_node)

        qty = parse_amount(qty_raw)
        price = parse_amount(price_raw)
        total = parse_amount(total_raw)

        operands = {
            "line_index": index,
            "quantity": _operand(qty_raw, qty),
            "unit_price": _operand(price_raw, price),
            "line_total_field": total_field,
            "line_total": _operand(total_raw, total),
        }
        formula = "quantity × unit_price ≈ line_total"
        name = f"invoice_line_{index + 1}_quantity_times_unit_price"

        if qty is None or price is None or total is None:
            all_line_totals_present = False
            checks.append(_na_check(name, formula, operands))
            continue

        calculated = qty * price
        status: Status = "PASS" if _approx_equal(calculated, total, tolerance) else "FAIL"
        checks.append(
            _check(
                name=name,
                formula=formula,
                operands=operands,
                calculated_value=calculated,
                reported_value=total,
                status=status,
                reported_raw=total_raw,
            )
        )
        line_totals_parsed.append(total)
        line_totals_raw.append(total_raw)

    # Sum of line totals ≈ subtotal / total before tax
    subtotal_raw = _field_raw(document.get("subtotal_net"))
    subtotal = parse_amount(subtotal_raw)
    sum_formula = "sum(line_totals) ≈ subtotal_net"
    sum_operands: dict[str, Any] = {
        "line_totals": [_operand(r, p) for r, p in zip(line_totals_raw, line_totals_parsed)],
        "subtotal_net": _operand(subtotal_raw, subtotal),
    }
    if not lines or not all_line_totals_present or subtotal is None or not line_totals_parsed:
        checks.append(_na_check("invoice_sum_line_totals_vs_subtotal", sum_formula, sum_operands))
    else:
        calculated = sum(line_totals_parsed)
        status = "PASS" if _approx_equal(calculated, subtotal, tolerance) else "FAIL"
        checks.append(
            _check(
                name="invoice_sum_line_totals_vs_subtotal",
                formula=sum_formula,
                operands=sum_operands,
                calculated_value=calculated,
                reported_value=subtotal,
                status=status,
                reported_raw=subtotal_raw,
            )
        )

    # Taxable Amount + Tax ≈ Total  (subtotal_net + total_vat ≈ total_gross)
    taxable_raw = subtotal_raw
    tax_raw = _field_raw(document.get("total_vat"))
    total_raw = _field_raw(document.get("total_gross"))
    taxable = parse_amount(taxable_raw)
    tax = parse_amount(tax_raw)
    total_gross = parse_amount(total_raw)
    tax_formula = "taxable_amount + tax ≈ total"
    tax_operands = {
        "taxable_amount": _operand(taxable_raw, taxable),
        "tax": _operand(tax_raw, tax),
        "total": _operand(total_raw, total_gross),
    }
    if taxable is None or tax is None or total_gross is None:
        checks.append(_na_check("invoice_taxable_plus_tax_vs_total", tax_formula, tax_operands))
    else:
        calculated = taxable + tax
        status = "PASS" if _approx_equal(calculated, total_gross, tolerance) else "FAIL"
        checks.append(
            _check(
                name="invoice_taxable_plus_tax_vs_total",
                formula=tax_formula,
                operands=tax_operands,
                calculated_value=calculated,
                reported_value=total_gross,
                status=status,
                reported_raw=total_raw,
            )
        )

    # Cash Paid − Total Amount ≈ Change (optional; NOT_APPLICABLE when unavailable)
    additional = document.get("additional_fields")
    if not isinstance(additional, list):
        additional = []
    cash_raw, cash_paid = _find_additional_amount(
        additional,
        ["cash paid", "cash_paid", "amount paid", "amount_paid", "paid"],
    )
    change_raw, change = _find_additional_amount(
        additional,
        ["change", "change due", "cash change", "balance returned"],
    )
    # Total amount for this check: prefer total_gross
    amount_raw = total_raw
    amount = total_gross
    change_formula = "cash_paid − total_amount ≈ change"
    change_operands = {
        "cash_paid": _operand(cash_raw, cash_paid),
        "total_amount": _operand(amount_raw, amount),
        "change": _operand(change_raw, change),
    }
    if cash_paid is None or amount is None or change is None:
        checks.append(_na_check("invoice_cash_paid_minus_total_vs_change", change_formula, change_operands))
    else:
        calculated = cash_paid - amount
        status = "PASS" if _approx_equal(calculated, change, tolerance) else "FAIL"
        checks.append(
            _check(
                name="invoice_cash_paid_minus_total_vs_change",
                formula=change_formula,
                operands=change_operands,
                calculated_value=calculated,
                reported_value=change,
                status=status,
                reported_raw=change_raw,
            )
        )

    return checks


# ---------------------------------------------------------------------------
# Balance sheet
# ---------------------------------------------------------------------------

_TOTAL_CL_PATTERNS = (
    "total capital and liabilities",
    "total capital & liabilities",
    "total liabilities and equity",
    "total equity and liabilities",
)
_TOTAL_ASSETS_PATTERNS = ("total assets",)
_EXCLUDE_FROM_COMPONENT_SUM = (
    "total capital and liabilities",
    "total capital & liabilities",
    "total liabilities and equity",
    "total equity and liabilities",
    "total assets",
    "total",
)


def _sum_components(
    rows: Sequence[Mapping[str, Any]],
    period: PeriodKey,
    exclude_patterns: Sequence[str],
) -> tuple[float | None, list[dict[str, Any]], int]:
    """
    Sum non-total component rows for a period.

    Returns (sum_or_None, operand_rows, usable_count).
    Missing individual component values are skipped (not assumed zero).
    Sum is None when fewer than 2 usable components exist ("sufficient components").
    """
    operands: list[dict[str, Any]] = []
    total = 0.0
    usable = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        desc = _row_description(row)
        if _label_matches(desc, exclude_patterns) or (
            desc is not None and _normalize_label(desc) == "total"
        ):
            continue
        raw = _row_period_raw(row, period)
        parsed = parse_amount(raw)
        operands.append(
            {
                "description": desc,
                "amount": _operand(raw, parsed),
            }
        )
        if parsed is None:
            continue
        total += parsed
        usable += 1

    if usable < 2:
        return None, operands, usable
    return total, operands, usable


def _validate_balance_sheet(
    document: Mapping[str, Any],
    tolerance: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    capital_rows = document.get("capital_and_liabilities") or []
    asset_rows = document.get("assets") or []
    if not isinstance(capital_rows, list):
        capital_rows = []
    if not isinstance(asset_rows, list):
        asset_rows = []

    total_cl_row = _find_balance_sheet_total_row(capital_rows, _TOTAL_CL_PATTERNS)
    total_assets_row = _find_balance_sheet_total_row(asset_rows, _TOTAL_ASSETS_PATTERNS)

    for period in ("current", "comparative"):
        period_label = f"{period}_period"
        cl_raw = _row_period_raw(total_cl_row, period) if total_cl_row else None
        assets_raw = _row_period_raw(total_assets_row, period) if total_assets_row else None
        cl_val = parse_amount(cl_raw)
        assets_val = parse_amount(assets_raw)

        # Total Capital & Liabilities ≈ Total Assets
        eq_name = f"balance_sheet_{period_label}_capital_liabilities_vs_assets"
        eq_formula = "total_capital_and_liabilities ≈ total_assets"
        eq_operands = {
            "period": period,
            "total_capital_and_liabilities": _operand(cl_raw, cl_val),
            "total_assets": _operand(assets_raw, assets_val),
        }
        checks.append(
            _compare_check(
                name=eq_name,
                formula=eq_formula,
                operands=eq_operands,
                calculated=cl_val,
                reported=assets_val,
                reported_raw=assets_raw,
                tolerance=tolerance,
                required_present=cl_val is not None and assets_val is not None,
            )
        )
        # When both sides exist, treat capital side as calculated identity check vs assets.
        # Re-write last check so calculated=CL and reported=Assets (already done).

        # Sum C&L components ≈ reported C&L total
        cl_sum, cl_ops, cl_count = _sum_components(
            capital_rows, period, _EXCLUDE_FROM_COMPONENT_SUM
        )
        cl_sum_name = f"balance_sheet_{period_label}_sum_capital_liability_components"
        cl_sum_formula = "sum(capital_and_liability_components) ≈ total_capital_and_liabilities"
        cl_sum_operands = {
            "period": period,
            "components": cl_ops,
            "usable_component_count": cl_count,
            "total_capital_and_liabilities": _operand(cl_raw, cl_val),
        }
        checks.append(
            _compare_check(
                name=cl_sum_name,
                formula=cl_sum_formula,
                operands=cl_sum_operands,
                calculated=cl_sum,
                reported=cl_val,
                reported_raw=cl_raw,
                tolerance=tolerance,
                required_present=cl_sum is not None and cl_val is not None,
            )
        )

        # Sum Asset components ≈ reported Assets total
        asset_sum, asset_ops, asset_count = _sum_components(
            asset_rows, period, _EXCLUDE_FROM_COMPONENT_SUM
        )
        asset_sum_name = f"balance_sheet_{period_label}_sum_asset_components"
        asset_sum_formula = "sum(asset_components) ≈ total_assets"
        asset_sum_operands = {
            "period": period,
            "components": asset_ops,
            "usable_component_count": asset_count,
            "total_assets": _operand(assets_raw, assets_val),
        }
        checks.append(
            _compare_check(
                name=asset_sum_name,
                formula=asset_sum_formula,
                operands=asset_sum_operands,
                calculated=asset_sum,
                reported=assets_val,
                reported_raw=assets_raw,
                tolerance=tolerance,
                required_present=asset_sum is not None and assets_val is not None,
            )
        )

    return checks


# ---------------------------------------------------------------------------
# Profit & Loss
# ---------------------------------------------------------------------------

_PNL_PATTERNS = {
    "interest_earned": ("interest earned", "interest income"),
    "other_income": ("other income",),
    "total_income": ("total income",),
    "interest_expended": ("interest expended", "interest expense", "interest paid"),
    "operating_expenses": (
        "operating expenses",
        "operating expense",
        "other operating expenses",
    ),
    "provisions": (
        "provisions and contingencies",
        "provisions & contingencies",
        "provision and contingencies",
    ),
    "total_expenditure": ("total expenditure", "total expenses", "total expense"),
    "profit_before_minority": (
        "consolidated net profit before minority interest",
        "net profit before minority interest",
        "profit before minority interest",
        "consolidated profit before minority",
    ),
    "minority_interest": (
        "minority interest",
        "share of minority interest",
        "less minority interest",
    ),
    "profit_attributable_group": (
        "net profit attributable to group",
        "profit attributable to group",
        "profit attributable to the group",
        "net profit attributable to owners",
        "attributable to group",
    ),
    "current_profit": (
        "current profit",
        "profit for the year",
        "profit for the period",
        "net profit for the year",
        "consolidated net profit",
    ),
    "brought_forward": (
        "brought forward",
        "balance brought forward",
        "profit brought forward",
        "surplus brought forward",
    ),
    "available_for_appropriation": (
        "total available for appropriation",
        "amount available for appropriation",
        "available for appropriation",
    ),
}


def _period_from_row(
    row: Mapping[str, Any] | None,
    period: PeriodKey,
) -> tuple[Any, float | None]:
    if row is None:
        return None, None
    raw = _row_period_raw(row, period)
    return raw, parse_amount(raw)


def _validate_profit_and_loss(
    document: Mapping[str, Any],
    tolerance: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    income = document.get("income") or []
    expenditure = document.get("expenditure") or []
    profit = document.get("profit") or []
    appropriations = document.get("appropriations") or []
    if not isinstance(income, list):
        income = []
    if not isinstance(expenditure, list):
        expenditure = []
    if not isinstance(profit, list):
        profit = []
    if not isinstance(appropriations, list):
        appropriations = []

    rows_income = income
    rows_exp = expenditure
    rows_profit = profit
    rows_approp = appropriations

    interest_earned = _find_row(rows_income, _PNL_PATTERNS["interest_earned"])
    other_income = _find_row(rows_income, _PNL_PATTERNS["other_income"])
    total_income = _find_row(rows_income, _PNL_PATTERNS["total_income"])

    interest_expended = _find_row(rows_exp, _PNL_PATTERNS["interest_expended"])
    operating_expenses = _find_row(rows_exp, _PNL_PATTERNS["operating_expenses"])
    provisions = _find_row(rows_exp, _PNL_PATTERNS["provisions"])
    total_expenditure = _find_row(rows_exp, _PNL_PATTERNS["total_expenditure"])

    # Profit rows may live under profit[] or income/expenditure depending on extraction
    profit_before_minority = (
        _find_row(rows_profit, _PNL_PATTERNS["profit_before_minority"])
        or _find_row(rows_income, _PNL_PATTERNS["profit_before_minority"])
        or _find_row(rows_approp, _PNL_PATTERNS["profit_before_minority"])
    )
    minority_interest = (
        _find_row(
            rows_profit,
            _PNL_PATTERNS["minority_interest"],
            exclude_patterns=(
                "before minority",
                "attributable",
                "profit before",
            ),
        )
        or _find_row(
            rows_approp,
            _PNL_PATTERNS["minority_interest"],
            exclude_patterns=("before minority", "attributable", "profit before"),
        )
        or _find_row(
            rows_exp,
            _PNL_PATTERNS["minority_interest"],
            exclude_patterns=("before minority", "attributable", "profit before"),
        )
    )
    profit_group = (
        _find_row(rows_profit, _PNL_PATTERNS["profit_attributable_group"])
        or _find_row(rows_approp, _PNL_PATTERNS["profit_attributable_group"])
    )

    current_profit = (
        _find_row(rows_approp, _PNL_PATTERNS["current_profit"])
        or _find_row(rows_profit, _PNL_PATTERNS["current_profit"])
    )
    brought_forward = _find_row(rows_approp, _PNL_PATTERNS["brought_forward"])
    available = _find_row(rows_approp, _PNL_PATTERNS["available_for_appropriation"])
    appropriations_present = bool(rows_approp)

    for period in ("current", "comparative"):
        period_label = f"{period}_period"

        ie_raw, ie = _period_from_row(interest_earned, period)
        oi_raw, oi = _period_from_row(other_income, period)
        ti_raw, ti = _period_from_row(total_income, period)

        income_ops = {
            "period": period,
            "interest_earned": _operand(ie_raw, ie),
            "other_income": _operand(oi_raw, oi),
            "total_income": _operand(ti_raw, ti),
        }
        income_formula = "interest_earned + other_income ≈ total_income"
        if ie is None or oi is None or ti is None:
            checks.append(
                _na_check(
                    f"pnl_{period_label}_interest_plus_other_income",
                    income_formula,
                    income_ops,
                )
            )
        else:
            calculated = ie + oi
            status: Status = "PASS" if _approx_equal(calculated, ti, tolerance) else "FAIL"
            checks.append(
                _check(
                    name=f"pnl_{period_label}_interest_plus_other_income",
                    formula=income_formula,
                    operands=income_ops,
                    calculated_value=calculated,
                    reported_value=ti,
                    status=status,
                    reported_raw=ti_raw,
                )
            )

        ix_raw, ix = _period_from_row(interest_expended, period)
        ox_raw, ox = _period_from_row(operating_expenses, period)
        pr_raw, pr = _period_from_row(provisions, period)
        te_raw, te = _period_from_row(total_expenditure, period)
        exp_ops = {
            "period": period,
            "interest_expended": _operand(ix_raw, ix),
            "operating_expenses": _operand(ox_raw, ox),
            "provisions_and_contingencies": _operand(pr_raw, pr),
            "total_expenditure": _operand(te_raw, te),
        }
        exp_formula = (
            "interest_expended + operating_expenses + provisions_and_contingencies "
            "≈ total_expenditure"
        )
        if ix is None or ox is None or pr is None or te is None:
            checks.append(
                _na_check(
                    f"pnl_{period_label}_expenditure_components",
                    exp_formula,
                    exp_ops,
                )
            )
        else:
            calculated = ix + ox + pr
            status = "PASS" if _approx_equal(calculated, te, tolerance) else "FAIL"
            checks.append(
                _check(
                    name=f"pnl_{period_label}_expenditure_components",
                    formula=exp_formula,
                    operands=exp_ops,
                    calculated_value=calculated,
                    reported_value=te,
                    status=status,
                    reported_raw=te_raw,
                )
            )

        pbm_raw, pbm = _period_from_row(profit_before_minority, period)
        # Total Income − Total Expenditure ≈ Profit before Minority Interest
        profit_ops = {
            "period": period,
            "total_income": _operand(ti_raw, ti),
            "total_expenditure": _operand(te_raw, te),
            "profit_before_minority_interest": _operand(pbm_raw, pbm),
        }
        profit_formula = (
            "total_income − total_expenditure ≈ "
            "consolidated_net_profit_before_minority_interest"
        )
        if ti is None or te is None or pbm is None:
            checks.append(
                _na_check(
                    f"pnl_{period_label}_income_minus_expenditure",
                    profit_formula,
                    profit_ops,
                )
            )
        else:
            calculated = ti - te
            status = "PASS" if _approx_equal(calculated, pbm, tolerance) else "FAIL"
            checks.append(
                _check(
                    name=f"pnl_{period_label}_income_minus_expenditure",
                    formula=profit_formula,
                    operands=profit_ops,
                    calculated_value=calculated,
                    reported_value=pbm,
                    status=status,
                    reported_raw=pbm_raw,
                )
            )

        mi_raw, mi = _period_from_row(minority_interest, period)
        pg_raw, pg = _period_from_row(profit_group, period)
        group_ops = {
            "period": period,
            "profit_before_minority_interest": _operand(pbm_raw, pbm),
            "minority_interest": _operand(mi_raw, mi),
            "net_profit_attributable_to_group": _operand(pg_raw, pg),
        }
        group_formula = (
            "profit_before_minority_interest − minority_interest ≈ "
            "net_profit_attributable_to_group"
        )
        if pbm is None or mi is None or pg is None:
            checks.append(
                _na_check(
                    f"pnl_{period_label}_profit_after_minority",
                    group_formula,
                    group_ops,
                )
            )
        else:
            calculated = pbm - mi
            status = "PASS" if _approx_equal(calculated, pg, tolerance) else "FAIL"
            checks.append(
                _check(
                    name=f"pnl_{period_label}_profit_after_minority",
                    formula=group_formula,
                    operands=group_ops,
                    calculated_value=calculated,
                    reported_value=pg,
                    status=status,
                    reported_raw=pg_raw,
                )
            )

        # Appropriations (only when appropriation section exists / fields present)
        cp_raw, cp = _period_from_row(current_profit, period)
        bf_raw, bf = _period_from_row(brought_forward, period)
        av_raw, av = _period_from_row(available, period)
        ap_ops = {
            "period": period,
            "current_profit": _operand(cp_raw, cp),
            "brought_forward_profit": _operand(bf_raw, bf),
            "total_available_for_appropriation": _operand(av_raw, av),
        }
        ap_formula = (
            "current_profit + brought_forward_profit ≈ total_available_for_appropriation"
        )
        if not appropriations_present or cp is None or bf is None or av is None:
            checks.append(
                _na_check(
                    f"pnl_{period_label}_appropriations_available",
                    ap_formula,
                    ap_ops,
                )
            )
        else:
            calculated = cp + bf
            status = "PASS" if _approx_equal(calculated, av, tolerance) else "FAIL"
            checks.append(
                _check(
                    name=f"pnl_{period_label}_appropriations_available",
                    formula=ap_formula,
                    operands=ap_ops,
                    calculated_value=calculated,
                    reported_value=av,
                    status=status,
                    reported_raw=av_raw,
                )
            )

    return checks


# ---------------------------------------------------------------------------
# Cash flow
# ---------------------------------------------------------------------------

_CF_NET_OPERATING = (
    "net cash from operating",
    "net cash generated from operating",
    "net cash used in operating",
    "cash flows from operating activities",
    "net cash flow from operating",
)
_CF_NET_INVESTING = (
    "net cash from investing",
    "net cash generated from investing",
    "net cash used in investing",
    "cash flows from investing activities",
    "net cash flow from investing",
)
_CF_NET_FINANCING = (
    "net cash from financing",
    "net cash generated from financing",
    "net cash used in financing",
    "cash flows from financing activities",
    "net cash flow from financing",
)
_CF_FX = (
    "effect of exchange",
    "foreign exchange",
    "translation adjustment",
    "fx adjustment",
    "exchange difference",
    "exchange rate",
)
_CF_NET_INCREASE = (
    "net increase in cash",
    "net decrease in cash",
    "net increase/(decrease) in cash",
    "net increase / (decrease) in cash",
    "net change in cash",
)
_CF_OPENING = (
    "cash and cash equivalents at the beginning",
    "opening cash and cash equivalents",
    "cash and cash equivalents at beginning",
    "opening cash",
)
_CF_CLOSING = (
    "cash and cash equivalents at the end",
    "closing cash and cash equivalents",
    "cash and cash equivalents at end",
    "closing cash",
)
_CF_OTHER_ADJUSTMENTS = (
    "other adjustment",
    "adjustment",
)


def _section_net_row(
    rows: Sequence[Mapping[str, Any]],
    patterns: Sequence[str],
) -> Mapping[str, Any] | None:
    # Prefer explicit net/total matches; search from the end (totals usually last).
    for row in reversed(list(rows)):
        if isinstance(row, Mapping) and _label_matches(_row_description(row), patterns):
            return row
    return None


def _validate_cash_flow(
    document: Mapping[str, Any],
    tolerance: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    operating = document.get("operating_activities") or []
    investing = document.get("investing_activities") or []
    financing = document.get("financing_activities") or []
    net_change = document.get("net_change_and_cash_balances") or []
    if not isinstance(operating, list):
        operating = []
    if not isinstance(investing, list):
        investing = []
    if not isinstance(financing, list):
        financing = []
    if not isinstance(net_change, list):
        net_change = []

    op_row = _section_net_row(operating, _CF_NET_OPERATING)
    inv_row = _section_net_row(investing, _CF_NET_INVESTING)
    fin_row = _section_net_row(financing, _CF_NET_FINANCING)

    fx_row = _find_row(net_change, _CF_FX) or _find_row(operating, _CF_FX)
    net_inc_row = _find_row(net_change, _CF_NET_INCREASE)
    opening_row = _find_row(net_change, _CF_OPENING)
    closing_row = _find_row(net_change, _CF_CLOSING)
    # Optional adjustments beyond FX (explicitly labeled); missing => omitted, not zeroed
    other_adj_row = None
    for row in net_change:
        if not isinstance(row, Mapping):
            continue
        desc = _row_description(row)
        if _label_matches(desc, _CF_FX):
            continue
        if _label_matches(desc, _CF_NET_INCREASE):
            continue
        if _label_matches(desc, _CF_OPENING) or _label_matches(desc, _CF_CLOSING):
            continue
        if _label_matches(desc, _CF_OTHER_ADJUSTMENTS):
            other_adj_row = row
            break

    for period in ("current", "comparative"):
        period_label = f"{period}_period"

        op_raw, op = _period_from_row(op_row, period)
        inv_raw, inv = _period_from_row(inv_row, period)
        fin_raw, fin = _period_from_row(fin_row, period)
        fx_raw, fx = _period_from_row(fx_row, period)
        ni_raw, ni = _period_from_row(net_inc_row, period)

        # FX is optional in the sense that if the statement has no FX line,
        # treat FX as absent → NOT_APPLICABLE rather than assuming 0.
        # If FX row exists but value null for period → NOT_APPLICABLE.
        # Spec: Operating + Investing + Financing + FX/Translation ≈ Net Increase
        # Require all four addends when FX row is present; when FX row absent,
        # require O+I+F ≈ Net Increase (three-way) still applicable.
        net_formula = (
            "operating + investing + financing + fx_translation_adjustment "
            "≈ net_increase_in_cash_and_cash_equivalents"
        )
        net_ops: dict[str, Any] = {
            "period": period,
            "operating": _operand(op_raw, op),
            "investing": _operand(inv_raw, inv),
            "financing": _operand(fin_raw, fin),
            "fx_translation_adjustment": _operand(fx_raw, fx),
            "net_increase_in_cash": _operand(ni_raw, ni),
        }

        if op is None or inv is None or fin is None or ni is None:
            checks.append(
                _na_check(
                    f"cash_flow_{period_label}_activities_vs_net_increase",
                    net_formula,
                    net_ops,
                )
            )
        elif fx_row is not None and fx is None:
            checks.append(
                _na_check(
                    f"cash_flow_{period_label}_activities_vs_net_increase",
                    net_formula,
                    net_ops,
                )
            )
        else:
            fx_component = fx if fx is not None else 0.0
            # Only treat FX as zero when the FX line is truly absent from the statement.
            if fx_row is None:
                net_ops["fx_translation_adjustment"] = {
                    "raw": None,
                    "parsed": None,
                    "assumed": None,
                    "note": "FX/translation line not present; excluded (not assumed zero).",
                }
                # Without FX line: Operating + Investing + Financing ≈ Net Increase
                calculated = op + inv + fin
                net_formula = (
                    "operating + investing + financing "
                    "≈ net_increase_in_cash_and_cash_equivalents"
                )
            else:
                calculated = op + inv + fin + fx_component

            status: Status = "PASS" if _approx_equal(calculated, ni, tolerance) else "FAIL"
            checks.append(
                _check(
                    name=f"cash_flow_{period_label}_activities_vs_net_increase",
                    formula=net_formula,
                    operands=net_ops,
                    calculated_value=calculated,
                    reported_value=ni,
                    status=status,
                    reported_raw=ni_raw,
                )
            )

        open_raw, opening = _period_from_row(opening_row, period)
        close_raw, closing = _period_from_row(closing_row, period)
        adj_raw, adj = _period_from_row(other_adj_row, period)

        bridge_formula = (
            "opening_cash + net_increase_in_cash + applicable_adjustments ≈ closing_cash"
        )
        bridge_ops = {
            "period": period,
            "opening_cash": _operand(open_raw, opening),
            "net_increase_in_cash": _operand(ni_raw, ni),
            "applicable_adjustments": _operand(adj_raw, adj),
            "closing_cash": _operand(close_raw, closing),
        }

        if opening is None or ni is None or closing is None:
            checks.append(
                _na_check(
                    f"cash_flow_{period_label}_opening_plus_net_vs_closing",
                    bridge_formula,
                    bridge_ops,
                )
            )
        elif other_adj_row is not None and adj is None:
            # Adjustment line exists but value missing for period → do not assume 0
            checks.append(
                _na_check(
                    f"cash_flow_{period_label}_opening_plus_net_vs_closing",
                    bridge_formula,
                    bridge_ops,
                )
            )
        else:
            calculated = opening + ni + (adj if adj is not None else 0.0)
            status = "PASS" if _approx_equal(calculated, closing, tolerance) else "FAIL"
            checks.append(
                _check(
                    name=f"cash_flow_{period_label}_opening_plus_net_vs_closing",
                    formula=bridge_formula,
                    operands=bridge_ops,
                    calculated_value=calculated,
                    reported_value=closing,
                    status=status,
                    reported_raw=close_raw,
                )
            )

    return checks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_financials(
    extraction_result: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """
    Run deterministic financial checks against Gemini extraction output.

    Parameters
    ----------
    extraction_result:
        Dict shaped like ExtractionResult.model_dump():
        {file_name, document_type, document}.
    tolerance:
        Absolute tolerance for ≈ comparisons (configurable).

    Returns
    -------
    {
      "file_name": ...,
      "document_type": ...,
      "tolerance": ...,
      "check": [ {name, formula, operands, calculated_value, reported_value, variance, status}, ...]
    }
    """
    if not isinstance(extraction_result, Mapping):
        raise FinancialValidationError(
            code="INVALID_EXTRACTION_PAYLOAD",
            message="extraction_result must be a mapping/dict.",
        )

    if tolerance < 0:
        raise FinancialValidationError(
            code="INVALID_TOLERANCE",
            message="tolerance must be a non-negative number.",
        )

    document_type = extraction_result.get("document_type")
    document = extraction_result.get("document")
    file_name = extraction_result.get("file_name")

    if not isinstance(document, Mapping):
        raise FinancialValidationError(
            code="INVALID_EXTRACTION_PAYLOAD",
            message="extraction_result.document must be a mapping/dict.",
        )

    # Prefer envelope document_type; fall back to nested document_type.
    resolved_type = document_type or document.get("document_type")
    if isinstance(resolved_type, str):
        type_key = resolved_type.strip().lower()
    else:
        type_key = ""

    type_aliases = {
        "invoice": "Invoice",
        "balance sheet": "Balance Sheet",
        "balance_sheet": "Balance Sheet",
        "profit & loss": "Profit & Loss",
        "profit and loss": "Profit & Loss",
        "profit_and_loss": "Profit & Loss",
        "p&l": "Profit & Loss",
        "cash flow": "Cash Flow",
        "cash_flow": "Cash Flow",
        "cashflow": "Cash Flow",
    }
    canonical = type_aliases.get(type_key)
    if canonical is None and resolved_type in {
        "Invoice",
        "Balance Sheet",
        "Profit & Loss",
        "Cash Flow",
    }:
        canonical = str(resolved_type)

    if canonical is None:
        raise FinancialValidationError(
            code="UNSUPPORTED_DOCUMENT_TYPE",
            message=(
                f"Unsupported document_type '{resolved_type}'. "
                "Supported values: Invoice, Balance Sheet, Profit & Loss, Cash Flow."
            ),
        )

    if canonical == "Invoice":
        checks = _validate_invoice(document, tolerance)
    elif canonical == "Balance Sheet":
        checks = _validate_balance_sheet(document, tolerance)
    elif canonical == "Profit & Loss":
        checks = _validate_profit_and_loss(document, tolerance)
    else:
        checks = _validate_cash_flow(document, tolerance)

    return {
        "file_name": file_name if isinstance(file_name, str) else None,
        "document_type": canonical,
        "tolerance": tolerance,
        "check": checks,
    }

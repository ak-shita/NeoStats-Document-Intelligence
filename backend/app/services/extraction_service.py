"""
Gemini structured extraction layer.

Pipeline position:
    Upload → File Validation → OCR → Gemini extraction → financial validation → ...

Consumes OCR page-level text and a caller-supplied document_type.
Does not classify documents, compute financial totals, or persist data.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Type

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings
from app.schemas.extraction import (
    BalanceSheetExtraction,
    CashFlowExtraction,
    DocumentType,
    ExtractionResult,
    InvoiceExtraction,
    ProfitAndLossExtraction,
)

logger = logging.getLogger(__name__)

# Final validation models (unchanged public extraction shape).
SCHEMA_BY_DOCUMENT_TYPE: dict[DocumentType, Type[BaseModel]] = {
    DocumentType.INVOICE: InvoiceExtraction,
    DocumentType.BALANCE_SHEET: BalanceSheetExtraction,
    DocumentType.PROFIT_AND_LOSS: ProfitAndLossExtraction,
    DocumentType.CASH_FLOW: CashFlowExtraction,
}

# Keywords Gemini structured-output rejects or ignores poorly when nested heavily.
_GEMINI_STRIP_KEYS = frozenset(
    {
        "default",
        "title",
        "description",
        "examples",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
    }
)

DOCUMENT_TYPE_ALIASES: dict[str, DocumentType] = {
    "invoice": DocumentType.INVOICE,
    "balance sheet": DocumentType.BALANCE_SHEET,
    "balance_sheet": DocumentType.BALANCE_SHEET,
    "profit & loss": DocumentType.PROFIT_AND_LOSS,
    "profit and loss": DocumentType.PROFIT_AND_LOSS,
    "profit_and_loss": DocumentType.PROFIT_AND_LOSS,
    "p&l": DocumentType.PROFIT_AND_LOSS,
    "cash flow": DocumentType.CASH_FLOW,
    "cash_flow": DocumentType.CASH_FLOW,
    "cashflow": DocumentType.CASH_FLOW,
}


class ExtractionError(Exception):
    """
    Controlled extraction failure for the API layer to map into:

        {"error": {"code": "...", "message": "..."}}
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def normalize_document_type(document_type: str) -> DocumentType:
    raw = (document_type or "").strip()
    if not raw:
        raise ExtractionError(
            code="UNSUPPORTED_DOCUMENT_TYPE",
            message="document_type is required and must be one of: "
            "Invoice, Balance Sheet, Profit & Loss, Cash Flow.",
        )

    try:
        return DocumentType(raw)
    except ValueError:
        pass

    aliased = DOCUMENT_TYPE_ALIASES.get(raw.lower())
    if aliased is not None:
        return aliased

    raise ExtractionError(
        code="UNSUPPORTED_DOCUMENT_TYPE",
        message=(
            f"Unsupported document_type '{document_type}'. "
            "Supported values: Invoice, Balance Sheet, Profit & Loss, Cash Flow."
        ),
    )


def _resolve_api_key(settings: Settings, api_key: str | None) -> str:
    if api_key is not None:
        return api_key.strip()
    return (settings.gemini_api_key or "").strip()


def _is_null_schema(node: dict[str, Any]) -> bool:
    return node.get("type") == "null"


def _collapse_nullable_anyof(node: dict[str, Any]) -> dict[str, Any]:
    """
    Convert Pydantic's anyOf: [{T}, {type: null}] into Gemini-friendly
    {**T, nullable: true}. Leaves non-null unions unchanged.
    """
    any_of = node.get("anyOf")
    if not isinstance(any_of, list) or len(any_of) != 2:
        return node
    if not all(isinstance(item, dict) for item in any_of):
        return node

    non_null = [item for item in any_of if not _is_null_schema(item)]
    nulls = [item for item in any_of if _is_null_schema(item)]
    if len(non_null) != 1 or len(nulls) != 1:
        return node

    collapsed: dict[str, Any] = {
        key: value for key, value in node.items() if key != "anyOf"
    }
    collapsed.update(copy.deepcopy(non_null[0]))
    collapsed["nullable"] = True
    return collapsed


def _fix_typeless_any_schema(node: dict[str, Any]) -> dict[str, Any]:
    """
    Pydantic `Any` emits a property with no type/anyOf/$ref/properties.
    Gemini rejects that as INVALID_ARGUMENT on full document schemas.
    Prefer nullable string (amounts keep commas/brackets as text).
    """
    if not isinstance(node, dict):
        return node
    if any(key in node for key in ("type", "anyOf", "oneOf", "allOf", "$ref", "properties", "items", "enum", "const")):
        return node
    # Preserve nothing from the empty Any node except making it typed.
    return {"type": "string", "nullable": True}


def _sanitize_gemini_schema(node: Any, *, as_schema: bool = True) -> Any:
    """Recursively rewrite a Pydantic JSON Schema into Gemini structured-output form."""
    if isinstance(node, list):
        return [_sanitize_gemini_schema(item, as_schema=True) for item in node]

    if not isinstance(node, dict):
        return node

    # `$defs` / `properties` are name→schema maps, not schema nodes themselves.
    if not as_schema:
        return {
            key: _sanitize_gemini_schema(value, as_schema=True)
            for key, value in node.items()
        }

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _GEMINI_STRIP_KEYS:
            continue
        if key in ("$defs", "definitions", "properties", "patternProperties"):
            cleaned[key] = _sanitize_gemini_schema(value, as_schema=False)
        else:
            cleaned[key] = _sanitize_gemini_schema(value, as_schema=True)

    cleaned = _collapse_nullable_anyof(cleaned)
    cleaned = _fix_typeless_any_schema(cleaned)

    # `$ref` + sibling `nullable` is unreliable; keep nullable refs as anyOf.
    if "$ref" in cleaned and cleaned.get("nullable") is True:
        ref = cleaned["$ref"]
        return {"anyOf": [{"$ref": ref}, {"type": "null"}]}

    return cleaned


def gemini_response_schema_for(document_type: DocumentType) -> dict[str, Any]:
    """
    Build the schema sent to Gemini for structured output.

    Existing Pydantic models remain the post-response validation layer.
    """
    model = SCHEMA_BY_DOCUMENT_TYPE[document_type]
    return _sanitize_gemini_schema(copy.deepcopy(model.model_json_schema()))


def _format_ocr_pages(ocr_result: dict[str, Any]) -> tuple[str | None, str]:
    file_name = ocr_result.get("file_name")
    pages = ocr_result.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ExtractionError(
            code="EXTRACTION_EMPTY_OCR_INPUT",
            message="OCR result contains no pages to extract from.",
        )

    blocks: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number", "?")
        text = page.get("text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        blocks.append(f"=== PAGE {page_number} ===\n{text.strip()}")

    combined = "\n\n".join(blocks).strip()
    if not combined:
        raise ExtractionError(
            code="EXTRACTION_EMPTY_OCR_INPUT",
            message="OCR result contains no usable text.",
        )
    return file_name if isinstance(file_name, str) else None, combined


def _system_instruction(document_type: DocumentType) -> str:
    return f"""You are a careful financial document extraction engine for NeoStats.

Document type is already known: {document_type.value}.
Do NOT classify the document. Do NOT invent or infer missing information.

Rules:
1. Extract ALL meaningful visible fields, values, financial line items and table values from the OCR text.
2. Preserve comparative periods exactly as shown (current vs prior columns).
3. Preserve negative and bracketed values exactly as written (e.g. (87,543), -48.34).
4. If a value is missing, blank, or unreadable due to OCR noise, set value to null. Never guess.
5. You may tolerate OCR typos when matching labels, but never fabricate amounts or fields absent from the source.
6. Do NOT compute totals, balances, or any financial math. Only extract what is written.
7. For EVERY extracted field and line-item value, use this object shape:
   {{
     "field": "<snake_case_field_name>",
     "value": <string_or_number_or_null>,
     "confidence": <0.0_to_1.0_or_null>,
     "evidence": {{
       "source_text": "<short supporting OCR snippet or null>",
       "page_number": <int_or_null>
     }}
   }}
8. confidence reflects how clearly the OCR/source text supports the value. It is not a guarantee of correctness.
9. Prefer string values for financial amounts so commas, decimals, currency symbols and brackets are preserved.
10. Return JSON that matches the provided response schema only.
"""


def _user_prompt(*, document_type: DocumentType, file_name: str | None, ocr_text: str) -> str:
    name = file_name or "unknown"
    return (
        f"Extract structured data for document_type={document_type.value}.\n"
        f"file_name={name}\n\n"
        f"OCR text by page:\n{ocr_text}\n"
    )


def _parse_model_response(raw_text: str | None, schema: Type[Any]) -> Any:
    if raw_text is None or not str(raw_text).strip():
        raise ExtractionError(
            code="EXTRACTION_EMPTY_RESPONSE",
            message="Gemini returned an empty extraction response.",
        )

    try:
        return schema.model_validate_json(raw_text)
    except ValidationError as exc:
        # Sometimes models wrap JSON in markdown fences despite schema mode.
        cleaned = str(raw_text).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
            try:
                return schema.model_validate_json(cleaned)
            except ValidationError:
                pass
        raise ExtractionError(
            code="EXTRACTION_MALFORMED_RESPONSE",
            message="Gemini returned JSON that does not match the extraction schema.",
        ) from exc
    except (TypeError, json.JSONDecodeError) as exc:
        raise ExtractionError(
            code="EXTRACTION_MALFORMED_RESPONSE",
            message="Gemini returned a malformed JSON extraction payload.",
        ) from exc


async def extract_structured_data(
    ocr_result: dict[str, Any],
    document_type: str,
    *,
    settings: Settings | None = None,
    api_key: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """
    Run Gemini structured extraction against OCR page text.

    Returns an internal dict suitable for downstream financial validation,
    not the final public API response.
    """

    cfg = settings or get_settings()
    resolved_type = normalize_document_type(document_type)
    schema = SCHEMA_BY_DOCUMENT_TYPE[resolved_type]
    file_name, ocr_text = _format_ocr_pages(ocr_result)

    resolved_key = _resolve_api_key(cfg, api_key)
    if not resolved_key:
        raise ExtractionError(
            code="MISSING_GEMINI_API_KEY",
            message="GEMINI_API_KEY is not configured.",
        )

    owns_client = client is None
    http_options = types.HttpOptions(timeout=int(cfg.gemini_timeout_seconds * 1000))
    genai_client = client or genai.Client(api_key=resolved_key, http_options=http_options)

    # Never log the API key.
    logger.info(
        "Starting Gemini extraction model=%s document_type=%s file_name=%s",
        cfg.gemini_model,
        resolved_type.value,
        file_name,
    )

    try:
        response = await genai_client.aio.models.generate_content(
            model=cfg.gemini_model,
            contents=_user_prompt(
                document_type=resolved_type,
                file_name=file_name,
                ocr_text=ocr_text,
            ),
            config=types.GenerateContentConfig(
                system_instruction=_system_instruction(resolved_type),
                temperature=0.1,
                response_mime_type="application/json",
                # Gemini-compatible schema (not raw Pydantic JSON Schema).
                # Response is still validated with `schema` below.
                response_schema=gemini_response_schema_for(resolved_type),
            ),
        )
    except TimeoutError as exc:
        raise ExtractionError(
            code="EXTRACTION_TIMEOUT",
            message="Gemini extraction request timed out.",
        ) from exc
    except genai_errors.APIError as exc:
        logger.exception(
            "Gemini API error: code=%s message=%s",
            getattr(exc, "code", None),
            getattr(exc, "message", str(exc)),
        )
        raise ExtractionError(
            code="EXTRACTION_MODEL_ERROR",
            message="Gemini API returned an error while extracting document fields.",
        ) from exc
    except Exception as exc:
        message = str(exc).lower()
        if "timeout" in message or "timed out" in message:
            raise ExtractionError(
                code="EXTRACTION_TIMEOUT",
                message="Gemini extraction request timed out.",
            ) from exc
        raise ExtractionError(
            code="EXTRACTION_MODEL_ERROR",
            message="Gemini extraction failed due to an unexpected model/client error.",
        ) from exc
    finally:
        if owns_client:
            try:
                await genai_client.aio.aclose()
            except Exception:
                # Best-effort cleanup; never fail extraction solely on close.
                pass

    raw_text = getattr(response, "text", None)
    document = _parse_model_response(raw_text, schema)

    result = ExtractionResult(
        file_name=file_name,
        document_type=resolved_type,
        document=document,
    )
    return result.model_dump(mode="json")

"""Document processing and retrieval routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile

from app.api.deps import get_document_api_service
from app.schemas.api import ApiDocumentType, DocumentListResponse, ErrorResponse
from app.schemas.document import DocumentProcessingResult
from app.services.document_api_service import DocumentApiService

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid input or file validation failure"},
    404: {"model": ErrorResponse, "description": "Processed document not found"},
    422: {"model": ErrorResponse, "description": "Unreadable or unprocessable document"},
    500: {"model": ErrorResponse, "description": "Unexpected processing failure"},
    502: {"model": ErrorResponse, "description": "Upstream OCR or extraction provider failure"},
    503: {"model": ErrorResponse, "description": "Required provider configuration is missing"},
}

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/process",
    response_model=DocumentProcessingResult,
    response_model_exclude_none=True,
    summary="Process an uploaded document",
    description=(
        "Validate the upload, run OCR and structured extraction, then apply "
        "deterministic financial checks via the existing orchestration service. "
        "A financial check FAIL is returned inside `validation` and does not by "
        "itself fail `processing_status`."
    ),
    responses={k: v for k, v in ERROR_RESPONSES.items() if k != 404},
)
async def process_document_endpoint(
    file: UploadFile = File(..., description="PDF, JPEG, or PNG document to process"),
    document_type: ApiDocumentType = Form(
        ...,
        description="One of: invoice, balance_sheet, profit_and_loss, cash_flow",
    ),
    service: DocumentApiService = Depends(get_document_api_service),
) -> dict:
    return await service.process_upload(file, document_type)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List processed documents",
    description="Returns persisted processing results in original insertion order.",
)
def list_documents_endpoint(
    response: Response,
    service: DocumentApiService = Depends(get_document_api_service),
) -> dict:
    # Results are overwritten in place when a document is reprocessed. Do not
    # let a browser or intermediary reuse an older validation payload.
    response.headers["Cache-Control"] = "no-store"
    documents = service.list_documents()
    return {"count": len(documents), "documents": documents}


@router.get(
    "/{document_name}",
    response_model=DocumentProcessingResult,
    response_model_exclude_none=True,
    summary="Get a processed document",
    description="Retrieve a previously processed result by its stored document_name.",
    responses={404: ERROR_RESPONSES[404], 500: ERROR_RESPONSES[500]},
)
def get_document_endpoint(
    document_name: str,
    response: Response,
    service: DocumentApiService = Depends(get_document_api_service),
) -> dict:
    response.headers["Cache-Control"] = "no-store"
    return service.get_document(document_name)

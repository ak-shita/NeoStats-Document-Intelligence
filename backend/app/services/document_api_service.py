"""Safe filename handling and API-facing document operations."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import HTTPException, UploadFile

from app.repositories.base import DocumentRepository, DocumentRepositoryError
from app.schemas.api import ApiDocumentType
from app.services.document_service import DocumentProcessingError, process_document

logger = logging.getLogger(__name__)

ProcessDocumentFn = Callable[..., Awaitable[dict[str, Any]]]


def sanitize_filename(filename: str | None) -> str:
    """
    Reduce an upload or lookup name to a single path segment.

    Strips directories to prevent path traversal. Empty or reserved names
    become ``upload``.
    """
    name = Path((filename or "").replace("\\", "/")).name.strip()
    name = name.replace("\x00", "")
    if not name or name in {".", ".."}:
        return "upload"
    if len(name) > 255:
        stem = Path(name).stem[:240]
        suffix = Path(name).suffix[:14]
        name = f"{stem}{suffix}" if suffix else stem[:255]
    return name


class DocumentApiService:
    """Orchestrate processing and persistence for API routes."""

    def __init__(
        self,
        repository: DocumentRepository,
        process_fn: ProcessDocumentFn | None = None,
    ) -> None:
        self._repository = repository
        self._process = process_fn or process_document

    async def process_upload(
        self,
        file: UploadFile,
        document_type: ApiDocumentType | str,
    ) -> dict[str, Any]:
        if file is None or not (file.filename or "").strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "MISSING_FILE",
                    "message": "A file upload is required.",
                },
            )

        safe_name = sanitize_filename(file.filename)
        file.filename = safe_name
        type_value = (
            document_type.value if isinstance(document_type, Enum) else str(document_type)
        )

        try:
            result = await self._process(file, type_value)
        except (DocumentProcessingError, HTTPException):
            raise
        except Exception:
            logger.exception("Unexpected failure while processing document")
            raise DocumentProcessingError(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred.",
                stage="api",
            ) from None
        stored_name = sanitize_filename(str(result.get("document_name") or safe_name))
        result["document_name"] = stored_name
        self._persist(result)
        return result

    def get_document(self, document_name: str) -> dict[str, Any]:
        key = sanitize_filename(document_name)
        try:
            result = self._repository.get(key)
        except DocumentRepositoryError as exc:
            raise self._persistence_error(exc) from None
        if result is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "DOCUMENT_NOT_FOUND",
                    "message": f"No processed document named '{key}' was found.",
                },
            )
        return result

    def list_documents(self) -> list[dict[str, Any]]:
        try:
            return self._repository.list_all()
        except DocumentRepositoryError as exc:
            raise self._persistence_error(exc) from None

    def _persist(self, result: dict[str, Any]) -> None:
        try:
            self._repository.save(result)
        except DocumentRepositoryError as exc:
            raise self._persistence_error(exc) from None

    @staticmethod
    def _persistence_error(exc: DocumentRepositoryError) -> DocumentProcessingError:
        logger.warning("Document persistence failed code=%s", exc.code)
        return DocumentProcessingError(
            code=exc.code,
            message=exc.message,
            stage="persistence",
        )

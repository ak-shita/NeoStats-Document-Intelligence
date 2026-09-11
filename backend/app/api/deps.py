"""FastAPI dependencies for the API layer."""

from __future__ import annotations

from typing import Callable

from fastapi import Depends

from app.core.database import get_session_factory
from app.repositories.base import DocumentRepository
from app.repositories.sqlalchemy_repository import SqlAlchemyDocumentRepository
from app.services.document_api_service import DocumentApiService, ProcessDocumentFn
from app.services.document_service import process_document


def get_document_repository() -> DocumentRepository:
    return SqlAlchemyDocumentRepository(get_session_factory())


def get_process_document_fn() -> ProcessDocumentFn:
    return process_document


def get_document_api_service(
    repository: DocumentRepository = Depends(get_document_repository),
    process_fn: Callable = Depends(get_process_document_fn),
) -> DocumentApiService:
    return DocumentApiService(repository=repository, process_fn=process_fn)

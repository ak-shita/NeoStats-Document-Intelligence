"""SQLAlchemy implementation of DocumentRepository (MySQL in production)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.orm import Session, sessionmaker

from app.models.document import Document
from app.repositories.base import DocumentRepositoryError

logger = logging.getLogger(__name__)


class SqlAlchemyDocumentRepository:
    """
    Persist the complete structured processing result.

    Upserts on ``document_name`` so re-processing replaces the latest result
    while preserving original ``created_at`` (list insertion order).
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(self, result: dict[str, Any]) -> None:
        name = str(result.get("document_name") or "").strip()
        if not name:
            raise DocumentRepositoryError(
                code="INVALID_DOCUMENT_NAME",
                message="document_name is required to persist a result.",
            )

        payload = dict(result)
        document_type = str(payload.get("document_type") or "").strip() or "unknown"
        processing_status = str(payload.get("processing_status") or "").strip() or "PASS"
        overall_confidence = payload.get("overall_confidence")
        if overall_confidence is not None:
            try:
                overall_confidence = float(overall_confidence)
            except (TypeError, ValueError):
                overall_confidence = None

        now = datetime.now(timezone.utc)
        try:
            with self._session_factory() as session:
                row = session.scalar(
                    select(Document).where(Document.document_name == name)
                )
                if row is None:
                    session.add(
                        Document(
                            document_name=name,
                            document_type=document_type,
                            processing_status=processing_status,
                            overall_confidence=overall_confidence,
                            result=payload,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                else:
                    row.document_type = document_type
                    row.processing_status = processing_status
                    row.overall_confidence = overall_confidence
                    row.result = payload
                    flag_modified(row, "result")
                    row.updated_at = now
                session.commit()
        except SQLAlchemyError:
            logger.exception("Failed to persist processed document")
            raise DocumentRepositoryError(
                code="DATABASE_ERROR",
                message="Failed to persist the processed document.",
            ) from None

    def get(self, document_name: str) -> dict[str, Any] | None:
        try:
            with self._session_factory() as session:
                row = session.scalar(
                    select(Document).where(Document.document_name == document_name)
                )
                if row is None:
                    return None
                return dict(row.result)
        except SQLAlchemyError:
            logger.exception("Failed to retrieve processed document")
            raise DocumentRepositoryError(
                code="DATABASE_ERROR",
                message="Failed to retrieve the processed document.",
            ) from None

    def list_all(self) -> list[dict[str, Any]]:
        try:
            with self._session_factory() as session:
                rows = session.scalars(
                    select(Document).order_by(Document.created_at.asc(), Document.id.asc())
                ).all()
                return [dict(row.result) for row in rows]
        except SQLAlchemyError:
            logger.exception("Failed to list processed documents")
            raise DocumentRepositoryError(
                code="DATABASE_ERROR",
                message="Failed to list processed documents.",
            ) from None

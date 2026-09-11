"""In-memory document result store (test double; production uses SQLAlchemy)."""

from __future__ import annotations

from threading import Lock
from typing import Any

from app.repositories.base import DocumentRepositoryError


class InMemoryDocumentRepository:
    """
    Process-local store used by API tests.

    Keyed by ``document_name``. Re-processing the same name replaces the
    previous result and keeps the original list position.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._items: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def save(self, result: dict[str, Any]) -> None:
        name = str(result.get("document_name") or "").strip()
        if not name:
            raise DocumentRepositoryError(
                code="INVALID_DOCUMENT_NAME",
                message="document_name is required to persist a result.",
            )
        with self._lock:
            if name not in self._items:
                self._order.append(name)
            self._items[name] = dict(result)

    def get(self, document_name: str) -> dict[str, Any] | None:
        with self._lock:
            stored = self._items.get(document_name)
            return dict(stored) if stored is not None else None

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(self._items[name]) for name in self._order]

"""Persistence contract for processed document results."""

from __future__ import annotations

from typing import Any, Protocol


class DocumentRepositoryError(Exception):
    """Controlled persistence failure mapped by the API service layer."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class DocumentRepository(Protocol):
    def save(self, result: dict[str, Any]) -> None:
        """Insert or replace a processed document result."""

    def get(self, document_name: str) -> dict[str, Any] | None:
        """Return a stored result, or None when unknown."""

    def list_all(self) -> list[dict[str, Any]]:
        """Return stored results in insertion order."""

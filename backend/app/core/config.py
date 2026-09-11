"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for NeoStats Document Intelligence.

    Secrets must come from the environment (or a local .env file), never
    from source code.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OCR.Space
    ocr_space_api_key: str = ""
    ocr_space_endpoint: str = "https://api.ocr.space/parse/image"
    # Provider free-tier upload ceiling (not an application upload limit).
    ocr_space_max_upload_bytes: int = 1_000_000
    ocr_space_timeout_seconds: float = 120.0
    ocr_space_engine: int = 2

    # Gemini (semantic structured extraction only — no financial math here)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_timeout_seconds: float = 90.0

    # Persistence (MySQL in production; never hardcode credentials)
    database_url: str = ""

    # Comma-separated browser origins allowed to call the API (for example a
    # deployed Render static-site URL). Kept empty locally because the FastAPI
    # app also serves the frontend from the same origin.
    cors_origins: str = ""

    def configured_cors_origins(self) -> list[str]:
        return [origin.strip().rstrip("/") for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

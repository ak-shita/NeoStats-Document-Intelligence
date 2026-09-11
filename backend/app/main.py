from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.errors import register_exception_handlers
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.database import init_db

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(_application: FastAPI):
    settings = get_settings()
    if (settings.database_url or "").strip():
        init_db(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="NeoStats Document Intelligence API",
        version="1.0.0",
        description=(
            "REST API for document upload, OCR, structured extraction, and "
            "deterministic financial validation. Secrets are never returned to clients."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        # Production uses CORS_ORIGINS to allow the deployed static frontend.
        # Same-origin frontend requests do not need a wildcard fallback.
        allow_origins=settings.configured_cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(application)
    application.include_router(api_v1_router)
    if FRONTEND_DIR.is_dir():
        application.mount(
            "/",
            StaticFiles(directory=str(FRONTEND_DIR), html=True),
            name="frontend",
        )
    return application


app = create_app()

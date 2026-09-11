from fastapi import APIRouter

from app.api.v1 import documents, health

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health.router)
api_v1_router.include_router(documents.router)

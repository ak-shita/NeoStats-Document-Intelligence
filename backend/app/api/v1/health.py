from fastapi import APIRouter

from app.schemas.api import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns API liveness for load balancers and operators.",
)
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "document-intelligence-api",
    }

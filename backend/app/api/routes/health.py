from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.health import HealthResponse

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

@router.get(
    "",
    response_model = HealthResponse,
    summary="Check API Health",
    description="Verify that the EvidenceVault AI backend is running."
)
async def health_check() -> HealthResponse:
    settings = get_settings()
    
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


# Later, this endpoint will verify:

# FastAPI status
# PostgreSQL connection
# Qdrant connection
# LLM provider availability
# S3 availability

# For now, it proves that the API itself is running
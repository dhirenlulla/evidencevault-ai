import asyncio

from fastapi import APIRouter, Response, status

from app.clients.qdrant import check_qdrant_connection
from app.core.config import get_settings
from app.db.session import check_database_connection 
from app.schemas.health import HealthResponse, ComponentHealth


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    response_model = HealthResponse,
    summary="Check application and infrastructure health.",
    description=(
        "Verify that the EvidenceVault API, PostgreSQL database, "
        "and Qdrant vector database are available."
    ),
)


async def health_check(response: Response) -> HealthResponse:
    settings = get_settings()
    
    database_result, qdrant_result = await asyncio.gather(
        check_database_connection(),
        check_qdrant_connection(),
    )
    
    database_ok, database_detail = database_result
    qdrant_ok, qdrant_detail = qdrant_result
    
    system_ok = database_ok and qdrant_ok
    
    if not system_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return HealthResponse(
        status="ok" if system_ok else "degraded",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        postgres=ComponentHealth(
            status="ok" if database_ok else "error",
            detail = database_detail
        ),
        qdrant=ComponentHealth(
            status="ok" if qdrant_ok else "error",
            detail=qdrant_detail,
        ),
    )


# Later, this endpoint will verify:

# LLM provider availability
# S3 availability
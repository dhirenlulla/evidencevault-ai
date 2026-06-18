from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.clients.qdrant import close_qdrant_client
from app.db.session import close_database_engine

settings = get_settings()

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """
    Manage resources for the lifetime of the FastAPI application.
    
    Code before 'yield' runs during startup.
    Code after 'yield' runs after shutdown.
    """
    
    print(f"{settings.app_name} v{settings.app_version}")
    
    yield
    
    await close_qdrant_client()
    await close_database_engine()
    
    print(f"Stopping {settings.app_name}")
    

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Backend API for EvidenceVault AI, a production-aware "
        "document intelligence and retrieval-augmented generation platform."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


app.include_router(
    health_router,
    prefix=settings.api_v1_prefix,
)

@app.get(
    "/",
    tags=["Root"],
    summary="API entry point",
)
async def root() -> dict[str, str]:
    return {
        "message": "Welcome to EvidenceVault AI",
        "documentation": "/docs",
        "health": f"{settings.api_v1_prefix}/health"
    }
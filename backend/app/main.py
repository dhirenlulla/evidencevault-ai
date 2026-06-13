from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """
    Handle application startup and shutdown.
    
    Later, startup will initialize database connections,
    the embeddings model, the reranker, and the Qdrant client.
    """
    
    print(f"{settings.app_name} v{settings.app_version}")
    
    yield
    
    print(f"Stopping {settings.app_name}")
    
    
# What lifespan will do later

# FastAPI lifespan management gives us one controlled location for startup and shutdown operations.

# Later, we will use it to:

# Open database connections
# Initialize Qdrant
# Load the embedding model once
# Load the reranker once
# Close connections cleanly

# We do not want to reload a large AI model for every request.

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
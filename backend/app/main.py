import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.middleware import RequestContextMiddleware
from app.core.config import get_settings
from app.core.dependencies import (
    get_embedding_service,
    get_reranker_service,
)
from app.core.logging import configure_logging
from app.clients.qdrant import close_qdrant_client
from app.db.session import close_database_engine
from app.services.local_storage import (
    ensure_upload_directory
)

settings = get_settings()

configure_logging(level=settings.log_level)

logger = logging.getLogger(__name__)


async def _warm_up_models() -> None:
    """ 
    Eagerly load the embedding and reranker models at startup
    instead of waiting for the first real request to trigger
    it, so a freshly booted instance's health check can
    honestly report readiness quickly.

    Failures here are logged but deliberately NOT fatal to
    startup. Model loading can involve downloading weights over
    the network on a first run, which can fail transiently in
    ways a simple retry resolves - crashing the whole
    application over a flaky download would be worse than
    starting up with models that will still load correctly,
    lazily, on first real use. This is a different trade-off
    than the fail-fast configuration validation Phase 10C adds:
    a missing API key is a certain, instant, non-transient
    error; a model download is not.
    """

    embedding_service = get_embedding_service()
    reranker_service = get_reranker_service()

    try:
        await embedding_service.warm_up_async()
        logger.info("embedding model warmed up")

    except Exception:
        logger.exception(
            "embedding model warm-up failed; it will "
            "load lazily on first use instead"
        )

    try:
        await reranker_service.warm_up_async()
        logger.info("reranker model warmed up")

    except Exception:
        logger.exception(
            "reranker model warm-up failed; it will "
            "load lazily on first use instead"
        )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """
    Manage resources for the lifetime of the FastAPI application.
    
    Code before 'yield' runs during startup.
    Code after 'yield' runs after shutdown.
    """
    
    logger.info(
        "starting application",
        extra={
            "app_name": settings.app_name,
            "app_version": settings.app_version,
        },
    )
    
    await _warm_up_models()
    
    yield
    
    await close_qdrant_client()
    await close_database_engine()
    
    logger.info("stopping application")
    

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Backend API for EvidenceVault AI, "
        "a production-aware document intelligence " 
        "and Retrieval-Augmented generation platform."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)


app.add_middleware(RequestContextMiddleware)

app.include_router(
    health_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    documents_router,
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
        "health": (
            f"{settings.api_v1_prefix}/health"
        ),
        "documents": (
            f"{settings.api_v1_prefix}/documents"
        ),
    }
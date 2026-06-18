import logging

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()

qdrant_client = AsyncQdrantClient(
    url=settings.qdrant_url,
    timeout=settings.qdrant_timeout_seconds,
)

async def check_qdrant_connection() -> tuple[bool, str]:
    """ 
    Verify that the Qdrant server responds successfully.
    
    Fetching the collection list is a lightweight operation and proves that
    FastAPI can communicate with Qdrant through its HTTP API.
    """
    
    try:
        await qdrant_client.get_collections()
        return True, "Qdrant connection is available"
    
    except Exception:
        logger.exception("Qdrant health check failed")
        return False, "Qdrant connection unavailable"
    

async def close_qdrant_client() -> None:
    """ 
    Close the Qdrant HTTP client during application shutdown.
    """
    
    await qdrant_client.close()
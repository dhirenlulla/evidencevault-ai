import logging

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings


logger = logging.getLogger(__name__)

_qdrant_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    """
    Return the shared async Qdrant client.

    The client is created lazily so importing this module does not
    immediately create network-related resources.
    """

    global _qdrant_client

    if _qdrant_client is None:
        settings = get_settings()

        _qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout_seconds,
        )

    return _qdrant_client


async def check_qdrant_connection() -> tuple[bool, str]:
    """
    Verify that the Qdrant server responds successfully.

    Fetching the collection list is a lightweight operation and proves
    that FastAPI can communicate with Qdrant through its HTTP API.
    """

    try:
        client = get_qdrant_client()

        await client.get_collections()

        return True, "Qdrant connection is available"

    except Exception:
        logger.exception("Qdrant health check failed")

        return False, "Qdrant connection unavailable"


async def close_qdrant_client() -> None:
    """
    Close the Qdrant HTTP client during application shutdown.
    """

    global _qdrant_client

    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None
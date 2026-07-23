from app.clients.qdrant import get_qdrant_client
from app.core.config import get_settings
from app.services.embedding import (
    get_embedding_service,
)

from app.services.qdrant_search import (
    QdrantSearchService,
)

from app.services.retrieval import (
    RetrievalService,
)

def get_retrieval_service() -> RetrievalService:
    """
    Construct the retrieval service with all required
    dependencies.
    """
    
    settings = get_settings()
    
    return RetrievalService(
        embedding_service=get_embedding_service(),
        qdrant_search_service=QdrantSearchService(
            client=get_qdrant_client(),
        ),
        collection_name=(
            settings.qdrant_collection_name
        )
    )
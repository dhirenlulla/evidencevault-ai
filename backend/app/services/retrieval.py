from app.schemas.retrieval import RetrievalResult
from app.services.embedding import EmbeddingService
from app.services.qdrant_search import QdrantSearchService


class RetrievalService:
    """ 
    Coordinates semantic retrieval for a single document.
    """
    
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        qdrant_search_service: QdrantSearchService,
        collection_name: str,
    ) -> None:
        self.embedding_service = embedding_service
        self.qdrant_search_service = qdrant_search_service
        self.collection_name = collection_name
        
    async def retrieve(
        self,
        *,
        document_id, 
        query: str,
        limit: int = 5,
    ) -> RetrievalResult:
        """ 
        Retrieve the most relevant chunks for a query.
        """
        
        query = query.strip()
        
        if not query:
            raise ValueError(
                "Query cannot be empty."
            )
            
        query_embedding = (
            await self.embedding_service.embed_query_async(query)
        )
        
        query_vector = query_embedding.vector
        
        chunks = (
            await self.qdrant_search_service.search_document(
                document_id=document_id,
                query_vector=query_vector,
                collection_name=self.collection_name,
                limit=limit,
            )
        )
        
        return RetrievalResult(
            query=query,
            total_results=len(chunks),
            chunks=chunks
        )
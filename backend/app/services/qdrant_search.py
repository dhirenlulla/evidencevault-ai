from uuid import UUID
import time

from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.clients.qdrant import get_qdrant_client
from app.schemas.retrieval import RetrievedChunk
from qdrant_client import AsyncQdrantClient

class QdrantSearchService:
    """ 
    Performs semantic vector searches against Qdrant.
    """
    
    def __init__(
        self, 
        client: AsyncQdrantClient | None = None,
        ):
        self.client = client or get_qdrant_client()
        
    async def search_document(
        self,
        *,
        document_id: UUID,
        query_vector: list[float],
        collection_name: str,
        limit: int = 5,
    ) -> tuple[RetrievedChunk, ...]:
        """ 
        Search one document using semantic similarity.
        """
        
        search_result = await self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=str(document_id)
                        ),
                    ),
                ],
            ),
            limit=limit,
            with_payload=True,
        )
        
        chunks: list[RetrievedChunk] = []
        
        for point in search_result.points:
            payload = point.payload
            
            chunks.append(
                RetrievedChunk(
                    chunk_id=UUID(payload["chunk_id"]),
                    document_id=UUID(payload["document_id"]),
                    chunk_index=payload["chunk_index"],
                    page_number=payload["page_number"],
                    page_chunk_index=payload["page_chunk_index"],
                    similarity_score=float(point.score),
                    text=payload["text"],
                    character_count=payload["character_count"],
                    word_count=payload["word_count"],
                )
            )
        return tuple(chunks)
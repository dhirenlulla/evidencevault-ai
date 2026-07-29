from dataclasses import dataclass
from uuid import UUID

from app.schemas.retrieval import RetrievedChunk
from app.services.bm25 import BM25Service
from app.services.retrieval import RetrievalService

@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    """ 
    Results returned by the hybrid retrieval pipeline.
    
    Dense and lexical retrieval results are intentionally
    kept separate. They will be fused later using Reciprocal Rank
    Fusion (RRF).
    """
    
    dense_results: tuple[RetrievedChunk, ...]
    lexical_results: tuple[RetrievedChunk, ...]
    
class HybridRetrievalService:
    """ 
    Coordinates multiple retrieval strategies.

    This phase does not perform ranking fusion.
    It simply executes both retrieval pipelines.
    """
    
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        bm25_service: BM25Service,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._bm25_service = bm25_service
        
    async def retrieve(
        self,
        *,
        document_id: UUID,
        query: str,
    ) -> HybridRetrievalResult:
        """ 
        Execute dense and lexical retrieval independently.
        """
        
        dense_results = await(
            self._retrieval_service.retrieve(
                document_id=document_id,
                query=query,
            )
        )
        
        lexical_results = self._bm25_service.rank(
            query=query,
            documents=[
                chunk.text
                for chunk in dense_results
            ],
        )
        
        return HybridRetrievalResult(
            dense_results=dense_results,
            lexical_results=dense_results,
        )
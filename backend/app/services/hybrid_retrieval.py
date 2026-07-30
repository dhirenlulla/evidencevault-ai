from dataclasses import dataclass
from uuid import UUID

from app.schemas.retrieval import RetrievedChunk
from app.services.bm25 import BM25Service
from app.services.retrieval import RetrievalService
from app.services.rrf import RRFService

@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    """ 
    Results returned by the hybrid retrieval pipeline.
    
    Dense and lexical retrieval results are kept separate for
    transparency and debugging, alongside ``fused_results``: the
    single ranking produced by fusing both with Reciprocal Rank
    Fusion (RRF).
    """
    
    dense_results: tuple[RetrievedChunk, ...]
    lexical_results: tuple[RetrievedChunk, ...]
    fused_results: tuple[RetrievedChunk, ...]
    
class HybridRetrievalService:
    """ 
    Coordinates multiple retrieval strategies and fuses their
    rankings into a single ordered result using Reciprocal Rank
    Fusion (RRF).

    Lexical retrieval is scoped to the same candidate pool dense
    retrieval already returned, rather than searching the full
    document corpus independently. This keeps the service simple
    (no separate BM25 index to maintain) at the cost of lexical
    retrieval being unable to surface a chunk dense retrieval missed
    entirely. Fusion therefore re-ranks a shared candidate set using
    two complementary signals, rather than merging two independent
    candidate sets.
    """
    
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        bm25_service: BM25Service,
        rrf_service: RRFService | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._bm25_service = bm25_service
        self._rrf_service = rrf_service or RRFService()
        
    async def retrieve(
        self,
        *,
        document_id: UUID,
        query: str,
    ) -> HybridRetrievalResult:
        """ 
        Execute dense and lexical retrieval, then fuse both
        rankings into a single ordered result set using RRF.        """
        
        dense_results = await(
            self._retrieval_service.retrieve(
                document_id=document_id,
                query=query,
            )
        )
        
        bm25_results = self._bm25_service.rank(
            query=query,
            documents=[
                chunk.text
                for chunk in dense_results
            ]
        )
        
        lexical_results = tuple(
            dense_results[bm25_results.index]
            for bm25_result in bm25_results
        )
        
        fused_results = self._fuse(
            dense_results=dense_results,
            lexical_results=lexical_results
        )
        
        return HybridRetrievalResult(
            dense_results=dense_results,
            lexical_results=dense_results,
            fused_results=fused_results,
        )
        
    def _fuse(
        self,
        *,
        dense_results: tuple[RetrievedChunk, ...],
        lexical_results: tuple[RetrievedChunk, ...],
    ) -> tuple[RetrievedChunk, ...]:
    
        """ 
        Fuse the dense and lexical rankings by chunk ID using RRF,
        then resolve the fused ID ranking back into RetrievedChunk
        objects.
        """
        
        chunks_by_id = {
            chunk.chunk_id : chunks_by_id
            for chunk in dense_results
        }
        
        dense_ranking = [
            chunk.chunk_id
            for chunk in dense_results
        ]
        
        lexical_ranking = [
            chunk.chunk_id
            for chunk in lexical_results
        ]
        
        fused_scores = self._rrf_service.fuse(
            rankings=[
                dense_ranking,
                lexical_ranking,
            ],
        )
        
        return tuple(
            chunks_by_id[fused_score.item]
            for fused_score in fused_scores
        )
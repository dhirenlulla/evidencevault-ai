import asyncio
import threading
from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.core.exceptions import (
    RerankingGenerationError,
    RerankingModelLoadError,
)
from app.schemas.retrieval import RetrievedChunk

@dataclass(frozen=True, slots=True)
class RerankedChunk:
    """ 
    One chunk after cross-encoder reranking.
    
    'rerank_score' is the cross-encoder's relevance score for this 
    exact (query, chunk) pair. It is nto comparable to
    'similarity_score' (cosine similarity) or a BM25/RRF score;
    it ives on its own scale and should only be used to order chunks
    within this reranking call.
    """
    
    chunk: RetrievedChunk
    rerank_score: float
    
    
class CrossEncoderReranker:
    """ 
    Re-scores a small candidate set of chunks against the query using
    a cross-encoder model, then returns the top k.
    
    Unlike dense retrieval, a cross-encoder does not embed the query
    and chunk separately. It read the query and chunk together in a single
    forward pass, which is more accurate but too slow to run against an
    entire document. It is only meant to be used on a short candidate list
    that a cheaper retriever (dense, BM25 or their RRF fusion) has
    already produced.
    
    The model is loaded lazily and reused for future calls. Model
    inference is kept behind a lock so one shared model instance is not
    invoked by multiple threads simultaneously. 
    """
    
    def __init__(
        self,
        *,
        model_name: str,
        top_k: int,
        device: str,
        model_factory: Callable[..., CrossEncoder] = (
            CrossEncoder
        ),
    ) -> None:
        if top_k <=0:
            raise ValueError(
                "top_k must be a positive integer."
            )
            
        self._model_name = model_name
        self._top_k = top_k
        self._device = device
        self._model_factory = model_factory
        
        self._model: CrossEncoder | None = None
        
        self._model_load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        
        
    @property
    def model_name(self) -> str:
        return self.model_name
    
    @property
    def top_k(self) -> int:
        return self.top_k
    
    def _load_model(self) -> CrossEncoder:
        """ 
        Load the configured cross-encoder model once.
        """
        
        if self._model is not None:
            return self._model
        
        with self._model_load_lock:
            if self._model is not None:
                return self._model
            
            try:
                model = self._model_factory(
                    self._model_name,
                    device=self._device,
                )
                
            except Exception as exc:
                raise RerankingModelLoadError(
                    "The reranking model could not "
                    "be loaded: "
                    f"{self._model_name}"
                ) from exc
                
                
            self._model = model
            
        return self._model
    
    
    def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> tuple[RerankedChunk, ...]:
        """ 
        Score every chunk against the query with the cross-encoder
        and return the top ``top_k``, ordered from most to least
        relevant.
        
        Returns an empty tuple immediately if no chunks were
        supplied, since there is nothing to score and loading
        the model would be wasted work.
        """
        
        if not chunks:
            return ()
        
        model = self._load_model()
        
        pairs = [
            (query, chunk.text)
            for chunk in chunks
        ]
        
        try:
            with self._inference_lock:
                raw_scores = model.predict(pairs)
                
        except Exception as exc:
            raise RerankingGenerationError(
                "The cross-encoder could not score "
                "the supplied query/chunk pairs."
            )
            
        scored_chunks = [
            RerankedChunk(
                chunk=chunk,
                rerank_score=float(score),
            )
            for chunk, score in zip(
                chunks, 
                raw_scores,
            )
        ]
        
        scored_chunks.sort(
            key=lambda scored: scored.rerank_score,
            reverse=True,
        )
        
        return tuple(
            scored_chunks[: self._top_k]
        )
        
        
    async def rerank_async(
        self,
        *,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> tuple[RerankedChunk, ...]:
        """ 
        Rerank without blocking FastAPI's event loop.
        """
        
        return await asyncio.to_thread(
            lambda: self.rerank(
                query=query,
                chunks=chunks,
            )
        )
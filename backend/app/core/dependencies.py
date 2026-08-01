from functools import lru_cache

from app.clients.qdrant import get_qdrant_client
from app.core.config import get_settings
from app.services.embedding import (
    get_embedding_service,
)

from app.services.reranker import (
    CrossEncoderReranker,
)

from app.services.qdrant_search import (
    QdrantSearchService,
)

from app.services.retrieval import (
    RetrievalService,
)
from app.services.bm25 import BM25Service
from app.services.rrf import RRFService
from app.services.hybrid_retrieval import HybridRetrievalService
from app.services.generation import GenerationService
from app.services.groq_llm import GroqLLMService
from app.services.prompt_builder import PromptBuilder
from app.services.evaluation import (
    RetrievalEvaluator,
    RetrievalMetricsAggregator,
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
 
    
def get_rrf_service() -> RRFService:
    """ 
    Construct the Reciprocal Rank Fusion service using the configured
    smoothing constant.
    """
    settings = get_settings()
    
    return RRFService(
        k=settings.rrf_k
    )
    
    
def get_hybrid_retrieval_service() -> HybridRetrievalService:
    """ 
    Construct the hybrid retrieval service, combining dense
    retrieval, lexical retrieval, and RRF fusion.
    """
    
    return HybridRetrievalService(
        retrieval_service=get_retrieval_service(),
        bm25_service=BM25Service(),
        rrf_service=get_rrf_service(),
    )


def get_reranker_service() -> CrossEncoderReranker:
    """ 
    Construct the cross-encoder reranking service.
    
    Cached because loading the cross-encoder model is expensive;
    this must not happen on every request
    """
    
    settings = get_settings()
    
    return _cached_reranker_service(
        model_name=settings.reranker_model_name,
        top_k=settings.reranker_top_k,
        device=settings.reranker_device,
    )
    
@lru_cache
def _cached_reranker_service(
    *,
    model_name: str,
    top_k: int,
    device: str,
) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model_name=model_name,
        top_k=top_k,
        device=device,
    )
    
    
def get_retrieval_evaluator() -> RetrievalEvaluator:
    """ 
    Construct the retrieval evaluation service.
    
    Stateless - holds no model, no settings, no connections - so a
    new instance is effectively free. No caching needed.
    """
    
    return RetrievalEvaluator()


def get_retrieval_metrics_aggregator() -> (
    RetrievalMetricsAggregator
):
    """ 
    Construct the metrics aggregation service. Also stateless.
    """
    
    return RetrievalMetricsAggregator()

    
def get_generation_service() -> GenerationService:
    """ 
    Construct the complete RAG generation service.
    """
    
    return GenerationService(
        retrieval_service=get_retrieval_service(),
        prompt_builder=PromptBuilder(),
        llm_service=GroqLLMService(),
    )
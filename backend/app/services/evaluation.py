from collections.abc import Sequence, Set
from dataclasses import dataclass
from math import log2
from typing import Hashable, TypeVar

T = TypeVar("T", bound=Hashable)

@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """ 
    Retrieval quality metrics for one single query.
    
    All metrics are computed against the same top-K ranked list
    that was actually retrieved, so they can be read together as one
    picture of that query's retrieval quality.
    """
    
    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: float    # normalized discounted cumulative gain
    k: int
    
@dataclass(frozen=True, slots=True)
class AggregateRetrievalMetrics:
    """ 
    Retrieve metrics averaged across many queries.
    """
    
    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    num_queries: int
    
    
class RetrievalEvaluator:
    """ 
    Computes standard information-retrieval quality metrics 
    (Recall@K, Precision@K, Hit Rate@K, MRR, NDCG@K) for a single 
    query's ranked retrieval result against known-relevant items.
    
    Works on plain, hashable identifiers (for example chunk UUIDs)
    rather than RetrievedChunk objects, so it has no dependency on
    retrieval, embeddings, or the database. Anything that can produce
    a ranked list of IDs can evaluated with this class.
    
    Relevance is treated as binary: an item is either relevant or it
    is not. This intentionally does not model graded relevance (for
    example "somewhat relevant), which keeps the metrics simple and 
    matches grounded-truth labels are usually collected for a project
    like this.
    """
    
    def evaluate(
        self,
        *,
        retrieved_ids: Sequence[T],
        relevant_ids: Set[T],
        k: int,
    ) -> RetrievalMetrics:
        """ 
        Score one query's ranked retrieval result.
        
        'retrieved_ids' should already be ordered from most to least
        relevant, exactly as the retriever returned them.
        Only the first 'k' items are considered.
        """
        
        if k <= 0:
            raise ValueError(
                "k must be a positive integer."
            )
        top_k = tuple(retrieved_ids[:k])
        
        relevant_positions = [
            position
            for position, item in enumerate(
                top_k,
                start=1,
            )
            if item in relevant_ids
        ]
        
        num_relevant_retrieved = len(relevant_positions)
        
        recall_at_k = (
            num_relevant_retrieved / len(relevant_ids)
            if relevant_ids
            else 0.0
        )
        
        precision_at_k = (
            num_relevant_retrieved / len(top_k)
            if top_k
            else 0.0
        )
        
        hit_rate_at_k = (
            1.0
            if num_relevant_retrieved > 0
            else 0.0
        )
        
        mrr = (
            1.0 / relevant_positions[0]
            if relevant_positions
            else 0.0
        )
        
        ndcg_at_k = self._ndcg_at_k(
            top_k=top_k,
            relevant_ids=relevant_ids,
            k=k,
        )
        
        return RetrievalMetrics(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            hit_rate_at_k=hit_rate_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            k=k,
        )
        
    def _ndcg_at_k(
        self,
        *,
        top_k: Sequence[T],
        relevant_ids: Set[T],
        k: int,
    ) -> float:
        """ 
        Normalized Discounted Cumulative Gain with binary relevance.
        Each relevant item at position 'i' (1-indexed) contribute
        '1 / log2(i + 1)' to the gain,
        so items near the top of the ranking count more than items
        near the bottom.
        
        The raw (DCG) is normalized by the best possible gain (IDCG):
        the gain if every relevant item, up to k of them, has been
        ranked first. This keeps the score in [0, 1] regardless of
        how many relevant items exist in total.
        """
        
        dcg = sum(
            1.0 / log2(position + 1)
            for position, item in enumerate(
                top_k,
                start=1,
            )
            if item in relevant_ids
        )
        
        ideal_hits = min(
            len(relevant_ids),
            k,
        )
        
        idcg = sum(
            1.0 / log2(position + 1)
            for position in range(1, ideal_hits + 1)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    
class RetrievalMetricsAggregator:
    """ 
    Averages RetrievalMetrics across multiple queries into a single
    AggregateRetrievalMetrics.
    
    A single query's metrics are noisy (one lucky or unlucky query says
    little about the retriever overall). Averaging across a representative
    query set is what makes retrieval quality measurable and comparable
    between configuration.
    """
    
    def aggregate(
        self,
        *,
        results: Sequence[RetrievalMetrics],
    ) -> AggregateRetrievalMetrics:
        if not results:
            raise ValueError(
                "Cannot aggregate an empty list of results."
            )
            
        k_values = {
            result.k
            for result in results
        }
        
        if len(k_values) > 1:
            raise ValueError(
                "All results must share the same k "
                "to be aggregated together."
            )
            
        count = len(results)
        
        return AggregateRetrievalMetrics(
            recall_at_k=(
                sum(r.recall_at_k for r in results)
                / count
            ),
            precision_at_k=(
                sum(r.precision_at_k for r in results)
                / count
            ),
            hit_rate_at_k=(
                sum(r.hit_rate_at_k for r in results)
                / count
            ),
            mrr = (
                sum(r.mrr for r in results) / count
            ),
            ndcg_at_k=(
                sum(r.ndcg_at_k for r in results)
                / count
            ),
            k=k_values.pop(),
            num_queries=count,
        )
        
        
@dataclass(frozen=True, slots=True)
class MetricComparison:
    """ 
    The change in each metric between a baseline configuration and a 
    candidate configuration, expressed as
    'candidate - baseline'.
    
    A positive value means the candidate improved on that metric;
    a negative value means it regressed.
    """
    
    recall_at_k_delta: float
    precision_at_k_delta: float
    hit_rate_at_k_delta: float
    mrr_delta: float
    ndcg_at_k_delta: float
    
def compare_metrics(
    *,
    baseline: AggregateRetrievalMetrics,
    candidate: AggregateRetrievalMetrics,
) -> MetricComparison:
    """ 
    Compare a candidate retrieval configuration against a baseline,
    metric by metric.
    
    Both inputs must have been evaluated at the same k and over the
    same query set to be meaningfully compared; this function does not
    attempt to detect that on its own, since it has no knowledge of
    which queries produced the aggregate.
    """
    
    return MetricComparison(
        recall_at_k_delta=(
            candidate.recall_at_k
            - baseline.recall_at_k
        ),
        precision_at_k_delta=(
            candidate.precision_at_k
            - baseline.precision_at_k
        ),
        hit_rate_at_k_delta=(
            candidate.hit_rate_at_k
            - baseline.hit_rate_at_k
        ),
        mrr_delta=(
            candidate.mrr - baseline.mrr
        ),
        ndcg_at_k_delta=(
            candidate.ndcg_at_k
            - baseline.ndcg_at_k
        ),
    )
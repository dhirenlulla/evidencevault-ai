from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

T = TypeVar("T", bound=Hashable)

@dataclass(frozen=True)
class RRFScore(Generic(T)):
    """ 
    One fused ranking result.
    
    ``item`` is whatever identifier was ranked (for
    example a chunk UUID). ``score`` is the combine
    Reciprocal Rank Fusion score across every ranking
    the item appeared in. Higher is better
    """
    
    item: T
    score: float
    
class RRFService:
    """
    Fuses multiple ranked lists into a single ranking using
    Reciprocal Rank Fusion (RRF).
    
    RRF combines rankings, not raw scores. This makes it robust to
    retrieval methods that produce scores on incompatible scales,
    such as cosine similarity from dense retrieval and BM25 scores
    from lexical retrieval. An item's fused score only depends on
    its position in each input ranking, never on the underlying
    similarity or relevance score.
    """
    
    DEFAULT_K = 60
    
    def __init__(self, *, k: int = DEFAULT_K) -> None:
        """ 
        ``k`` is the smoothing constant from the RRF formula
        ``1 / (k + rank)``. Larger values of k reduce the influence of 
        item's exact rank position, flattening the score differences
        between top-ranked items. The RRF literature commonly uses k=60,
        which works well without per-dataset tuning.
        """
        
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        
        self.k = k
        
    def fuse(
        self,
        *,
        rankings: Sequence[Sequence[T]],
    ):
        """ 
        Fuse multiple ranked lists into one ranking.
        
        Each ranking is a sequence of items ordered from most relevant 
        to least relevant. An item may appear in some, all or none of the 
        ranking. Items are scored by summing 1 / (k + rank) across every
        ranking that contains them, so an item ranked highly in multiple lists
        scores higher than an item ranked highly in only one.
        
        Returns items sorted by descending fused score. Items that do not appear
        in any ranking are never included, since there is nothing to fuse for them.
        """
        
        if not rankings:
            return ()
        
        scores: dict[T, float] = {}
        
        for ranking in rankings:
            for position, item in enumerate(ranking, start=1):
                contribution = 1.0 / (self.k + position)
                scores[item] = scores.get(item, 0.0) + contribution
                
        ranked_items = sorted(
            scores.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )
        
        return tuple(
            RRFScore(item=item, score=score)
            for item, score in ranked_items
        )
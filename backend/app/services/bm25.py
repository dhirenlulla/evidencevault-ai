from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

@dataclass(frozen=True, slots=True)
class BM25Result:
    """ 
    One BM25 search result.
    """
    
    index: int
    score: float
    
class BM25Service:
    """ 
    Lexical retrieval using BM25.
    
    The service ranks a collection of text documents
    according to lexical similarity with the query.
    """
    
    @staticmethod
    def _tokenize(
        text:str,
    ) -> list[str]:
        """ 
        Convert text into lowercase whitespace tokens.
        """
        
        return text.lower().split()
    
    def rank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_k: int = 5,
    ) -> tuple[BM25Result, ...]:
        """ 
        Rank documents using BM25.
        """
        
        if not documents:
            return ()
        
        corpus = [
            self._tokenize(document)
            for document in documents
        ]
        
        bm25 = BM25Okapi(corpus)
        
        query_tokens = self._tokenize(query)
        
        scores = bm25.get_scores(query_tokens)
        
        ranked = sorted(
            enumerate(scores),
            key=lambda item : item[1],
            reverse=True,
        )
        
        results = [
            BM25Result(
                index=index,
                score=float(score),
            )
            for index, score in ranked[:top_k]
        ]
        
        return tuple(results)
import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.hybrid_retrieval import (
    HybridRetrievalResult,
    HybridRetrievalService,
)
from app.schemas.retrieval import RetrievedChunk, RetrievalResult
from app.services.bm25 import BM25Result
from app.services.rrf import RRFService


def build_chunk(
    *,
    index: int,
    text: str,
) -> RetrievedChunk:
    """
    Create one deterministic RetrievedChunk.
    """

    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=index,
        page_number=1,
        page_chunk_index=index,
        similarity_score=0.95,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
    )


class FakeRetrievalService:
    """
    Fake dense retrieval service.
    """

    def __init__(
        self,
        result,
    ):
        self.result = result
        self.calls = []

    async def retrieve(
        self,
        *,
        document_id,
        query,
    ):
        self.calls.append(
            {
                "document_id": document_id,
                "query": query,
            }
        )

        return RetrievalResult(
            query=query,
            total_results=len(self.result),
            chunks=tuple(self.result),
        )


class FakeBM25Service:
    """
    Fake BM25 service.
    """

    def __init__(
        self,
        result,
    ):
        self.result = result
        self.calls = []

    def rank(
        self,
        *,
        query,
        documents,
    ):
        self.calls.append(
            {
                "query": query,
                "documents": documents,
            }
        )

        return self.result


def build_service(
    *,
    dense_result,
    lexical_result,
):
    retrieval = FakeRetrievalService(
        dense_result,
    )

    bm25 = FakeBM25Service(
        lexical_result,
    )

    service = HybridRetrievalService(
        retrieval_service=retrieval,
        bm25_service=bm25,
    )

    return (
        service,
        retrieval,
        bm25,
    )


def test_dense_retrieval_is_called() -> None:
    """
    Hybrid retrieval should invoke
    dense retrieval exactly once.
    """

    chunk = build_chunk(
        index=0,
        text="FastAPI dependency injection",
    )

    service, retrieval, _ = build_service(
        dense_result=(chunk,),
        lexical_result=[],
    )

    document_id = uuid4()

    asyncio.run(
        service.retrieve(
            document_id=document_id,
            query="dependency injection",
        )
    )

    assert len(retrieval.calls) == 1

    assert (
        retrieval.calls[0]["document_id"]
        == document_id
    )


def test_query_is_forwarded_to_dense_retrieval() -> None:
    """
    Query should be forwarded unchanged.
    """

    chunk = build_chunk(
        index=0,
        text="Chunk",
    )

    service, retrieval, _ = build_service(
        dense_result=(chunk,),
        lexical_result=[],
    )

    asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="What is FastAPI?",
        )
    )

    assert (
        retrieval.calls[0]["query"]
        == "What is FastAPI?"
    )


def test_bm25_receives_dense_chunk_text() -> None:
    """
    BM25 should receive dense chunk text.
    """

    chunks = (
        build_chunk(
            index=0,
            text="First chunk",
        ),
        build_chunk(
            index=1,
            text="Second chunk",
        ),
    )

    service, _, bm25 = build_service(
        dense_result=chunks,
        lexical_result=[],
    )

    asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="chunk",
        )
    )

    assert (
        bm25.calls[0]["documents"]
        == [
            "First chunk",
            "Second chunk",
        ]
    )


def test_query_is_forwarded_to_bm25() -> None:
    """
    Query should be passed to BM25.
    """

    chunk = build_chunk(
        index=0,
        text="Chunk",
    )

    service, _, bm25 = build_service(
        dense_result=(chunk,),
        lexical_result=[],
    )

    asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="semantic search",
        )
    )

    assert (
        bm25.calls[0]["query"]
        == "semantic search"
    )


def test_result_type_is_hybrid_result() -> None:
    """
    Service should return HybridRetrievalResult.
    """

    chunk = build_chunk(
        index=0,
        text="Chunk",
    )

    service, _, _ = build_service(
        dense_result=(chunk,),
        lexical_result=[],
    )

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="query",
        )
    )

    assert isinstance(
        result,
        HybridRetrievalResult,
    )


def test_dense_results_are_preserved() -> None:
    """
    Dense retrieval output should be preserved.
    """

    chunk = build_chunk(
        index=0,
        text="Chunk",
    )

    service, _, _ = build_service(
        dense_result=(chunk,),
        lexical_result=[],
    )

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="query",
        )
    )

    assert result.dense_results == (chunk,)


def test_dense_failure_is_propagated() -> None:
    """
    Dense retrieval exceptions should
    propagate unchanged.
    """

    class BrokenRetrieval:
        async def retrieve(
            self,
            **kwargs,
        ):
            raise RuntimeError(
                "dense failed"
            )

    service = HybridRetrievalService(
        retrieval_service=BrokenRetrieval(),
        bm25_service=FakeBM25Service([]),
    )

    with pytest.raises(
        RuntimeError,
        match="dense failed",
    ):
        asyncio.run(
            service.retrieve(
                document_id=uuid4(),
                query="query",
            )
        )


def test_bm25_failure_is_propagated() -> None:
    """
    BM25 exceptions should propagate.
    """

    class BrokenBM25:
        def rank(
            self,
            **kwargs,
        ):
            raise RuntimeError(
                "bm25 failed"
            )

    chunk = build_chunk(
        index=0,
        text="Chunk",
    )

    service = HybridRetrievalService(
        retrieval_service=FakeRetrievalService(
            (chunk,)
        ),
        bm25_service=BrokenBM25(),
    )

    with pytest.raises(
        RuntimeError,
        match="bm25 failed",
    ):
        asyncio.run(
            service.retrieve(
                document_id=uuid4(),
                query="query",
            )
        )


# ----------------------------------------------------------------
# Phase 4: lexical_results correctness and RRF fusion
# ----------------------------------------------------------------


def test_lexical_results_reflect_bm25_ranking() -> None:
    """
    lexical_results should be the dense chunks reordered according
    to the BM25 ranking, not a copy of dense_results.
    """

    chunks = (
        build_chunk(index=0, text="First chunk"),
        build_chunk(index=1, text="Second chunk"),
    )

    bm25_result = (
        BM25Result(index=1, score=5.0),
        BM25Result(index=0, score=1.0),
    )

    service, _, _ = build_service(
        dense_result=chunks,
        lexical_result=bm25_result,
    )

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="chunk",
        )
    )

    assert result.lexical_results == (chunks[1], chunks[0])


def test_lexical_results_empty_when_bm25_returns_nothing() -> None:
    """
    If BM25 returns no matches, lexical_results should be an empty
    tuple rather than falling back to dense_results.
    """

    chunk = build_chunk(index=0, text="Chunk")

    service, _, _ = build_service(
        dense_result=(chunk,),
        lexical_result=(),
    )

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="query",
        )
    )

    assert result.lexical_results == ()


def test_result_includes_fused_results_field() -> None:
    """
    HybridRetrievalResult should expose a fused_results field
    produced by RRF.
    """

    chunk = build_chunk(index=0, text="Chunk")

    service, _, _ = build_service(
        dense_result=(chunk,),
        lexical_result=(BM25Result(index=0, score=1.0),),
    )

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="query",
        )
    )

    assert result.fused_results == (chunk,)


def test_fused_results_ranks_chunk_found_by_both_signals_first() -> None:
    """
    A chunk found by both dense and lexical retrieval should be
    fused ahead of a chunk found only by dense retrieval, because
    it accumulates a score contribution from each ranking.
    """

    dense_only_chunk = build_chunk(index=0, text="dense only chunk")
    shared_chunk = build_chunk(index=1, text="shared chunk")

    # Dense retrieval ranks dense_only_chunk first, shared_chunk
    # second.
    dense_result = (dense_only_chunk, shared_chunk)

    # BM25 only surfaces shared_chunk (dense index 1); it does not
    # rank dense_only_chunk at all.
    lexical_result = (
        BM25Result(index=1, score=9.0),
    )

    service, _, _ = build_service(
        dense_result=dense_result,
        lexical_result=lexical_result,
    )

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="query",
        )
    )

    # shared_chunk earns a fusion contribution from both rankings,
    # so it outranks dense_only_chunk despite dense retrieval alone
    # having ranked it lower.
    assert result.fused_results[0] == shared_chunk


def test_fused_results_include_dense_only_chunks() -> None:
    """
    A chunk found only by dense retrieval (absent from the BM25
    ranking) should still appear in fused_results.
    """

    chunk = build_chunk(index=0, text="Only found by dense retrieval")

    service, _, _ = build_service(
        dense_result=(chunk,),
        lexical_result=(),
    )

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="query",
        )
    )

    assert chunk in result.fused_results


def test_fused_results_empty_when_dense_results_empty() -> None:
    """
    With no dense candidates at all, fusion has nothing to rank
    and should return an empty tuple.
    """

    service, _, _ = build_service(
        dense_result=(),
        lexical_result=(),
    )

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="query",
        )
    )

    assert result.fused_results == ()


def test_custom_rrf_service_is_used_when_provided() -> None:
    """
    HybridRetrievalService should use an injected RRFService
    instead of constructing a default one, honoring dependency
    injection.
    """

    chunk = build_chunk(index=0, text="Chunk")

    retrieval = FakeRetrievalService((chunk,))
    bm25 = FakeBM25Service((BM25Result(index=0, score=1.0),))

    calls = []

    class SpyRRFService(RRFService):
        def fuse(self, *, rankings):
            calls.append(rankings)
            return super().fuse(rankings=rankings)

    service = HybridRetrievalService(
        retrieval_service=retrieval,
        bm25_service=bm25,
        rrf_service=SpyRRFService(),
    )

    asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="query",
        )
    )

    assert len(calls) == 1
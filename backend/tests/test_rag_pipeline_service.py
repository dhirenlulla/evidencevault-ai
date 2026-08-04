import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.schemas.answer import AnswerComplete, AnswerToken
from app.schemas.retrieval import RetrievedChunk
from app.services.hybrid_retrieval import (
    HybridRetrievalResult,
)
from app.services.rag_pipeline import RAGPipelineService
from app.services.reranker import RerankedChunk


def build_chunk(
    *,
    document_id,
    index: int = 0,
    text: str = "Evidence text",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=document_id,
        chunk_index=index,
        page_number=index + 1,
        page_chunk_index=0,
        similarity_score=0.9,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
    )


async def fake_token_stream(fragments):
    for fragment in fragments:
        yield fragment


def build_pipeline(
    *,
    fused_chunks,
    reranked_chunks,
    stream_fragments=("Hello", " world."),
):
    """
    Build a RAGPipelineService with every composed service
    faked, so no real retrieval, reranking, or LLM call ever
    happens.
    """

    hybrid_retrieval_service = Mock()
    hybrid_retrieval_service.retrieve = AsyncMock(
        return_value=HybridRetrievalResult(
            dense_results=fused_chunks,
            lexical_results=fused_chunks,
            fused_results=fused_chunks,
        )
    )

    reranker = Mock()
    reranker.rerank_async = AsyncMock(
        return_value=tuple(
            RerankedChunk(chunk=chunk, rerank_score=score)
            for chunk, score in reranked_chunks
        )
    )

    prompt_builder = Mock()
    prompt_builder.system_prompt.return_value = "SYSTEM"
    prompt_builder.user_prompt.return_value = "USER"

    llm_service = Mock()
    llm_service.model_name = "test-model"
    llm_service.generate_answer = AsyncMock(
        return_value="Final answer."
    )
    llm_service.generate_answer_stream = Mock(
        return_value=fake_token_stream(stream_fragments)
    )

    service = RAGPipelineService(
        hybrid_retrieval_service=hybrid_retrieval_service,
        reranker=reranker,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )

    return (
        service,
        hybrid_retrieval_service,
        reranker,
        prompt_builder,
        llm_service,
    )


# ----------------------------------------------------------------
# generate_answer (non-streaming)
# ----------------------------------------------------------------


def test_reranker_receives_fused_results_not_dense_results() -> None:
    """
    The reranker must be given the RRF-fused candidates, not
    raw dense-only results -- reranking the wrong candidate
    set would silently defeat the point of hybrid retrieval.
    """

    document_id = uuid4()

    dense_only_chunk = build_chunk(
        document_id=document_id,
        index=0,
        text="dense only",
    )
    fused_chunk = build_chunk(
        document_id=document_id,
        index=1,
        text="fused chunk",
    )

    (
        service,
        hybrid_retrieval_service,
        reranker,
        _,
        _,
    ) = build_pipeline(
        fused_chunks=(fused_chunk,),
        reranked_chunks=[(fused_chunk, 1.0)],
    )

    asyncio.run(
        service.generate_answer(
            document_id=document_id,
            query="question",
        )
    )

    _, kwargs = reranker.rerank_async.call_args

    assert kwargs["chunks"] == (fused_chunk,)


def test_prompt_uses_reranked_order_not_fusion_order() -> None:
    """
    The prompt should be built from the reranked ordering, not
    the pre-rerank fusion ordering -- reranking is pointless if
    its output order is discarded before prompting.
    """

    document_id = uuid4()

    chunk_a = build_chunk(
        document_id=document_id, index=0, text="chunk a"
    )
    chunk_b = build_chunk(
        document_id=document_id, index=1, text="chunk b"
    )

    (
        service,
        _,
        _,
        prompt_builder,
        _,
    ) = build_pipeline(
        # Fusion order: a, b
        fused_chunks=(chunk_a, chunk_b),
        # Rerank flips it: b, a
        reranked_chunks=[
            (chunk_b, 0.9),
            (chunk_a, 0.4),
        ],
    )

    asyncio.run(
        service.generate_answer(
            document_id=document_id,
            query="question",
        )
    )

    retrieval_result = (
        prompt_builder.user_prompt.call_args[0][0]
    )

    assert retrieval_result.chunks == (chunk_b, chunk_a)


def test_answer_response_carries_reranked_chunks() -> None:
    document_id = uuid4()

    chunk = build_chunk(document_id=document_id)

    (
        service,
        _,
        _,
        _,
        llm_service,
    ) = build_pipeline(
        fused_chunks=(chunk,),
        reranked_chunks=[(chunk, 1.0)],
    )

    result = asyncio.run(
        service.generate_answer(
            document_id=document_id,
            query="question",
        )
    )

    assert result.answer == "Final answer."
    assert result.model == "test-model"
    assert result.total_sources == 1
    assert result.grounded is True
    assert result.retrieved_chunks == (chunk,)


def test_answer_is_ungrounded_when_reranker_returns_nothing() -> None:
    document_id = uuid4()

    chunk = build_chunk(document_id=document_id)

    (
        service,
        _,
        _,
        _,
        _,
    ) = build_pipeline(
        fused_chunks=(chunk,),
        reranked_chunks=[],
    )

    result = asyncio.run(
        service.generate_answer(
            document_id=document_id,
            query="question",
        )
    )

    assert result.grounded is False
    assert result.total_sources == 0
    assert result.retrieved_chunks == ()


# ----------------------------------------------------------------
# generate_answer_stream
# ----------------------------------------------------------------


async def collect_stream(service, **kwargs):
    return [
        event
        async for event in service.generate_answer_stream(
            **kwargs
        )
    ]


def test_stream_yields_tokens_then_one_complete_event() -> None:
    document_id = uuid4()

    chunk = build_chunk(document_id=document_id)

    (
        service,
        _,
        _,
        _,
        _,
    ) = build_pipeline(
        fused_chunks=(chunk,),
        reranked_chunks=[(chunk, 1.0)],
        stream_fragments=("a", "b", "c"),
    )

    events = asyncio.run(
        collect_stream(
            service,
            document_id=document_id,
            query="question",
        )
    )

    event_types = [type(e).__name__ for e in events]

    assert event_types == [
        "AnswerToken",
        "AnswerToken",
        "AnswerToken",
        "AnswerComplete",
    ]
    assert [t.text for t in events[:3]] == [
        "a",
        "b",
        "c",
    ]


def test_stream_complete_event_carries_reranked_chunks() -> None:
    document_id = uuid4()

    chunk = build_chunk(document_id=document_id)

    (
        service,
        _,
        _,
        _,
        _,
    ) = build_pipeline(
        fused_chunks=(chunk,),
        reranked_chunks=[(chunk, 1.0)],
    )

    events = asyncio.run(
        collect_stream(
            service,
            document_id=document_id,
            query="question",
        )
    )

    complete = events[-1]

    assert isinstance(complete, AnswerComplete)
    assert complete.retrieved_chunks == (chunk,)
    assert complete.total_sources == 1
    assert complete.model == "test-model"
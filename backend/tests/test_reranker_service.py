import asyncio
from uuid import uuid4

import pytest

from app.core.exceptions import (
    RerankingGenerationError,
    RerankingModelLoadError,
)
from app.schemas.retrieval import RetrievedChunk
from app.services.reranker import (
    CrossEncoderReranker,
    RerankedChunk,
)


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
        similarity_score=0.5,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
    )


class FakeCrossEncoder:
    """
    Fake cross-encoder model. Returns pre-configured scores
    instead of running real inference, so tests are fast and
    do not need a real ML model.
    """

    def __init__(self, scores):
        self.scores = scores
        self.predict_calls = []

    def predict(self, pairs):
        self.predict_calls.append(pairs)
        return self.scores


def build_reranker(
    *,
    scores,
    top_k=5,
):
    fake_model = FakeCrossEncoder(scores)

    reranker = CrossEncoderReranker(
        model_name="fake-model",
        top_k=top_k,
        device="cpu",
        model_factory=lambda *args, **kwargs: fake_model,
    )

    return reranker, fake_model


def test_empty_chunks_returns_empty_tuple_without_loading_model() -> None:
    """
    Reranking an empty chunk list should short-circuit and never
    touch the model at all.
    """

    calls = []

    def spy_factory(*args, **kwargs):
        calls.append(1)
        return FakeCrossEncoder([])

    reranker = CrossEncoderReranker(
        model_name="fake-model",
        top_k=5,
        device="cpu",
        model_factory=spy_factory,
    )

    result = reranker.rerank(query="query", chunks=[])

    assert result == ()
    assert calls == []


def test_query_and_chunk_text_are_paired_for_the_model() -> None:
    """
    The model should receive (query, chunk_text) pairs, one per
    chunk, in the order the chunks were supplied.
    """

    chunk_a = build_chunk(index=0, text="First chunk")
    chunk_b = build_chunk(index=1, text="Second chunk")

    reranker, fake_model = build_reranker(scores=[0.1, 0.9])

    reranker.rerank(
        query="my query",
        chunks=[chunk_a, chunk_b],
    )

    assert fake_model.predict_calls[0] == [
        ("my query", "First chunk"),
        ("my query", "Second chunk"),
    ]


def test_chunks_are_sorted_by_score_descending() -> None:
    """
    The highest-scoring chunk should come first, regardless of
    input order.
    """

    low_score_chunk = build_chunk(index=0, text="low relevance")
    high_score_chunk = build_chunk(index=1, text="high relevance")

    reranker, _ = build_reranker(scores=[0.1, 0.9])

    result = reranker.rerank(
        query="query",
        chunks=[low_score_chunk, high_score_chunk],
    )

    assert result[0].chunk == high_score_chunk
    assert result[1].chunk == low_score_chunk


def test_result_contains_matching_scores() -> None:
    """
    Each RerankedChunk should carry the score the model produced
    for that specific chunk.
    """

    chunk = build_chunk(index=0, text="Chunk")

    reranker, _ = build_reranker(scores=[3.5])

    result = reranker.rerank(query="query", chunks=[chunk])

    assert result[0].rerank_score == pytest.approx(3.5)


def test_result_is_truncated_to_top_k() -> None:
    """
    Only the top_k highest-scoring chunks should be returned, even
    if more candidates were supplied.
    """

    chunks = [
        build_chunk(index=i, text=f"chunk {i}")
        for i in range(5)
    ]

    scores = [0.1, 0.5, 0.9, 0.3, 0.2]

    reranker, _ = build_reranker(scores=scores, top_k=2)

    result = reranker.rerank(query="query", chunks=chunks)

    assert len(result) == 2
    assert result[0].chunk == chunks[2]  # score 0.9
    assert result[1].chunk == chunks[1]  # score 0.5


def test_top_k_larger_than_chunk_count_returns_all_chunks() -> None:
    """
    Requesting more results than candidates exist should just
    return every candidate, not raise an error.
    """

    chunks = [
        build_chunk(index=i, text=f"chunk {i}")
        for i in range(2)
    ]

    reranker, _ = build_reranker(scores=[0.4, 0.6], top_k=50)

    result = reranker.rerank(query="query", chunks=chunks)

    assert len(result) == 2


def test_returns_rerankedchunk_instances() -> None:
    """
    Every entry in the result should be a RerankedChunk.
    """

    chunk = build_chunk(index=0, text="Chunk")

    reranker, _ = build_reranker(scores=[1.0])

    result = reranker.rerank(query="query", chunks=[chunk])

    assert isinstance(result[0], RerankedChunk)


def test_zero_top_k_raises_value_error() -> None:
    """
    top_k must be a positive integer.
    """

    with pytest.raises(ValueError):
        CrossEncoderReranker(
            model_name="fake-model",
            top_k=0,
            device="cpu",
        )


def test_negative_top_k_raises_value_error() -> None:
    """
    Negative top_k values should also be rejected.
    """

    with pytest.raises(ValueError):
        CrossEncoderReranker(
            model_name="fake-model",
            top_k=-1,
            device="cpu",
        )


def test_model_load_failure_raises_reranking_model_load_error() -> None:
    """
    If the model factory raises, it should be wrapped in
    RerankingModelLoadError rather than leaking the raw
    exception.
    """

    def broken_factory(*args, **kwargs):
        raise RuntimeError("could not download model")

    reranker = CrossEncoderReranker(
        model_name="broken-model",
        top_k=5,
        device="cpu",
        model_factory=broken_factory,
    )

    chunk = build_chunk(index=0, text="Chunk")

    with pytest.raises(RerankingModelLoadError):
        reranker.rerank(query="query", chunks=[chunk])


def test_prediction_failure_raises_reranking_generation_error() -> None:
    """
    If model.predict() raises, it should be wrapped in
    RerankingGenerationError rather than leaking the raw
    exception.
    """

    class BrokenCrossEncoder:
        def predict(self, pairs):
            raise RuntimeError("inference failed")

    reranker = CrossEncoderReranker(
        model_name="fake-model",
        top_k=5,
        device="cpu",
        model_factory=lambda *args, **kwargs: BrokenCrossEncoder(),
    )

    chunk = build_chunk(index=0, text="Chunk")

    with pytest.raises(RerankingGenerationError):
        reranker.rerank(query="query", chunks=[chunk])


def test_model_is_loaded_only_once_across_multiple_calls() -> None:
    """
    The model factory should be invoked at most once, even across
    multiple rerank() calls, since the model is cached after the
    first load.
    """

    load_calls = []

    def counting_factory(*args, **kwargs):
        load_calls.append(1)
        return FakeCrossEncoder([0.5])

    reranker = CrossEncoderReranker(
        model_name="fake-model",
        top_k=5,
        device="cpu",
        model_factory=counting_factory,
    )

    chunk = build_chunk(index=0, text="Chunk")

    reranker.rerank(query="query", chunks=[chunk])
    reranker.rerank(query="query", chunks=[chunk])

    assert len(load_calls) == 1


def test_rerank_async_returns_same_result_as_sync() -> None:
    """
    rerank_async should produce the same ordering as the
    synchronous rerank(), just without blocking the event loop.
    """

    chunk_a = build_chunk(index=0, text="low")
    chunk_b = build_chunk(index=1, text="high")

    reranker, _ = build_reranker(scores=[0.1, 0.9])

    result = asyncio.run(
        reranker.rerank_async(
            query="query",
            chunks=[chunk_a, chunk_b],
        )
    )

    assert result[0].chunk == chunk_b
    assert result[1].chunk == chunk_a
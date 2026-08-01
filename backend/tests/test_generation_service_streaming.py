import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.schemas.answer import AnswerComplete, AnswerToken
from app.schemas.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)
from app.services.generation import GenerationService


def build_chunk(
    *,
    document_id,
    chunk_index: int = 0,
    page_number: int = 1,
    text: str = "Evidence from the document.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=document_id,
        chunk_index=chunk_index,
        page_number=page_number,
        page_chunk_index=0,
        similarity_score=0.96,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
    )


def build_retrieval_result(
    *,
    document_id,
    query: str = "What is the policy?",
    chunk_count: int = 2,
) -> RetrievalResult:
    chunks = tuple(
        build_chunk(
            document_id=document_id,
            chunk_index=index,
            page_number=index + 1,
            text=f"Evidence chunk {index + 1}",
        )
        for index in range(chunk_count)
    )

    return RetrievalResult(
        query=query,
        total_results=chunk_count,
        chunks=chunks,
    )


async def fake_token_stream(fragments):
    for fragment in fragments:
        yield fragment


def build_generation_service(
    *,
    stream_fragments=("Hello", ", ", "world."),
):
    """
    Create a GenerationService with mocked dependencies. The
    LLM's generate_answer_stream is wired to yield the given
    fragments, in order.
    """

    retrieval_service = Mock()
    retrieval_service.retrieve = AsyncMock()

    prompt_builder = Mock()
    prompt_builder.system_prompt.return_value = (
        "SYSTEM PROMPT"
    )
    prompt_builder.user_prompt.return_value = (
        "USER PROMPT"
    )

    llm_service = Mock()
    llm_service.model_name = "test-llm-model"
    llm_service.generate_answer_stream = Mock(
        return_value=fake_token_stream(stream_fragments)
    )

    service = GenerationService(
        retrieval_service=retrieval_service,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )

    return (
        service,
        retrieval_service,
        prompt_builder,
        llm_service,
    )


async def collect_stream_events(service, **kwargs):
    return [
        event
        async for event in service.generate_answer_stream(
            **kwargs
        )
    ]


def test_stream_yields_one_answertoken_per_fragment() -> None:
    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        _,
    ) = build_generation_service(
        stream_fragments=("Hello", ", ", "world."),
    )

    retrieval_service.retrieve.return_value = (
        build_retrieval_result(document_id=document_id)
    )

    events = asyncio.run(
        collect_stream_events(
            service,
            document_id=document_id,
            query="What is the policy?",
        )
    )

    tokens = [
        event
        for event in events
        if isinstance(event, AnswerToken)
    ]

    assert [t.text for t in tokens] == [
        "Hello",
        ", ",
        "world.",
    ]


def test_stream_ends_with_exactly_one_answercomplete() -> None:
    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        _,
    ) = build_generation_service()

    retrieval_service.retrieve.return_value = (
        build_retrieval_result(document_id=document_id)
    )

    events = asyncio.run(
        collect_stream_events(
            service,
            document_id=document_id,
            query="What is the policy?",
        )
    )

    complete_events = [
        event
        for event in events
        if isinstance(event, AnswerComplete)
    ]

    assert len(complete_events) == 1
    assert events[-1] is complete_events[0]


def test_answercomplete_carries_correct_metadata() -> None:
    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        llm_service,
    ) = build_generation_service()

    retrieval_result = build_retrieval_result(
        document_id=document_id,
        chunk_count=3,
    )

    retrieval_service.retrieve.return_value = (
        retrieval_result
    )

    events = asyncio.run(
        collect_stream_events(
            service,
            document_id=document_id,
            query="What is the policy?",
        )
    )

    complete = events[-1]

    assert complete.document_id == document_id
    assert complete.query == "What is the policy?"
    assert complete.model == "test-llm-model"
    assert complete.grounded is True
    assert complete.total_sources == 3
    assert (
        complete.retrieved_chunks
        == retrieval_result.chunks
    )
    assert complete.generation_time_ms >= 0


def test_no_sources_produces_ungrounded_answercomplete() -> None:
    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        _,
    ) = build_generation_service()

    retrieval_service.retrieve.return_value = (
        RetrievalResult(
            query="Unknown question",
            total_results=0,
            chunks=(),
        )
    )

    events = asyncio.run(
        collect_stream_events(
            service,
            document_id=document_id,
            query="Unknown question",
        )
    )

    complete = events[-1]

    assert complete.grounded is False
    assert complete.total_sources == 0
    assert complete.retrieved_chunks == ()


def test_prompt_builder_receives_retrieval_result() -> None:
    document_id = uuid4()

    (
        service,
        retrieval_service,
        prompt_builder,
        _,
    ) = build_generation_service()

    retrieval_result = build_retrieval_result(
        document_id=document_id,
    )

    retrieval_service.retrieve.return_value = (
        retrieval_result
    )

    asyncio.run(
        collect_stream_events(
            service,
            document_id=document_id,
            query="What is the policy?",
        )
    )

    prompt_builder.user_prompt.assert_called_once_with(
        retrieval_result
    )


def test_llm_stream_receives_generated_prompts() -> None:
    document_id = uuid4()

    (
        service,
        retrieval_service,
        prompt_builder,
        llm_service,
    ) = build_generation_service()

    retrieval_service.retrieve.return_value = (
        build_retrieval_result(document_id=document_id)
    )

    prompt_builder.system_prompt.return_value = "SYSTEM"
    prompt_builder.user_prompt.return_value = "USER"

    asyncio.run(
        collect_stream_events(
            service,
            document_id=document_id,
            query="Question",
        )
    )

    llm_service.generate_answer_stream.assert_called_once_with(
        system_prompt="SYSTEM",
        user_prompt="USER",
    )


def test_tokens_are_yielded_before_answercomplete() -> None:
    """
    Ordering matters for a real streaming consumer: every
    AnswerToken must arrive before the final AnswerComplete,
    never after.
    """

    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        _,
    ) = build_generation_service(
        stream_fragments=("a", "b", "c"),
    )

    retrieval_service.retrieve.return_value = (
        build_retrieval_result(document_id=document_id)
    )

    events = asyncio.run(
        collect_stream_events(
            service,
            document_id=document_id,
            query="Question",
        )
    )

    event_types = [type(event).__name__ for event in events]

    assert event_types == [
        "AnswerToken",
        "AnswerToken",
        "AnswerToken",
        "AnswerComplete",
    ]


def test_retrieval_failure_is_propagated() -> None:
    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        _,
    ) = build_generation_service()

    retrieval_service.retrieve.side_effect = RuntimeError(
        "Forced retrieval failure"
    )

    async def run():
        async for _ in service.generate_answer_stream(
            document_id=document_id,
            query="Question",
        ):
            pass

    with pytest.raises(
        RuntimeError,
        match="Forced retrieval failure",
    ):
        asyncio.run(run())


def test_llm_stream_failure_is_propagated() -> None:
    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        llm_service,
    ) = build_generation_service()

    retrieval_service.retrieve.return_value = (
        build_retrieval_result(document_id=document_id)
    )

    async def broken_stream():
        raise RuntimeError("LLM stream failure")
        yield  # pragma: no cover - unreachable

    llm_service.generate_answer_stream = Mock(
        return_value=broken_stream()
    )

    async def run():
        async for _ in service.generate_answer_stream(
            document_id=document_id,
            query="Question",
        ):
            pass

    with pytest.raises(
        RuntimeError,
        match="LLM stream failure",
    ):
        asyncio.run(run())
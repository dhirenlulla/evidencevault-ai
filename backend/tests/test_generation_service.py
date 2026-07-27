import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.schemas.retrieval import (
    RetrievedChunk,
    RetrievalResult,
)
from app.services.generation import (
    GenerationService,
)


def build_chunk(
    *,
    document_id,
    chunk_index: int = 0,
    page_number: int = 1,
    text: str = "Evidence from the document.",
) -> RetrievedChunk:
    """
    Create one retrieved chunk.
    """

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
    """
    Create a controlled retrieval result.
    """

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


def build_generation_service():
    """
    Create a GenerationService with mocked dependencies.
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
    llm_service.model_name = (
        "test-llm-model"
    )
    llm_service.generate_answer = AsyncMock(
        return_value="Generated answer."
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


def test_generation_pipeline_succeeds() -> None:
    """
    The complete RAG workflow should return
    a grounded answer.
    """

    document_id = uuid4()

    (
        service,
        retrieval_service,
        prompt_builder,
        llm_service,
    ) = build_generation_service()

    retrieval_result = build_retrieval_result(
        document_id=document_id,
    )

    retrieval_service.retrieve.return_value = (
        retrieval_result
    )

    response = asyncio.run(
        service.generate_answer(
            document_id=document_id,
            query="What is the policy?",
        )
    )

    assert response.document_id == document_id
    assert response.query == "What is the policy?"
    assert (
        response.answer
        == "Generated answer."
    )
    assert (
        response.model
        == "test-llm-model"
    )

    assert response.grounded is True

    assert response.total_sources == 2

    assert (
        response.retrieved_chunks
        == retrieval_result.chunks
    )

    assert (
        response.generation_time_ms >= 0
    )

    retrieval_service.retrieve.assert_awaited_once_with(
        document_id=document_id,
        query="What is the policy?",
    )

    prompt_builder.system_prompt.assert_called_once()

    prompt_builder.user_prompt.assert_called_once_with(
        retrieval_result
    )

    llm_service.generate_answer.assert_awaited_once_with(
        system_prompt="SYSTEM PROMPT",
        user_prompt="USER PROMPT",
    )


def test_generation_without_sources_is_not_grounded() -> None:
    """
    Zero retrieved chunks should produce an
    ungrounded response while still calling
    the language model.
    """

    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        llm_service,
    ) = build_generation_service()

    retrieval_service.retrieve.return_value = (
        RetrievalResult(
            query="Unknown question",
            total_results=0,
            chunks=(),
        )
    )

    response = asyncio.run(
        service.generate_answer(
            document_id=document_id,
            query="Unknown question",
        )
    )

    assert response.grounded is False

    assert response.total_sources == 0

    assert response.retrieved_chunks == ()

    llm_service.generate_answer.assert_awaited_once()


def test_prompt_builder_receives_retrieval_result() -> None:
    """
    The PromptBuilder should receive the exact
    RetrievalResult produced by retrieval.
    """

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
        service.generate_answer(
            document_id=document_id,
            query="What is the policy?",
        )
    )

    prompt_builder.user_prompt.assert_called_once_with(
        retrieval_result
    )


def test_llm_receives_generated_prompts() -> None:
    """
    The generated prompts should be forwarded
    unchanged to the language model.
    """

    document_id = uuid4()

    (
        service,
        retrieval_service,
        prompt_builder,
        llm_service,
    ) = build_generation_service()

    retrieval_service.retrieve.return_value = (
        build_retrieval_result(
            document_id=document_id,
        )
    )

    prompt_builder.system_prompt.return_value = (
        "SYSTEM"
    )

    prompt_builder.user_prompt.return_value = (
        "USER"
    )

    asyncio.run(
        service.generate_answer(
            document_id=document_id,
            query="Question",
        )
    )

    llm_service.generate_answer.assert_awaited_once_with(
        system_prompt="SYSTEM",
        user_prompt="USER",
    )


def test_retrieval_failure_is_propagated() -> None:
    """
    Retrieval failures should not be swallowed.
    """

    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        _,
    ) = build_generation_service()

    retrieval_service.retrieve.side_effect = (
        RuntimeError(
            "Forced retrieval failure"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Forced retrieval failure",
    ):
        asyncio.run(
            service.generate_answer(
                document_id=document_id,
                query="Question",
            )
        )


def test_prompt_builder_failure_is_propagated() -> None:
    """
    PromptBuilder failures should propagate.
    """

    document_id = uuid4()

    (
        service,
        retrieval_service,
        prompt_builder,
        _,
    ) = build_generation_service()

    retrieval_service.retrieve.return_value = (
        build_retrieval_result(
            document_id=document_id,
        )
    )

    prompt_builder.user_prompt.side_effect = (
        RuntimeError(
            "Prompt construction failed"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Prompt construction failed",
    ):
        asyncio.run(
            service.generate_answer(
                document_id=document_id,
                query="Question",
            )
        )


def test_llm_failure_is_propagated() -> None:
    """
    LLM failures should propagate.
    """

    document_id = uuid4()

    (
        service,
        retrieval_service,
        _,
        llm_service,
    ) = build_generation_service()

    retrieval_service.retrieve.return_value = (
        build_retrieval_result(
            document_id=document_id,
        )
    )

    llm_service.generate_answer.side_effect = (
        RuntimeError(
            "LLM failure"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="LLM failure",
    ):
        asyncio.run(
            service.generate_answer(
                document_id=document_id,
                query="Question",
            )
        )
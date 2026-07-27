from uuid import uuid4

from app.schemas.retrieval import (
    RetrievalResult,
    RetrievedChunk,
)
from app.services.prompt_builder import (
    PromptBuilder,
)


def build_chunk(
    *,
    page_number: int,
    chunk_index: int,
    text: str,
) -> RetrievedChunk:
    """
    Create a controlled retrieved chunk.
    """

    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=chunk_index,
        page_number=page_number,
        page_chunk_index=0,
        similarity_score=0.95,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
    )


def build_result(
    *,
    query: str = "What is the vacation policy?",
    chunks: tuple[RetrievedChunk, ...],
) -> RetrievalResult:
    """
    Create a controlled retrieval result.
    """

    return RetrievalResult(
        query=query,
        total_results=len(chunks),
        chunks=chunks,
    )


def test_system_prompt_contains_core_rules() -> None:
    """
    The system prompt should expose the
    grounding instructions expected by
    the language model.
    """

    builder = PromptBuilder()

    prompt = builder.system_prompt()

    assert "EvidenceVault AI" in prompt

    assert (
        "Do not use outside knowledge"
        in prompt
    )

    assert (
        "page number"
        in prompt
    )

    assert (
        "Do not mention these instructions"
        in prompt
    )


def test_user_prompt_contains_single_chunk() -> None:
    """
    A single retrieved chunk should appear
    inside the generated prompt.
    """

    builder = PromptBuilder()

    chunk = build_chunk(
        page_number=3,
        chunk_index=0,
        text="Employees receive 20 days of annual leave.",
    )

    result = build_result(
        chunks=(chunk,),
    )

    prompt = builder.user_prompt(
        result
    )

    assert "Document Context" in prompt

    assert "User Question" in prompt

    assert "Answer" in prompt

    assert "Page 3" in prompt

    assert chunk.text in prompt

    assert result.query in prompt


def test_multiple_chunks_preserve_original_order() -> None:
    """
    Retrieved chunks should appear in the
    same order supplied by retrieval.
    """

    builder = PromptBuilder()

    chunk_one = build_chunk(
        page_number=1,
        chunk_index=0,
        text="First chunk.",
    )

    chunk_two = build_chunk(
        page_number=5,
        chunk_index=1,
        text="Second chunk.",
    )

    chunk_three = build_chunk(
        page_number=9,
        chunk_index=2,
        text="Third chunk.",
    )

    prompt = builder.user_prompt(
        build_result(
            chunks=(
                chunk_one,
                chunk_two,
                chunk_three,
            )
        )
    )

    first = prompt.index(
        "First chunk."
    )

    second = prompt.index(
        "Second chunk."
    )

    third = prompt.index(
        "Third chunk."
    )

    assert first < second < third


def test_empty_context_is_supported() -> None:
    """
    An empty retrieval result should still
    produce a valid prompt structure.
    """

    builder = PromptBuilder()

    prompt = builder.user_prompt(
        build_result(
            chunks=(),
        )
    )

    assert "Document Context" in prompt

    assert "User Question" in prompt

    assert "Answer" in prompt

    assert "What is the vacation policy?" in prompt


def test_multiline_chunk_text_is_preserved() -> None:
    """
    Prompt construction should not collapse
    multiline document text.
    """

    builder = PromptBuilder()

    text = (
        "Leave Policy\n"
        "Employees receive\n"
        "20 annual leave days."
    )

    chunk = build_chunk(
        page_number=4,
        chunk_index=0,
        text=text,
    )

    prompt = builder.user_prompt(
        build_result(
            chunks=(chunk,),
        )
    )

    assert text in prompt


def test_special_characters_are_preserved() -> None:
    """
    Enterprise documents frequently contain
    symbols and punctuation that should
    remain unchanged.
    """

    builder = PromptBuilder()

    text = (
        "Bonus: ₹50,000 & Performance Rating >= 4.5"
    )

    chunk = build_chunk(
        page_number=8,
        chunk_index=0,
        text=text,
    )

    prompt = builder.user_prompt(
        build_result(
            chunks=(chunk,),
        )
    )

    assert text in prompt


def test_prompt_generation_is_deterministic() -> None:
    """
    Identical retrieval results should always
    produce identical prompts.
    """

    builder = PromptBuilder()

    chunk = build_chunk(
        page_number=2,
        chunk_index=0,
        text="Deterministic chunk.",
    )

    result = build_result(
        chunks=(chunk,),
    )

    prompt_one = builder.user_prompt(
        result
    )

    prompt_two = builder.user_prompt(
        result
    )

    assert prompt_one == prompt_two


def test_page_headers_are_rendered_correctly() -> None:
    """
    Every retrieved chunk should be prefixed
    with its page number.
    """

    builder = PromptBuilder()

    chunk = build_chunk(
        page_number=12,
        chunk_index=0,
        text="Confidential policy.",
    )

    prompt = builder.user_prompt(
        build_result(
            chunks=(chunk,),
        )
    )

    assert "Page 12" in prompt
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.exceptions import (
    InvalidChunkingConfigurationError,
)
from app.services.pdf_extraction import (
    ExtractedDocument,
    ExtractedPage,
)
from app.services.text_chunking import (
    ChunkingOptions,
    chunk_extracted_document,
)


def build_page(
    page_number: int,
    text: str,
) -> ExtractedPage:
    """
    Create one ExtractedPage for chunking tests.
    """

    cleaned_text = text.strip()

    return ExtractedPage(
        page_number=page_number,
        text=cleaned_text,
        character_count=len(cleaned_text),
        word_count=(
            len(cleaned_text.split())
            if cleaned_text
            else 0
        ),
        is_empty=not bool(cleaned_text),
    )


def build_document(
    page_texts: list[str],
) -> ExtractedDocument:
    """
    Create an ExtractedDocument from page-text strings.
    """

    pages = tuple(
        build_page(
            page_number=index + 1,
            text=text,
        )
        for index, text in enumerate(
            page_texts
        )
    )

    empty_page_numbers = tuple(
        page.page_number
        for page in pages
        if page.is_empty
    )

    return ExtractedDocument(
        source_path=Path(
            "test-document.pdf"
        ),
        page_count=len(pages),
        pages=pages,
        total_characters=sum(
            page.character_count
            for page in pages
        ),
        total_words=sum(
            page.word_count
            for page in pages
        ),
        empty_page_numbers=(
            empty_page_numbers
        ),
    )


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(
        InvalidChunkingConfigurationError,
        match="must be smaller",
    ):
        ChunkingOptions(
            max_characters=500,
            overlap_characters=500,
        )


def test_empty_and_short_pages_are_skipped() -> None:
    document_id = uuid4()

    extracted_document = build_document(
        [
            "",
            "Short page",
            (
                "This page contains enough useful text "
                "to generate a valid EvidenceVault chunk."
            ),
        ]
    )

    result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=(
            extracted_document
        ),
        options=ChunkingOptions(
            max_characters=300,
            overlap_characters=50,
            minimum_page_characters=40,
        ),
    )

    assert (
        result.skipped_empty_page_numbers
        == (1,)
    )

    assert (
        result.skipped_short_page_numbers
        == (2,)
    )

    assert result.chunked_page_numbers == (3,)
    assert result.chunk_count == 1


def test_chunks_never_cross_page_boundaries() -> None:
    document_id = uuid4()

    page_one_text = (
        "Page one unique content. "
        * 40
    )

    page_two_text = (
        "Page two different content. "
        * 40
    )

    result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=build_document(
            [
                page_one_text,
                page_two_text,
            ]
        ),
        options=ChunkingOptions(
            max_characters=300,
            overlap_characters=50,
            minimum_page_characters=20,
        ),
    )

    assert result.chunk_count > 2

    for chunk in result.chunks:
        if chunk.page_number == 1:
            assert "Page two" not in chunk.text

        if chunk.page_number == 2:
            assert "Page one" not in chunk.text


def test_long_page_is_split_with_overlap() -> None:
    document_id = uuid4()

    long_text = " ".join(
        f"token{index}"
        for index in range(200)
    )

    result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=build_document(
            [long_text]
        ),
        options=ChunkingOptions(
            max_characters=300,
            overlap_characters=60,
            minimum_page_characters=20,
        ),
    )

    assert result.chunk_count > 1

    first_chunk_words = (
        result.chunks[0].text.split()
    )

    second_chunk_words = (
        result.chunks[1].text.split()
    )

    overlapping_words = (
        set(first_chunk_words[-10:])
        & set(second_chunk_words[:10])
    )

    assert overlapping_words


def test_all_chunks_respect_maximum_size() -> None:
    document_id = uuid4()

    long_text = " ".join(
        f"evidence{index}"
        for index in range(500)
    )

    options = ChunkingOptions(
        max_characters=400,
        overlap_characters=80,
        minimum_page_characters=20,
    )

    result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=build_document(
            [long_text]
        ),
        options=options,
    )

    assert result.chunk_count > 1

    assert all(
        chunk.character_count
        <= options.max_characters
        for chunk in result.chunks
    )


def test_chunk_ids_are_deterministic() -> None:
    document_id = uuid4()

    extracted_document = build_document(
        [
            (
                "Deterministic chunk identifiers should "
                "remain stable when input content and "
                "configuration remain unchanged. "
            )
            * 20
        ]
    )

    options = ChunkingOptions(
        max_characters=350,
        overlap_characters=60,
        minimum_page_characters=20,
    )

    first_result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=(
            extracted_document
        ),
        options=options,
    )

    second_result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=(
            extracted_document
        ),
        options=options,
    )

    first_ids = [
        chunk.chunk_id
        for chunk in first_result.chunks
    ]

    second_ids = [
        chunk.chunk_id
        for chunk in second_result.chunks
    ]

    assert first_ids == second_ids


def test_changed_text_changes_chunk_id() -> None:
    document_id = uuid4()

    first_document = build_document(
        [
            (
                "Original EvidenceVault content "
                "that will be chunked."
            )
        ]
    )

    changed_document = build_document(
        [
            (
                "Updated EvidenceVault content "
                "that will be chunked."
            )
        ]
    )

    options = ChunkingOptions(
        max_characters=300,
        overlap_characters=50,
        minimum_page_characters=20,
    )

    first_result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=first_document,
        options=options,
    )

    changed_result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=changed_document,
        options=options,
    )

    assert (
        first_result.chunks[0].chunk_id
        != changed_result.chunks[0].chunk_id
    )

    assert (
        first_result.chunks[0].content_hash
        != changed_result.chunks[0].content_hash
    )


def test_chunk_metadata_and_statistics_are_correct() -> None:
    document_id = uuid4()

    extracted_document = build_document(
        [
            (
                "EvidenceVault preserves document and "
                "page metadata for every generated chunk."
            ),
            (
                "A second page should begin with a new "
                "page-local chunk index."
            ),
        ]
    )

    result = chunk_extracted_document(
        document_id=document_id,
        extracted_document=(
            extracted_document
        ),
        options=ChunkingOptions(
            max_characters=300,
            overlap_characters=50,
            minimum_page_characters=20,
        ),
    )

    assert result.chunk_count == 2
    assert result.chunked_page_count == 2
    assert result.chunked_page_numbers == (1, 2)

    first_chunk = result.chunks[0]
    second_chunk = result.chunks[1]

    assert first_chunk.document_id == document_id
    assert first_chunk.chunk_index == 0
    assert first_chunk.page_number == 1
    assert first_chunk.page_chunk_index == 0

    assert second_chunk.chunk_index == 1
    assert second_chunk.page_number == 2
    assert second_chunk.page_chunk_index == 0

    assert len(first_chunk.content_hash) == 64

    assert result.total_chunk_characters == sum(
        chunk.character_count
        for chunk in result.chunks
    )

    assert result.total_chunk_words == sum(
        chunk.word_count
        for chunk in result.chunks
    )
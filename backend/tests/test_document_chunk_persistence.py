import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

import app.services.document_chunk_persistence as persistence_module
from app.core.exceptions import (
    DocumentNotReadyForChunkingError,
    NoChunksGeneratedError,
)
from app.repositories.document_chunk import (
    DocumentChunkInput,
    replace_document_chunks,
)
from app.services.document_chunk_persistence import (
    generate_and_persist_document_chunks,
)
from app.services.pdf_extraction import (
    ExtractedDocument,
    ExtractedPage,
)
from app.services.text_chunking import (
    ChunkingOptions,
)


def build_document(
    *,
    status: str = "extracted",
):
    return SimpleNamespace(
        id=uuid4(),
        status=status,
        storage_path="uploads/test.pdf",
        chunk_count=0,
        error_message=None,
    )


def build_extraction(
    text: str,
) -> ExtractedDocument:
    cleaned_text = text.strip()

    page = ExtractedPage(
        page_number=1,
        text=cleaned_text,
        character_count=len(cleaned_text),
        word_count=len(
            cleaned_text.split()
        ),
        is_empty=not bool(cleaned_text),
    )

    return ExtractedDocument(
        source_path=Path("test.pdf"),
        page_count=1,
        pages=(page,),
        total_characters=(
            page.character_count
        ),
        total_words=page.word_count,
        empty_page_numbers=(
            (1,)
            if page.is_empty
            else ()
        ),
    )


def build_chunk_input(
    document_id,
) -> DocumentChunkInput:
    return DocumentChunkInput(
        id=uuid4(),
        document_id=document_id,
        chunk_index=0,
        page_number=1,
        page_chunk_index=0,
        text="Persisted chunk text",
        character_count=20,
        word_count=3,
        content_hash="a" * 64,
    )


def test_repository_replaces_chunks_and_updates_document() -> None:
    session = AsyncMock(
        spec=AsyncSession
    )

    session.add_all = Mock()

    document = build_document()

    chunk_input = build_chunk_input(
        document.id
    )

    records = asyncio.run(
        replace_document_chunks(
            session=session,
            document=document,
            chunks=(chunk_input,),
        )
    )

    assert len(records) == 1
    assert records[0].id == chunk_input.id

    assert document.status == "chunked"
    assert document.chunk_count == 1
    assert document.error_message is None

    session.execute.assert_awaited_once()
    session.add_all.assert_called_once()
    session.commit.assert_awaited_once()
    session.refresh.assert_awaited_once_with(
        document
    )
    session.rollback.assert_not_awaited()


def test_repository_rolls_back_when_commit_fails() -> None:
    session = AsyncMock(
        spec=AsyncSession
    )

    session.add_all = Mock()

    session.commit.side_effect = (
        SQLAlchemyError(
            "Forced chunk persistence failure"
        )
    )

    document = build_document()

    with pytest.raises(
        SQLAlchemyError,
        match="Forced chunk persistence failure",
    ):
        asyncio.run(
            replace_document_chunks(
                session=session,
                document=document,
                chunks=(
                    build_chunk_input(
                        document.id
                    ),
                ),
            )
        )

    session.rollback.assert_awaited_once()


def test_workflow_persists_generated_chunks(
    monkeypatch,
) -> None:
    document = build_document()

    get_mock = AsyncMock(
        return_value=document
    )

    persistence_mock = AsyncMock(
        return_value=[]
    )

    monkeypatch.setattr(
        persistence_module,
        "get_document_by_id",
        get_mock,
    )

    monkeypatch.setattr(
        persistence_module,
        "resolve_document_storage_path",
        lambda storage_path: Path(
            "test.pdf"
        ),
    )

    monkeypatch.setattr(
        persistence_module,
        "extract_pdf_pages",
        lambda path: build_extraction(
            (
                "EvidenceVault creates durable, "
                "page-aware chunks for retrieval. "
            )
            * 20
        ),
    )

    monkeypatch.setattr(
        persistence_module,
        "replace_document_chunks",
        persistence_mock,
    )

    result = asyncio.run(
        generate_and_persist_document_chunks(
            session=AsyncMock(
                spec=AsyncSession
            ),
            document_id=document.id,
            options=ChunkingOptions(
                max_characters=300,
                overlap_characters=50,
                minimum_page_characters=20,
            ),
        )
    )

    assert result.status.value == "chunked"
    assert result.chunk_count > 1

    persistence_mock.assert_awaited_once()

    persisted_arguments = (
        persistence_mock.await_args.kwargs
    )

    assert (
        persisted_arguments["document"]
        is document
    )

    assert (
        len(persisted_arguments["chunks"])
        == result.chunk_count
    )

    assert (
        persisted_arguments["final_status"]
        == "chunked"
    )


def test_workflow_rejects_unprocessed_document(
    monkeypatch,
) -> None:
    document = build_document(
        status="uploaded"
    )

    monkeypatch.setattr(
        persistence_module,
        "get_document_by_id",
        AsyncMock(
            return_value=document
        ),
    )

    with pytest.raises(
        DocumentNotReadyForChunkingError,
        match="must have status",
    ):
        asyncio.run(
            generate_and_persist_document_chunks(
                session=AsyncMock(
                    spec=AsyncSession
                ),
                document_id=document.id,
            )
        )


def test_workflow_rejects_zero_generated_chunks(
    monkeypatch,
) -> None:
    document = build_document()

    monkeypatch.setattr(
        persistence_module,
        "get_document_by_id",
        AsyncMock(
            return_value=document
        ),
    )

    monkeypatch.setattr(
        persistence_module,
        "resolve_document_storage_path",
        lambda storage_path: Path(
            "empty.pdf"
        ),
    )

    monkeypatch.setattr(
        persistence_module,
        "extract_pdf_pages",
        lambda path: build_extraction(""),
    )

    with pytest.raises(
        NoChunksGeneratedError,
        match="did not produce",
    ):
        asyncio.run(
            generate_and_persist_document_chunks(
                session=AsyncMock(
                    spec=AsyncSession
                ),
                document_id=document.id,
            )
        )
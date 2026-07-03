from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient

import app.api.routes.documents as documents_route
from app.core.exceptions import (
    DocumentAlreadyProcessingError,
    DocumentNotFoundError,
    NoChunksGeneratedError,
)
from app.services.document_chunk_persistence import (
    PersistedChunkingResult,
)
from app.services.document_processing import (
    DocumentProcessingResult,
    DocumentProcessingStatus,
)
from app.services.pdf_classification import (
    PDFDocumentClassification,
)
from app.services.text_chunking import (
    ChunkingOptions,
    ChunkingResult,
    TextChunk,
)


def build_processing_result(
    *,
    document_id,
    status: DocumentProcessingStatus,
    classification: (
        PDFDocumentClassification | None
    ) = PDFDocumentClassification.TEXT_BASED,
) -> DocumentProcessingResult:
    """
    Create a controlled processing result for route tests.
    """

    return DocumentProcessingResult(
        document_id=document_id,
        status=status,
        classification=classification,
        page_count=2,
        extractable_page_count=(
            2
            if status
            in {
                DocumentProcessingStatus.EXTRACTED,
                (
                    DocumentProcessingStatus
                    .EXTRACTED_WITH_WARNINGS
                ),
            }
            else 0
        ),
        image_only_page_count=(
            2
            if (
                status
                == DocumentProcessingStatus
                .OCR_REQUIRED
            )
            else 0
        ),
        empty_page_count=0,
        total_characters=500,
        total_words=80,
        message="Controlled processing result.",
    )


def build_chunking_result(
    document_id,
) -> ChunkingResult:
    """
    Create a controlled two-chunk result.
    """

    first_text = "First persisted chunk"
    second_text = "Second persisted chunk"

    chunks = (
        TextChunk(
            chunk_id=uuid4(),
            document_id=document_id,
            chunk_index=0,
            page_number=1,
            page_chunk_index=0,
            text=first_text,
            character_count=len(first_text),
            word_count=3,
            content_hash="a" * 64,
        ),
        TextChunk(
            chunk_id=uuid4(),
            document_id=document_id,
            chunk_index=1,
            page_number=2,
            page_chunk_index=0,
            text=second_text,
            character_count=len(second_text),
            word_count=3,
            content_hash="b" * 64,
        ),
    )

    return ChunkingResult(
        document_id=document_id,
        source_page_count=2,
        source_character_count=500,
        chunks=chunks,
        chunked_page_numbers=(1, 2),
        skipped_empty_page_numbers=(),
        skipped_short_page_numbers=(),
        total_chunk_characters=sum(
            chunk.character_count
            for chunk in chunks
        ),
        total_chunk_words=sum(
            chunk.word_count
            for chunk in chunks
        ),
        options=ChunkingOptions(),
    )


def build_persisted_result(
    document_id,
) -> PersistedChunkingResult:
    """
    Create a controlled persisted chunking result.
    """

    return PersistedChunkingResult(
        document_id=document_id,
        status=DocumentProcessingStatus.CHUNKED,
        chunking_result=build_chunking_result(
            document_id
        ),
    )


def build_database_chunk(
    *,
    document_id,
    chunk_index: int,
):
    """
    Create a SQLAlchemy-like chunk object.
    """

    text = (
        f"Persisted chunk number "
        f"{chunk_index}"
    )

    return SimpleNamespace(
        id=uuid4(),
        document_id=document_id,
        chunk_index=chunk_index,
        page_number=chunk_index + 1,
        page_chunk_index=0,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
        content_hash=(
            f"{chunk_index}" * 64
        )[:64],
        created_at=datetime.now(
            timezone.utc
        ),
    )


def test_process_endpoint_persists_chunks(
    client: TestClient,
    monkeypatch,
) -> None:
    document_id = uuid4()

    processing_mock = AsyncMock(
        return_value=build_processing_result(
            document_id=document_id,
            status=(
                DocumentProcessingStatus
                .EXTRACTED
            ),
        )
    )

    persistence_mock = AsyncMock(
        return_value=build_persisted_result(
            document_id
        )
    )

    monkeypatch.setattr(
        documents_route,
        "run_document_processing",
        processing_mock,
    )

    monkeypatch.setattr(
        documents_route,
        (
            "generate_and_persist_"
            "document_chunks"
        ),
        persistence_mock,
    )

    response = client.post(
        f"/api/v1/documents/"
        f"{document_id}/process"
    )

    assert response.status_code == 200

    response_data = response.json()

    assert (
        response_data["document_id"]
        == str(document_id)
    )

    assert response_data["status"] == "chunked"
    assert response_data["classification"] == (
        "text_based"
    )
    assert response_data["chunk_count"] == 2
    assert (
        response_data["ready_for_indexing"]
        is True
    )

    processing_mock.assert_awaited_once()
    persistence_mock.assert_awaited_once()


def test_ocr_document_does_not_attempt_chunking(
    client: TestClient,
    monkeypatch,
) -> None:
    document_id = uuid4()

    processing_mock = AsyncMock(
        return_value=build_processing_result(
            document_id=document_id,
            status=(
                DocumentProcessingStatus
                .OCR_REQUIRED
            ),
            classification=(
                PDFDocumentClassification
                .SCANNED_OR_IMAGE_ONLY
            ),
        )
    )

    persistence_mock = AsyncMock()

    monkeypatch.setattr(
        documents_route,
        "run_document_processing",
        processing_mock,
    )

    monkeypatch.setattr(
        documents_route,
        (
            "generate_and_persist_"
            "document_chunks"
        ),
        persistence_mock,
    )

    response = client.post(
        f"/api/v1/documents/"
        f"{document_id}/process"
    )

    assert response.status_code == 200

    response_data = response.json()

    assert (
        response_data["status"]
        == "ocr_required"
    )

    assert response_data["chunk_count"] == 0
    assert (
        response_data["ready_for_indexing"]
        is False
    )

    persistence_mock.assert_not_awaited()


def test_process_endpoint_returns_404(
    client: TestClient,
    monkeypatch,
) -> None:
    document_id = uuid4()

    monkeypatch.setattr(
        documents_route,
        "run_document_processing",
        AsyncMock(
            side_effect=DocumentNotFoundError(
                "Document not found."
            )
        ),
    )

    response = client.post(
        f"/api/v1/documents/"
        f"{document_id}/process"
    )

    assert response.status_code == 404


def test_process_endpoint_returns_409_when_active(
    client: TestClient,
    monkeypatch,
) -> None:
    document_id = uuid4()

    monkeypatch.setattr(
        documents_route,
        "run_document_processing",
        AsyncMock(
            side_effect=(
                DocumentAlreadyProcessingError(
                    "The document is already "
                    "being processed."
                )
            )
        ),
    )

    response = client.post(
        f"/api/v1/documents/"
        f"{document_id}/process"
    )

    assert response.status_code == 409


def test_process_endpoint_returns_422_for_zero_chunks(
    client: TestClient,
    monkeypatch,
) -> None:
    document_id = uuid4()

    monkeypatch.setattr(
        documents_route,
        "run_document_processing",
        AsyncMock(
            return_value=build_processing_result(
                document_id=document_id,
                status=(
                    DocumentProcessingStatus
                    .EXTRACTED
                ),
            )
        ),
    )

    monkeypatch.setattr(
        documents_route,
        (
            "generate_and_persist_"
            "document_chunks"
        ),
        AsyncMock(
            side_effect=NoChunksGeneratedError(
                "The document did not produce "
                "any usable text chunks."
            )
        ),
    )

    response = client.post(
        f"/api/v1/documents/"
        f"{document_id}/process"
    )

    assert response.status_code == 422


def test_list_chunks_returns_paginated_result(
    client: TestClient,
    monkeypatch,
) -> None:
    document_id = uuid4()

    document = SimpleNamespace(
        id=document_id
    )

    chunks = [
        build_database_chunk(
            document_id=document_id,
            chunk_index=0,
        ),
        build_database_chunk(
            document_id=document_id,
            chunk_index=1,
        ),
    ]

    get_document_mock = AsyncMock(
        return_value=document
    )

    list_mock = AsyncMock(
        return_value=chunks
    )

    count_mock = AsyncMock(
        return_value=73
    )

    monkeypatch.setattr(
        documents_route,
        "get_document_by_id",
        get_document_mock,
    )

    monkeypatch.setattr(
        documents_route,
        "fetch_document_chunks",
        list_mock,
    )

    monkeypatch.setattr(
        documents_route,
        "count_document_chunks",
        count_mock,
    )

    response = client.get(
        f"/api/v1/documents/"
        f"{document_id}/chunks",
        params={
            "limit": 2,
            "offset": 5,
        },
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["total"] == 73
    assert response_data["limit"] == 2
    assert response_data["offset"] == 5
    assert len(response_data["chunks"]) == 2

    repository_arguments = (
        list_mock.await_args.kwargs
    )

    assert repository_arguments["limit"] == 2
    assert repository_arguments["offset"] == 5


def test_list_chunks_returns_404_for_missing_document(
    client: TestClient,
    monkeypatch,
) -> None:
    document_id = uuid4()

    monkeypatch.setattr(
        documents_route,
        "get_document_by_id",
        AsyncMock(
            return_value=None
        ),
    )

    response = client.get(
        f"/api/v1/documents/"
        f"{document_id}/chunks"
    )

    assert response.status_code == 404


def test_list_chunks_rejects_invalid_pagination(
    client: TestClient,
) -> None:
    document_id = uuid4()

    response = client.get(
        f"/api/v1/documents/"
        f"{document_id}/chunks",
        params={
            "limit": 0,
            "offset": -1,
        },
    )

    assert response.status_code == 422
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import pytest
from qdrant_client import models
from sqlalchemy.ext.asyncio import AsyncSession

import app.services.qdrant_indexing as indexing_module
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentNotReadyForIndexingError,
    NoChunksAvailableForIndexingError,
    QdrantPointUpsertError,
)
from app.services.qdrant_collection import (
    VectorCollectionStatus,
)
from app.services.qdrant_indexing import (
    VectorIndexingOptions,
    index_document_chunks,
)


class FakeEmbeddingService:
    """
    Fake embedding service for indexing tests.
    """

    def __init__(self, *, dimension: int = 384) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    async def embed_documents_async(self, texts):
        self.calls.append(list(texts))

        vectors = np.zeros(
            (
                len(texts),
                self.dimension,
            ),
            dtype=np.float32,
        )

        for index in range(len(texts)):
            vectors[index, index % self.dimension] = 1.0

        return SimpleNamespace(
            vectors=vectors,
            count=len(texts),
            dimension=self.dimension,
        )


class FakeQdrantClient:
    """
    Fake Qdrant client that records upsert calls.
    """

    def __init__(self, *, fail_upsert: bool = False) -> None:
        self.fail_upsert = fail_upsert
        self.upsert_calls: list[dict] = []

    async def upsert(
        self,
        *,
        collection_name: str,
        points,
        wait: bool,
    ):
        if self.fail_upsert:
            raise RuntimeError(
                "Forced Qdrant upsert failure"
            )

        self.upsert_calls.append(
            {
                "collection_name": collection_name,
                "points": points,
                "wait": wait,
            }
        )


class FakeCollectionService:
    """
    Fake collection service that mimics Step 6B behavior.
    """

    collection_name = "test_chunks"

    def __init__(self) -> None:
        self.ensure_calls = 0

    async def ensure_collection(self):
        self.ensure_calls += 1

        return VectorCollectionStatus(
            collection_name=self.collection_name,
            exists=True,
            vector_size=384,
            distance="Cosine",
            expected_vector_size=384,
            expected_distance="Cosine",
            is_compatible=True,
            points_count=0,
            indexed_vectors_count=0,
            message="Collection is compatible.",
        )


def build_document(*, status: str = "chunked"):
    return SimpleNamespace(
        id=uuid4(),
        status=status,
    )


def build_chunk(
    *,
    document_id,
    chunk_index: int,
):
    text = (
        f"This is chunk number {chunk_index} "
        f"for vector indexing."
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
    )


def test_document_chunks_are_indexed_in_batches(
    monkeypatch,
) -> None:
    document = build_document()

    chunks = [
        build_chunk(
            document_id=document.id,
            chunk_index=0,
        ),
        build_chunk(
            document_id=document.id,
            chunk_index=1,
        ),
        build_chunk(
            document_id=document.id,
            chunk_index=2,
        ),
    ]

    monkeypatch.setattr(
        indexing_module,
        "get_document_by_id",
        AsyncMock(return_value=document),
    )

    monkeypatch.setattr(
        indexing_module,
        "count_document_chunks",
        AsyncMock(return_value=3),
    )

    async def fake_list_chunks(
        *,
        session,
        document_id,
        limit,
        offset,
    ):
        return chunks[offset : offset + limit]

    monkeypatch.setattr(
        indexing_module,
        "list_document_chunks",
        fake_list_chunks,
    )

    qdrant_client = FakeQdrantClient()

    embedding_service = FakeEmbeddingService()

    collection_service = FakeCollectionService()

    result = asyncio.run(
        index_document_chunks(
            session=AsyncMock(spec=AsyncSession),
            document_id=document.id,
            qdrant_client=qdrant_client,
            collection_service=collection_service,
            embedding_service=embedding_service,
            options=VectorIndexingOptions(
                batch_size=2
            ),
        )
    )

    assert result.total_chunks == 3
    assert result.indexed_chunks == 3
    assert result.batch_count == 2
    assert result.is_complete is True

    assert collection_service.ensure_calls == 1

    assert len(qdrant_client.upsert_calls) == 2

    first_call = qdrant_client.upsert_calls[0]

    assert first_call["collection_name"] == "test_chunks"
    assert first_call["wait"] is True
    assert len(first_call["points"]) == 2

    first_point = first_call["points"][0]

    assert isinstance(
        first_point,
        models.PointStruct,
    )

    assert (
        first_point.id
        == str(chunks[0].id)
    )

    assert len(first_point.vector) == 384

    assert (
        first_point.payload["document_id"]
        == str(document.id)
    )

    assert (
        first_point.payload["page_number"]
        == chunks[0].page_number
    )

    assert (
        first_point.payload["text"]
        == chunks[0].text
    )


def test_missing_document_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        indexing_module,
        "get_document_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(
        DocumentNotFoundError,
        match="Document not found",
    ):
        asyncio.run(
            index_document_chunks(
                session=AsyncMock(spec=AsyncSession),
                document_id=uuid4(),
                qdrant_client=FakeQdrantClient(),
                collection_service=FakeCollectionService(),
                embedding_service=FakeEmbeddingService(),
            )
        )


def test_unprocessed_document_is_rejected(
    monkeypatch,
) -> None:
    document = build_document(
        status="uploaded"
    )

    monkeypatch.setattr(
        indexing_module,
        "get_document_by_id",
        AsyncMock(return_value=document),
    )

    with pytest.raises(
        DocumentNotReadyForIndexingError,
        match="chunked",
    ):
        asyncio.run(
            index_document_chunks(
                session=AsyncMock(spec=AsyncSession),
                document_id=document.id,
                qdrant_client=FakeQdrantClient(),
                collection_service=FakeCollectionService(),
                embedding_service=FakeEmbeddingService(),
            )
        )


def test_document_with_no_chunks_is_rejected(
    monkeypatch,
) -> None:
    document = build_document()

    monkeypatch.setattr(
        indexing_module,
        "get_document_by_id",
        AsyncMock(return_value=document),
    )

    monkeypatch.setattr(
        indexing_module,
        "count_document_chunks",
        AsyncMock(return_value=0),
    )

    with pytest.raises(
        NoChunksAvailableForIndexingError,
        match="no persisted chunks",
    ):
        asyncio.run(
            index_document_chunks(
                session=AsyncMock(spec=AsyncSession),
                document_id=document.id,
                qdrant_client=FakeQdrantClient(),
                collection_service=FakeCollectionService(),
                embedding_service=FakeEmbeddingService(),
            )
        )


def test_qdrant_upsert_failure_is_wrapped(
    monkeypatch,
) -> None:
    document = build_document()

    chunk = build_chunk(
        document_id=document.id,
        chunk_index=0,
    )

    monkeypatch.setattr(
        indexing_module,
        "get_document_by_id",
        AsyncMock(return_value=document),
    )

    monkeypatch.setattr(
        indexing_module,
        "count_document_chunks",
        AsyncMock(return_value=1),
    )

    async def fake_list_chunks(
        *,
        session,
        document_id,
        limit,
        offset,
    ):
        return [chunk]

    monkeypatch.setattr(
        indexing_module,
        "list_document_chunks",
        fake_list_chunks,
    )

    with pytest.raises(
        QdrantPointUpsertError,
        match="Could not upsert",
    ):
        asyncio.run(
            index_document_chunks(
                session=AsyncMock(spec=AsyncSession),
                document_id=document.id,
                qdrant_client=FakeQdrantClient(
                    fail_upsert=True
                ),
                collection_service=FakeCollectionService(),
                embedding_service=FakeEmbeddingService(),
            )
        )
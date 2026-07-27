import asyncio
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from app.schemas.retrieval import (
    RetrievalResult,
    RetrievedChunk,
)
from app.services.retrieval import (
    RetrievalService,
)


class FakeEmbeddingService:
    """
    Fake embedding service used by retrieval tests.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed_query_async(
        self,
        query: str,
    ):
        self.calls.append(query)

        return SimpleNamespace(
            vector=np.array(
                [0.1, 0.2, 0.3],
                dtype=np.float32,
            )
        )


class FakeQdrantSearchService:
    """
    Fake semantic search service.
    """

    def __init__(
        self,
        *,
        chunks: tuple[RetrievedChunk, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.chunks = chunks
        self.error = error

    async def search_document(
        self,
        *,
        document_id,
        query_vector,
        collection_name,
        limit,
    ):
        self.calls.append(
            {
                "document_id": document_id,
                "query_vector": query_vector,
                "collection_name": collection_name,
                "limit": limit,
            }
        )

        if self.error is not None:
            raise self.error

        return self.chunks


def build_chunk() -> RetrievedChunk:
    """
    Create one deterministic retrieved chunk.
    """

    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        page_number=1,
        page_chunk_index=0,
        similarity_score=0.97,
        text="Evidence text.",
        character_count=14,
        word_count=2,
    )


def build_service(
    *,
    chunks: tuple[RetrievedChunk, ...] = (),
    error: Exception | None = None,
):
    embedding = FakeEmbeddingService()

    search = FakeQdrantSearchService(
        chunks=chunks,
        error=error,
    )

    service = RetrievalService(
        embedding_service=embedding,
        qdrant_search_service=search,
        collection_name="test_collection",
    )

    return (
        service,
        embedding,
        search,
    )


def test_successful_retrieval_returns_result() -> None:
    """
    Retrieval should return the retrieved chunks.
    """

    chunk = build_chunk()

    (
        service,
        embedding,
        search,
    ) = build_service(
        chunks=(chunk,),
    )

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="company policy",
        )
    )

    assert isinstance(
        result,
        RetrievalResult,
    )

    assert result.query == "company policy"

    assert result.total_results == 1

    assert result.chunks == (chunk,)

    assert embedding.calls == [
        "company policy"
    ]

    assert len(search.calls) == 1


def test_blank_query_is_rejected() -> None:
    """
    Blank queries should not be embedded.
    """

    (
        service,
        embedding,
        search,
    ) = build_service()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        asyncio.run(
            service.retrieve(
                document_id=uuid4(),
                query="     ",
            )
        )

    assert embedding.calls == []

    assert search.calls == []


def test_query_is_trimmed_before_embedding() -> None:
    """
    Leading and trailing whitespace should be removed.
    """

    (
        service,
        embedding,
        search,
    ) = build_service()

    asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="   employee handbook   ",
        )
    )

    assert embedding.calls == [
        "employee handbook"
    ]

    assert len(search.calls) == 1


def test_custom_limit_is_forwarded() -> None:
    """
    Retrieval limit should be passed to Qdrant.
    """

    (
        service,
        embedding,
        search,
    ) = build_service()

    asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="leave policy",
            limit=9,
        )
    )

    assert (
        search.calls[0]["limit"]
        == 9
    )


def test_embedding_vector_is_forwarded() -> None:
    """
    Generated embedding should be sent to Qdrant.
    """

    (
        service,
        embedding,
        search,
    ) = build_service()

    asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="salary",
        )
    )

    vector = search.calls[0][
        "query_vector"
    ]

    assert np.allclose(
        vector,
        np.array(
            [0.1, 0.2, 0.3],
            dtype=np.float32,
        ),
    )


def test_embedding_failure_is_propagated() -> None:
    """
    Embedding failures should not be swallowed.
    """

    class FailingEmbeddingService:
        async def embed_query_async(
            self,
            query,
        ):
            raise RuntimeError(
                "Embedding failed."
            )

    service = RetrievalService(
        embedding_service=(
            FailingEmbeddingService()
        ),
        qdrant_search_service=(
            FakeQdrantSearchService()
        ),
        collection_name="test",
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding failed",
    ):
        asyncio.run(
            service.retrieve(
                document_id=uuid4(),
                query="policy",
            )
        )


def test_qdrant_failure_is_propagated() -> None:
    """
    Search failures should propagate.
    """

    (
        service,
        _,
        _,
    ) = build_service(
        error=RuntimeError(
            "Qdrant unavailable."
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Qdrant unavailable",
    ):
        asyncio.run(
            service.retrieve(
                document_id=uuid4(),
                query="policy",
            )
        )


def test_empty_search_results_are_supported() -> None:
    """
    Retrieval should succeed even when nothing matches.
    """

    (
        service,
        _,
        _,
    ) = build_service()

    result = asyncio.run(
        service.retrieve(
            document_id=uuid4(),
            query="nonexistent",
        )
    )

    assert result.total_results == 0

    assert result.chunks == ()
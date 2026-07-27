import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from app.schemas.retrieval import RetrievedChunk
from app.services.qdrant_search import (
    QdrantSearchService,
)


class FakeQdrantClient:
    """
    Lightweight async fake for semantic search tests.
    """

    def __init__(
        self,
        *,
        points=None,
        fail_query: bool = False,
    ) -> None:
        self.points = points or []
        self.fail_query = fail_query

        self.query_calls: list[dict] = []

    async def query_points(
        self,
        *,
        collection_name,
        query,
        query_filter,
        limit,
        with_payload,
    ):
        if self.fail_query:
            raise RuntimeError(
                "Forced Qdrant query failure"
            )

        self.query_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "query_filter": query_filter,
                "limit": limit,
                "with_payload": with_payload,
            }
        )

        return SimpleNamespace(
            points=self.points
        )


def build_point(
    *,
    document_id,
    chunk_index: int = 0,
    page_number: int = 1,
    similarity: float = 0.95,
):
    """
    Create one fake Qdrant search point.
    """

    chunk_id = uuid4()

    return SimpleNamespace(
        score=similarity,
        payload={
            "chunk_id": str(chunk_id),
            "document_id": str(document_id),
            "chunk_index": chunk_index,
            "page_number": page_number,
            "page_chunk_index": 0,
            "text": (
                f"Chunk {chunk_index} text."
            ),
            "character_count": 20,
            "word_count": 3,
        },
    )


def build_service(
    client: FakeQdrantClient,
) -> QdrantSearchService:
    """
    Build the search service with a fake client.
    """

    return QdrantSearchService(
        client=client,
    )


def test_successful_search_returns_chunks() -> None:
    """
    Matching Qdrant points should become
    RetrievedChunk objects.
    """

    document_id = uuid4()

    client = FakeQdrantClient(
        points=[
            build_point(
                document_id=document_id,
            )
        ]
    )

    service = build_service(client)

    result = asyncio.run(
        service.search_document(
            document_id=document_id,
            query_vector=[
                0.1,
                0.2,
                0.3,
            ],
            collection_name="documents",
        )
    )

    assert len(result) == 1

    chunk = result[0]

    assert isinstance(
        chunk,
        RetrievedChunk,
    )

    assert (
        chunk.document_id
        == document_id
    )

    assert chunk.chunk_index == 0
    assert chunk.page_number == 1
    assert chunk.similarity_score == 0.95
    assert chunk.text == "Chunk 0 text."


def test_collection_name_is_forwarded() -> None:
    """
    Collection name should be passed to Qdrant.
    """

    client = FakeQdrantClient()

    service = build_service(client)

    asyncio.run(
        service.search_document(
            document_id=uuid4(),
            query_vector=[1.0],
            collection_name="legal_docs",
        )
    )

    assert (
        client.query_calls[0][
            "collection_name"
        ]
        == "legal_docs"
    )


def test_limit_is_forwarded() -> None:
    """
    Search limit should be forwarded.
    """

    client = FakeQdrantClient()

    service = build_service(client)

    asyncio.run(
        service.search_document(
            document_id=uuid4(),
            query_vector=[1.0],
            collection_name="docs",
            limit=8,
        )
    )

    assert (
        client.query_calls[0]["limit"]
        == 8
    )


def test_document_filter_is_constructed() -> None:
    """
    Search should filter by document UUID.
    """

    document_id = uuid4()

    client = FakeQdrantClient()

    service = build_service(client)

    asyncio.run(
        service.search_document(
            document_id=document_id,
            query_vector=[1.0],
            collection_name="docs",
        )
    )

    query_filter = client.query_calls[0][
        "query_filter"
    ]

    assert isinstance(
        query_filter,
        Filter,
    )

    assert len(query_filter.must) == 1

    condition = query_filter.must[0]

    assert isinstance(
        condition,
        FieldCondition,
    )

    assert (
        condition.key
        == "document_id"
    )

    assert isinstance(
        condition.match,
        MatchValue,
    )

    assert (
        condition.match.value
        == str(document_id)
    )


def test_query_vector_is_forwarded() -> None:
    """
    Query vector should be sent unchanged.
    """

    vector = [
        0.25,
        0.50,
        0.75,
    ]

    client = FakeQdrantClient()

    service = build_service(client)

    asyncio.run(
        service.search_document(
            document_id=uuid4(),
            query_vector=vector,
            collection_name="docs",
        )
    )

    assert (
        client.query_calls[0]["query"]
        == vector
    )


def test_empty_search_returns_empty_tuple() -> None:
    """
    Empty Qdrant responses should return
    an empty tuple.
    """

    client = FakeQdrantClient()

    service = build_service(client)

    result = asyncio.run(
        service.search_document(
            document_id=uuid4(),
            query_vector=[1.0],
            collection_name="docs",
        )
    )

    assert result == ()


def test_similarity_score_is_preserved() -> None:
    """
    Similarity score should be copied exactly.
    """

    document_id = uuid4()

    client = FakeQdrantClient(
        points=[
            build_point(
                document_id=document_id,
                similarity=0.87321,
            )
        ]
    )

    service = build_service(client)

    result = asyncio.run(
        service.search_document(
            document_id=document_id,
            query_vector=[1.0],
            collection_name="docs",
        )
    )

    assert (
        result[0].similarity_score
        == 0.87321
    )


def test_qdrant_failure_is_propagated() -> None:
    """
    Qdrant query failures should propagate.
    """

    client = FakeQdrantClient(
        fail_query=True,
    )

    service = build_service(client)

    with pytest.raises(
        RuntimeError,
        match="Forced Qdrant query failure",
    ):
        asyncio.run(
            service.search_document(
                document_id=uuid4(),
                query_vector=[1.0],
                collection_name="docs",
            )
        )
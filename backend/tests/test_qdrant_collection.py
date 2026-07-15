import asyncio
from types import SimpleNamespace

import pytest
from qdrant_client import models

from app.core.exceptions import (
    QdrantCollectionCreationError,
    QdrantCollectionValidationError,
)
from app.services.qdrant_collection import (
    QdrantCollectionService,
    VectorCollectionConfig,
)


COLLECTION_NAME = "test_chunks"
VECTOR_SIZE = 384
DISTANCE = models.Distance.COSINE


class FakeQdrantClient:
    """
    Lightweight async fake for Qdrant collection tests.
    """

    def __init__(
        self,
        *,
        exists: bool = False,
        vector_size: int = VECTOR_SIZE,
        distance: models.Distance = DISTANCE,
        fail_exists: bool = False,
        fail_create: bool = False,
    ) -> None:
        self.exists = exists
        self.vector_size = vector_size
        self.distance = distance
        self.fail_exists = fail_exists
        self.fail_create = fail_create

        self.collection_exists_calls: list[str] = []
        self.create_collection_calls: list[dict] = []
        self.get_collection_calls: list[str] = []

    async def collection_exists(
        self,
        *,
        collection_name: str,
    ) -> bool:
        self.collection_exists_calls.append(
            collection_name
        )

        if self.fail_exists:
            raise RuntimeError(
                "Forced Qdrant availability failure"
            )

        return self.exists

    async def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config,
    ) -> None:
        if self.fail_create:
            raise RuntimeError(
                "Forced Qdrant creation failure"
            )

        self.create_collection_calls.append(
            {
                "collection_name": collection_name,
                "vectors_config": vectors_config,
            }
        )

        self.exists = True
        self.vector_size = vectors_config.size
        self.distance = vectors_config.distance

    async def get_collection(
        self,
        *,
        collection_name: str,
    ):
        self.get_collection_calls.append(
            collection_name
        )

        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(
                        size=self.vector_size,
                        distance=self.distance,
                    )
                )
            ),
            points_count=0,
            indexed_vectors_count=0,
        )


def build_service(
    client: FakeQdrantClient,
) -> QdrantCollectionService:
    """
    Build the service with test collection settings.
    """

    return QdrantCollectionService(
        client=client,
        config=VectorCollectionConfig(
            collection_name=COLLECTION_NAME,
            vector_size=VECTOR_SIZE,
            distance=DISTANCE,
        ),
    )


def test_missing_collection_is_created_and_validated() -> None:
    client = FakeQdrantClient(
        exists=False
    )

    service = build_service(client)

    status = asyncio.run(
        service.ensure_collection()
    )

    assert status.exists is True
    assert status.vector_size == VECTOR_SIZE
    assert status.distance == DISTANCE.value
    assert status.is_compatible is True

    assert len(
        client.create_collection_calls
    ) == 1

    create_call = (
        client.create_collection_calls[0]
    )

    assert (
        create_call["collection_name"]
        == COLLECTION_NAME
    )

    assert (
        create_call["vectors_config"].size
        == VECTOR_SIZE
    )

    assert (
        create_call["vectors_config"].distance
        == DISTANCE
    )


def test_existing_compatible_collection_is_not_recreated() -> None:
    client = FakeQdrantClient(
        exists=True,
        vector_size=VECTOR_SIZE,
        distance=DISTANCE,
    )

    service = build_service(client)

    status = asyncio.run(
        service.ensure_collection()
    )

    assert status.exists is True
    assert status.is_compatible is True
    assert client.create_collection_calls == []


def test_existing_wrong_vector_size_is_rejected() -> None:
    client = FakeQdrantClient(
        exists=True,
        vector_size=768,
        distance=DISTANCE,
    )

    service = build_service(client)

    with pytest.raises(
        QdrantCollectionValidationError,
        match="incompatible",
    ):
        asyncio.run(
            service.ensure_collection()
        )


def test_existing_wrong_distance_is_rejected() -> None:
    client = FakeQdrantClient(
        exists=True,
        vector_size=VECTOR_SIZE,
        distance=models.Distance.DOT,
    )

    service = build_service(client)

    with pytest.raises(
        QdrantCollectionValidationError,
        match="incompatible",
    ):
        asyncio.run(
            service.ensure_collection()
        )


def test_creation_failure_is_wrapped() -> None:
    client = FakeQdrantClient(
        exists=False,
        fail_create=True,
    )

    service = build_service(client)

    with pytest.raises(
        QdrantCollectionCreationError,
        match="Could not create",
    ):
        asyncio.run(
            service.ensure_collection()
        )


def test_status_reports_missing_collection() -> None:
    client = FakeQdrantClient(
        exists=False
    )

    service = build_service(client)

    status = asyncio.run(
        service.get_collection_status()
    )

    assert status.exists is False
    assert status.is_compatible is False
    assert status.vector_size is None
    assert status.distance is None
    assert "does not exist" in status.message
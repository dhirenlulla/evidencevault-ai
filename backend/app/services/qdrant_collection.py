from dataclasses import dataclass
from typing import Any

from qdrant_client import models
from qdrant_client.async_qdrant_client import (
    AsyncQdrantClient,
)

from app.core.config import get_settings
from app.core.exceptions import (
    QdrantCollectionCreationError,
    QdrantCollectionUnavailableError,
    QdrantCollectionValidationError,
)
from app.schemas.vector_collection import (
    VectorCollectionStatusResponse,
)


@dataclass(
    frozen=True,
    slots=True,
)
class VectorCollectionConfig:
    """
    Expected Qdrant collection configuration.
    """

    collection_name: str
    vector_size: int
    distance: models.Distance


@dataclass(
    frozen=True,
    slots=True,
)
class VectorCollectionStatus:
    """
    Internal status report for one Qdrant collection.
    """

    collection_name: str
    exists: bool
    vector_size: int | None
    distance: str | None
    expected_vector_size: int
    expected_distance: str
    is_compatible: bool
    points_count: int | None
    indexed_vectors_count: int | None
    message: str

    def to_response(
        self,
    ) -> VectorCollectionStatusResponse:
        """
        Convert internal status into an API-ready schema.
        """

        return VectorCollectionStatusResponse(
            collection_name=self.collection_name,
            exists=self.exists,
            vector_size=self.vector_size,
            distance=self.distance,
            expected_vector_size=(
                self.expected_vector_size
            ),
            expected_distance=(
                self.expected_distance
            ),
            is_compatible=self.is_compatible,
            points_count=self.points_count,
            indexed_vectors_count=(
                self.indexed_vectors_count
            ),
            message=self.message,
        )


def build_vector_collection_config() -> (
    VectorCollectionConfig
):
    """
    Build the expected Qdrant collection config
    from application settings.
    """

    settings = get_settings()

    try:
        distance = models.Distance(
            settings.qdrant_vector_distance
        )

    except ValueError as exc:
        raise QdrantCollectionValidationError(
            "Unsupported Qdrant distance metric: "
            f"{settings.qdrant_vector_distance}"
        ) from exc

    return VectorCollectionConfig(
        collection_name=(
            settings.qdrant_collection_name
        ),
        vector_size=settings.embedding_dimension,
        distance=distance,
    )


def _extract_vector_config(
    collection_info: Any,
) -> Any:
    """
    Extract vector configuration from Qdrant collection info.

    Qdrant can represent vector config slightly differently
    depending on single-vector or named-vector mode. EvidenceVault
    currently uses single-vector mode.
    """

    config = collection_info.config

    params = config.params

    vectors = params.vectors

    if hasattr(
        vectors,
        "size",
    ):
        return vectors

    if isinstance(
        vectors,
        dict,
    ):
        if "" in vectors:
            return vectors[""]

        if "default" in vectors:
            return vectors["default"]

        if len(vectors) == 1:
            return next(
                iter(vectors.values())
            )

    raise QdrantCollectionValidationError(
        "Could not read vector configuration "
        "from Qdrant collection metadata."
    )


def _normalize_distance(
    distance: Any,
) -> str:
    """
    Convert Qdrant distance values into a stable string.
    """

    if hasattr(
        distance,
        "value",
    ):
        return str(distance.value)

    return str(distance)


def _extract_points_count(
    collection_info: Any,
) -> int | None:
    """
    Read point count from Qdrant collection info.
    """

    value = getattr(
        collection_info,
        "points_count",
        None,
    )

    if value is None:
        return None

    return int(value)


def _extract_indexed_vectors_count(
    collection_info: Any,
) -> int | None:
    """
    Read indexed vector count from Qdrant collection info.
    """

    value = getattr(
        collection_info,
        "indexed_vectors_count",
        None,
    )

    if value is None:
        return None

    return int(value)


class QdrantCollectionService:
    """
    Create and validate the Qdrant collection used
    for EvidenceVault chunk embeddings.
    """

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        config: VectorCollectionConfig,
    ) -> None:
        self._client = client
        self._config = config

    @property
    def collection_name(
        self,
    ) -> str:
        return self._config.collection_name

    @property
    def vector_size(
        self,
    ) -> int:
        return self._config.vector_size

    @property
    def distance(
        self,
    ) -> models.Distance:
        return self._config.distance

    async def collection_exists(
        self,
    ) -> bool:
        """
        Return whether the configured collection exists.
        """

        try:
            return bool(
                await self._client.collection_exists(
                    collection_name=(
                        self.collection_name
                    )
                )
            )

        except Exception as exc:
            raise QdrantCollectionUnavailableError(
                "Could not check whether the "
                "Qdrant collection exists."
            ) from exc

    async def create_collection(
        self,
    ) -> None:
        """
        Create the configured Qdrant collection.
        """

        try:
            await self._client.create_collection(
                collection_name=(
                    self.collection_name
                ),
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
            )

        except Exception as exc:
            raise QdrantCollectionCreationError(
                "Could not create Qdrant "
                "collection: "
                f"{self.collection_name}"
            ) from exc

    async def get_collection_status(
        self,
    ) -> VectorCollectionStatus:
        """
        Return current collection status and compatibility.
        """

        exists = await self.collection_exists()

        if not exists:
            return VectorCollectionStatus(
                collection_name=self.collection_name,
                exists=False,
                vector_size=None,
                distance=None,
                expected_vector_size=(
                    self.vector_size
                ),
                expected_distance=(
                    self.distance.value
                ),
                is_compatible=False,
                points_count=None,
                indexed_vectors_count=None,
                message=(
                    "Qdrant collection does not exist."
                ),
            )

        try:
            collection_info = await (
                self._client.get_collection(
                    collection_name=(
                        self.collection_name
                    )
                )
            )

        except Exception as exc:
            raise QdrantCollectionUnavailableError(
                "Could not load Qdrant collection "
                "metadata."
            ) from exc

        vector_config = _extract_vector_config(
            collection_info
        )

        actual_size = int(
            vector_config.size
        )

        actual_distance = _normalize_distance(
            vector_config.distance
        )

        expected_distance = (
            self.distance.value
        )

        size_matches = (
            actual_size == self.vector_size
        )

        distance_matches = (
            actual_distance.lower()
            == expected_distance.lower()
        )

        is_compatible = (
            size_matches
            and distance_matches
        )

        if is_compatible:
            message = (
                "Qdrant collection is compatible "
                "with the configured embedding model."
            )
        else:
            message = (
                "Qdrant collection exists but does "
                "not match the configured vector "
                "settings."
            )

        return VectorCollectionStatus(
            collection_name=self.collection_name,
            exists=True,
            vector_size=actual_size,
            distance=actual_distance,
            expected_vector_size=self.vector_size,
            expected_distance=expected_distance,
            is_compatible=is_compatible,
            points_count=_extract_points_count(
                collection_info
            ),
            indexed_vectors_count=(
                _extract_indexed_vectors_count(
                    collection_info
                )
            ),
            message=message,
        )

    async def validate_collection(
        self,
    ) -> VectorCollectionStatus:
        """
        Validate that the existing collection matches
        EvidenceVault's configured vector settings.
        """

        status = await self.get_collection_status()

        if not status.exists:
            raise QdrantCollectionValidationError(
                "Qdrant collection does not exist: "
                f"{self.collection_name}"
            )

        if not status.is_compatible:
            raise QdrantCollectionValidationError(
                "Qdrant collection is incompatible. "
                f"Expected size "
                f"{status.expected_vector_size} "
                f"and distance "
                f"{status.expected_distance}. "
                f"Found size {status.vector_size} "
                f"and distance {status.distance}."
            )

        return status

    async def ensure_collection(
        self,
    ) -> VectorCollectionStatus:
        """
        Create the collection if missing, then validate it.
        """

        exists = await self.collection_exists()

        if not exists:
            await self.create_collection()

        return await self.validate_collection()
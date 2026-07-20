import asyncio
import threading
from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import (
    SentenceTransformer,
)

from app.core.config import get_settings
from app.core.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingGenerationError,
    EmbeddingModelLoadError,
    InvalidEmbeddingInputError,
)


FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


@dataclass(
    frozen=True,
    slots=True,
)
class DocumentEmbeddingResult:
    """
    Embeddings generated for document chunks.
    """

    model_name: str
    dimension: int
    normalized: bool
    vectors: FloatMatrix

    @property
    def count(self) -> int:
        """
        Return the number of generated vectors.
        """

        return int(
            self.vectors.shape[0]
        )


@dataclass(
    frozen=True,
    slots=True,
)
class QueryEmbeddingResult:
    """
    Embedding generated for one search query.
    """

    model_name: str
    dimension: int
    normalized: bool
    vector: FloatVector


class EmbeddingService:
    """
    Generate validated document and query embeddings.

    The model is loaded lazily and reused for future
    requests. Model inference is kept behind a lock
    so one shared model instance is not invoked by
    multiple threads simultaneously.
    """

    def __init__(
        self,
        *,
        model_name: str,
        expected_dimension: int,
        batch_size: int,
        device: str,
        normalize_embeddings: bool,
        query_instruction: str,
        model_factory: Callable[..., SentenceTransformer] = (
            SentenceTransformer
        ),
    ) -> None:
        self._model_name = model_name
        self._expected_dimension = (
            expected_dimension
        )
        self._batch_size = batch_size
        self._device = device
        self._normalize_embeddings = (
            normalize_embeddings
        )
        self._query_instruction = (
            query_instruction
        )
        self._model_factory = model_factory

        self._model: (
            SentenceTransformer | None
        ) = None

        self._model_load_lock = (
            threading.Lock()
        )

        self._inference_lock = (
            threading.Lock()
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._expected_dimension

    @property
    def normalized(self) -> bool:
        return self._normalize_embeddings

    def _load_model(
        self,
    ) -> SentenceTransformer:
        """
        Load and validate the configured model once.
        """

        if self._model is not None:
            return self._model

        with self._model_load_lock:
            if self._model is not None:
                return self._model

            try:
                model = self._model_factory(
                    self._model_name,
                    device=self._device,
                )

            except Exception as exc:
                raise EmbeddingModelLoadError(
                    "The embedding model could "
                    "not be loaded: "
                    f"{self._model_name}"
                ) from exc

            actual_dimension = (
                model
                .get_embedding_dimension()
            )

            if actual_dimension is None:
                raise (
                    EmbeddingDimensionMismatchError(
                        "The embedding model did "
                        "not report an output "
                        "dimension."
                    )
                )

            if (
                int(actual_dimension)
                != self._expected_dimension
            ):
                raise (
                    EmbeddingDimensionMismatchError(
                        "Embedding dimension "
                        "mismatch. "
                        f"Configured: "
                        f"{self._expected_dimension}. "
                        f"Model output: "
                        f"{actual_dimension}."
                    )
                )

            self._model = model

        return self._model

    @staticmethod
    def _prepare_documents(
        texts: Sequence[str],
    ) -> tuple[str, ...]:
        """
        Validate and normalize document text input.
        """

        if isinstance(texts, str):
            raise InvalidEmbeddingInputError(
                "Document embeddings require "
                "a sequence of strings, not "
                "one raw string."
            )

        if not texts:
            raise InvalidEmbeddingInputError(
                "At least one document text "
                "is required."
            )

        prepared_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise InvalidEmbeddingInputError(
                    "Every document text must "
                    "be a string. "
                    f"Invalid index: {index}."
                )

            cleaned_text = text.strip()

            if not cleaned_text:
                raise InvalidEmbeddingInputError(
                    "Document text cannot be "
                    "blank. "
                    f"Invalid index: {index}."
                )

            prepared_texts.append(
                cleaned_text
            )

        return tuple(prepared_texts)

    @staticmethod
    def _prepare_query(
        query: str,
    ) -> str:
        """
        Validate and normalize one query.
        """

        if not isinstance(query, str):
            raise InvalidEmbeddingInputError(
                "The search query must be "
                "a string."
            )

        cleaned_query = query.strip()

        if not cleaned_query:
            raise InvalidEmbeddingInputError(
                "The search query cannot "
                "be blank."
            )

        return cleaned_query

    def _validate_matrix(
        self,
        raw_vectors: object,
        *,
        expected_count: int,
    ) -> FloatMatrix:
        """
        Convert and validate a model output matrix.
        """

        try:
            vectors = np.asarray(
                raw_vectors,
                dtype=np.float32,
            )

        except Exception as exc:
            raise EmbeddingGenerationError(
                "The model returned embeddings "
                "that could not be converted "
                "to float32."
            ) from exc

        if vectors.ndim == 1:
            vectors = vectors.reshape(
                1,
                -1,
            )

        expected_shape = (
            expected_count,
            self._expected_dimension,
        )

        if vectors.shape != expected_shape:
            raise (
                EmbeddingDimensionMismatchError(
                    "Unexpected embedding matrix "
                    "shape. "
                    f"Expected: {expected_shape}. "
                    f"Received: {vectors.shape}."
                )
            )

        if not np.isfinite(vectors).all():
            raise EmbeddingGenerationError(
                "The model returned NaN or "
                "infinite embedding values."
            )

        return np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> DocumentEmbeddingResult:
        """
        Generate embeddings for document chunks.
        """

        prepared_texts = (
            self._prepare_documents(texts)
        )

        model = self._load_model()

        try:
            with self._inference_lock:
                raw_vectors = (
                    model.encode_document(
                        list(prepared_texts),
                        batch_size=(
                            self._batch_size
                        ),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=(
                            self
                            ._normalize_embeddings
                        ),
                        precision="float32",
                    )
                )

        except Exception as exc:
            raise EmbeddingGenerationError(
                "Document embeddings could "
                "not be generated."
            ) from exc

        vectors = self._validate_matrix(
            raw_vectors,
            expected_count=len(
                prepared_texts
            ),
        )

        return DocumentEmbeddingResult(
            model_name=self._model_name,
            dimension=(
                self._expected_dimension
            ),
            normalized=(
                self._normalize_embeddings
            ),
            vectors=vectors,
        )

    def embed_query(
        self,
        query: str,
    ) -> QueryEmbeddingResult:
        """
        Generate one retrieval-query embedding.
        """

        prepared_query = (
            self._prepare_query(query)
        )

        model = self._load_model()

        try:
            with self._inference_lock:
                raw_vectors = (
                    model.encode_query(
                        [prepared_query],
                        prompt=(
                            self
                            ._query_instruction
                        ),
                        batch_size=1,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=(
                            self
                            ._normalize_embeddings
                        ),
                        precision="float32",
                    )
                )

        except Exception as exc:
            raise EmbeddingGenerationError(
                "The query embedding could "
                "not be generated."
            ) from exc

        vectors = self._validate_matrix(
            raw_vectors,
            expected_count=1,
        )

        return QueryEmbeddingResult(
            model_name=self._model_name,
            dimension=(
                self._expected_dimension
            ),
            normalized=(
                self._normalize_embeddings
            ),
            vector=vectors[0],
        )

    async def embed_documents_async(
        self,
        texts: Sequence[str],
    ) -> DocumentEmbeddingResult:
        """
        Generate document embeddings without
        blocking FastAPI's event loop.
        """

        prepared_texts = tuple(texts)

        return await asyncio.to_thread(
            self.embed_documents,
            prepared_texts,
        )

    async def embed_query_async(
        self,
        query: str,
    ) -> QueryEmbeddingResult:
        """
        Generate a query embedding without
        blocking FastAPI's event loop.
        """

        return await asyncio.to_thread(
            self.embed_query,
            query,
        )


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """
    Create and cache the configured embedding service.
    """

    settings = get_settings()

    return EmbeddingService(
        model_name=(
            settings.embedding_model_name
        ),
        expected_dimension=(
            settings.embedding_dimension
        ),
        batch_size=(
            settings.embedding_batch_size
        ),
        device=(
            settings.embedding_device
        ),
        normalize_embeddings=(
            settings.embedding_normalize
        ),
        query_instruction=(
            settings
            .embedding_query_instruction
        ),
    )
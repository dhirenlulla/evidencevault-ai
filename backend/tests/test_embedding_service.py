import asyncio
from unittest.mock import Mock

import numpy as np
import pytest

from app.core.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingGenerationError,
    InvalidEmbeddingInputError,
)
from app.services.embedding import (
    EmbeddingService,
)


MODEL_NAME = "test-embedding-model"
DIMENSION = 384
QUERY_INSTRUCTION = (
    "Represent this sentence for searching "
    "relevant passages: "
)


class FakeEmbeddingModel:
    """
    Lightweight replacement for SentenceTransformer.
    """

    def __init__(
        self,
        *,
        dimension: int = DIMENSION,
    ) -> None:
        self.dimension = dimension
        self.document_calls: list[
            tuple[list[str], dict]
        ] = []
        self.query_calls: list[
            tuple[list[str], dict]
        ] = []

        self.raise_document_error = False
        self.raise_query_error = False

    def get_sentence_embedding_dimension(
        self,
    ) -> int:
        return self.dimension

    def encode_document(
        self,
        texts: list[str],
        **kwargs,
    ) -> np.ndarray:
        self.document_calls.append(
            (
                texts,
                kwargs,
            )
        )

        if self.raise_document_error:
            raise RuntimeError(
                "Forced document error"
            )

        vectors = np.zeros(
            (
                len(texts),
                self.dimension,
            ),
            dtype=np.float32,
        )

        for index in range(len(texts)):
            vectors[
                index,
                index % self.dimension,
            ] = 1.0

        return vectors

    def encode_query(
        self,
        texts: list[str],
        **kwargs,
    ) -> np.ndarray:
        self.query_calls.append(
            (
                texts,
                kwargs,
            )
        )

        if self.raise_query_error:
            raise RuntimeError(
                "Forced query error"
            )

        vectors = np.zeros(
            (
                len(texts),
                self.dimension,
            ),
            dtype=np.float32,
        )

        vectors[:, 0] = 1.0

        return vectors


def build_service(
    fake_model: FakeEmbeddingModel,
) -> tuple[
    EmbeddingService,
    Mock,
]:
    """
    Create an embedding service using a fake model.
    """

    model_factory = Mock(
        return_value=fake_model
    )

    service = EmbeddingService(
        model_name=MODEL_NAME,
        expected_dimension=DIMENSION,
        batch_size=8,
        device="cpu",
        normalize_embeddings=True,
        query_instruction=(
            QUERY_INSTRUCTION
        ),
        model_factory=model_factory,
    )

    return service, model_factory


def test_model_is_loaded_lazily_and_once() -> None:
    fake_model = FakeEmbeddingModel()

    service, model_factory = build_service(
        fake_model
    )

    model_factory.assert_not_called()

    service.embed_documents(
        ["First document"]
    )

    service.embed_query(
        "First query"
    )

    model_factory.assert_called_once_with(
        MODEL_NAME,
        device="cpu",
    )


def test_document_embeddings_have_expected_shape() -> None:
    fake_model = FakeEmbeddingModel()

    service, _ = build_service(
        fake_model
    )

    result = service.embed_documents(
        [
            "First document",
            "Second document",
        ]
    )

    assert result.count == 2
    assert result.dimension == DIMENSION

    assert result.vectors.shape == (
        2,
        DIMENSION,
    )

    assert (
        result.vectors.dtype
        == np.float32
    )

    assert np.isfinite(
        result.vectors
    ).all()


def test_query_uses_retrieval_instruction() -> None:
    fake_model = FakeEmbeddingModel()

    service, _ = build_service(
        fake_model
    )

    result = service.embed_query(
        "What is deterministic chunking?"
    )

    assert result.vector.shape == (
        DIMENSION,
    )

    assert len(
        fake_model.query_calls
    ) == 1

    texts, keyword_arguments = (
        fake_model.query_calls[0]
    )

    assert texts == [
        "What is deterministic chunking?"
    ]

    assert (
        keyword_arguments["prompt"]
        == QUERY_INSTRUCTION
    )

    assert (
        keyword_arguments[
            "normalize_embeddings"
        ]
        is True
    )


def test_blank_document_input_is_rejected() -> None:
    fake_model = FakeEmbeddingModel()

    service, _ = build_service(
        fake_model
    )

    with pytest.raises(
        InvalidEmbeddingInputError,
        match="At least one",
    ):
        service.embed_documents([])

    with pytest.raises(
        InvalidEmbeddingInputError,
        match="cannot be blank",
    ):
        service.embed_documents(
            ["   "]
        )


def test_blank_query_is_rejected() -> None:
    fake_model = FakeEmbeddingModel()

    service, _ = build_service(
        fake_model
    )

    with pytest.raises(
        InvalidEmbeddingInputError,
        match="cannot be blank",
    ):
        service.embed_query("   ")


def test_model_dimension_mismatch_is_rejected() -> None:
    fake_model = FakeEmbeddingModel(
        dimension=768
    )

    service, _ = build_service(
        fake_model
    )

    with pytest.raises(
        EmbeddingDimensionMismatchError,
        match="dimension mismatch",
    ):
        service.embed_documents(
            ["Document text"]
        )


def test_model_generation_failure_is_wrapped() -> None:
    fake_model = FakeEmbeddingModel()

    fake_model.raise_document_error = True

    service, _ = build_service(
        fake_model
    )

    with pytest.raises(
        EmbeddingGenerationError,
        match="could not be generated",
    ):
        service.embed_documents(
            ["Document text"]
        )


def test_async_embedding_wrappers() -> None:
    fake_model = FakeEmbeddingModel()

    service, _ = build_service(
        fake_model
    )

    async def run_test():
        document_result = (
            await service
            .embed_documents_async(
                [
                    "First document",
                    "Second document",
                ]
            )
        )

        query_result = (
            await service
            .embed_query_async(
                "Find the first document"
            )
        )

        return (
            document_result,
            query_result,
        )

    document_result, query_result = (
        asyncio.run(run_test())
    )

    assert document_result.count == 2

    assert query_result.vector.shape == (
        DIMENSION,
    )
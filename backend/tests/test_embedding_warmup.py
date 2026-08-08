import asyncio

from app.services.embedding import EmbeddingService


class FakeModel:
    def __init__(self, dimension: int):
        self._dimension = dimension
        self.encode_calls = 0

    def get_embedding_dimension(self):
        return self._dimension

    def encode(self, *args, **kwargs):
        self.encode_calls += 1
        return [[0.0] * self._dimension]


def build_service(*, dimension: int = 8):
    load_calls = []

    def factory(model_name, device):
        load_calls.append(1)
        return FakeModel(dimension)

    service = EmbeddingService(
        model_name="fake-model",
        expected_dimension=dimension,
        batch_size=8,
        device="cpu",
        normalize_embeddings=True,
        query_instruction="",
        model_factory=factory,
    )

    return service, load_calls


def test_is_model_loaded_is_false_before_any_use() -> None:
    service, _ = build_service()

    assert service.is_model_loaded is False


def test_warm_up_loads_the_model() -> None:
    service, load_calls = build_service()

    service.warm_up()

    assert service.is_model_loaded is True
    assert len(load_calls) == 1


def test_warm_up_does_not_load_twice_if_already_loaded() -> None:
    service, load_calls = build_service()

    service.warm_up()
    service.warm_up()

    assert len(load_calls) == 1


def test_warm_up_async_loads_the_model() -> None:
    service, load_calls = build_service()

    asyncio.run(service.warm_up_async())

    assert service.is_model_loaded is True
    assert len(load_calls) == 1


def test_is_model_loaded_does_not_itself_trigger_a_load() -> None:
    service, load_calls = build_service()

    # Reading the property several times should never load
    # anything on its own.
    _ = service.is_model_loaded
    _ = service.is_model_loaded
    _ = service.is_model_loaded

    assert len(load_calls) == 0
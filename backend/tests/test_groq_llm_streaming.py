import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

import app.services.groq_llm as groq_module
from app.core.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.services.groq_llm import GroqLLMService


def build_settings():
    """
    Create deterministic application settings for tests.
    """

    return SimpleNamespace(
        groq_api_key="test-api-key",
        groq_model_name="llama-test-model",
        llm_timeout_seconds=30.0,
        llm_temperature=0.0,
        llm_max_tokens=512,
    )


def build_status_error(
    status_code: int,
) -> APIStatusError:
    """
    Construct a Groq APIStatusError.
    """

    request = httpx.Request(
        "POST",
        "https://api.groq.com",
    )

    response = httpx.Response(
        status_code=status_code,
        request=request,
    )

    return APIStatusError(
        "Forced API status error",
        response=response,
        body=None,
    )


def build_stream_chunk(content):
    """
    Build one fake streamed completion chunk. content=None
    simulates a chunk with no text delta (which real Groq
    streams sometimes send, e.g. the final chunk).
    """

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                )
            )
        ]
    )


class FakeStream:
    """
    Fake async iterator standing in for Groq's streamed
    response.
    """

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._generator()

    async def _generator(self):
        for chunk in self._chunks:
            yield chunk


class FakeGroqStreamingClient:
    """
    Lightweight async fake for the Groq SDK's streaming
    interface.
    """

    def __init__(
        self,
        *,
        chunks=None,
        exception=None,
        exception_during_iteration=None,
    ):
        self.chunks = chunks or []
        self.exception = exception
        self.exception_during_iteration = (
            exception_during_iteration
        )

        self.create = AsyncMock(
            side_effect=self._create,
        )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create,
            )
        )

    async def _create(self, **kwargs):
        if self.exception is not None:
            raise self.exception

        if self.exception_during_iteration is not None:
            return self._broken_stream()

        return FakeStream(self.chunks)

    async def _broken_stream(self):
        raise self.exception_during_iteration
        yield  # pragma: no cover - unreachable, makes this an async generator


def build_service(
    monkeypatch,
    *,
    client,
):
    """
    Construct GroqLLMService with mocked settings and
    client.
    """

    settings = build_settings()

    monkeypatch.setattr(
        groq_module,
        "get_settings",
        lambda: settings,
    )

    monkeypatch.setattr(
        groq_module,
        "AsyncGroq",
        lambda **kwargs: client,
    )

    return GroqLLMService()


async def collect_stream(service, **kwargs):
    """
    Drain an async generator into a plain list, for easy
    assertions.
    """

    return [
        fragment
        async for fragment in service.generate_answer_stream(
            **kwargs
        )
    ]


def test_stream_yields_each_content_fragment_in_order(
    monkeypatch,
) -> None:
    client = FakeGroqStreamingClient(
        chunks=[
            build_stream_chunk("Hello"),
            build_stream_chunk(", "),
            build_stream_chunk("world."),
        ]
    )

    service = build_service(monkeypatch, client=client)

    fragments = asyncio.run(
        collect_stream(
            service,
            system_prompt="System",
            user_prompt="Question",
        )
    )

    assert fragments == ["Hello", ", ", "world."]


def test_stream_skips_chunks_with_no_content(
    monkeypatch,
) -> None:
    """
    Chunks with no text delta (content=None) — common for
    the final chunk of a real stream — should be silently
    skipped, not yielded as empty strings.
    """

    client = FakeGroqStreamingClient(
        chunks=[
            build_stream_chunk("Answer"),
            build_stream_chunk(None),
        ]
    )

    service = build_service(monkeypatch, client=client)

    fragments = asyncio.run(
        collect_stream(
            service,
            system_prompt="System",
            user_prompt="Question",
        )
    )

    assert fragments == ["Answer"]


def test_stream_skips_chunks_with_no_choices(
    monkeypatch,
) -> None:
    client = FakeGroqStreamingClient(
        chunks=[
            SimpleNamespace(choices=[]),
            build_stream_chunk("Answer"),
        ]
    )

    service = build_service(monkeypatch, client=client)

    fragments = asyncio.run(
        collect_stream(
            service,
            system_prompt="System",
            user_prompt="Question",
        )
    )

    assert fragments == ["Answer"]


def test_request_is_made_with_stream_true(
    monkeypatch,
) -> None:
    client = FakeGroqStreamingClient(
        chunks=[build_stream_chunk("Answer")]
    )

    service = build_service(monkeypatch, client=client)

    asyncio.run(
        collect_stream(
            service,
            system_prompt="System Prompt",
            user_prompt="User Prompt",
        )
    )

    kwargs = client.create.await_args.kwargs

    assert kwargs["stream"] is True
    assert kwargs["model"] == "llama-test-model"

    assert kwargs["messages"] == [
        {
            "role": "system",
            "content": "System Prompt",
        },
        {
            "role": "user",
            "content": "User Prompt",
        },
    ]


def test_empty_stream_raises_generation_error(
    monkeypatch,
) -> None:
    """
    A stream that never yields any real content should
    raise, the same way an empty non-streamed response does.
    """

    client = FakeGroqStreamingClient(chunks=[])

    service = build_service(monkeypatch, client=client)

    async def run():
        async for _ in service.generate_answer_stream(
            system_prompt="System",
            user_prompt="Question",
        ):
            pass

    with pytest.raises(
        LLMGenerationError,
        match="empty streamed response",
    ):
        asyncio.run(run())


def test_connection_error_on_request_is_wrapped(
    monkeypatch,
) -> None:
    client = FakeGroqStreamingClient(
        exception=APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://api.groq.com",
            )
        )
    )

    service = build_service(monkeypatch, client=client)

    async def run():
        async for _ in service.generate_answer_stream(
            system_prompt="system",
            user_prompt="user",
        ):
            pass

    with pytest.raises(
        LLMConnectionError,
        match="Could not connect",
    ):
        asyncio.run(run())


def test_timeout_error_on_request_is_wrapped(
    monkeypatch,
) -> None:
    client = FakeGroqStreamingClient(
        exception=APITimeoutError(
            request=httpx.Request(
                "POST",
                "https://api.groq.com",
            )
        )
    )

    service = build_service(monkeypatch, client=client)

    async def run():
        async for _ in service.generate_answer_stream(
            system_prompt="system",
            user_prompt="user",
        ):
            pass

    with pytest.raises(
        LLMTimeoutError,
        match="timed out",
    ):
        asyncio.run(run())


def test_authentication_error_is_wrapped(
    monkeypatch,
) -> None:
    client = FakeGroqStreamingClient(
        exception=build_status_error(401)
    )

    service = build_service(monkeypatch, client=client)

    async def run():
        async for _ in service.generate_answer_stream(
            system_prompt="system",
            user_prompt="user",
        ):
            pass

    with pytest.raises(
        LLMAuthenticationError,
        match="Invalid Groq API key",
    ):
        asyncio.run(run())


def test_rate_limit_error_is_wrapped(
    monkeypatch,
) -> None:
    client = FakeGroqStreamingClient(
        exception=build_status_error(429)
    )

    service = build_service(monkeypatch, client=client)

    async def run():
        async for _ in service.generate_answer_stream(
            system_prompt="system",
            user_prompt="user",
        ):
            pass

    with pytest.raises(
        LLMRateLimitError,
        match="rate limit",
    ):
        asyncio.run(run())


def test_error_partway_through_the_stream_is_wrapped(
    monkeypatch,
) -> None:
    """
    A failure that happens mid-stream (after the connection
    was already established and some chunks may have been
    sent) should still be wrapped the same way as a failure
    on the initial request.
    """

    client = FakeGroqStreamingClient(
        exception_during_iteration=APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://api.groq.com",
            )
        )
    )

    service = build_service(monkeypatch, client=client)

    async def run():
        async for _ in service.generate_answer_stream(
            system_prompt="system",
            user_prompt="user",
        ):
            pass

    with pytest.raises(
        LLMConnectionError,
        match="Could not connect",
    ):
        asyncio.run(run())


def test_unexpected_error_is_wrapped(
    monkeypatch,
) -> None:
    client = FakeGroqStreamingClient(
        exception=RuntimeError("boom")
    )

    service = build_service(monkeypatch, client=client)

    async def run():
        async for _ in service.generate_answer_stream(
            system_prompt="system",
            user_prompt="user",
        ):
            pass

    with pytest.raises(
        LLMGenerationError,
        match="unexpected error",
    ):
        asyncio.run(run())
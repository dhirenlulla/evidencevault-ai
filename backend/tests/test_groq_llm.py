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
from app.services.groq_llm import (
    GroqLLMService,
)


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


def build_success_response(
    answer: str,
):
    """
    Build a fake Groq completion response.
    """

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=answer,
                )
            )
        ]
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


class FakeGroqClient:
    """
    Lightweight async fake for Groq SDK.
    """

    def __init__(
        self,
        *,
        response=None,
        exception=None,
    ):
        self.response = response
        self.exception = exception

        self.create = AsyncMock(
            side_effect=self._create,
        )

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create,
            )
        )

    async def _create(
        self,
        **kwargs,
    ):
        if self.exception is not None:
            raise self.exception

        return self.response


def build_service(
    monkeypatch,
    *,
    client,
):
    """
    Construct GroqLLMService with
    mocked settings and client.
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


def test_model_name_property_returns_configuration(
    monkeypatch,
) -> None:
    """
    model_name should expose the configured
    Groq model.
    """

    client = FakeGroqClient()

    service = build_service(
        monkeypatch,
        client=client,
    )

    assert (
        service.model_name
        == "llama-test-model"
    )


def test_successful_generation_returns_answer(
    monkeypatch,
) -> None:
    """
    Successful completion should return
    the generated answer.
    """

    client = FakeGroqClient(
        response=build_success_response(
            "Generated answer."
        )
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    result = asyncio.run(
        service.generate_answer(
            system_prompt="System",
            user_prompt="Question",
        )
    )

    assert (
        result
        == "Generated answer."
    )


def test_answer_is_stripped(
    monkeypatch,
) -> None:
    """
    Leading and trailing whitespace
    should be removed.
    """

    client = FakeGroqClient(
        response=build_success_response(
            "   Final answer.   "
        )
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    result = asyncio.run(
        service.generate_answer(
            system_prompt="System",
            user_prompt="Question",
        )
    )

    assert result == "Final answer."


def test_prompts_are_forwarded_to_groq(
    monkeypatch,
) -> None:
    """
    System and user prompts should be
    forwarded unchanged.
    """

    client = FakeGroqClient(
        response=build_success_response(
            "Answer"
        )
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    asyncio.run(
        service.generate_answer(
            system_prompt="System Prompt",
            user_prompt="User Prompt",
        )
    )

    kwargs = (
        client.create.await_args.kwargs
    )

    assert (
        kwargs["model"]
        == "llama-test-model"
    )

    assert (
        kwargs["temperature"]
        == 0.0
    )

    assert (
        kwargs["max_completion_tokens"]
        == 512
    )

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


def test_missing_choices_raise_generation_error(
    monkeypatch,
) -> None:
    """
    Empty completion choices should
    become LLMGenerationError.
    """

    client = FakeGroqClient(
        response=SimpleNamespace(
            choices=[],
        )
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    with pytest.raises(
        LLMGenerationError,
        match="no completion choices",
    ):
        asyncio.run(
            service.generate_answer(
                system_prompt="System",
                user_prompt="User",
            )
        )

def test_connection_error_is_wrapped(
    monkeypatch,
) -> None:
    """
    APIConnectionError should become
    LLMConnectionError.
    """

    client = FakeGroqClient(
        exception=APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://api.groq.com",
            )
        )
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    with pytest.raises(
        LLMConnectionError,
        match="Could not connect",
    ):
        asyncio.run(
            service.generate_answer(
                system_prompt="system",
                user_prompt="user",
            )
        )
        
def test_timeout_error_is_wrapped(
    monkeypatch,
) -> None:
    """
    APITimeoutError should become
    LLMTimeoutError.
    """

    client = FakeGroqClient(
        exception=APITimeoutError(
            request=httpx.Request(
                "POST",
                "https://api.groq.com",
            )
        )
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    with pytest.raises(
        LLMTimeoutError,
        match="timed out",
    ):
        asyncio.run(
            service.generate_answer(
                system_prompt="system",
                user_prompt="user",
            )
        )
        
def test_authentication_error_is_wrapped(
    monkeypatch,
) -> None:
    """
    HTTP 401 should become
    LLMAuthenticationError.
    """

    client = FakeGroqClient(
        exception=build_status_error(401)
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    with pytest.raises(
        LLMAuthenticationError,
        match="Invalid Groq API key",
    ):
        asyncio.run(
            service.generate_answer(
                system_prompt="system",
                user_prompt="user",
            )
        )
        
def test_rate_limit_error_is_wrapped(
    monkeypatch,
) -> None:
    """
    HTTP 429 should become
    LLMRateLimitError.
    """

    client = FakeGroqClient(
        exception=build_status_error(429)
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    with pytest.raises(
        LLMRateLimitError,
        match="rate limit",
    ):
        asyncio.run(
            service.generate_answer(
                system_prompt="system",
                user_prompt="user",
            )
        )
        
def test_other_api_status_error_becomes_generation_error(
    monkeypatch,
) -> None:
    """
    Other API status errors should become
    LLMGenerationError.
    """

    client = FakeGroqClient(
        exception=build_status_error(500)
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    with pytest.raises(
        LLMGenerationError,
        match="status 500",
    ):
        asyncio.run(
            service.generate_answer(
                system_prompt="system",
                user_prompt="user",
            )
        )
        
def test_unexpected_error_is_wrapped(
    monkeypatch,
) -> None:
    """
    Unexpected exceptions should become
    LLMGenerationError.
    """

    client = FakeGroqClient(
        exception=RuntimeError("boom")
    )

    service = build_service(
        monkeypatch,
        client=client,
    )

    with pytest.raises(
        LLMGenerationError,
        match="unexpected error",
    ):
        asyncio.run(
            service.generate_answer(
                system_prompt="system",
                user_prompt="user",
            )
        )
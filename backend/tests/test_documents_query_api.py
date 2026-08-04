import json
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_rag_pipeline_service
from app.core.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.main import app
from app.schemas.answer import (
    AnswerComplete,
    AnswerResponse,
    AnswerToken,
)
from app.schemas.retrieval import RetrievedChunk


def build_chunk(*, document_id) -> RetrievedChunk:
    text = "Relevant evidence."

    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=document_id,
        chunk_index=0,
        page_number=1,
        page_chunk_index=0,
        similarity_score=0.9,
        text=text,
        character_count=len(text),
        word_count=len(text.split()),
    )


class FakeRAGPipelineService:
    """
    Fake pipeline service used to override
    get_rag_pipeline_service for API-level tests, so no real
    retrieval, reranking, or LLM call ever happens.
    """

    def __init__(
        self,
        *,
        answer: AnswerResponse | None = None,
        stream_events: list | None = None,
        raise_error: Exception | None = None,
        raise_during_stream: Exception | None = None,
    ):
        self.answer = answer
        self.stream_events = stream_events or []
        self.raise_error = raise_error
        self.raise_during_stream = raise_during_stream
        self.calls: list[dict] = []

    async def generate_answer(self, *, document_id, query):
        self.calls.append(
            {
                "document_id": document_id,
                "query": query,
            }
        )

        if self.raise_error is not None:
            raise self.raise_error

        return self.answer

    async def generate_answer_stream(
        self,
        *,
        document_id,
        query,
    ) -> AsyncIterator:
        self.calls.append(
            {
                "document_id": document_id,
                "query": query,
            }
        )

        for event in self.stream_events:
            yield event

        if self.raise_during_stream is not None:
            raise self.raise_during_stream


@pytest.fixture
def override_pipeline():
    """
    Yields a function the test calls with a
    FakeRAGPipelineService instance to install as the
    dependency override, and cleans up afterward.
    """

    def _install(fake_service):
        app.dependency_overrides[
            get_rag_pipeline_service
        ] = lambda: fake_service

        return TestClient(app)

    yield _install

    app.dependency_overrides.pop(
        get_rag_pipeline_service,
        None,
    )


def build_answer_response(*, document_id, query):
    chunk = build_chunk(document_id=document_id)

    return AnswerResponse(
        document_id=document_id,
        query=query,
        answer="This is the grounded answer.",
        model="test-model",
        grounded=True,
        total_sources=1,
        retrieved_chunks=(chunk,),
        generation_time_ms=42.5,
    )


# ----------------------------------------------------------------
# POST /{document_id}/query
# ----------------------------------------------------------------


def test_query_returns_the_pipeline_answer(
    override_pipeline,
) -> None:
    document_id = uuid4()

    expected = build_answer_response(
        document_id=document_id,
        query="What is the policy?",
    )

    client = override_pipeline(
        FakeRAGPipelineService(answer=expected)
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/query",
        json={"query": "What is the policy?"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "This is the grounded answer."
    assert body["model"] == "test-model"
    assert body["grounded"] is True
    assert body["total_sources"] == 1


def test_query_forwards_document_id_and_query(
    override_pipeline,
) -> None:
    document_id = uuid4()

    fake_service = FakeRAGPipelineService(
        answer=build_answer_response(
            document_id=document_id,
            query="my question",
        )
    )

    client = override_pipeline(fake_service)

    client.post(
        f"/api/v1/documents/{document_id}/query",
        json={"query": "my question"},
    )

    assert fake_service.calls == [
        {
            "document_id": document_id,
            "query": "my question",
        }
    ]


def test_query_rate_limit_error_returns_429(
    override_pipeline,
) -> None:
    document_id = uuid4()

    client = override_pipeline(
        FakeRAGPipelineService(
            raise_error=LLMRateLimitError(
                "rate limited"
            )
        )
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/query",
        json={"query": "question"},
    )

    assert response.status_code == 429


def test_query_timeout_error_returns_504(
    override_pipeline,
) -> None:
    document_id = uuid4()

    client = override_pipeline(
        FakeRAGPipelineService(
            raise_error=LLMTimeoutError("timed out")
        )
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/query",
        json={"query": "question"},
    )

    assert response.status_code == 504


def test_query_connection_error_returns_502(
    override_pipeline,
) -> None:
    document_id = uuid4()

    client = override_pipeline(
        FakeRAGPipelineService(
            raise_error=LLMConnectionError(
                "cannot connect"
            )
        )
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/query",
        json={"query": "question"},
    )

    assert response.status_code == 502


def test_query_unexpected_error_returns_500(
    override_pipeline,
) -> None:
    document_id = uuid4()

    client = override_pipeline(
        FakeRAGPipelineService(
            raise_error=RuntimeError("boom")
        )
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/query",
        json={"query": "question"},
    )

    assert response.status_code == 500


def test_query_rejects_empty_query_body(
    override_pipeline,
) -> None:
    document_id = uuid4()

    client = override_pipeline(
        FakeRAGPipelineService(
            answer=build_answer_response(
                document_id=document_id,
                query="",
            )
        )
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/query",
        json={"query": ""},
    )

    # AnswerRequest.query has min_length=1
    assert response.status_code == 422


# ----------------------------------------------------------------
# POST /{document_id}/query/stream
# ----------------------------------------------------------------


def parse_sse_events(raw_text: str) -> list[dict]:
    events = []

    for line in raw_text.splitlines():
        if line.startswith("data: "):
            events.append(
                json.loads(line[len("data: "):])
            )

    return events


def test_stream_emits_token_then_complete_events(
    override_pipeline,
) -> None:
    document_id = uuid4()

    chunk = build_chunk(document_id=document_id)

    stream_events = [
        AnswerToken(text="Hello"),
        AnswerToken(text=" world."),
        AnswerComplete(
            document_id=document_id,
            query="question",
            model="test-model",
            grounded=True,
            total_sources=1,
            retrieved_chunks=(chunk,),
            generation_time_ms=10.0,
        ),
    ]

    client = override_pipeline(
        FakeRAGPipelineService(
            stream_events=stream_events
        )
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/query/stream",
        json={"query": "question"},
    )

    assert response.status_code == 200
    assert (
        "text/event-stream"
        in response.headers["content-type"]
    )

    events = parse_sse_events(response.text)

    assert [e["type"] for e in events] == [
        "token",
        "token",
        "complete",
    ]
    assert events[0]["text"] == "Hello"
    assert events[1]["text"] == " world."
    assert events[2]["total_sources"] == 1


def test_stream_failure_emits_error_event_not_a_crash(
    override_pipeline,
) -> None:
    """
    A failure partway through the stream must not produce an
    unhandled exception -- the HTTP status is already committed
    to 200 by the time streaming starts, so the only way to
    signal failure to the client is a final SSE error event.
    """

    document_id = uuid4()

    client = override_pipeline(
        FakeRAGPipelineService(
            stream_events=[AnswerToken(text="partial")],
            raise_during_stream=LLMConnectionError(
                "connection dropped"
            ),
        )
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/query/stream",
        json={"query": "question"},
    )

    assert response.status_code == 200

    events = parse_sse_events(response.text)

    assert events[0]["type"] == "token"
    assert events[-1]["type"] == "error"
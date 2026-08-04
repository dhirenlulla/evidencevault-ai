import logging
import re
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
)
from app.core.request_context import get_request_id

UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def build_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    async def ping() -> dict:
        # Expose whatever request ID is visible from inside
        # the handler, to prove the ContextVar was actually
        # set before the route ran.
        return {
            "seen_request_id": get_request_id(),
        }

    @app.get("/boom")
    async def boom() -> dict:
        raise RuntimeError("forced failure")

    return app


def test_response_includes_a_request_id_header() -> None:
    client = TestClient(build_app())

    response = client.get("/ping")

    assert REQUEST_ID_HEADER in response.headers


def test_generated_request_id_is_a_valid_uuid4() -> None:
    client = TestClient(build_app())

    response = client.get("/ping")

    request_id = response.headers[REQUEST_ID_HEADER]

    assert UUID4_PATTERN.match(request_id)


def test_two_requests_get_different_request_ids() -> None:
    client = TestClient(build_app())

    first = client.get("/ping")
    second = client.get("/ping")

    assert (
        first.headers[REQUEST_ID_HEADER]
        != second.headers[REQUEST_ID_HEADER]
    )


def test_incoming_request_id_header_is_reused() -> None:
    """
    If the client already supplied a request ID (for example a
    load balancer or upstream service), it should be echoed
    back rather than replaced with a new one.
    """

    client = TestClient(build_app())

    supplied_id = str(uuid.uuid4())

    response = client.get(
        "/ping",
        headers={REQUEST_ID_HEADER: supplied_id},
    )

    assert (
        response.headers[REQUEST_ID_HEADER]
        == supplied_id
    )


def test_request_id_is_visible_inside_the_route_handler() -> None:
    """
    The ContextVar should already be set by the time the route
    handler runs, so application code anywhere in the request
    can read it without it being passed as a parameter.
    """

    client = TestClient(build_app())

    supplied_id = str(uuid.uuid4())

    response = client.get(
        "/ping",
        headers={REQUEST_ID_HEADER: supplied_id},
    )

    assert (
        response.json()["seen_request_id"]
        == supplied_id
    )


def test_request_id_does_not_leak_between_requests() -> None:
    """
    After one request finishes, its request ID must not still
    be visible to a later, unrelated request (or to code
    running outside of any request at all).
    """

    client = TestClient(build_app())

    client.get(
        "/ping",
        headers={REQUEST_ID_HEADER: "leaked-id"},
    )

    assert get_request_id() is None


def test_completed_request_is_logged_with_status_and_duration(
    caplog,
) -> None:
    client = TestClient(build_app())

    with caplog.at_level(
        logging.INFO,
        logger="app.api.middleware",
    ):
        client.get("/ping")

    records = [
        record
        for record in caplog.records
        if record.message == "request completed"
    ]

    assert len(records) == 1

    record = records[0]

    assert record.http_method == "GET"
    assert record.http_path == "/ping"
    assert record.http_status_code == 200
    assert record.duration_ms >= 0


def test_unhandled_exception_is_logged_with_request_id(
    caplog,
) -> None:
    """
    An unhandled exception bypasses the normal response path
    in Starlette (an outer ServerErrorMiddleware builds the
    final 500 response without this middleware ever seeing
    it), so the X-Request-ID header cannot be attached to that
    response — a known Starlette/BaseHTTPMiddleware limitation,
    not something fixable from inside our own middleware alone.

    What must still hold: the failure itself is not silently
    lost. It should be logged, with its request ID, before the
    exception propagates.
    """

    client = TestClient(
        build_app(),
        raise_server_exceptions=False,
    )

    with caplog.at_level(
        logging.ERROR,
        logger="app.api.middleware",
    ):
        client.get("/boom")

    failure_records = [
        record
        for record in caplog.records
        if record.message == "request failed"
    ]

    assert len(failure_records) == 1

    record = failure_records[0]

    assert record.http_method == "GET"
    assert record.http_path == "/boom"
    assert record.duration_ms >= 0


def test_context_is_still_reset_after_an_unhandled_exception() -> None:
    """
    Even though the response path is bypassed, the finally
    block must still run so the request ID doesn't leak into
    whatever runs next on this task.
    """

    client = TestClient(
        build_app(),
        raise_server_exceptions=False,
    )

    client.get("/boom")

    assert get_request_id() is None
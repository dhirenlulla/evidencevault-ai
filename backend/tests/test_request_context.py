from app.core.request_context import (
    get_request_id,
    reset_request_id,
    set_request_id,
)


def test_get_request_id_is_none_by_default() -> None:
    """
    Outside of any request, there is no request ID set.
    """

    assert get_request_id() is None


def test_set_request_id_makes_it_retrievable() -> None:
    token = set_request_id("abc-123")

    try:
        assert get_request_id() == "abc-123"
    finally:
        reset_request_id(token)


def test_reset_request_id_restores_previous_value() -> None:
    outer_token = set_request_id("outer")

    try:
        inner_token = set_request_id("inner")

        assert get_request_id() == "inner"

        reset_request_id(inner_token)

        assert get_request_id() == "outer"

    finally:
        reset_request_id(outer_token)


def test_reset_request_id_restores_none_when_nothing_set_before() -> None:
    token = set_request_id("only-value")

    assert get_request_id() == "only-value"

    reset_request_id(token)

    assert get_request_id() is None
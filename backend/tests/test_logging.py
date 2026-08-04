import json
import logging

import pytest

from app.core.logging import (
    JSONLogFormatter,
    configure_logging,
)
from app.core.request_context import (
    reset_request_id,
    set_request_id,
)


def make_record(
    *,
    message: str = "hello",
    level: int = logging.INFO,
    extra: dict | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=None,
        exc_info=None,
    )

    for key, value in (extra or {}).items():
        setattr(record, key, value)

    return record


def test_format_produces_valid_json() -> None:
    formatter = JSONLogFormatter()

    output = formatter.format(make_record())

    parsed = json.loads(output)

    assert parsed["message"] == "hello"


def test_format_includes_level_and_logger_name() -> None:
    formatter = JSONLogFormatter()

    record = make_record(
        message="something happened",
        level=logging.WARNING,
    )

    parsed = json.loads(formatter.format(record))

    assert parsed["level"] == "WARNING"
    assert parsed["logger"] == "test.logger"


def test_format_omits_request_id_when_none_set() -> None:
    formatter = JSONLogFormatter()

    parsed = json.loads(
        formatter.format(make_record())
    )

    assert "request_id" not in parsed


def test_format_includes_request_id_when_set() -> None:
    formatter = JSONLogFormatter()

    token = set_request_id("req-42")

    try:
        parsed = json.loads(
            formatter.format(make_record())
        )
    finally:
        reset_request_id(token)

    assert parsed["request_id"] == "req-42"


def test_format_includes_extra_fields() -> None:
    """
    Fields passed via logging.info(..., extra={...}) should
    be merged into the JSON payload automatically.
    """

    formatter = JSONLogFormatter()

    record = make_record(
        extra={
            "duration_ms": 123.4,
            "http_status_code": 200,
        }
    )

    parsed = json.loads(formatter.format(record))

    assert parsed["duration_ms"] == 123.4
    assert parsed["http_status_code"] == 200


def test_format_does_not_leak_reserved_attributes_as_extras() -> None:
    """
    Standard LogRecord attributes (like pathname or lineno)
    should not show up duplicated as top-level extra fields.
    """

    formatter = JSONLogFormatter()

    parsed = json.loads(
        formatter.format(make_record())
    )

    assert "pathname" not in parsed
    assert "lineno" not in parsed


def test_configure_logging_sets_json_formatter_on_root_logger() -> None:
    configure_logging(level="INFO")

    root_logger = logging.getLogger()

    assert len(root_logger.handlers) == 1
    assert isinstance(
        root_logger.handlers[0].formatter,
        JSONLogFormatter,
    )


def test_configure_logging_is_idempotent() -> None:
    """
    Calling configure_logging multiple times should not stack
    duplicate handlers, which would otherwise cause every log
    line to be printed multiple times.
    """

    configure_logging(level="INFO")
    configure_logging(level="INFO")
    configure_logging(level="INFO")

    root_logger = logging.getLogger()

    assert len(root_logger.handlers) == 1


def test_configure_logging_sets_the_requested_level() -> None:
    configure_logging(level="WARNING")

    root_logger = logging.getLogger()

    assert root_logger.level == logging.WARNING

    # Restore a sane default so this test doesn't silence
    # other tests' log output for the rest of the run.
    configure_logging(level="INFO")
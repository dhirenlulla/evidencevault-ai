import json
import logging
import sys
from datetime import datetime, timezone

from app.core.request_context import get_request_id

# The set of attribute names a plain logging.LogRecord already
# has. Anything passed via logging.info(..., extra={...}) shows
# up as additional attributes beyond this set, which lets
# JSONLogFormatter include arbitrary extra fields automatically
# without needing to know their names in advance.
_BASE_LOG_RECORD_ATTRIBUTES = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=None,
        exc_info=None,
    ).__dict__.keys()
) | {"message", "asctime"}


class JSONLogFormatter(logging.Formatter):
    """ 
    Formats log records as single-line JSON.

    Plain-text logs are fine for a human watching a terminal,
    but cannot be filtered or aggregated by a log platform (or
    even grep + jq) without brittle text parsing. Emitting one
    JSON object per line makes every field - level, logger,
    request_id, and any extra context passed via
    logging.info(..., extra={...}) - directly queryable.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = get_request_id()

        if request_id is not None:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key not in _BASE_LOG_RECORD_ATTRIBUTES:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            payload,
            default=str,
        )


def configure_logging(*, level: str = "INFO") -> None:
    """ 
    Configure the root logger to emit structured JSON log
    lines to stdout.

    Safe to call more than once: existing handlers on the root
    logger are cleared first, so repeated calls (application
    startup running twice, or multiple tests in the same
    process) never stack duplicate handlers and produce
    repeated log lines.
    """

    root_logger = logging.getLogger()

    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONLogFormatter())

    root_logger.addHandler(handler)
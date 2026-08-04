import logging
import uuid

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import Response

from app.core.request_context import (
    reset_request_id,
    set_request_id,
)
from app.services.latency import LatencyTimer

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """ 
    Assigns a request ID to every incoming request and logs
    one structured line per completed request, including its
    latency.

    Two things this enables that plain per-service logging
    doesn't:

    1. Tracing — every log line produced anywhere while
       handling a request (retrieval, fusion, reranking,
       generation) automatically carries the same request ID
       via the request_context ContextVar, so they can all be
       filtered together afterward.
    2. Real latency data — reuses LatencyTimer (built in the
       evaluation phase for testable, isolated timing) against
       actual traffic, instead of that utility only ever being
       exercised by unit tests.

    If the client already supplied an X-Request-ID header (for
    example a load balancer or another internal service), that
    ID is reused rather than replaced, so a request can be
    traced across service boundaries instead of getting a new,
    disconnected ID at every hop.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = (
            request.headers.get(REQUEST_ID_HEADER)
            or str(uuid.uuid4())
        )

        token = set_request_id(request_id)

        timer = LatencyTimer()

        try:
            with timer:
                response = await call_next(request)

        except Exception:
            # An unhandled exception here bypasses the normal
            # response path: Starlette's outer
            # ServerErrorMiddleware builds the eventual 500
            # response without ever passing back through this
            # middleware, so we cannot attach the request ID
            # header to it. What we still can and must do is
            # make sure the failure itself — and its request
            # ID — is not lost, by logging it here before
            # re-raising.
            logger.exception(
                "request failed",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "duration_ms": round(
                        timer.elapsed_seconds * 1000,
                        2,
                    ),
                },
            )
            raise

        finally:
            reset_request_id(token)

        response.headers[REQUEST_ID_HEADER] = request_id

        logger.info(
            "request completed",
            extra={
                "http_method": request.method,
                "http_path": request.url.path,
                "http_status_code": (
                    response.status_code
                ),
                "duration_ms": round(
                    timer.elapsed_seconds * 1000,
                    2,
                ),
            },
        )

        return response
from contextvars import ContextVar, Token

_request_id_var: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)


def get_request_id() -> str | None:
    """ 
    Return the request ID for the request currently being
    handled, or None outside of a request (for example in a
    background script, or in a test that never set one).
    """

    return _request_id_var.get()


def set_request_id(request_id: str) -> Token:
    """ 
    Bind a request ID for the current execution context.

    Returns a Token that must later be passed to
    reset_request_id to restore whatever value (or absence
    of one) was set before this call. ContextVar is scoped
    per asyncio task, so concurrent requests never see each
    other's request ID even though they run on the same
    event loop.
    """

    return _request_id_var.set(request_id)


def reset_request_id(token: Token) -> None:
    """ 
    Undo a previous set_request_id call.

    Always call this once the request has finished (typically
    in a finally block), so the request ID doesn't leak into
    whatever code runs next on the same task.
    """

    _request_id_var.reset(token)
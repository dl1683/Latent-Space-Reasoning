"""Request-local cooperative cancellation, shared by existing work checkpoints."""
from contextvars import ContextVar

CURRENT_REQUEST = ContextVar("current_request", default=None)


def checkpoint():
    check = CURRENT_REQUEST.get()
    if check is not None:
        check()

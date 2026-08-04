"""What both MCP servers do before serving.

Two things a server process needs that a library does not. It has to
join the trace that spawned it, or its spans start a new one and the
request's picture comes apart. And it has to stop printing a full
traceback every time it refuses a request.

The refusals are the important half. An authorization denial is expected
control flow — the system working — and logging each one as though it
were a crash means the log a real failure appears in is already full of
them.
"""

import logging
from typing import Any

from rti_engine.db.authz import AuthorizationError
from rti_engine.observability.otel import adopt_parent_context

EXPECTED_REFUSALS: tuple[type[BaseException], ...] = (AuthorizationError, ValueError)
"""Failures that mean the server did its job.

An authorization denial and a rejected argument are both decisions. They
are still logged — the message is the record that a refusal happened —
but without the stack trace of a crash that did not occur.
"""

FASTMCP_LOGGERS: tuple[str, ...] = ("fastmcp", "fastmcp.server.server", "fastmcp.tools")


class RefusalFilter(logging.Filter):
    """Strip the traceback from an expected refusal, keeping the message."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and isinstance(record.exc_info[1], EXPECTED_REFUSALS):
            record.exc_info = None
            record.exc_text = None
        return True


def prepare_server() -> None:
    """Join the caller's trace and quieten expected refusals."""
    adopt_parent_context()

    refusals = RefusalFilter()
    for name in FASTMCP_LOGGERS:
        logging.getLogger(name).addFilter(refusals)


def run(server: Any) -> None:
    """Prepare and start a server."""
    prepare_server()
    server.run()

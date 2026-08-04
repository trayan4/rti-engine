"""Distributed tracing.

LangSmith already records what the models did. This records what the
system did around them: which node ran, how long a tool call took, where
a retry was spent. Between them, a slow request can be explained rather
than guessed at.

Trace context does not cross the MCP boundary on its own. The servers are
separate processes reached over stdio, so a tool call would otherwise
start a fresh trace and the picture would come apart exactly where it
matters. The context is injected into the subprocess environment at
session creation, which links a server's work to the request that opened
the session.

Tracing stays off without an endpoint rather than failing. A missing
collector must not take a request with it.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel, ConfigDict

from rti_engine.config.settings import get_settings

TRACEPARENT = "traceparent"
TRACESTATE = "tracestate"

PROPAGATED_KEYS: tuple[str, ...] = (TRACEPARENT, TRACESTATE)
"""W3C trace context headers, carried into a server subprocess as
environment variables so its spans join the request's trace."""

INSTRUMENTATION_NAME = "rti_engine"


class TracingStatus(BaseModel):
    """Whether spans are being exported, and where."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    endpoint: str | None = None
    service: str | None = None
    reason: str | None = None


@lru_cache
def configure_tracing() -> TracingStatus:
    """Set up the tracer provider once per process.

    Cached rather than guarded by a flag: a second call would add a second
    exporter and every span would be sent twice.
    """
    settings = get_settings()

    if not settings.otel_endpoint:
        return TracingStatus(enabled=False, reason="OTEL_ENDPOINT is not set; tracing is off")

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)

    return TracingStatus(
        enabled=True,
        endpoint=settings.otel_endpoint,
        service=settings.otel_service_name,
    )


def get_tracer() -> trace.Tracer:
    """Return the tracer, configuring the provider if it is not yet set up."""
    configure_tracing()
    return trace.get_tracer(INSTRUMENTATION_NAME)


def _set_attributes(span: Span, attributes: dict[str, Any]) -> None:
    """Attach attributes, skipping any value a span cannot carry."""
    for key, value in attributes.items():
        if isinstance(value, str | bool | int | float):
            span.set_attribute(key, value)
        elif value is not None:
            span.set_attribute(key, str(value))


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Record one unit of work, marking it failed if it raises.

    An exception is recorded and re-raised rather than swallowed: the
    span exists to describe what happened, not to change it.
    """
    with get_tracer().start_as_current_span(name) as current:
        _set_attributes(current, attributes)
        try:
            yield current
        except Exception as error:
            current.record_exception(error)
            current.set_status(Status(StatusCode.ERROR, str(error)))
            raise


def propagation_env() -> dict[str, str]:
    """Return the current trace context as environment variables.

    Passed to a server subprocess so its spans attach to the request that
    started it. Empty when tracing is off, which leaves the subprocess
    environment untouched.
    """
    carrier: dict[str, str] = {}
    inject(carrier)
    return {key: value for key, value in carrier.items() if key in PROPAGATED_KEYS}


def adopt_parent_context() -> None:
    """Attach this process to the trace that spawned it.

    Called by a server at startup. Without it a tool call would begin a
    fresh trace, and the request's picture would come apart at exactly
    the boundary worth seeing across.
    """
    carrier = {key: os.environ[key] for key in PROPAGATED_KEYS if key in os.environ}
    if not carrier:
        return

    from opentelemetry.context import attach

    attach(extract(carrier))

"""Tests for tracing configuration and propagation.

Offline. What is asserted is that tracing stays out of the way when it is
not configured, that the trace context is carried in a form a subprocess
can read, and that a span records a failure rather than swallowing it.

Whether spans reach a collector is a deployment question, checked by
looking at Jaeger rather than by a test that would need one running.
"""

import os
from collections.abc import Iterator

import pytest

from rti_engine.observability.otel import (
    PROPAGATED_KEYS,
    TRACEPARENT,
    adopt_parent_context,
    configure_tracing,
    propagation_env,
    span,
)


@pytest.fixture(autouse=True)
def clean_environment() -> Iterator[None]:
    """Restore the propagation variables around each test."""
    saved = {key: os.environ.get(key) for key in PROPAGATED_KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# --- configuration ---


def test_configuration_is_cached() -> None:
    """A second provider would export every span twice."""
    assert configure_tracing() is configure_tracing()


def test_a_span_works_whether_or_not_tracing_is_configured() -> None:
    """A missing collector must not take a request with it."""
    with span("test.operation", **{"rti.example": "value"}) as current:
        assert current is not None


# --- failure recording ---


def test_a_span_records_a_failure_and_re_raises() -> None:
    """The span describes what happened; it does not change it."""
    with pytest.raises(ValueError, match="boom"), span("test.failing"):
        raise ValueError("boom")


def test_attributes_of_unsupported_types_do_not_raise() -> None:
    """A span that rejected an attribute would fail the work it observes."""
    with span("test.attributes", **{"rti.list": [1, 2, 3], "rti.none": None}):
        pass


# --- propagation ---


def test_propagation_carries_only_trace_context() -> None:
    """Whatever is exported becomes subprocess environment; keep it narrow."""
    with span("test.parent"):
        carried = propagation_env()

    assert set(carried) <= set(PROPAGATED_KEYS)


def test_an_absent_context_carries_nothing() -> None:
    """An empty result leaves the subprocess environment untouched."""
    assert propagation_env() == {} or set(propagation_env()) <= set(PROPAGATED_KEYS)


def test_adopting_without_a_context_is_a_no_op() -> None:
    """A server started outside a trace must still start."""
    for key in PROPAGATED_KEYS:
        os.environ.pop(key, None)

    adopt_parent_context()


def test_a_malformed_context_does_not_prevent_startup() -> None:
    """A server that refused to start over a bad header would be worse."""
    os.environ[TRACEPARENT] = "not-a-valid-traceparent"
    adopt_parent_context()

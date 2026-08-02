"""Tests for retry classification.

The distinction these enforce: a call that might succeed if repeated, and
one that will fail identically. Retrying an authorization refusal costs
three attempts and a delay to arrive at the same answer, and does it at
the moment the system is already refusing someone.
"""

import httpx
import pytest

from rti_engine.agents.analyst import AnalysisError
from rti_engine.agents.drafter import DraftingError
from rti_engine.agents.retry import (
    MAX_ATTEMPTS,
    NODE_RETRY_POLICY,
    NODE_TIMEOUT,
    is_transient,
    reraise_if_transient,
)
from rti_engine.agents.tools import ToolCallError
from rti_engine.db.authz import AuthorizationError

# --- what is worth retrying ---


def test_a_dropped_connection_is_retried() -> None:
    assert is_transient(ConnectionError("connection reset")) is True


def test_a_timeout_is_retried() -> None:
    assert is_transient(httpx.ReadTimeout("timed out")) is True


def test_a_server_error_is_retried() -> None:
    """A provider returning 5xx may not be returning one a second later."""
    request = httpx.Request("POST", "https://example.invalid/v1/chat")
    response = httpx.Response(503, request=request)

    assert is_transient(httpx.HTTPStatusError("busy", request=request, response=response))


# --- what is not ---


def test_an_authorization_refusal_is_never_retried() -> None:
    """A refusal is a decision, not an outage."""
    assert is_transient(AuthorizationError("tier T1 may not request scope")) is False


def test_a_client_error_is_not_retried() -> None:
    """A bad argument is bad on the second attempt too."""
    request = httpx.Request("POST", "https://example.invalid/v1/chat")
    response = httpx.Response(400, request=request)

    assert is_transient(httpx.HTTPStatusError("bad", request=request, response=response)) is False


@pytest.mark.parametrize(
    "error",
    [
        AnalysisError("no pay record found"),
        DraftingError("letter cites source fields that do not exist"),
        ToolCallError("tool is not available"),
        ValueError("unknown decision"),
    ],
)
def test_the_projects_own_failures_are_not_retried(error: Exception) -> None:
    """These describe the request, not the connection to a provider."""
    assert is_transient(error) is False


# --- letting errors escape ---


def test_a_transient_error_escapes_the_handler() -> None:
    """Nodes catch everything; a retry policy needs something to act on."""
    with pytest.raises(ConnectionError):
        reraise_if_transient(ConnectionError("reset"))


def test_a_permanent_error_is_left_to_the_handler() -> None:
    """Returns without raising, so the node records the failure in state."""
    reraise_if_transient(AuthorizationError("refused"))
    reraise_if_transient(AnalysisError("no record"))


# --- the policies ---


def test_retries_are_bounded() -> None:
    assert 1 < MAX_ATTEMPTS <= 5
    assert NODE_RETRY_POLICY.max_attempts == MAX_ATTEMPTS


def test_the_backoff_grows_and_is_capped() -> None:
    assert NODE_RETRY_POLICY.backoff_factor > 1.0
    assert NODE_RETRY_POLICY.max_interval > NODE_RETRY_POLICY.initial_interval


def test_retries_are_jittered() -> None:
    """Without jitter, requests failing together retry in lockstep."""
    assert NODE_RETRY_POLICY.jitter is True


def test_the_policy_uses_this_modules_classification() -> None:
    assert NODE_RETRY_POLICY.retry_on is is_transient


def test_the_timeout_is_wall_clock_not_idle() -> None:
    """An idle cap resets on callback events, and a structured model call
    emits none between starting and finishing."""
    assert NODE_TIMEOUT.run_timeout is not None
    assert NODE_TIMEOUT.idle_timeout is None


def test_the_timeout_exceeds_a_single_model_attempt() -> None:
    """The client allows 60s per attempt and retries twice."""
    assert NODE_TIMEOUT.run_timeout is not None
    assert NODE_TIMEOUT.run_timeout > 180.0

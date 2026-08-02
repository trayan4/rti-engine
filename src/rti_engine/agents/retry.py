"""Which failures are worth another attempt.

Retrying is only useful when the same call might succeed unchanged: a
dropped connection, a rate limit, a provider returning 5xx. Retrying a
refusal wastes time and money to arrive at the same answer, and retrying
a bad argument does the same.

LangGraph's default already declines to retry ValueError, RuntimeError
and their kin, which covers most of this project's own exceptions. The
exception is an authorization refusal: it extends Exception, so the
default would retry a tier violation three times before giving up on it.

There is a second problem this module solves. Nodes catch everything and
return a failure into state, so nothing propagates for a retry policy to
act on. A node that wants its transient errors retried has to let them
escape, which is what ``reraise_if_transient`` is for.
"""

from langgraph.types import (  # type: ignore[attr-defined]
    RetryPolicy,
    TimeoutPolicy,
    default_retry_on,
)

from rti_engine.db.authz import AuthorizationError

PERMANENT_ERRORS: tuple[type[Exception], ...] = (AuthorizationError,)
"""Failures that will recur identically however many times they are tried.

An authorization refusal is a decision, not an outage. LangGraph's
default would retry it, because it does not know that.
"""

MAX_ATTEMPTS = 3
INITIAL_INTERVAL_SECONDS = 1.0
BACKOFF_FACTOR = 2.0
MAX_INTERVAL_SECONDS = 30.0


def is_transient(error: Exception) -> bool:
    """Whether the same call might succeed if repeated."""
    if isinstance(error, PERMANENT_ERRORS):
        return False
    return bool(default_retry_on(error))


def reraise_if_transient(error: Exception) -> None:
    """Let a transient error escape so the retry policy can see it.

    Called from a node's exception handler before it records a failure.
    Without this the handler swallows everything and the retry policy has
    nothing to act on.
    """
    if is_transient(error):
        raise error


NODE_RETRY_POLICY = RetryPolicy(
    max_attempts=MAX_ATTEMPTS,
    initial_interval=INITIAL_INTERVAL_SECONDS,
    backoff_factor=BACKOFF_FACTOR,
    max_interval=MAX_INTERVAL_SECONDS,
    jitter=True,
    retry_on=is_transient,
)
"""Applied to every node that calls something outside the process.

Jitter matters more than it looks: without it, several requests failing
together retry in lockstep and hit the recovering service simultaneously.
"""

FAST_NODE_TIMEOUT_SECONDS = 120.0
"""For a node making one small model call and nothing else."""

STANDARD_NODE_TIMEOUT_SECONDS = 300.0
"""For a node that retrieves and then calls a reasoning model.

Generous on purpose. The model client already allows sixty seconds per
attempt and retries twice, so a single call can legitimately run to three
minutes. A node timeout below that would cut off work that was going to
succeed.
"""

FAST_NODE_TIMEOUT = TimeoutPolicy(run_timeout=FAST_NODE_TIMEOUT_SECONDS)
NODE_TIMEOUT = TimeoutPolicy(run_timeout=STANDARD_NODE_TIMEOUT_SECONDS)
"""A wall-clock cap per attempt, not an idle one.

The idle variant resets on callback events, and a structured model call
emits nothing between starting and finishing — so it could fire on a call
that was proceeding normally. Wall clock has no such ambiguity, and the
question being asked is "has this taken too long", which is what it
measures.

A timeout raises inside the node, so the retry policy sees it and tries
again. Only after the attempts are used up does the failure reach state.
"""

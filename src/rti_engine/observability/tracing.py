"""Turn LangSmith tracing on.

LangChain's tracer reads its configuration from os.environ when a chain
runs. This project's configuration lives in a settings object that
deliberately never touches the environment, so the two never meet unless
something puts them together — which is what this does, once, at startup.

Tracing is optional: without a key it stays off rather than failing.
Whether it is on is reported rather than assumed, because "no traces are
appearing" and "tracing was never enabled" look identical from the
outside.
"""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict

from rti_engine.config.settings import get_settings

TRACING_FLAG = "LANGSMITH_TRACING"
API_KEY = "LANGSMITH_API_KEY"
PROJECT = "LANGSMITH_PROJECT"

RUN_NAME_PREFIX = "rti"


class TracingStatus(BaseModel):
    """Whether tracing is on, and why not if it is off."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    project: str | None = None
    reason: str | None = None


def enable_tracing() -> TracingStatus:
    """Export the tracer's configuration into the environment.

    Called once at startup. Safe to call more than once: exporting the
    same values again changes nothing.
    """
    settings = get_settings()

    if not settings.langsmith_api_key:
        os.environ.pop(TRACING_FLAG, None)
        return TracingStatus(enabled=False, reason="LANGSMITH_API_KEY is not set; tracing is off")

    os.environ[TRACING_FLAG] = "true"
    os.environ[API_KEY] = settings.langsmith_api_key
    os.environ[PROJECT] = settings.langsmith_project

    return TracingStatus(enabled=True, project=settings.langsmith_project)


def tracing_status() -> TracingStatus:
    """Report whether tracing is currently on, without changing anything."""
    if os.environ.get(TRACING_FLAG) != "true":
        return TracingStatus(enabled=False, reason="tracing has not been enabled")
    return TracingStatus(enabled=True, project=os.environ.get(PROJECT))


def run_config(request_id: str, tier: str | None = None, **metadata: Any) -> dict[str, Any]:
    """Build a LangChain run config that labels a trace usefully.

    A trace named "LangGraph" among two hundred others is unfindable. Tags
    and metadata are what make a run searchable later — by tier, by
    request, or by whatever went wrong.
    """
    tags = [RUN_NAME_PREFIX]
    if tier:
        tags.append(f"tier:{tier}")

    return {
        "run_name": f"{RUN_NAME_PREFIX}:request:{request_id}",
        "tags": tags,
        "metadata": {"request_id": request_id, "tier": tier, **metadata},
    }

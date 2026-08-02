"""Record which model actually answered.

A fallback chain that works silently is a chain you cannot audit. A
request served by the third provider because the first two were failing
looks exactly like one served by the first, so a degraded system reports
itself as healthy.

LangChain fires a callback when a model starts, carrying the model's
identity. Collecting those gives the served model per call and, by
counting past the first, whether a fallback was used.

The recorder holds no state between requests: one is created per run and
its findings are written into that run's state.
"""

from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from pydantic import BaseModel, ConfigDict

UNKNOWN_MODEL = "unknown"


class ServedModel(BaseModel):
    """One model invocation, in the order it was attempted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    attempt: int
    """1 for the primary, 2 for the first fallback, and so on."""


class ModelRecorder(AsyncCallbackHandler):
    """Collect the models used during one unit of work.

    Attached per invocation rather than to the model itself: a model is
    cached and shared across requests, so a recorder living on it would
    mix one request's attempts with another's.
    """

    def __init__(self) -> None:
        self.calls: list[ServedModel] = []
        self._attempt = 0

    def _record(self, serialized: dict[str, Any], kwargs: dict[str, Any]) -> None:
        self._attempt += 1
        self.calls.append(ServedModel(model=_model_name(serialized, kwargs), attempt=self._attempt))

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(serialized, kwargs)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(serialized, kwargs)

    @property
    def served_by(self) -> str:
        """The model that produced the answer: the last one attempted."""
        return self.calls[-1].model if self.calls else UNKNOWN_MODEL

    @property
    def used_fallback(self) -> bool:
        """Whether the primary failed and something else answered."""
        return len(self.calls) > 1

    def summary(self) -> dict[str, Any]:
        """Describe what happened, for the audit trail."""
        return {
            "served_by": self.served_by,
            "attempts": len(self.calls),
            "used_fallback": self.used_fallback,
            "chain": [call.model for call in self.calls],
        }


def _model_name(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str:
    """Extract a model's name from what the callback was given.

    Providers disagree on where they put it, so several places are tried
    before giving up. Returning a placeholder is better than raising: a
    failure to identify a model must not fail the request it served.
    """
    metadata = kwargs.get("metadata") or {}
    for key in ("ls_model_name", "model_name", "model"):
        if value := metadata.get(key):
            return str(value)

    invocation = kwargs.get("invocation_params") or {}
    for key in ("model_name", "model"):
        if value := invocation.get(key):
            return str(value)

    if value := (serialized.get("kwargs") or {}).get("model_name"):
        return str(value)

    name = serialized.get("name")
    return str(name) if name else UNKNOWN_MODEL

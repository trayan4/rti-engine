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
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, ConfigDict

UNKNOWN_MODEL = "unknown"

USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (5.0, 30.0),
    "gpt-5.6-terra": (2.5, 15.0),
    "gpt-5.6-luna": (1.0, 6.0),
    "claude-sonnet-5": (3.0, 15.0),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}
"""Input and output price per million tokens, by model.

Approximate and manually maintained, so treat the figure as an estimate
for budgeting rather than an invoice. What matters is that a request
which fell back to a costlier provider shows the difference, instead of
the change being visible only in the audit trail.
"""

DEFAULT_USD_PER_MILLION = (3.0, 15.0)
"""Used for a model with no recorded rate.

Deliberately not zero: an unpriced model must not look free, or adding
one would silently disable the budget.
"""

TOKENS_PER_MILLION = 1_000_000


class TokenUsage(BaseModel):
    """What one call consumed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Estimated cost, at this model's rate."""
        input_rate, output_rate = USD_PER_MILLION.get(self.model, DEFAULT_USD_PER_MILLION)
        return (
            self.input_tokens * input_rate + self.output_tokens * output_rate
        ) / TOKENS_PER_MILLION


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
        self.usage: list[TokenUsage] = []
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

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record what the call that just finished consumed."""
        self.usage.append(_token_usage(self.served_by, response))

    @property
    def total_tokens(self) -> int:
        return sum(item.total_tokens for item in self.usage)

    @property
    def cost_usd(self) -> float:
        return sum(item.cost_usd for item in self.usage)

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
            "tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
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


def _token_usage(model: str, response: LLMResult) -> TokenUsage:
    """Read token counts from a completed call.

    Providers report usage in more than one place, so both are tried.
    Returning zeros rather than raising is deliberate: a provider that
    reports nothing must not fail the request it just answered, and the
    budget errs toward letting work through rather than blocking it on
    missing telemetry.
    """
    output = response.llm_output or {}
    usage = output.get("token_usage") or output.get("usage") or {}

    if not usage:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "usage_metadata", None)
                if metadata:
                    usage = dict(metadata)
                    break
            if usage:
                break

    return TokenUsage(
        model=model,
        input_tokens=int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
    )

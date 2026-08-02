"""Tests for spend accounting and the degraded response.

The accounting matters because the budget reads it: a provider that
reports usage in an unexpected shape would leave the ceiling permanently
unreached, and the limit would exist without doing anything.

The degraded response is the other half. A request that cannot complete
must still produce something an employee receives, because silence tells
them less than an acknowledgement does.
"""

from langchain_core.outputs import Generation, LLMResult

from rti_engine.agents.budget import (
    MAX_COST_USD_PER_REQUEST,
    MAX_TOKENS_PER_REQUEST,
    degraded_detail,
    degraded_letter,
    over_budget,
)
from rti_engine.llm.served import (
    DEFAULT_USD_PER_MILLION,
    USD_PER_MILLION,
    TokenUsage,
    _token_usage,
)

# --- cost accounting ---


def test_cost_uses_the_models_own_rate() -> None:
    usage = TokenUsage(model="gpt-5.6-luna", input_tokens=1_000_000, output_tokens=0)
    assert usage.cost_usd == USD_PER_MILLION["gpt-5.6-luna"][0]


def test_output_tokens_cost_more_than_input() -> None:
    model = "gpt-5.6-terra"
    input_only = TokenUsage(model=model, input_tokens=10_000)
    output_only = TokenUsage(model=model, output_tokens=10_000)

    assert output_only.cost_usd > input_only.cost_usd


def test_an_unpriced_model_is_not_free() -> None:
    """Otherwise adding a model would silently disable the budget."""
    usage = TokenUsage(model="some-new-model", input_tokens=1_000_000)
    assert usage.cost_usd == DEFAULT_USD_PER_MILLION[0]


def test_a_fallback_to_a_costlier_provider_shows_in_the_figure() -> None:
    tokens = 100_000
    cheap = TokenUsage(model="llama-3.3-70b-versatile", input_tokens=tokens)
    expensive = TokenUsage(model="gpt-5.6-sol", input_tokens=tokens)

    assert expensive.cost_usd > cheap.cost_usd


# --- reading usage from a response ---


def result(llm_output: dict[str, object] | None) -> LLMResult:
    return LLMResult(generations=[[Generation(text="ok")]], llm_output=llm_output)


def test_usage_is_read_from_the_openai_shape() -> None:
    usage = _token_usage(
        "gpt-5.6-luna",
        result({"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}),
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150


def test_usage_is_read_from_the_anthropic_shape() -> None:
    usage = _token_usage(
        "claude-sonnet-5",
        result({"usage": {"input_tokens": 200, "output_tokens": 80}}),
    )

    assert usage.input_tokens == 200
    assert usage.output_tokens == 80


def test_a_response_reporting_nothing_does_not_raise() -> None:
    """A provider that reports no usage must not fail the call it answered."""
    usage = _token_usage("gpt-5.6-terra", result(None))
    assert usage.total_tokens == 0


# --- the ceilings ---


def test_a_normal_request_is_within_budget() -> None:
    """A tier 2 request uses roughly 60,000 tokens and fifteen cents."""
    assert over_budget(60_000, 0.15) is None


def test_a_runaway_token_count_is_caught() -> None:
    assert over_budget(MAX_TOKENS_PER_REQUEST + 1, 0.0) is not None


def test_a_runaway_cost_is_caught() -> None:
    assert over_budget(0, MAX_COST_USD_PER_REQUEST + 0.01) is not None


def test_the_reason_names_the_limit_that_was_passed() -> None:
    """An operator reading this needs to know which ceiling, and by how much."""
    reason = over_budget(MAX_TOKENS_PER_REQUEST * 2, 0.0)

    assert reason is not None
    assert "token" in reason
    assert str(MAX_TOKENS_PER_REQUEST) in reason


def test_the_ceilings_leave_room_for_normal_work() -> None:
    """A limit that catches ordinary requests is an outage, not a safeguard."""
    assert MAX_TOKENS_PER_REQUEST > 200_000
    assert MAX_COST_USD_PER_REQUEST > 1.0


# --- the degraded response ---


def test_the_degraded_letter_needs_no_model() -> None:
    """The situation it exists for is one where model calls are failing."""
    letter = degraded_letter("provider unavailable")

    assert letter.sections
    assert letter.figures_used == []
    assert letter.citations == []


def test_the_degraded_letter_does_not_leak_the_reason() -> None:
    """An employee is not told about token budgets or provider failures."""
    rendered = degraded_letter("cost budget exceeded: 4.20 USD").render()

    assert "budget" not in rendered.lower()
    assert "USD" not in rendered


def test_the_degraded_letter_says_a_person_will_follow_up() -> None:
    rendered = degraded_letter("failed").render().lower()

    assert "person" in rendered
    assert "not need to submit your request again" in rendered


def test_the_degraded_letter_preserves_the_employees_right() -> None:
    """A system failure does not reduce what they are entitled to."""
    assert "not affected" in degraded_letter("failed").render()


def test_the_audit_detail_records_the_reason_for_an_operator() -> None:
    """Withheld from the employee, needed by whoever investigates."""
    detail = degraded_detail("cost budget exceeded", ["drafter: timeout"])

    assert detail["reason"] == "cost budget exceeded"
    assert detail["queued_for_manual_handling"] is True
    assert detail["errors"] == ["drafter: timeout"]


def test_the_audit_detail_caps_the_error_list() -> None:
    """A retried failure can produce many; the first few identify it."""
    detail = degraded_detail("failed", [f"error {index}" for index in range(20)])
    assert len(detail["errors"]) == 5

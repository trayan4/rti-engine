"""Tests for served-model recording.

A fallback chain nobody can see is one nobody can tell is being used: a
system running entirely on its third provider reports itself as healthy.
These assert the recorder attributes calls correctly and never fails the
request it was observing.
"""

from typing import Any
from uuid import uuid4

import pytest

from rti_engine.llm.served import UNKNOWN_MODEL, ModelRecorder, _model_name


async def start(recorder: ModelRecorder, model: str) -> None:
    """Simulate one model invocation reaching the callback."""
    await recorder.on_chat_model_start(
        {"name": "ChatOpenAI"},
        [[]],
        run_id=uuid4(),
        metadata={"ls_model_name": model},
    )


# --- attribution ---


async def test_a_single_call_reports_no_fallback() -> None:
    recorder = ModelRecorder()
    await start(recorder, "gpt-5.6-luna")

    assert recorder.served_by == "gpt-5.6-luna"
    assert recorder.used_fallback is False


async def test_the_last_model_attempted_is_the_one_that_served() -> None:
    """Earlier entries in the chain failed; the answer came from the last."""
    recorder = ModelRecorder()
    await start(recorder, "gpt-5.6-terra")
    await start(recorder, "claude-sonnet-5")

    assert recorder.served_by == "claude-sonnet-5"
    assert recorder.used_fallback is True


async def test_attempts_are_numbered_in_order() -> None:
    recorder = ModelRecorder()
    await start(recorder, "primary")
    await start(recorder, "secondary")

    assert [call.attempt for call in recorder.calls] == [1, 2]
    assert [call.model for call in recorder.calls] == ["primary", "secondary"]


async def test_the_summary_names_the_whole_chain() -> None:
    """The audit needs what was tried, not only what succeeded."""
    recorder = ModelRecorder()
    await start(recorder, "gpt-5.6-terra")
    await start(recorder, "llama-3.3-70b-versatile")

    summary = recorder.summary()
    assert summary["attempts"] == 2
    assert summary["used_fallback"] is True
    assert summary["chain"] == ["gpt-5.6-terra", "llama-3.3-70b-versatile"]


def test_a_recorder_that_saw_nothing_says_so() -> None:
    """Silence must not read as a successful call to an unnamed model."""
    recorder = ModelRecorder()

    assert recorder.served_by == UNKNOWN_MODEL
    assert recorder.used_fallback is False
    assert recorder.summary()["attempts"] == 0


async def test_recorders_do_not_share_state() -> None:
    """One per request: a shared recorder would mix requests together."""
    first, second = ModelRecorder(), ModelRecorder()
    await start(first, "primary")

    assert len(first.calls) == 1
    assert second.calls == []


# --- name extraction ---


@pytest.mark.parametrize(
    ("serialized", "kwargs"),
    [
        ({}, {"metadata": {"ls_model_name": "gpt-5.6-luna"}}),
        ({}, {"metadata": {"model_name": "gpt-5.6-luna"}}),
        ({}, {"invocation_params": {"model": "gpt-5.6-luna"}}),
        ({"kwargs": {"model_name": "gpt-5.6-luna"}}, {}),
    ],
)
def test_a_model_name_is_found_wherever_a_provider_puts_it(
    serialized: dict[str, Any], kwargs: dict[str, Any]
) -> None:
    """Providers disagree on where the name goes."""
    assert _model_name(serialized, kwargs) == "gpt-5.6-luna"


def test_an_unidentifiable_model_does_not_raise() -> None:
    """Failing to name a model must not fail the request it served."""
    assert _model_name({}, {}) == UNKNOWN_MODEL


def test_a_serialized_name_is_the_last_resort() -> None:
    assert _model_name({"name": "ChatGroq"}, {}) == "ChatGroq"

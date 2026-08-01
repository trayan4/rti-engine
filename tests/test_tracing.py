"""Tests for tracing configuration.

The tracer reads os.environ; this project's configuration does not live
there. What is asserted is that the bridge between them works, and that
an absent key disables tracing rather than breaking a run.
"""

import os
from collections.abc import Iterator
from typing import Any

import pytest

from rti_engine.observability.tracing import (
    API_KEY,
    PROJECT,
    TRACING_FLAG,
    enable_tracing,
    run_config,
    tracing_status,
)


@pytest.fixture(autouse=True)
def clean_environment() -> Iterator[None]:
    """Restore the tracer's environment variables around each test."""
    saved = {name: os.environ.get(name) for name in (TRACING_FLAG, API_KEY, PROJECT)}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_tracing_is_off_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing key means no tracing, not a failed run."""
    from rti_engine.config import settings as settings_module

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("LANGSMITH_API_KEY", "")

    status = enable_tracing()
    settings_module.get_settings.cache_clear()

    assert status.enabled is False
    assert status.reason is not None
    assert os.environ.get(TRACING_FLAG) != "true"


def test_status_reports_off_when_never_enabled() -> None:
    os.environ.pop(TRACING_FLAG, None)
    assert tracing_status().enabled is False


def test_status_reports_on_once_enabled() -> None:
    os.environ[TRACING_FLAG] = "true"
    os.environ[PROJECT] = "some-project"

    status = tracing_status()
    assert status.enabled is True
    assert status.project == "some-project"


# --- run configuration ---


def test_a_run_is_named_for_its_request() -> None:
    """A trace named "LangGraph" among two hundred others is unfindable."""
    config = run_config("req-42")
    assert config["run_name"] == "rti:request:req-42"


def test_the_tier_becomes_a_searchable_tag() -> None:
    config = run_config("req-42", tier="T2")
    assert "tier:T2" in config["tags"]
    assert "rti" in config["tags"]


def test_an_untiered_run_carries_no_tier_tag() -> None:
    """Before intake there is no tier, and inventing one would mislead."""
    config = run_config("req-42")
    assert not any(tag.startswith("tier:") for tag in config["tags"])


def test_extra_metadata_is_carried_through() -> None:
    """The eval harness labels runs by scenario; that must survive."""
    config: dict[str, Any] = run_config("req-42", tier="T1", scenario="S1")
    assert config["metadata"]["scenario"] == "S1"
    assert config["metadata"]["request_id"] == "req-42"

"""Tests for the analytical protocol.

Two halves. The result-handling functions are pure and tested with stub
tools: they decide whether a tool result is data or a refusal, and getting
that wrong means an agent narrates an authorization failure as though it
were a finding.

The protocol itself is tested live against the generated dataset, because
what it must get right — that S1's gap survives controls and S2's does
not — is a property of the whole path, not of any one function.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.tools import StructuredTool

from rti_engine.agents.analyst import (
    AnalysisError,
    GroupAnalysis,
    RequesterRecord,
    _call,
    _result_text,
    analyse_requester_group,
)
from rti_engine.db.models import AutonomyTier
from rti_engine.mcp.analytics_server import DATASET_PATH

needs_dataset = pytest.mark.skipif(
    not Path(DATASET_PATH).is_file(),
    reason="dataset not generated; run `make data`",
)


def stub_tool(name: str, payload: Any) -> StructuredTool:
    """A tool returning a fixed payload, shaped like an MCP content block."""

    async def _invoke(**_: Any) -> Any:
        return payload

    return StructuredTool(
        name=name, description=name, args_schema={"properties": {}}, coroutine=_invoke
    )


def content_block(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


# --- result handling ---


def test_content_blocks_are_unwrapped() -> None:
    assert _result_text(content_block('{"a": 1}')) == '{"a": 1}'


def test_a_plain_value_passes_through() -> None:
    assert _result_text("plain") == "plain"


async def test_a_successful_call_is_parsed() -> None:
    tools = {"t": stub_tool("t", content_block('{"found": true, "n": 3}'))}
    assert await _call(tools, "t") == {"found": True, "n": 3}


async def test_a_refusal_is_raised_not_returned() -> None:
    """An agent that treats a refusal as data narrates it as a finding."""
    refusal = "Error calling tool 'x': tier T1 may not request scope aggregate_group"
    tools = {"t": stub_tool("t", content_block(refusal))}

    with pytest.raises(AnalysisError, match="may not request scope"):
        await _call(tools, "t")


async def test_unparseable_output_is_refused() -> None:
    tools = {"t": stub_tool("t", content_block("not json at all"))}
    with pytest.raises(AnalysisError, match="unparseable"):
        await _call(tools, "t")


async def test_a_non_object_result_is_refused() -> None:
    tools = {"t": stub_tool("t", content_block("[1, 2, 3]"))}
    with pytest.raises(AnalysisError, match="expected an object"):
        await _call(tools, "t")


async def test_a_missing_tool_is_refused() -> None:
    with pytest.raises(AnalysisError, match="not available"):
        await _call({}, "absent")


# --- the requester record ---


def test_the_group_is_derived_from_the_requesters_own_record() -> None:
    """The comparator group is not chosen; it follows from the record."""
    record = RequesterRecord(
        employee_id="EMP-00001",
        country="DE",
        job_family="Sales",
        level="L3",
        working_pattern="full_time",
        fte=1.0,
        tenure_years=4.0,
        base_salary_fte_eur=70000.0,
        base_salary_actual_eur=70000.0,
        bonus_actual_eur=8000.0,
        total_comp_actual_eur=78000.0,
        currency="EUR",
    )
    assert record.group == "DE/Sales/L3"


def test_the_analysis_schema_is_closed() -> None:
    """An unmodelled field means a tool returned something unaccounted for."""
    with pytest.raises(ValueError):
        GroupAnalysis(unexpected=1)  # type: ignore[call-arg]


# --- the protocol, live ---


@pytest.fixture(scope="module")
def workforce() -> pd.DataFrame:
    return pd.read_parquet(DATASET_PATH)


def a_member_of(frame: pd.DataFrame, country: str, family: str, level: str) -> str:
    rows = frame[(frame.country == country) & (frame.job_family == family) & (frame.level == level)]
    return str(rows.iloc[0]["employee_id"])


@needs_dataset
async def test_an_unexplained_gap_survives_the_controls(workforce: pd.DataFrame) -> None:
    """S1: a real gap, and the protocol must report it as one."""
    employee = a_member_of(workforce, "DE", "Sales", "L3")
    result = await analyse_requester_group(employee, AutonomyTier.T2)

    assert result.group == "DE/Sales/L3"
    assert result.base_raw_gap_pct == pytest.approx(7.0, abs=0.1)
    assert result.base_significant is True
    assert result.exceeds_jpa_threshold is True


@needs_dataset
async def test_an_explained_gap_collapses_under_the_controls(
    workforce: pd.DataFrame,
) -> None:
    """S2: the false-positive trap. A large raw gap, fully explained."""
    employee = a_member_of(workforce, "FR", "Engineering", "L4")
    result = await analyse_requester_group(employee, AutonomyTier.T2)

    assert result.base_raw_gap_pct > 5.0
    assert abs(result.base_adjusted_gap_pct) < 3.0
    assert result.base_significant is False
    assert "tenure_years" in result.controls


@needs_dataset
async def test_an_age_specific_gap_needs_the_interaction_model(
    workforce: pd.DataFrame,
) -> None:
    """S5: invisible to a gender-only model, found by the interaction."""
    employee = a_member_of(workforce, "FR", "Engineering", "L3")
    result = await analyse_requester_group(employee, AutonomyTier.T2)

    assert result.base_significant is False
    assert result.interaction_significant is True
    assert result.older_gap_pct > result.base_adjusted_gap_pct


@needs_dataset
async def test_a_group_too_small_to_report_is_refused(workforce: pd.DataFrame) -> None:
    """S3: the protocol stops rather than reporting an unreliable figure."""
    employee = a_member_of(workforce, "ES", "Legal", "L5")

    with pytest.raises(AnalysisError, match="below the minimum"):
        await analyse_requester_group(employee, AutonomyTier.T2)


@needs_dataset
async def test_a_lower_tier_cannot_run_the_protocol(workforce: pd.DataFrame) -> None:
    """Every step past the own record concerns other employees."""
    employee = a_member_of(workforce, "DE", "Sales", "L3")

    with pytest.raises(AnalysisError, match="may not request scope"):
        await analyse_requester_group(employee, AutonomyTier.T1)


@needs_dataset
async def test_every_figure_is_traceable_to_a_tool(workforce: pd.DataFrame) -> None:
    """Nothing in the result is computed by this module."""
    employee = a_member_of(workforce, "DE", "Sales", "L3")
    result = await analyse_requester_group(employee, AutonomyTier.T2)

    assert set(result.tools_called) == {
        "get_own_pay_record",
        "describe_comparator_group",
        "compute_pay_gap_statistics",
        "check_age_interaction",
    }
    assert json.loads(result.model_dump_json())["group"] == "DE/Sales/L3"

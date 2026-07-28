"""Analyst: run the fixed analytical protocol for a requester's group.

The protocol is deterministic. A requester's comparator group is defined
by their own record — country, job family, level — and once that is known
the analyses to run follow from it. There is no judgment to exercise, so
no model is involved: one would add latency, cost and a way to go wrong
without adding anything.

Interpretation is a separate step. It receives this result as typed
fields, so the model never handles a number it could restate incorrectly.

Every figure here is quoted from a tool result, unchanged. Nothing in
this module computes.
"""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict

from rti_engine.db.models import AutonomyTier
from rti_engine.mcp.client import ANALYTICS, tool_session

BASE_METRIC = "base_salary_fte_eur"
TOTAL_COMP_METRIC = "total_comp_actual_eur"

TOOL_ERROR_PREFIX = "Error calling tool"


class AnalysisError(RuntimeError):
    """Raised when the protocol cannot be completed."""


class RequesterRecord(BaseModel):
    """The requester's own pay record, as returned by the analytics server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    employee_id: str
    country: str
    job_family: str
    level: str
    working_pattern: str
    fte: float
    tenure_years: float
    base_salary_fte_eur: float
    base_salary_actual_eur: float
    bonus_actual_eur: float
    total_comp_actual_eur: float
    currency: str

    @property
    def group(self) -> str:
        return f"{self.country}/{self.job_family}/{self.level}"


class GroupAnalysis(BaseModel):
    """Everything the protocol established about a requester's group.

    Field for field, this is what the tools returned. Nothing is derived
    here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    requester: RequesterRecord
    group: str

    n_total: int
    n_female: int
    n_male: int
    mean_female_eur: float
    mean_male_eur: float
    median_female_eur: float
    median_male_eur: float
    reportable: bool
    reportability_note: str

    base_raw_gap_pct: float
    base_median_gap_pct: float
    base_adjusted_gap_pct: float
    base_p_value: float
    base_significant: bool
    base_confidence_interval_pct: list[float]
    controls: list[str]
    exceeds_jpa_threshold: bool
    jpa_threshold_pct: float
    alpha: float

    total_comp_raw_gap_pct: float
    total_comp_adjusted_gap_pct: float
    total_comp_significant: bool

    age_cutoff: int
    younger_gap_pct: float
    older_gap_pct: float
    older_significant: bool
    interaction_p_value: float
    interaction_significant: bool

    tools_called: list[str]


def _result_text(result: Any) -> str:
    """Extract the payload from an MCP tool result.

    Results arrive as content blocks, and errors arrive the same way
    rather than as exceptions.
    """
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return str(result[0].get("text", ""))
    return str(result)


async def _call(tools: dict[str, BaseTool], name: str, **arguments: Any) -> dict[str, Any]:
    """Call one tool and return its parsed result, refusing an error payload."""
    tool = tools.get(name)
    if tool is None:
        raise AnalysisError(f"tool {name!r} is not available")

    text = _result_text(await tool.ainvoke(arguments))
    if text.startswith(TOOL_ERROR_PREFIX):
        raise AnalysisError(text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise AnalysisError(f"{name} returned unparseable output: {text[:200]}") from error

    if not isinstance(parsed, dict):
        raise AnalysisError(f"{name} returned {type(parsed).__name__}, expected an object")
    return parsed


async def _run_protocol(tools: dict[str, BaseTool]) -> GroupAnalysis:
    """Run the fixed sequence of analyses against an open tool session."""
    own = await _call(tools, "get_own_pay_record")
    if not own.get("found"):
        raise AnalysisError("no pay record found for this requester")

    requester = RequesterRecord.model_validate(
        {key: value for key, value in own.items() if key != "found"}
    )
    group = {
        "country": requester.country,
        "job_family": requester.job_family,
        "level": requester.level,
    }

    summary = await _call(tools, "describe_comparator_group", **group)
    base = await _call(tools, "compute_pay_gap_statistics", **group, metric=BASE_METRIC)
    total = await _call(tools, "compute_pay_gap_statistics", **group, metric=TOTAL_COMP_METRIC)
    interaction = await _call(tools, "check_age_interaction", **group)

    return GroupAnalysis(
        requester=requester,
        group=requester.group,
        n_total=summary["n_total"],
        n_female=summary["n_female"],
        n_male=summary["n_male"],
        mean_female_eur=summary["mean_female_eur"],
        mean_male_eur=summary["mean_male_eur"],
        median_female_eur=summary["median_female_eur"],
        median_male_eur=summary["median_male_eur"],
        reportable=summary["reportable"],
        reportability_note=summary["reportability_note"],
        base_raw_gap_pct=base["raw_mean_gap_pct"],
        base_median_gap_pct=base["raw_median_gap_pct"],
        base_adjusted_gap_pct=base["adjusted_gap_pct"],
        base_p_value=base["p_value"],
        base_significant=base["significant"],
        base_confidence_interval_pct=base["confidence_interval_pct"],
        controls=base["controls"],
        exceeds_jpa_threshold=base["exceeds_jpa_threshold"],
        jpa_threshold_pct=base["jpa_threshold_pct"],
        alpha=base["alpha"],
        total_comp_raw_gap_pct=total["raw_mean_gap_pct"],
        total_comp_adjusted_gap_pct=total["adjusted_gap_pct"],
        total_comp_significant=total["significant"],
        age_cutoff=interaction["age_cutoff"],
        younger_gap_pct=interaction["younger_gap_pct"],
        older_gap_pct=interaction["older_gap_pct"],
        older_significant=interaction["older_significant"],
        interaction_p_value=interaction["interaction_p_value"],
        interaction_significant=interaction["interaction_significant"],
        tools_called=[
            "get_own_pay_record",
            "describe_comparator_group",
            "compute_pay_gap_statistics",
            "check_age_interaction",
        ],
    )


async def analyse_requester_group(employee_id: str, tier: AutonomyTier) -> GroupAnalysis:
    """Run the full protocol for one requester's comparator group.

    Requires T2: every step after the requester's own record concerns
    other employees, and the tools refuse a lower tier regardless.

    One session serves the whole protocol, so the servers start once
    rather than once per call.
    """
    async with tool_session(employee_id, tier.value, servers=[ANALYTICS]) as tools:
        return await _run_protocol(tools)

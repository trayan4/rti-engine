"""The attack suite.

Each case asserts that a guarantee holds in code, not that a model
declined. The distinction is the point of the architecture: an attack
that talks its way past the instructions must still find nothing to call.

Classification behaviour under attack is measured in the evaluation
harness, where a live model call belongs. What is here runs offline or
against the tool boundary.
"""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from rti_engine.agents.intake import IntakeClassification, _apply_floor
from rti_engine.db.models import AutonomyTier
from rti_engine.evals.attacks import ALL_ATTACKS, Attack, by_defence
from rti_engine.guardrails.numbers import validate_numbers
from rti_engine.guardrails.pii import scan
from rti_engine.knowledge.graph_queries import GraphQueryError, run_query
from rti_engine.mcp.analytics_server import mcp
from rti_engine.mcp.client import IDENTITY_FIELDS

REQUESTER = "EMP-00001"


def attack(name: str) -> Attack:
    return next(item for item in ALL_ATTACKS if item.name == name)


# --- the catalog ---


def test_every_attack_names_a_defence_that_is_not_a_prompt() -> None:
    """A guarantee held only by instructions is not a guarantee."""
    assert ALL_ATTACKS
    assert all(item.defended_by for item in ALL_ATTACKS)
    assert all("prompt" not in item.defended_by for item in ALL_ATTACKS)


def test_attack_names_are_unique() -> None:
    names = [item.name for item in ALL_ATTACKS]
    assert len(names) == len(set(names))


def test_indirect_attacks_are_represented() -> None:
    """The harder case: instructions nobody typed, inside fetched content."""
    assert any(item.kind == "indirect" for item in ALL_ATTACKS)


# --- tier floor ---


def test_a_comparator_request_cannot_be_classified_downward() -> None:
    """However the request is dressed, comparator data forces T2."""
    for category in ("general_information", "own_pay", "unclear"):
        classification = IntakeClassification(
            category=category,  # type: ignore[arg-type]
            seeks_own_pay=True,
            seeks_comparator_data=True,
            ambiguous=False,
            rationale="attack",
        )
        tier, _ = _apply_floor(classification)
        assert tier is AutonomyTier.T2


# --- pii redaction ---


def test_a_named_colleague_is_removed_before_any_model_sees_it() -> None:
    result = scan(attack("named_colleague").payload)

    assert "PERSON" in result.entity_types
    assert "Maria Fernandez" not in result.redacted


def test_redaction_leaves_the_request_answerable() -> None:
    """A comparator request is still one without the name in it."""
    redacted = scan(attack("named_colleague").payload).redacted
    assert "earn" in redacted


# --- number validation ---


def test_an_estimated_figure_is_caught() -> None:
    """The payload asks for an estimate; the estimate has no source."""
    facts = {"base_salary_analysis": {"adjusted_gap_pct": 7.8}}
    result = validate_numbers("The sector benchmark is 14.5%.", [], facts)

    assert not result.grounded
    assert result.ungrounded[0].value == "14.5"


def test_a_zeroed_finding_is_caught_when_it_contradicts_the_facts() -> None:
    """The injected passage instructs a report of zero per cent."""
    facts = {"base_salary_analysis": {"adjusted_gap_pct": 7.8}}
    result = validate_numbers("The gap is 0.0% in all cases.", [], facts)

    assert not result.grounded


# --- query templates ---


def test_supplied_cypher_is_refused_as_an_unknown_query() -> None:
    """Agents choose from a menu; the payload is not on it."""
    with pytest.raises(GraphQueryError, match="unknown query"):
        run_query(attack("injected_cypher").payload)


def test_a_template_cannot_be_extended_with_extra_parameters() -> None:
    with pytest.raises(GraphQueryError, match="unexpected"):
        run_query("article_context", article=7, limit=9999)


# --- the tool surface ---


async def test_no_tool_returns_individual_pay_for_a_group() -> None:
    """The escalation the indirect attacks aim at has nothing to call."""
    async with Client(mcp) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert "list_employee_salaries" not in names
    assert "get_employee_record" not in names
    assert names == {
        "get_own_pay_record",
        "describe_comparator_group",
        "compute_pay_gap_statistics",
        "check_age_interaction",
        "optimize_remediation",
    }


async def test_a_group_below_the_reporting_minimum_is_refused() -> None:
    """The small-group probe: nine people in ES/Legal/L5."""
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="below the minimum"):
            await client.call_tool(
                "describe_comparator_group",
                {
                    "requester_employee_id": REQUESTER,
                    "tier": "T2",
                    "country": "ES",
                    "job_family": "Legal",
                    "level": "L5",
                },
            )


async def test_a_claimed_tier_does_not_grant_access() -> None:
    """The tier is a server parameter, not something a request asserts."""
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="aggregate_group"):
            await client.call_tool(
                "describe_comparator_group",
                {
                    "requester_employee_id": REQUESTER,
                    "tier": "T1",
                    "country": "DE",
                    "job_family": "Sales",
                    "level": "L3",
                },
            )


def test_identity_is_bound_rather_than_supplied() -> None:
    """The substitution attack changes an argument that is overwritten."""
    assert "requester_employee_id" in IDENTITY_FIELDS
    assert "tier" in IDENTITY_FIELDS


# --- coverage ---


def test_every_defence_has_at_least_one_attack() -> None:
    """A defence nothing attacks is a defence nothing tests."""
    for item in ALL_ATTACKS:
        assert by_defence(item.defended_by)

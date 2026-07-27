"""Tests for principal binding at the tool boundary.

The authorization module decides correctly and the servers enforce it,
but both operate on an identity that was, until this layer existed,
supplied by the agent. These assert the identity an agent sends is
discarded, and that the fields are not even present in the schema it is
shown.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from rti_engine.mcp.analytics_server import DATASET_PATH
from rti_engine.mcp.client import IDENTITY_FIELDS, load_tools_for

REQUESTER = "EMP-00001"
IMPERSONATED = "EMP-00042"

pytestmark = pytest.mark.skipif(
    not Path(DATASET_PATH).is_file(),
    reason="dataset not generated; run `make data`",
)


def result_text(result: Any) -> str:
    """MCP tools return content blocks, including for errors."""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return str(result[0].get("text", ""))
    return str(result)


async def tools_for(tier: str) -> dict[str, Any]:
    return {tool.name: tool for tool in await load_tools_for(REQUESTER, tier)}


async def test_identity_fields_are_hidden_from_the_model() -> None:
    """What the model cannot see, it cannot be persuaded to change."""
    tools = await tools_for("T1")
    schema = tools["get_own_pay_record"].args_schema

    assert isinstance(schema, dict)
    properties = schema.get("properties", {})
    assert not any(field in properties for field in IDENTITY_FIELDS)


async def test_tools_without_an_identity_keep_their_arguments() -> None:
    tools = await tools_for("T1")
    schema = tools["get_jurisdiction_status"].args_schema

    assert isinstance(schema, dict)
    assert "jurisdiction" in schema.get("properties", {})


async def test_a_supplied_employee_id_is_discarded() -> None:
    """An agent asking as someone else receives its own record."""
    tools = await tools_for("T1")
    result = await tools["get_own_pay_record"].ainvoke(
        {"requester_employee_id": IMPERSONATED, "tier": "T2"}
    )

    payload = json.loads(result_text(result))
    assert payload["employee_id"] == REQUESTER


async def test_a_supplied_tier_cannot_escalate() -> None:
    """The refusal must cite the authenticated tier, not the one sent."""
    tools = await tools_for("T1")
    result = await tools["describe_comparator_group"].ainvoke(
        {"tier": "T2", "country": "DE", "job_family": "Sales", "level": "L3"}
    )

    text = result_text(result)
    assert "may not request scope" in text
    assert "T1" in text


async def test_the_bound_tier_is_the_one_enforced() -> None:
    """Binding restricts; it does not grant. A T2 principal reaches the group."""
    tools = await tools_for("T2")
    result = await tools["describe_comparator_group"].ainvoke(
        {"country": "DE", "job_family": "Sales", "level": "L3"}
    )

    payload = json.loads(result_text(result))
    assert payload["n_total"] > 0
    assert "employee_id" not in payload


async def test_refusals_arrive_as_content_not_exceptions() -> None:
    """Recorded deliberately: agents must detect refusals in tool output.

    An agent that treats any returned string as success will narrate a
    refusal as though it were data.
    """
    tools = await tools_for("T0")
    result = await tools["get_own_pay_record"].ainvoke({})

    assert "Error calling tool" in result_text(result)

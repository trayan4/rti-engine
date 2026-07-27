"""Tests for the analytics MCP server.

The unit tests on the authorization module prove the rules are correct.
These prove the rules are actually reached: a tool that forgot to call
apply_scope would pass every test in test_authz.py and still hand an
agent another employee's salary.

So these exercise the tools through an MCP client, which is how an agent
will reach them.
"""

from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from rti_engine.mcp.analytics_server import (
    DATASET_PATH,
    PERMITTED_METRICS,
    mcp,
)

REQUESTER = "EMP-00001"
S1_GROUP = {"country": "DE", "job_family": "Sales", "level": "L3"}

pytestmark = pytest.mark.skipif(
    not Path(DATASET_PATH).is_file(),
    reason="dataset not generated; run `make data`",
)


async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call one tool through an MCP client and return its structured result."""
    async with Client(mcp) as client:
        result = await client.call_tool(name, arguments)
    data = result.data
    assert isinstance(data, dict)
    return data


async def test_every_tool_is_registered() -> None:
    async with Client(mcp) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert names == {
        "get_own_pay_record",
        "describe_comparator_group",
        "compute_pay_gap_statistics",
        "check_age_interaction",
        "optimize_remediation",
    }


async def test_t1_receives_its_own_record() -> None:
    data = await call_tool("get_own_pay_record", {"requester_employee_id": REQUESTER, "tier": "T1"})
    assert data["found"] is True
    assert data["employee_id"] == REQUESTER


async def test_t0_reaches_no_employee_data() -> None:
    with pytest.raises(ToolError, match="own_record"):
        await call_tool("get_own_pay_record", {"requester_employee_id": REQUESTER, "tier": "T0"})


async def test_t1_cannot_reach_a_comparator_group() -> None:
    """The tier boundary must hold at the tool, not only in the module."""
    with pytest.raises(ToolError, match="aggregate_group"):
        await call_tool(
            "describe_comparator_group",
            {"requester_employee_id": REQUESTER, "tier": "T1", **S1_GROUP},
        )


async def test_t2_receives_a_comparator_group_without_identities() -> None:
    data = await call_tool(
        "describe_comparator_group",
        {"requester_employee_id": REQUESTER, "tier": "T2", **S1_GROUP},
    )

    assert data["n_female"] > 0 and data["n_male"] > 0
    assert data["reportable"] is True
    assert "employee_id" not in data
    assert "full_name" not in data


async def test_a_planted_gap_is_measured_correctly() -> None:
    """S1 carries a calibrated 7% gap that survives the controls."""
    data = await call_tool(
        "compute_pay_gap_statistics",
        {"requester_employee_id": REQUESTER, "tier": "T2", **S1_GROUP},
    )

    assert data["raw_mean_gap_pct"] == pytest.approx(7.0, abs=0.1)
    assert data["significant"] is True
    assert data["exceeds_jpa_threshold"] is True
    assert "tenure_years" in data["controls"]


async def test_an_explained_gap_is_not_significant() -> None:
    """S2's raw gap disappears once tenure is controlled for."""
    data = await call_tool(
        "compute_pay_gap_statistics",
        {
            "requester_employee_id": REQUESTER,
            "tier": "T2",
            "country": "FR",
            "job_family": "Engineering",
            "level": "L4",
        },
    )

    assert data["raw_mean_gap_pct"] > 5.0
    assert abs(data["adjusted_gap_pct"]) < 3.0
    assert data["significant"] is False


async def test_a_group_below_the_minimum_size_is_refused() -> None:
    """S3 has nine people; aggregates must not become a way to read them."""
    with pytest.raises(ToolError, match="below the minimum"):
        await call_tool(
            "describe_comparator_group",
            {
                "requester_employee_id": REQUESTER,
                "tier": "T2",
                "country": "ES",
                "job_family": "Legal",
                "level": "L5",
            },
        )


async def test_only_permitted_metrics_may_be_measured() -> None:
    """Actual paid amounts across working patterns measure hours, not pay.

    Rejected at schema validation rather than in the function body: the
    tool's type makes the value unrepresentable, so the runtime check
    behind it never has to fire.
    """
    assert "base_salary_actual_eur" not in PERMITTED_METRICS

    with pytest.raises(ToolError, match="Input should be"):
        await call_tool(
            "compute_pay_gap_statistics",
            {
                "requester_employee_id": REQUESTER,
                "tier": "T2",
                **S1_GROUP,
                "metric": "base_salary_actual_eur",
            },
        )


async def test_an_unknown_tier_is_refused() -> None:
    with pytest.raises(ToolError, match="Input should be"):
        await call_tool("get_own_pay_record", {"requester_employee_id": REQUESTER, "tier": "T9"})


async def test_a_missing_identity_is_refused() -> None:
    with pytest.raises(ToolError):
        await call_tool("get_own_pay_record", {"requester_employee_id": "", "tier": "T1"})


async def test_remediation_requires_the_highest_tier() -> None:
    with pytest.raises(ToolError, match="requires tier T2"):
        await call_tool("optimize_remediation", {"requester_employee_id": REQUESTER, "tier": "T1"})


async def test_remediation_returns_no_individual_awards() -> None:
    """Awards name identifiable employees and must not cross the boundary."""
    data = await call_tool(
        "optimize_remediation", {"requester_employee_id": REQUESTER, "tier": "T2"}
    )

    assert data["feasible"] is True
    assert data["awards_count"] > 0
    assert "awards" not in data
    assert all(verdict["group"] for verdict in data["verdicts"])


async def test_only_unexplained_gaps_are_remediated() -> None:
    data = await call_tool(
        "optimize_remediation", {"requester_employee_id": REQUESTER, "tier": "T2"}
    )

    remediated = [v for v in data["verdicts"] if v["remediated"]]
    assert remediated
    assert all(v["verdict"] == "unexplained" for v in remediated)

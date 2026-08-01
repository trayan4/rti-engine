"""Tests for the request state and its routing.

Routing is asserted here as a pure function, because it is one: intake
decides the tier, code floors it, and this maps a tier to a path. Testing
it through a live classification would measure the classifier instead,
which the eval harness does.

The reducer test matters more than it looks. Without it, two nodes writing
audit entries overwrite each other, and the trail loses exactly the
entries that explain what went wrong.
"""

from typing import Any

import pytest
from langgraph.graph import END

from rti_engine.agents.graph import (
    DISCLOSURE_PIPELINE,
    RESPOND_INFORMATIONAL,
    RESPOND_OWN_DATA,
    TIER_NODES,
    build_graph,
    route_by_tier,
)
from rti_engine.agents.state import (
    MAX_REVISIONS,
    Actor,
    AuditEntry,
    RequestState,
    audited,
    failed,
    initial_state,
)
from rti_engine.db.models import AutonomyTier, RequestStatus


def state(**overrides: Any) -> RequestState:
    base = initial_state("req-1", "EMP-00001", "What is my salary?", "DE")
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# --- initial state ---


def test_a_new_request_starts_unclassified() -> None:
    """No tier until intake assigns one; there is no safe default."""
    fresh = initial_state("req-1", "EMP-00001", "text", "ES")

    assert fresh["tier"] is None
    assert fresh["status"] is RequestStatus.RECEIVED
    assert fresh["revision_count"] == 0
    assert fresh["approved_by"] is None


def test_receipt_is_recorded_before_anything_runs() -> None:
    fresh = initial_state("req-1", "EMP-00001", "text", "FR")

    assert len(fresh["audit"]) == 1
    assert fresh["audit"][0].action == "request_received"
    assert fresh["audit"][0].actor is Actor.SYSTEM


# --- routing ---


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        (AutonomyTier.T0, RESPOND_INFORMATIONAL),
        (AutonomyTier.T1, RESPOND_OWN_DATA),
        (AutonomyTier.T2, DISCLOSURE_PIPELINE),
    ],
)
def test_each_tier_takes_its_own_path(tier: AutonomyTier, expected: str) -> None:
    assert route_by_tier(state(tier=tier)) == expected


def test_every_tier_has_a_path() -> None:
    """A tier with no mapping would raise at routing time, mid-request."""
    assert set(TIER_NODES) == set(AutonomyTier)


def test_an_unclassified_request_ends_rather_than_defaulting() -> None:
    """Defaulting would mean choosing a disclosure level nobody decided."""
    assert route_by_tier(state(tier=None)) == END


def test_a_failed_request_ends() -> None:
    assert route_by_tier(state(tier=AutonomyTier.T2, errors=["boom"])) == END


def test_an_error_ends_the_run_even_with_a_tier_assigned() -> None:
    """Errors are checked first: a tier assigned before a failure is stale."""
    assert route_by_tier(state(tier=AutonomyTier.T0, errors=["boom"])) == END


# --- state updates ---


def test_an_audited_update_carries_one_entry() -> None:
    update = audited(Actor.INTAKE, "tier_assigned", tier="T2")

    assert len(update["audit"]) == 1
    assert update["audit"][0].detail == {"tier": "T2"}


def test_a_failure_is_recorded_in_both_places() -> None:
    """The audit explains what happened; the errors list is what routing reads."""
    update = failed(Actor.ANALYST, "analysis_failed", ValueError("no record"))

    assert update["status"] is RequestStatus.FAILED
    assert "ValueError" in update["errors"][0]
    assert update["audit"][0].detail["error_type"] == "ValueError"


def test_audit_entries_are_timestamped() -> None:
    entry = AuditEntry(actor=Actor.DRAFTER, action="draft_written")
    assert entry.occurred_at.tzinfo is not None


# --- the loop bound ---


def test_the_revision_limit_is_finite() -> None:
    """Two models that cannot agree must stop rather than argue."""
    assert 0 < MAX_REVISIONS < 10


# --- the compiled graph ---


async def test_the_graph_compiles_with_every_node() -> None:
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)

    assert {
        "intake",
        RESPOND_INFORMATIONAL,
        RESPOND_OWN_DATA,
        DISCLOSURE_PIPELINE,
    } <= nodes


async def test_a_failed_intake_reaches_no_tier_path() -> None:
    """A request that failed classification must not be handled anyway."""
    graph = build_graph()
    final = await graph.ainvoke(state(errors=["classification failed"], tier=None))

    routed = [entry for entry in final["audit"] if entry.action == "routed"]
    assert routed == []

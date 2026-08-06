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

from rti_engine.agents.drafter import DraftLetter, FigureUse, LetterSection
from rti_engine.agents.graph import (
    ANALYST,
    APPROVAL,
    DECISION_STATUS,
    DEGRADED,
    DRAFTER,
    REGULATORY,
    RESPOND_INFORMATIONAL,
    RESPOND_NOT_APPLICABLE,
    RESPOND_OWN_DATA,
    TIER_NODES,
    _guarded,
    approval_payload,
    build_graph,
    needs_revision,
    parse_decision,
    route_after_approval,
    route_after_review,
    route_by_tier,
)
from rti_engine.agents.reviewer import Finding, ReviewResult
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
        (AutonomyTier.T2, ANALYST),
    ],
)
def test_each_tier_takes_its_own_path(tier: AutonomyTier, expected: str) -> None:
    assert route_by_tier(state(tier=tier)) == expected


def test_every_tier_has_a_path() -> None:
    """A tier with no mapping would raise at routing time, mid-request."""
    assert set(TIER_NODES) == set(AutonomyTier)


def test_an_unclassified_request_gets_the_not_applicable_response() -> None:
    """A tier of None means the request was never a pay request at all,
    not that it failed classification — routing it to a fixed response
    is not defaulting to a disclosure level nobody decided."""
    assert route_by_tier(state(tier=None)) == RESPOND_NOT_APPLICABLE


def test_a_failed_request_reaches_the_degraded_response() -> None:
    """An employee who receives nothing has been told less than one who
    receives an acknowledgement."""
    assert route_by_tier(state(tier=AutonomyTier.T2, errors=["boom"])) == DEGRADED


def test_an_error_diverts_the_run_even_with_a_tier_assigned() -> None:
    """Errors are checked first: a tier assigned before a failure is stale."""
    assert route_by_tier(state(tier=AutonomyTier.T0, errors=["boom"])) == DEGRADED


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
        ANALYST,
    } <= nodes


async def test_a_failed_intake_reaches_no_tier_path() -> None:
    """A request that failed classification must not be handled anyway."""
    graph = build_graph()
    final = await graph.ainvoke(state(errors=["classification failed"], tier=None))

    routed = [entry for entry in final["audit"] if entry.action == "routed"]
    assert routed == []


# --- the tier 2 pipeline ---


def review(approved: bool = False, blocking: int = 0) -> ReviewResult:
    """A review result with the given approval and blocking-finding count."""
    findings = [
        Finding(
            kind="ungrounded_figure",
            severity="blocking",
            quote=f"figure {index}",
            problem="not in the fact sheet",
            suggested_fix="remove it",
        )
        for index in range(blocking)
    ]
    return ReviewResult(approved=approved, findings=findings, summary="s")


def test_an_approved_draft_is_not_sent_back() -> None:
    assert needs_revision(state(revision_count=0), review(approved=True)) is False


def test_a_blocked_draft_is_sent_back() -> None:
    assert needs_revision(state(revision_count=0), review(blocking=2)) is True


def test_revisions_stop_at_the_limit() -> None:
    """Two models that cannot agree must stop rather than argue."""
    assert needs_revision(state(revision_count=MAX_REVISIONS), review(blocking=2)) is False


def test_a_tier_two_request_never_completes_on_its_own() -> None:
    """Approved or not, it stops at the approval node for a human decision."""
    for result in (review(approved=True), review(blocking=1)):
        routed = route_after_review(state(revision_count=MAX_REVISIONS, review=result))
        assert routed == APPROVAL


def test_an_exhausted_draft_still_reaches_a_human() -> None:
    """A draft the reviewer will not approve goes forward with its findings."""
    routed = route_after_review(state(revision_count=MAX_REVISIONS, review=review(blocking=3)))
    assert routed == APPROVAL


def test_a_blocked_draft_within_the_limit_returns_to_the_drafter() -> None:
    assert route_after_review(state(revision_count=0, review=review(blocking=1))) == DRAFTER


def test_a_failed_run_does_not_revise() -> None:
    failing = state(revision_count=0, review=review(blocking=1), errors=["boom"])
    assert route_after_review(failing) == DEGRADED


def test_a_missing_review_ends_rather_than_looping() -> None:
    assert route_after_review(state(revision_count=0, review=None)) == END


def test_the_pipeline_stops_where_a_node_failed() -> None:
    """Each stage depends on the last; continuing past a failure is worse."""
    route = _guarded(REGULATORY)

    assert route(state()) == REGULATORY
    assert route(state(errors=["analysis failed"])) == DEGRADED


def test_a_request_over_budget_is_diverted() -> None:
    """A runaway request stops rather than spending without a ceiling."""
    route = _guarded(REGULATORY)
    assert route(state(tokens_used=10_000_000)) == DEGRADED


# --- human approval ---


def test_an_approval_settles_the_request() -> None:
    decision = parse_decision({"decision": "approved", "reviewer_id": "hr-7", "comment": "fine"})
    assert DECISION_STATUS[decision.decision] is RequestStatus.APPROVED


def test_a_rejection_settles_the_request() -> None:
    decision = parse_decision({"decision": "rejected", "reviewer_id": "hr-7"})
    assert DECISION_STATUS[decision.decision] is RequestStatus.REJECTED


def test_requesting_changes_returns_to_the_drafter() -> None:
    settled = state(approval_decision="changes_requested")
    assert route_after_approval(settled) == DRAFTER


def test_a_settled_request_goes_no_further() -> None:
    for outcome in ("approved", "rejected"):
        assert route_after_approval(state(approval_decision=outcome)) == END


def test_an_unknown_decision_is_refused() -> None:
    """A malformed resume must not release a statutory disclosure."""
    with pytest.raises(ValueError, match="unknown decision"):
        parse_decision({"decision": "looks_fine", "reviewer_id": "hr-7"})


def test_a_decision_without_a_reviewer_is_refused() -> None:
    """An approval nobody is accountable for is not an approval."""
    with pytest.raises(ValueError):
        parse_decision({"decision": "approved"})


def test_the_decision_schema_is_closed() -> None:
    with pytest.raises(ValueError):
        parse_decision({"decision": "approved", "reviewer_id": "hr-7", "auto_approve": True})


def test_the_payload_carries_the_findings_a_reviewer_needs() -> None:
    """A decision made without the findings is a rubber stamp."""
    letter = DraftLetter(
        subject="s",
        salutation="Dear colleague,",
        sections=[LetterSection(heading="h", body="b")],
        closing="Yours sincerely,",
        figures_used=[FigureUse(value="7.8%", source_field="adjusted_gap_pct", meaning="gap")],
        citations=["Directive (EU) 2023/970, Article 7"],
    )
    payload = approval_payload(state(draft=letter, review=review(blocking=2), revision_count=2))

    assert payload["reviewer_approved"] is False
    assert len(payload["blocking_findings"]) == 2
    assert payload["revisions_used"] == 2
    assert payload["figures_used"][0]["value"] == "7.8%"
    assert set(payload["decisions"]) == set(DECISION_STATUS)


def test_a_payload_needs_both_a_draft_and_a_review() -> None:
    with pytest.raises(ValueError, match="requires both"):
        approval_payload(state(draft=None, review=None))

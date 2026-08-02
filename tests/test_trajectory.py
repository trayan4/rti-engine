"""Tests for path checking.

These assert what an outcome cannot: that the request got where it went
by a legitimate route. A letter with no ungrounded figures says nothing
about whether the validator ran, and a tier 0 response that mentions no
comparator says nothing about whether the analytics server was reached.

Audit sequences are built by hand rather than run, so every rule can be
exercised including the ones that should never fire in practice.
"""

from rti_engine.agents.state import Actor, AuditEntry
from rti_engine.db.models import AutonomyTier
from rti_engine.evals.trajectory import check_trajectory, trajectory_summary

T2_HAPPY = [
    "request_received",
    "tier_assigned",
    "group_analysed",
    "position_established",
    "draft_written",
    "figures_validated",
    "draft_reviewed",
    "approval_decided",
]

T1_HAPPY = ["request_received", "tier_assigned", "response_written"]


def audit(*actions: str) -> list[AuditEntry]:
    """Build an audit trail from a sequence of action names."""
    return [AuditEntry(actor=Actor.SYSTEM, action=action) for action in actions]


def violations(actions: list[str], tier: AutonomyTier | None) -> set[str]:
    return {item.rule for item in check_trajectory(audit(*actions), tier)}


# --- legitimate paths ---


def test_a_complete_disclosure_path_is_valid() -> None:
    assert check_trajectory(audit(*T2_HAPPY), AutonomyTier.T2) == []


def test_an_autonomous_path_is_valid() -> None:
    assert check_trajectory(audit(*T1_HAPPY), AutonomyTier.T1) == []


def test_a_revision_loop_is_valid() -> None:
    actions = [
        "request_received",
        "tier_assigned",
        "group_analysed",
        "position_established",
        "draft_written",
        "figures_validated",
        "draft_reviewed",
        "draft_revised",
        "figures_validated",
        "draft_reviewed",
        "approval_decided",
    ]
    assert check_trajectory(audit(*actions), AutonomyTier.T2) == []


def test_a_degraded_path_does_not_require_the_full_pipeline() -> None:
    """A request that failed early never reached the stages it skipped."""
    actions = ["request_received", "tier_assigned", "group_analysed", "degraded_response_issued"]
    assert "disclosure_runs_in_order" not in violations(actions, AutonomyTier.T2)


# --- the tier boundary ---


def test_an_autonomous_tier_may_not_reach_the_pipeline() -> None:
    """The trajectory form of the tier guarantee."""
    actions = ["request_received", "tier_assigned", "group_analysed"]

    assert "tier_stays_within_its_path" in violations(actions, AutonomyTier.T1)
    assert "tier_stays_within_its_path" in violations(actions, AutonomyTier.T0)


def test_a_disclosure_may_not_complete_autonomously() -> None:
    actions = ["request_received", "tier_assigned", "response_written"]
    assert "a_disclosure_never_completes_alone" in violations(actions, AutonomyTier.T2)


def test_an_autonomous_request_sent_for_approval_is_a_defect() -> None:
    """Not unsafe, but the tier distinction is not doing its job."""
    actions = ["request_received", "tier_assigned", "response_written", "approval_decided"]
    assert "autonomous_paths_do_not_await_approval" in violations(actions, AutonomyTier.T1)


# --- ordering ---


def test_nothing_runs_before_classification() -> None:
    actions = ["request_received", "group_analysed", "tier_assigned"]
    assert "starts_with_intake" in violations(actions, AutonomyTier.T2)


def test_an_empty_trail_is_a_violation() -> None:
    assert "starts_with_intake" in violations([], AutonomyTier.T2)


def test_the_stages_must_run_in_dependency_order() -> None:
    """Each consumes the last one's output."""
    actions = [
        "request_received",
        "tier_assigned",
        "position_established",
        "group_analysed",
        "draft_written",
        "figures_validated",
        "draft_reviewed",
    ]
    assert "disclosure_runs_in_order" in violations(actions, AutonomyTier.T2)


def test_a_missing_stage_is_reported() -> None:
    actions = [
        "request_received",
        "tier_assigned",
        "group_analysed",
        "draft_written",
        "figures_validated",
        "draft_reviewed",
    ]
    assert "disclosure_runs_in_order" in violations(actions, AutonomyTier.T2)


# --- validation before review ---


def test_a_draft_reviewed_without_validation_is_caught() -> None:
    """The validator is deterministic; the reviewer can be persuaded."""
    actions = [
        "request_received",
        "tier_assigned",
        "group_analysed",
        "position_established",
        "draft_written",
        "draft_reviewed",
    ]
    assert "every_draft_is_validated" in violations(actions, AutonomyTier.T2)


def test_a_revision_that_skipped_validation_is_caught() -> None:
    actions = [
        "request_received",
        "tier_assigned",
        "group_analysed",
        "position_established",
        "draft_written",
        "figures_validated",
        "draft_reviewed",
        "draft_revised",
        "draft_reviewed",
    ]
    assert "every_draft_is_validated" in violations(actions, AutonomyTier.T2)


# --- approval ---


def test_a_decision_with_no_review_before_it_is_caught() -> None:
    """A person asked to approve without findings is rubber-stamping."""
    actions = [
        "request_received",
        "tier_assigned",
        "group_analysed",
        "position_established",
        "draft_written",
        "figures_validated",
        "approval_decided",
    ]
    assert "approval_follows_a_review" in violations(actions, AutonomyTier.T2)


# --- reporting ---


def test_the_summary_names_the_rules_that_failed() -> None:
    found = check_trajectory(audit("request_received", "group_analysed"), AutonomyTier.T2)
    summary = trajectory_summary(found)

    assert summary["trajectory_valid"] is False
    assert "starts_with_intake" in summary["violations"]


def test_a_valid_path_summarises_cleanly() -> None:
    summary = trajectory_summary(check_trajectory(audit(*T2_HAPPY), AutonomyTier.T2))

    assert summary["trajectory_valid"] is True
    assert summary["violations"] == []

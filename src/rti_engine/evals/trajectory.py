"""Check the path a request took, not just where it ended.

An outcome can be right for the wrong reason. A letter with no ungrounded
figures proves nothing about whether the validator ran; a tier 0 response
that mentions no comparator proves nothing about whether the analytics
server was reachable. The audit trail records every node, so the path is
checkable directly.

Deterministic and free. These read a list of recorded actions and assert
properties of the sequence, with no model involved.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from rti_engine.agents.state import AuditEntry
from rti_engine.db.models import AutonomyTier

RECEIVED = "request_received"
TIER_ASSIGNED = "tier_assigned"
GROUP_ANALYSED = "group_analysed"
POSITION_ESTABLISHED = "position_established"
DRAFT_WRITTEN = "draft_written"
DRAFT_REVISED = "draft_revised"
FIGURES_VALIDATED = "figures_validated"
DRAFT_REVIEWED = "draft_reviewed"
APPROVAL_DECIDED = "approval_decided"
RESPONSE_WRITTEN = "response_written"
DEGRADED = "degraded_response_issued"

DRAFTING_ACTIONS = frozenset({DRAFT_WRITTEN, DRAFT_REVISED})

DISCLOSURE_ONLY = frozenset({GROUP_ANALYSED, POSITION_ESTABLISHED, DRAFT_WRITTEN, DRAFT_REVISED})
"""Actions that may only occur on the statutory disclosure path.

An informational or own-data request reaching any of these has been
handled at a level its tier does not permit, whatever the response said.
"""


class Violation(BaseModel):
    """One way a request's path was not what it should have been."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    detail: str


Rule = Callable[[list[str], AutonomyTier | None], Violation | None]


def actions_of(audit: list[AuditEntry]) -> list[str]:
    """Reduce an audit trail to the sequence of actions it records."""
    return [entry.action for entry in audit]


def _index(actions: list[str], action: str) -> int:
    """Where an action first occurs, or -1 if it never did."""
    return actions.index(action) if action in actions else -1


def _starts_with_intake(actions: list[str], tier: AutonomyTier | None) -> Violation | None:
    """Nothing may run before the request has been classified."""
    if not actions:
        return Violation(rule="starts_with_intake", detail="no actions were recorded")

    if actions[0] != RECEIVED:
        return Violation(rule="starts_with_intake", detail=f"began with {actions[0]!r}")

    classified = _index(actions, TIER_ASSIGNED)
    for action in DISCLOSURE_ONLY:
        position = _index(actions, action)
        if position >= 0 and (classified < 0 or position < classified):
            return Violation(
                rule="starts_with_intake",
                detail=f"{action} ran before the request was classified",
            )
    return None


def _tier_stays_within_its_path(actions: list[str], tier: AutonomyTier | None) -> Violation | None:
    """An informational or own-data request never reaches the pipeline.

    This is the trajectory form of the tier guarantee. The tools refuse
    the data, but a request that got as far as calling them was routed
    somewhere it should not have been.
    """
    if tier is AutonomyTier.T2:
        return None

    reached = sorted(set(actions) & DISCLOSURE_ONLY)
    if reached:
        return Violation(
            rule="tier_stays_within_its_path",
            detail=f"tier {tier.value if tier else 'none'} reached {', '.join(reached)}",
        )
    return None


def _disclosure_runs_in_order(actions: list[str], tier: AutonomyTier | None) -> Violation | None:
    """Analysis precedes the legal position, which precedes the draft.

    Each stage consumes the last one's output, so a draft written before
    the analysis was available was written from something else.
    """
    if tier is not AutonomyTier.T2 or DEGRADED in actions:
        return None

    stages = [GROUP_ANALYSED, POSITION_ESTABLISHED, DRAFT_WRITTEN]
    positions = [_index(actions, stage) for stage in stages]

    if any(position < 0 for position in positions):
        missing = [stage for stage, position in zip(stages, positions, strict=True) if position < 0]
        return Violation(
            rule="disclosure_runs_in_order",
            detail=f"never ran: {', '.join(missing)}",
        )

    if positions != sorted(positions):
        return Violation(
            rule="disclosure_runs_in_order",
            detail=f"ran out of order: {list(zip(stages, positions, strict=True))}",
        )
    return None


def _every_draft_is_validated(actions: list[str], tier: AutonomyTier | None) -> Violation | None:
    """No draft reaches the reviewer before its figures were checked.

    The validator is deterministic and the reviewer is not, so a draft
    that skipped it was judged only by something that can be persuaded.
    """
    drafts = sum(action in DRAFTING_ACTIONS for action in actions)
    validations = actions.count(FIGURES_VALIDATED)

    if drafts and validations < drafts:
        return Violation(
            rule="every_draft_is_validated",
            detail=f"{drafts} drafts but only {validations} validations",
        )

    for position, action in enumerate(actions):
        if action != DRAFT_REVIEWED:
            continue
        preceding = actions[:position]
        if FIGURES_VALIDATED not in preceding:
            return Violation(
                rule="every_draft_is_validated",
                detail="a draft was reviewed before any validation ran",
            )
    return None


def _approval_follows_a_review(actions: list[str], tier: AutonomyTier | None) -> Violation | None:
    """Nothing is decided that was not first drafted and reviewed.

    A person asked to approve without the reviewer's findings is being
    asked to rubber-stamp.
    """
    decided = _index(actions, APPROVAL_DECIDED)
    if decided < 0:
        return None

    if DRAFT_REVIEWED not in actions[:decided]:
        return Violation(
            rule="approval_follows_a_review",
            detail="a decision was recorded with no review before it",
        )
    return None


def _autonomous_paths_do_not_await_approval(
    actions: list[str], tier: AutonomyTier | None
) -> Violation | None:
    """A tier 0 or 1 request completes without a person.

    Sending one for approval is not unsafe, but it is a defect: it means
    the tier distinction is not doing what it exists for.
    """
    if tier is AutonomyTier.T2:
        return None

    if APPROVAL_DECIDED in actions:
        return Violation(
            rule="autonomous_paths_do_not_await_approval",
            detail=f"tier {tier.value if tier else 'none'} was sent for approval",
        )
    return None


def _a_disclosure_never_completes_alone(
    actions: list[str], tier: AutonomyTier | None
) -> Violation | None:
    """A statutory disclosure is never written on the autonomous path."""
    if tier is not AutonomyTier.T2:
        return None

    if RESPONSE_WRITTEN in actions:
        return Violation(
            rule="a_disclosure_never_completes_alone",
            detail="a tier 2 request produced an autonomous response",
        )
    return None


RULES: tuple[Rule, ...] = (
    _starts_with_intake,
    _tier_stays_within_its_path,
    _disclosure_runs_in_order,
    _every_draft_is_validated,
    _approval_follows_a_review,
    _autonomous_paths_do_not_await_approval,
    _a_disclosure_never_completes_alone,
)


def check_trajectory(audit: list[AuditEntry], tier: AutonomyTier | None) -> list[Violation]:
    """Return every way this request's path was not legitimate."""
    actions = actions_of(audit)
    found = [rule(actions, tier) for rule in RULES]
    return [violation for violation in found if violation is not None]


def trajectory_summary(violations: list[Violation]) -> dict[str, Any]:
    """Describe the check for the audit trail."""
    return {
        "trajectory_valid": not violations,
        "violations": [violation.rule for violation in violations],
    }

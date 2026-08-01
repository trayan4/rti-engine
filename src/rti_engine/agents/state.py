"""The state that flows through the request graph.

Each node receives this, does one thing, and returns a partial update.
LangGraph merges the updates and checkpoints the result, which is what
lets a Tier 2 request pause for human approval and resume later — possibly
in a different process, after a restart.

Values are the typed objects the agents already produce rather than loose
dictionaries, so a node cannot hand the next one a differently shaped
payload and have it discovered three steps later.

Accumulating fields carry reducers. Without them a node returning an audit
entry would replace the list rather than extend it, and the trail would
have holes in exactly the places something went wrong.
"""

import enum
from datetime import UTC, datetime
from operator import add
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from rti_engine.agents.analyst import GroupAnalysis
from rti_engine.agents.drafter import DraftLetter
from rti_engine.agents.intake import IntakeResult
from rti_engine.agents.regulatory import RegulatoryPosition
from rti_engine.agents.reviewer import ReviewResult
from rti_engine.db.models import AutonomyTier, RequestStatus

MAX_REVISIONS = 2
"""How many times the Drafter may be sent back by the Reviewer.

The loop must terminate. A draft the reviewer will not approve after this
many attempts goes to a human with its findings attached, which is a
better outcome than burning tokens on an argument between two models.
"""

Jurisdiction = Literal["DE", "FR", "ES"]


class Actor(enum.StrEnum):
    """Who took an audited action."""

    SUPERVISOR = "supervisor"
    INTAKE = "intake"
    ANALYST = "analyst"
    REGULATORY = "regulatory"
    DRAFTER = "drafter"
    REVIEWER = "reviewer"
    SYSTEM = "system"


class AuditEntry(BaseModel):
    """One recorded action, mirroring the audit_events table.

    Written during the run and persisted with it, so the trail survives a
    process that never reaches the database.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor: Actor
    action: str
    detail: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RequestState(TypedDict, total=False):
    """Everything known about one request as it moves through the graph."""

    request_id: str
    requester_employee_id: str
    request_text: str
    jurisdiction: Jurisdiction

    tier: AutonomyTier | None
    intake: IntakeResult | None

    analysis: GroupAnalysis | None
    position: RegulatoryPosition | None
    draft: DraftLetter | None
    review: ReviewResult | None

    revision_count: int
    pay_setting_criteria: str | None
    """Retrieved once and reused across revisions rather than re-fetched."""

    approved_by: str | None
    """Set when a human approves a Tier 2 response. Never set by an agent."""

    status: RequestStatus
    audit: Annotated[list[AuditEntry], add]
    errors: Annotated[list[str], add]


def initial_state(
    request_id: str,
    requester_employee_id: str,
    request_text: str,
    jurisdiction: Jurisdiction,
) -> RequestState:
    """Build the state a new request starts from."""
    return RequestState(
        request_id=request_id,
        requester_employee_id=requester_employee_id,
        request_text=request_text,
        jurisdiction=jurisdiction,
        tier=None,
        intake=None,
        analysis=None,
        position=None,
        draft=None,
        review=None,
        revision_count=0,
        pay_setting_criteria=None,
        approved_by=None,
        status=RequestStatus.RECEIVED,
        audit=[
            AuditEntry(
                actor=Actor.SYSTEM,
                action="request_received",
                detail={"jurisdiction": jurisdiction},
            )
        ],
        errors=[],
    )


def audited(actor: Actor, action: str, **detail: Any) -> dict[str, list[AuditEntry]]:
    """Return a state update recording one action.

    Merged into whatever else a node returns, so recording an action never
    means remembering to carry the existing list forward.
    """
    return {"audit": [AuditEntry(actor=actor, action=action, detail=detail)]}


def failed(actor: Actor, action: str, error: Exception) -> dict[str, Any]:
    """Return a state update recording a failure.

    The error is recorded in both places deliberately: the audit trail
    explains what happened, and the errors list is what routing decisions
    read.
    """
    message = f"{actor.value}: {type(error).__name__}: {error}"
    return {
        "errors": [message],
        "audit": [
            AuditEntry(
                actor=actor,
                action=action,
                detail={"error_type": type(error).__name__, "message": str(error)},
            )
        ],
        "status": RequestStatus.FAILED,
    }

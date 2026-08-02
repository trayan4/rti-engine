"""What crosses the HTTP boundary.

Deliberately not the graph's state. The state holds full analyses, legal
positions and review findings; an employee checking on their request
should see none of that, and a reviewer should see it in a shape built
for deciding rather than for computing.

Note what is absent from every request body: an employee id. Identity
comes from the authenticated session, so a caller cannot ask as someone
else — the same guarantee the agents operate under, applied at the edge.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SubmitRequest(BaseModel):
    """A new pay-information request."""

    model_config = ConfigDict(extra="forbid")

    request_text: str = Field(min_length=1, max_length=4000)


class RequestSummary(BaseModel):
    """Enough to list a request without loading its working state."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    request_id: str
    requester_employee_id: str
    tier: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class RequestDetail(BaseModel):
    """One request, with its response if it has produced one."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    tier: str | None
    status: str
    letter: str | None = None
    citations: list[str] = Field(default_factory=list)
    audit: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ApprovalItem(BaseModel):
    """A request awaiting a decision, with what the decision rests on."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    jurisdiction: str
    letter: str
    figures_used: list[dict[str, str]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    reviewer_approved: bool
    revisions_used: int
    blocking_findings: list[dict[str, str]] = Field(default_factory=list)
    advisory_findings: list[dict[str, str]] = Field(default_factory=list)


class DecisionRequest(BaseModel):
    """A reviewer's decision on a request awaiting approval.

    The reviewer's identity is not in the body. It comes from the session,
    so an approval is always attributable to whoever was authenticated.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected", "changes_requested"]
    comment: str | None = Field(default=None, max_length=2000)

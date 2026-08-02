"""Who is calling.

Identity comes from the request's authentication, never from its body.
That is the same rule the agents operate under, applied one layer out: a
caller cannot submit a request as another employee any more than an agent
can query for one.

Headers stand in for a real identity provider here. In deployment this
would validate a signed token and read the same fields from its claims —
what matters architecturally is that the principal is constructed by this
layer and passed down, never accepted from the caller's payload.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from rti_engine.db.authz import Principal

EMPLOYEE_HEADER = "X-Employee-Id"
REVIEWER_HEADER = "X-Reviewer"


def get_principal(
    employee_id: Annotated[str | None, Header(alias=EMPLOYEE_HEADER)] = None,
    reviewer: Annotated[str | None, Header(alias=REVIEWER_HEADER)] = None,
) -> Principal:
    """Build the principal for this request, or refuse it."""
    if not employee_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{EMPLOYEE_HEADER} is required",
        )

    return Principal(
        employee_id=employee_id,
        is_reviewer=(reviewer or "").lower() in {"true", "1", "yes"},
    )


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


def require_reviewer(principal: CurrentPrincipal) -> Principal:
    """Refuse a caller who is not a reviewer.

    Approving a statutory disclosure is not something a requester may do
    for their own request, so the check is a separate dependency rather
    than a condition inside an endpoint.
    """
    if not principal.is_reviewer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="this action requires a reviewer",
        )
    return principal


CurrentReviewer = Annotated[Principal, Depends(require_reviewer)]

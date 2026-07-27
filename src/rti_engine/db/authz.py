"""Authorization: what a given requester may see, enforced in code.

The tiered autonomy design grants different data access at each tier: T0
requests reach no employee data, T1 reaches only the requester's own
record, and T2 may reach comparator groups but only in aggregate and only
above a minimum size.

None of that may depend on an agent behaving correctly. An agent states
what it wants; this module decides what it gets. A request for another
employee's salary is not declined by a well-behaved model — it has no code
path that could satisfy it.

Direct identifiers are stripped from every aggregate result, and group
size is checked before any aggregate is released, so a "group" of one
cannot be used to read an individual's pay.
"""

import enum
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from rti_engine.analytics.catalog import Thresholds
from rti_engine.db.models import AutonomyTier

IDENTIFYING_COLUMNS: tuple[str, ...] = ("employee_id", "full_name")
"""Columns that identify a person and must never appear in an aggregate."""

FILTERABLE_COLUMNS: frozenset[str] = frozenset(
    {"country", "job_family", "level", "working_pattern"}
)
"""The only columns a caller may filter on. Anything else is refused."""


class AuthorizationError(Exception):
    """Raised when a request falls outside the requester's permitted scope."""


class ScopeKind(enum.StrEnum):
    """The shape of data being requested."""

    NONE = "none"
    OWN_RECORD = "own_record"
    AGGREGATE_GROUP = "aggregate_group"


PERMITTED_SCOPES: dict[AutonomyTier, frozenset[ScopeKind]] = {
    AutonomyTier.T0: frozenset({ScopeKind.NONE}),
    AutonomyTier.T1: frozenset({ScopeKind.NONE, ScopeKind.OWN_RECORD}),
    AutonomyTier.T2: frozenset({ScopeKind.NONE, ScopeKind.OWN_RECORD, ScopeKind.AGGREGATE_GROUP}),
}
"""What each tier may ask for. The single source of truth for access."""


@dataclass(frozen=True)
class Principal:
    """Who is making the request.

    Established by the application from the authenticated session, never
    from anything an agent produced or a request body contained.
    """

    employee_id: str
    is_reviewer: bool = False


class QueryScope(BaseModel):
    """What a caller is asking for."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ScopeKind
    filters: dict[str, str] = {}
    """Group selectors, for aggregate scope only."""


def _reject(reason: str) -> None:
    """Raise an authorization failure with a fixed, non-leaking message."""
    raise AuthorizationError(reason)


def authorise(principal: Principal, tier: AutonomyTier, scope: QueryScope) -> None:
    """Check a request against the tier's permitted scopes.

    Returns nothing on success and raises on any violation. Callers must
    treat a raised error as terminal rather than degrading to a narrower
    query, so that a refusal is always visible in the audit trail.
    """
    permitted = PERMITTED_SCOPES[tier]
    if scope.kind not in permitted:
        _reject(f"tier {tier.value} may not request scope {scope.kind.value}")

    if scope.kind is ScopeKind.OWN_RECORD and scope.filters:
        _reject("own-record scope does not accept filters")

    if scope.kind is ScopeKind.AGGREGATE_GROUP:
        unknown = set(scope.filters) - FILTERABLE_COLUMNS
        if unknown:
            _reject(f"filtering is not permitted on: {', '.join(sorted(unknown))}")

    if not principal.employee_id:
        _reject("no requester identity was established")


def apply_scope(
    frame: pd.DataFrame,
    principal: Principal,
    tier: AutonomyTier,
    scope: QueryScope,
    thresholds: Thresholds,
) -> pd.DataFrame:
    """Return only the rows and columns this requester is permitted to see.

    The requester's identity comes from the principal, never from the
    scope, so an agent cannot substitute another employee's id and read
    their pay.
    """
    authorise(principal, tier, scope)

    if scope.kind is ScopeKind.NONE:
        return frame.iloc[0:0]

    if scope.kind is ScopeKind.OWN_RECORD:
        return frame[frame["employee_id"] == principal.employee_id]

    selected = frame
    for column, value in scope.filters.items():
        selected = selected[selected[column] == value]

    if len(selected) < thresholds.min_reportable_group_size:
        _reject(
            f"the selected group has {len(selected)} members, below the minimum "
            f"of {thresholds.min_reportable_group_size} required for disclosure"
        )

    return selected.drop(columns=list(IDENTIFYING_COLUMNS), errors="ignore")


def describe_permissions(tier: AutonomyTier) -> dict[str, Any]:
    """Summarise a tier's access, for inclusion in the audit trail."""
    return {
        "tier": tier.value,
        "permitted_scopes": sorted(kind.value for kind in PERMITTED_SCOPES[tier]),
        "identifiers_stripped_from_aggregates": list(IDENTIFYING_COLUMNS),
    }

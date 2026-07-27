"""Tests for the authorization layer.

These assert the property the tiered autonomy design rests on: what a
requester may see is decided by code, not by what an agent asks for. A
prompt-injected agent requesting another employee's salary must find no
code path that returns it.
"""

import pandas as pd
import pytest

from rti_engine.analytics.catalog import Thresholds
from rti_engine.db.authz import (
    IDENTIFYING_COLUMNS,
    AuthorizationError,
    Principal,
    QueryScope,
    ScopeKind,
    apply_scope,
    authorise,
    describe_permissions,
)
from rti_engine.db.models import AutonomyTier

THRESHOLDS = Thresholds(jpa_trigger_pct=5.0, significance_alpha=0.05, min_reportable_group_size=10)


@pytest.fixture
def workforce() -> pd.DataFrame:
    """A small stand-in workforce: twelve in Sales, three in Legal."""
    rows = [
        {
            "employee_id": f"EMP-{index:05d}",
            "full_name": f"Person {index}",
            "country": "DE",
            "job_family": "Sales" if index <= 12 else "Legal",
            "level": "L3",
            "working_pattern": "full_time",
            "base_salary_fte_eur": 70000.0 + index * 100,
        }
        for index in range(1, 16)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def requester() -> Principal:
    return Principal(employee_id="EMP-00001")


def test_t0_may_not_reach_any_employee_data(requester: Principal) -> None:
    with pytest.raises(AuthorizationError):
        authorise(requester, AutonomyTier.T0, QueryScope(kind=ScopeKind.OWN_RECORD))


def test_t1_may_not_reach_group_data(requester: Principal) -> None:
    scope = QueryScope(kind=ScopeKind.AGGREGATE_GROUP, filters={"job_family": "Sales"})
    with pytest.raises(AuthorizationError):
        authorise(requester, AutonomyTier.T1, scope)


def test_t2_may_reach_group_data(requester: Principal) -> None:
    scope = QueryScope(kind=ScopeKind.AGGREGATE_GROUP, filters={"job_family": "Sales"})
    authorise(requester, AutonomyTier.T2, scope)


def test_t0_returns_no_rows(workforce: pd.DataFrame, requester: Principal) -> None:
    result = apply_scope(
        workforce, requester, AutonomyTier.T0, QueryScope(kind=ScopeKind.NONE), THRESHOLDS
    )
    assert len(result) == 0


def test_t1_returns_only_the_requesters_own_row(
    workforce: pd.DataFrame, requester: Principal
) -> None:
    result = apply_scope(
        workforce, requester, AutonomyTier.T1, QueryScope(kind=ScopeKind.OWN_RECORD), THRESHOLDS
    )
    assert len(result) == 1
    assert result.iloc[0]["employee_id"] == requester.employee_id


def test_identity_comes_from_the_principal_not_the_request(workforce: pd.DataFrame) -> None:
    """Two requesters asking identically must receive different rows."""
    scope = QueryScope(kind=ScopeKind.OWN_RECORD)

    first = apply_scope(
        workforce, Principal(employee_id="EMP-00003"), AutonomyTier.T1, scope, THRESHOLDS
    )
    second = apply_scope(
        workforce, Principal(employee_id="EMP-00007"), AutonomyTier.T1, scope, THRESHOLDS
    )

    assert first.iloc[0]["employee_id"] == "EMP-00003"
    assert second.iloc[0]["employee_id"] == "EMP-00007"


def test_own_record_scope_refuses_filters(requester: Principal) -> None:
    """Filters on own-record scope are a smuggling attempt, not a refinement."""
    scope = QueryScope(kind=ScopeKind.OWN_RECORD, filters={"job_family": "Sales"})
    with pytest.raises(AuthorizationError):
        authorise(requester, AutonomyTier.T1, scope)


def test_aggregate_strips_identifying_columns(
    workforce: pd.DataFrame, requester: Principal
) -> None:
    scope = QueryScope(kind=ScopeKind.AGGREGATE_GROUP, filters={"job_family": "Sales"})
    result = apply_scope(workforce, requester, AutonomyTier.T2, scope, THRESHOLDS)

    assert len(result) == 12
    for column in IDENTIFYING_COLUMNS:
        assert column not in result.columns


def test_aggregate_below_minimum_group_size_is_refused(
    workforce: pd.DataFrame, requester: Principal
) -> None:
    """A group of three cannot be used to read three people's pay."""
    scope = QueryScope(kind=ScopeKind.AGGREGATE_GROUP, filters={"job_family": "Legal"})
    with pytest.raises(AuthorizationError):
        apply_scope(workforce, requester, AutonomyTier.T2, scope, THRESHOLDS)


def test_filtering_on_an_unlisted_column_is_refused(requester: Principal) -> None:
    """Only the declared group selectors are filterable, nothing else."""
    scope = QueryScope(kind=ScopeKind.AGGREGATE_GROUP, filters={"employee_id": "EMP-00002"})
    with pytest.raises(AuthorizationError):
        authorise(requester, AutonomyTier.T2, scope)


def test_missing_identity_is_refused() -> None:
    scope = QueryScope(kind=ScopeKind.OWN_RECORD)
    with pytest.raises(AuthorizationError):
        authorise(Principal(employee_id=""), AutonomyTier.T1, scope)


def test_scope_model_rejects_unknown_fields() -> None:
    """The scope contract is closed: unexpected keys are a bug, not an extra."""
    with pytest.raises(ValueError):
        QueryScope(kind=ScopeKind.OWN_RECORD, bypass=True)  # type: ignore[call-arg]


def test_permissions_are_describable_for_the_audit_trail() -> None:
    described = describe_permissions(AutonomyTier.T1)
    assert described["tier"] == "T1"
    assert "aggregate_group" not in described["permitted_scopes"]

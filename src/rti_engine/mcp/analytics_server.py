"""MCP server exposing the deterministic analytics as tools.

Agents call these tools; they do not hold database credentials and they do
not compute. Every number that can reach a generated document originates
here, in pure Python that no model has influenced.

Authorization is applied inside every tool, before any row is read. The
requester's identity and tier are parameters supplied by the application
from the authenticated session — an agent passing a different employee id
changes nothing, because the identity used for scoping is never taken from
the agent's request body.

Tools return typed results rather than prose. A model that receives a
number as a structured field can quote it; a model that receives a
paragraph can paraphrase it into something subtly wrong.
"""

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastmcp import FastMCP

from rti_engine.analytics.catalog import Thresholds, load_catalog
from rti_engine.analytics.cleaning import clean_workforce
from rti_engine.analytics.gap_metrics import compute_pay_gap
from rti_engine.analytics.regression import (
    estimate_adjusted_gap,
    estimate_interaction_gap,
)
from rti_engine.analytics.remediation import optimise_remediation
from rti_engine.db.authz import (
    AuthorizationError,
    Principal,
    QueryScope,
    ScopeKind,
    apply_scope,
)
from rti_engine.db.models import AutonomyTier

DATASET_PATH = Path("data/generated/workforce.parquet")

BASE_FTE_COLUMN = "base_salary_fte_eur"
TOTAL_COMP_COLUMN = "total_comp_actual_eur"

PERMITTED_METRICS: frozenset[str] = frozenset({BASE_FTE_COLUMN, TOTAL_COMP_COLUMN})
"""The only columns a caller may measure.

Full-time-equivalent base pay and total compensation are the two the
directive is concerned with. Allowing arbitrary columns would let a caller
measure actual paid amounts across different working patterns, which
produces a difference reflecting hours rather than pay.
"""

Tier = Literal["T0", "T1", "T2"]
Country = Literal["DE", "FR", "ES"]
JobFamily = Literal["Sales", "Engineering", "Marketing", "Legal", "Operations"]
Level = Literal["L1", "L2", "L3", "L4", "L5"]
Metric = Literal["base_salary_fte_eur", "total_comp_actual_eur"]
"""Closed value sets, expressed in the tool schema.

A model given a bare string invents plausible values — "Spain" for a
country code, "Engineering Dept" for a job family. Constraining the schema
makes those unrepresentable rather than merely refused.
"""

mcp: FastMCP[Any] = FastMCP("rti-analytics")


@lru_cache
def _workforce() -> pd.DataFrame:
    """Load and clean the dataset once per process."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"dataset not found at {DATASET_PATH}; run `make data`")

    cleaned, _ = clean_workforce(pd.read_parquet(DATASET_PATH))
    return cleaned


@lru_cache
def _thresholds() -> Thresholds:
    """Return the statutory and statistical thresholds from the catalog."""
    return load_catalog().thresholds


def _tier(value: str) -> AutonomyTier:
    """Parse a tier, refusing anything outside the defined set."""
    try:
        return AutonomyTier(value)
    except ValueError as error:
        permitted = ", ".join(t.value for t in AutonomyTier)
        raise AuthorizationError(f"unknown tier {value!r}; permitted: {permitted}") from error


def _metric(column: str) -> str:
    """Validate the requested measure against the permitted set."""
    if column not in PERMITTED_METRICS:
        permitted = ", ".join(sorted(PERMITTED_METRICS))
        raise AuthorizationError(f"unknown metric {column!r}; permitted: {permitted}")
    return column


def _scoped_rows(
    requester_employee_id: str,
    tier: str,
    kind: ScopeKind,
    filters: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return the rows this requester is permitted to see, and no others."""
    return apply_scope(
        _workforce(),
        Principal(employee_id=requester_employee_id),
        _tier(tier),
        QueryScope(kind=kind, filters=dict(filters or {})),
        _thresholds(),
    )


@mcp.tool()
def get_own_pay_record(requester_employee_id: str, tier: Tier) -> dict[str, Any]:
    """Return the requester's own pay record.

    Available from tier T1 upward. The record returned is always the
    requester's own, regardless of any identifier appearing elsewhere in
    the request.
    """
    rows = _scoped_rows(requester_employee_id, tier, ScopeKind.OWN_RECORD)
    if rows.empty:
        return {"found": False, "employee_id": requester_employee_id}

    record = rows.iloc[0]
    return {
        "found": True,
        "employee_id": str(record["employee_id"]),
        "country": str(record["country"]),
        "job_family": str(record["job_family"]),
        "level": str(record["level"]),
        "working_pattern": str(record["working_pattern"]),
        "fte": float(record["fte"]),
        "tenure_years": float(record["tenure_years"]),
        "base_salary_fte_eur": float(record["base_salary_fte_eur"]),
        "base_salary_actual_eur": float(record["base_salary_actual_eur"]),
        "bonus_actual_eur": float(record["bonus_actual_eur"]),
        "total_comp_actual_eur": float(record["total_comp_actual_eur"]),
        "currency": "EUR",
    }


@mcp.tool()
def describe_comparator_group(
    requester_employee_id: str,
    tier: Tier,
    country: Country,
    job_family: JobFamily,
    level: Level,
) -> dict[str, Any]:
    """Describe a category of workers performing work of equal value.

    Requires tier T2. Returns headcounts and averages only: identifying
    columns are stripped, and a group below the minimum reportable size is
    refused rather than returned in a reduced form.
    """
    filters = {"country": country, "job_family": job_family, "level": level}
    rows = _scoped_rows(requester_employee_id, tier, ScopeKind.AGGREGATE_GROUP, filters)
    result = compute_pay_gap(rows, BASE_FTE_COLUMN, _thresholds())

    return {
        "group": f"{country}/{job_family}/{level}",
        "n_total": result.summary.n_total,
        "n_female": result.summary.n_female,
        "n_male": result.summary.n_male,
        "mean_female_eur": round(result.summary.mean_female, 2),
        "mean_male_eur": round(result.summary.mean_male, 2),
        "median_female_eur": round(result.summary.median_female, 2),
        "median_male_eur": round(result.summary.median_male, 2),
        "reportable": result.reportable,
        "reportability_note": result.reportability_note,
        "currency": "EUR",
    }


@mcp.tool()
def compute_pay_gap_statistics(
    requester_employee_id: str,
    tier: Tier,
    country: Country,
    job_family: JobFamily,
    level: Level,
    metric: Metric = "base_salary_fte_eur",
) -> dict[str, Any]:
    """Compute the raw and adjusted pay gap for a category.

    Requires tier T2. The adjusted figure controls for level, job family,
    country and tenure; the difference between the two is what separates a
    gap that is explained from one that is not.
    """
    column = _metric(metric)
    filters = {"country": country, "job_family": job_family, "level": level}
    rows = _scoped_rows(requester_employee_id, tier, ScopeKind.AGGREGATE_GROUP, filters)

    raw = compute_pay_gap(rows, column, _thresholds())
    adjusted = estimate_adjusted_gap(rows, column, _thresholds())

    return {
        "group": f"{country}/{job_family}/{level}",
        "metric": column,
        "n_observations": adjusted.n_observations,
        "raw_mean_gap_pct": raw.mean_gap_pct,
        "raw_median_gap_pct": raw.median_gap_pct,
        "adjusted_gap_pct": adjusted.adjusted_gap_pct,
        "p_value": adjusted.p_value,
        "significant": adjusted.significant,
        "confidence_interval_pct": [adjusted.ci_low_pct, adjusted.ci_high_pct],
        "controls": adjusted.controls,
        "r_squared": adjusted.r_squared,
        "reportable": raw.reportable,
        "reportability_note": raw.reportability_note,
        "exceeds_jpa_threshold": raw.exceeds_jpa_threshold,
        "jpa_threshold_pct": raw.jpa_threshold_pct,
        "alpha": adjusted.alpha,
    }


@mcp.tool()
def check_age_interaction(
    requester_employee_id: str,
    tier: Tier,
    country: Country,
    job_family: JobFamily,
    level: Level,
) -> dict[str, Any]:
    """Test whether a pay gap is concentrated in one age band.

    Requires tier T2. A gender-only model averages an age-specific penalty
    across all women and can miss it entirely; this estimates each band
    separately.
    """
    filters = {"country": country, "job_family": job_family, "level": level}
    rows = _scoped_rows(requester_employee_id, tier, ScopeKind.AGGREGATE_GROUP, filters)
    result = estimate_interaction_gap(rows, BASE_FTE_COLUMN, _thresholds())

    return {
        "group": f"{country}/{job_family}/{level}",
        "age_cutoff": result.age_cutoff,
        "n_observations": result.n_observations,
        "younger_gap_pct": result.younger_gap_pct,
        "younger_p_value": result.younger_p_value,
        "older_gap_pct": result.older_gap_pct,
        "older_p_value": result.older_p_value,
        "older_significant": result.older_significant,
        "interaction_p_value": result.interaction_p_value,
        "interaction_significant": result.interaction_significant,
        "alpha": result.alpha,
    }


@mcp.tool()
def optimize_remediation(
    requester_employee_id: str,
    tier: Tier,
    max_individual_raise_pct: float = 15.0,
) -> dict[str, Any]:
    """Compute the least-cost salary adjustments closing every unexplained gap.

    Requires tier T2. Only gaps that survive the controls and clear a
    correction for multiple comparisons are remediated; explained and
    inconclusive gaps are reported with wording that says what is true of
    them, and no money is proposed against either.

    Individual awards are not returned: they name identifiable employees.
    """
    if _tier(tier) is not AutonomyTier.T2:
        raise AuthorizationError("remediation planning requires tier T2")
    if not requester_employee_id:
        raise AuthorizationError("no requester identity was established")

    plan = optimise_remediation(
        _workforce(),
        _thresholds(),
        max_individual_raise_pct=max_individual_raise_pct,
    )

    return {
        "status": plan.status,
        "feasible": plan.feasible,
        "correction": plan.multiple_comparison_correction,
        "target_gap_pct": plan.target_gap_pct,
        "max_individual_raise_pct": plan.max_individual_raise_pct,
        "groups_assessed": plan.groups_assessed,
        "groups_remediated": plan.groups_remediated,
        "total_cost_eur": plan.total_cost_eur,
        "awards_count": len(plan.awards),
        "groups": [
            {
                "group": group.group,
                "gap_before_pct": group.gap_before_pct,
                "gap_after_pct": group.gap_after_pct,
                "cost_eur": group.cost_eur,
            }
            for group in plan.groups
        ],
        "verdicts": [
            {
                "group": verdict.group,
                "verdict": verdict.verdict,
                "raw_gap_pct": verdict.raw_gap_pct,
                "adjusted_gap_pct": verdict.adjusted_gap_pct,
                "q_value": verdict.q_value,
                "remediated": verdict.remediated,
                "note": verdict.note,
            }
            for verdict in plan.verdicts
            if verdict.exceeds_target
        ],
        "currency": "EUR",
    }


if __name__ == "__main__":
    mcp.run()

"""Least-cost remediation planning.

Answers the question that follows a finding: what is the cheapest set of
salary adjustments that brings every unexplained pay gap below the
statutory threshold?

Expressed as a linear program. Minimise total spend, subject to each
affected group's post-adjustment gap falling below the target, no salary
decreasing, and no individual receiving an implausibly large increase.

Two safeguards are enforced here rather than left to the caller, because
neither may depend on an agent behaving correctly. Only gaps that survive
the controls *and* clear a multiple-comparison correction are remediated:
spending money to close an explained gap misstates the finding, and
spending it on a false positive is money spent on a coin flip. Groups that
fail either test are still reported, with wording that says what is
actually true of them.

Nothing here targets individuals by gender. Every employee in an affected
group is eligible; the optimum concentrates on the underpaid side because
raising the better-paid side would widen the gap the constraint requires
closing.

Deterministic and free of any model call.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pulp
from pydantic import BaseModel, ConfigDict

from rti_engine.analytics.catalog import Thresholds
from rti_engine.analytics.inference import (
    EXPLAINED_RESIDUAL_PCT,
    GapVerdict,
    benjamini_hochberg,
    classify_gap,
)
from rti_engine.analytics.regression import estimate_adjusted_gap

BASE_FTE_COLUMN = "base_salary_fte_eur"
FEMALE = "F"
MALE = "M"

DEFAULT_GROUP_BY: tuple[str, ...] = ("country", "job_family", "level")
DEFAULT_MAX_INDIVIDUAL_RAISE_PCT = 15.0
AWARD_EPSILON_EUR = 0.01
"""Awards below this are solver noise rather than real adjustments."""


@dataclass(frozen=True)
class _Assessment:
    """Internal working record for one assessed group."""

    label: str
    members: pd.DataFrame
    n_female: int
    n_male: int
    raw_gap_pct: float
    adjusted_gap_pct: float
    p_value: float


class EmployeeAward(BaseModel):
    """One employee's salary adjustment under the plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    employee_id: str
    group: str
    current_salary_eur: float
    raise_eur: float
    new_salary_eur: float
    raise_pct: float


class GroupVerdict(BaseModel):
    """What was concluded about one group, whether or not it was remediated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str
    n_female: int
    n_male: int
    raw_gap_pct: float
    adjusted_gap_pct: float
    p_value: float
    q_value: float
    verdict: GapVerdict
    exceeds_target: bool
    remediated: bool
    note: str


class GroupRemediation(BaseModel):
    """The effect of the plan on one remediated group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str
    n_female: int
    n_male: int
    gap_before_pct: float
    gap_after_pct: float
    cost_eur: float


class RemediationPlan(BaseModel):
    """A complete least-cost plan, or the reason one could not be produced."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    feasible: bool
    unexplained_only: bool
    multiple_comparison_correction: str
    target_gap_pct: float
    alpha: float
    max_individual_raise_pct: float
    groups_assessed: int
    groups_remediated: int
    total_cost_eur: float
    verdicts: list[GroupVerdict]
    groups: list[GroupRemediation]
    awards: list[EmployeeAward]
    notes: list[str]


def _label(key: Any) -> str:
    """Render a groupby key as a readable group name."""
    parts = key if isinstance(key, tuple) else (key,)
    return "/".join(str(part) for part in parts)


def _variable_name(employee_id: str) -> str:
    """Solver-safe variable name for one employee's raise."""
    return "raise_" + employee_id.replace("-", "_")


def _gap_pct(male_mean: float, female_mean: float) -> float:
    """Percentage by which the male mean exceeds the female mean."""
    if male_mean == 0.0:
        return 0.0
    return 100.0 * (male_mean - female_mean) / male_mean


def _assess_groups(
    frame: pd.DataFrame,
    thresholds: Thresholds,
    column: str,
    group_by: Sequence[str],
) -> tuple[list[_Assessment], list[str]]:
    """Measure and model every group large enough to assess.

    Every assessable group is tested, not only those above the threshold,
    so that the multiple-comparison correction is applied across the full
    family of comparisons actually made.
    """
    assessments: list[_Assessment] = []
    notes: list[str] = []

    for key, members in frame.groupby(list(group_by), sort=True):
        label = _label(key)
        usable = members[members[column].notna() & (members[column] > 0)]
        female = usable[usable["gender"] == FEMALE]
        male = usable[usable["gender"] == MALE]

        smaller = min(len(female), len(male))
        if smaller < thresholds.min_reportable_group_size:
            notes.append(f"{label}: smallest gender group is {smaller}, too small to assess")
            continue

        raw_gap = _gap_pct(float(male[column].mean()), float(female[column].mean()))
        adjusted = estimate_adjusted_gap(usable, column, thresholds)

        assessments.append(
            _Assessment(
                label=label,
                members=usable,
                n_female=len(female),
                n_male=len(male),
                raw_gap_pct=raw_gap,
                adjusted_gap_pct=adjusted.adjusted_gap_pct,
                p_value=adjusted.p_value,
            )
        )

    return assessments, notes


def optimise_remediation(
    frame: pd.DataFrame,
    thresholds: Thresholds,
    column: str = BASE_FTE_COLUMN,
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
    max_individual_raise_pct: float = DEFAULT_MAX_INDIVIDUAL_RAISE_PCT,
    target_gap_pct: float | None = None,
    unexplained_only: bool = True,
    explained_residual_pct: float = EXPLAINED_RESIDUAL_PCT,
) -> RemediationPlan:
    """Find the cheapest set of raises that closes every actionable gap."""
    target = thresholds.jpa_trigger_pct if target_gap_pct is None else target_gap_pct
    target_ratio = 1.0 - target / 100.0
    cap_fraction = max_individual_raise_pct / 100.0

    assessments, notes = _assess_groups(frame, thresholds, column, group_by)
    q_values = benjamini_hochberg([item.p_value for item in assessments])

    verdicts: list[GroupVerdict] = []
    to_remediate: list[_Assessment] = []

    for assessment, q_value in zip(assessments, q_values, strict=True):
        classification = classify_gap(
            raw_gap_pct=assessment.raw_gap_pct,
            adjusted_gap_pct=assessment.adjusted_gap_pct,
            q_value=q_value,
            alpha=thresholds.significance_alpha,
            explained_residual_pct=explained_residual_pct,
        )

        exceeds_target = assessment.raw_gap_pct > target
        actionable = classification.actionable or not unexplained_only
        remediate = exceeds_target and actionable
        if remediate:
            to_remediate.append(assessment)

        verdicts.append(
            GroupVerdict(
                group=assessment.label,
                n_female=assessment.n_female,
                n_male=assessment.n_male,
                raw_gap_pct=round(assessment.raw_gap_pct, 4),
                adjusted_gap_pct=round(assessment.adjusted_gap_pct, 4),
                p_value=assessment.p_value,
                q_value=q_value,
                verdict=classification.verdict,
                exceeds_target=exceeds_target,
                remediated=remediate,
                note=classification.note,
            )
        )

    def _empty_plan(status: str, feasible: bool, extra_note: str) -> RemediationPlan:
        return RemediationPlan(
            status=status,
            feasible=feasible,
            unexplained_only=unexplained_only,
            multiple_comparison_correction="benjamini-hochberg",
            target_gap_pct=target,
            alpha=thresholds.significance_alpha,
            max_individual_raise_pct=max_individual_raise_pct,
            groups_assessed=len(assessments),
            groups_remediated=0,
            total_cost_eur=0.0,
            verdicts=verdicts,
            groups=[],
            awards=[],
            notes=[*notes, extra_note],
        )

    if not to_remediate:
        return _empty_plan("Optimal", True, "no group requires remediation")

    problem = pulp.LpProblem("pay_gap_remediation", pulp.LpMinimize)
    variables: dict[str, Any] = {}

    for assessment in to_remediate:
        members = assessment.members
        for employee_id, salary in zip(members["employee_id"], members[column], strict=True):
            variables[str(employee_id)] = pulp.LpVariable(
                _variable_name(str(employee_id)),
                lowBound=0.0,
                upBound=float(salary) * cap_fraction,
            )

        female = members[members["gender"] == FEMALE]
        male = members[members["gender"] == MALE]

        # Require mean_female_post >= target_ratio * mean_male_post.
        # Multiplied through by both headcounts to keep the constraint linear.
        female_terms = pulp.lpSum(
            [
                float(salary) + variables[str(employee_id)]
                for employee_id, salary in zip(female["employee_id"], female[column], strict=True)
            ]
        )
        male_terms = pulp.lpSum(
            [
                float(salary) + variables[str(employee_id)]
                for employee_id, salary in zip(male["employee_id"], male[column], strict=True)
            ]
        )
        problem += (
            len(male) * female_terms >= target_ratio * len(female) * male_terms,
            (f"gap_{assessment.label.replace('/', '_')}"),
        )

    problem += pulp.lpSum(list(variables.values())), "total_remediation_cost"
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = str(pulp.LpStatus[problem.status])

    if status != "Optimal":
        return _empty_plan(
            status,
            False,
            f"no plan exists within a {max_individual_raise_pct}% individual cap; "
            f"raise the cap or lower the target",
        )

    awards: list[EmployeeAward] = []
    group_results: list[GroupRemediation] = []
    total_cost = 0.0

    for assessment in to_remediate:
        members = assessment.members
        group_cost = 0.0
        post_salary: dict[str, float] = {}

        for employee_id, salary in zip(members["employee_id"], members[column], strict=True):
            key_id = str(employee_id)
            awarded = float(pulp.value(variables[key_id]) or 0.0)
            post_salary[key_id] = float(salary) + awarded
            if awarded < AWARD_EPSILON_EUR:
                continue

            group_cost += awarded
            awards.append(
                EmployeeAward(
                    employee_id=key_id,
                    group=assessment.label,
                    current_salary_eur=round(float(salary), 2),
                    raise_eur=round(awarded, 2),
                    new_salary_eur=round(float(salary) + awarded, 2),
                    raise_pct=round(100.0 * awarded / float(salary), 4),
                )
            )

        female_ids = [
            str(value) for value in members.loc[members["gender"] == FEMALE, "employee_id"]
        ]
        male_ids = [str(value) for value in members.loc[members["gender"] == MALE, "employee_id"]]
        female_mean = sum(post_salary[key] for key in female_ids) / len(female_ids)
        male_mean = sum(post_salary[key] for key in male_ids) / len(male_ids)

        total_cost += group_cost
        group_results.append(
            GroupRemediation(
                group=assessment.label,
                n_female=len(female_ids),
                n_male=len(male_ids),
                gap_before_pct=round(assessment.raw_gap_pct, 4),
                gap_after_pct=round(_gap_pct(male_mean, female_mean), 4),
                cost_eur=round(group_cost, 2),
            )
        )

    return RemediationPlan(
        status=status,
        feasible=True,
        unexplained_only=unexplained_only,
        multiple_comparison_correction="benjamini-hochberg",
        target_gap_pct=target,
        alpha=thresholds.significance_alpha,
        max_individual_raise_pct=max_individual_raise_pct,
        groups_assessed=len(assessments),
        groups_remediated=len(group_results),
        total_cost_eur=round(total_cost, 2),
        verdicts=verdicts,
        groups=group_results,
        awards=awards,
        notes=notes,
    )

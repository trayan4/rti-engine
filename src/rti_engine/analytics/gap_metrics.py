"""Headline pay-gap statistics required by the directive.

Computes the mean and median gender pay gap for a population, together
with the two gatekeeping checks that must precede any reported finding:
whether the group is large enough to support a conclusion at all, and
whether the gap crosses the threshold that triggers a joint pay
assessment.

Every function here is pure, deterministic and free of any model call.
These are the only numbers permitted to appear in a generated document.
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict

from rti_engine.analytics.catalog import Thresholds

FEMALE = "F"
MALE = "M"


class GroupSummary(BaseModel):
    """Headcounts and central tendencies for one population, by gender."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str
    n_total: int
    n_female: int
    n_male: int
    mean_female: float
    mean_male: float
    median_female: float
    median_male: float


class PayGapResult(BaseModel):
    """A computed pay gap and the conclusions permitted to follow from it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: GroupSummary
    mean_gap_pct: float
    median_gap_pct: float
    reportable: bool
    reportability_note: str
    exceeds_jpa_threshold: bool
    jpa_threshold_pct: float


def _gap_pct(male_value: float, female_value: float) -> float:
    """Percentage by which the male figure exceeds the female figure.

    This is the direction the directive specifies: a positive number means
    women are paid less. Returns 0.0 when the male figure is zero, since
    no meaningful ratio exists.
    """
    if male_value == 0.0:
        return 0.0
    return 100.0 * (male_value - female_value) / male_value


def summarise_group(frame: pd.DataFrame, column: str) -> GroupSummary:
    """Summarise one population by gender, ignoring rows missing the metric."""
    usable = frame[frame[column].notna()]
    female = usable.loc[usable["gender"] == FEMALE, column]
    male = usable.loc[usable["gender"] == MALE, column]

    return GroupSummary(
        column=column,
        n_total=len(usable),
        n_female=len(female),
        n_male=len(male),
        mean_female=float(female.mean()) if len(female) else 0.0,
        mean_male=float(male.mean()) if len(male) else 0.0,
        median_female=float(female.median()) if len(female) else 0.0,
        median_male=float(male.median()) if len(male) else 0.0,
    )


def compute_pay_gap(
    frame: pd.DataFrame,
    column: str,
    thresholds: Thresholds,
) -> PayGapResult:
    """Compute the mean and median pay gap for a population.

    The gap is always computed and returned, but `reportable` records
    whether the group is large enough for that gap to support a
    conclusion. A gap measured across too few people is a number, not a
    finding, and callers must respect the distinction.
    """
    summary = summarise_group(frame, column)

    mean_gap = _gap_pct(summary.mean_male, summary.mean_female)
    median_gap = _gap_pct(summary.median_male, summary.median_female)

    smaller_group = min(summary.n_female, summary.n_male)
    reportable = smaller_group >= thresholds.min_reportable_group_size

    if summary.n_female == 0 or summary.n_male == 0:
        note = "one gender is absent from this population; no comparison is possible"
    elif reportable:
        note = f"both gender groups meet the minimum size of {thresholds.min_reportable_group_size}"
    else:
        note = (
            f"smallest gender group has {smaller_group} members, below the "
            f"minimum of {thresholds.min_reportable_group_size}; any gap shown "
            f"is indicative only"
        )

    return PayGapResult(
        summary=summary,
        mean_gap_pct=round(mean_gap, 4),
        median_gap_pct=round(median_gap, 4),
        reportable=reportable,
        reportability_note=note,
        exceeds_jpa_threshold=mean_gap >= thresholds.jpa_trigger_pct,
        jpa_threshold_pct=thresholds.jpa_trigger_pct,
    )

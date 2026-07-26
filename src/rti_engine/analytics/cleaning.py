"""Detect and normalise data-quality defects before any statistic is computed.

Every defect found is recorded in a report that travels with the cleaned
data and ends up in the audit bundle. The rule this module enforces is
that nothing is ever silently computed over: a defect is either normalised
and logged, or flagged and excluded, but never ignored.

Missing values are deliberately not imputed. Filling them in would be
inventing data, which is the failure mode this system exists to prevent.
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

MONTHS_PER_YEAR = 12

MONETARY_COLUMNS: list[str] = [
    "base_salary_fte_eur",
    "base_salary_actual_eur",
    "bonus_actual_eur",
    "total_comp_actual_eur",
]
"""Currency columns, which must always be converted together.

Defined here rather than imported from the generator: cleaning is part of
the production analysis path and must not depend on test-data tooling.
"""

FindingCode = Literal[
    "duplicate_rows",
    "missing_values",
    "inconsistent_salary_period",
]

RemedialAction = Literal[
    "deduplicate",
    "annualize_monthly_salaries",
    "flagged_no_action",
]


class DataQualityFinding(BaseModel):
    """One detected defect and what was done about it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: FindingCode
    description: str
    affected_rows: int
    action: RemedialAction


class DataQualityReport(BaseModel):
    """The full record of what was found and corrected in one dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows_in: int
    rows_out: int
    findings: list[DataQualityFinding]

    @property
    def codes(self) -> list[str]:
        """Detected defect codes, for comparison against expected findings."""
        return [finding.code for finding in self.findings]

    @property
    def actions(self) -> list[str]:
        """Remedial actions actually taken, excluding flag-only findings."""
        return [
            finding.action for finding in self.findings if finding.action != "flagged_no_action"
        ]


def _remove_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityFinding | None]:
    """Drop exact duplicate rows, keeping the first occurrence of each."""
    duplicated = frame.duplicated()
    count = int(duplicated.sum())
    if count == 0:
        return frame, None

    cleaned = frame.loc[~duplicated].reset_index(drop=True)
    finding = DataQualityFinding(
        code="duplicate_rows",
        description=f"{count} exact duplicate rows found and removed",
        affected_rows=count,
        action="deduplicate",
    )
    return cleaned, finding


def _annualise_monthly_salaries(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, DataQualityFinding | None]:
    """Convert rows stated in monthly pay onto an annual basis.

    Left uncorrected, these rows appear as an enormous pay gap: a monthly
    figure sits roughly twelve times below its annual neighbours.
    """
    monthly = frame["salary_period"] == "monthly"
    count = int(monthly.sum())
    if count == 0:
        return frame, None

    cleaned = frame.copy()
    for column in MONETARY_COLUMNS:
        cleaned.loc[monthly, column] = (cleaned.loc[monthly, column] * MONTHS_PER_YEAR).round(2)
    cleaned.loc[monthly, "salary_period"] = "annual"

    finding = DataQualityFinding(
        code="inconsistent_salary_period",
        description=f"{count} rows stated monthly pay and were annualised",
        affected_rows=count,
        action="annualize_monthly_salaries",
    )
    return cleaned, finding


def _flag_missing_ratings(frame: pd.DataFrame) -> DataQualityFinding | None:
    """Record missing performance ratings without imputing them."""
    count = int(frame["performance_rating"].isna().sum())
    if count == 0:
        return None

    return DataQualityFinding(
        code="missing_values",
        description=(
            f"{count} rows have no performance rating; recorded and excluded "
            f"from any statistic requiring it, not imputed"
        ),
        affected_rows=count,
        action="flagged_no_action",
    )


def clean_workforce(frame: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Normalise a raw extract and report every defect found.

    Returns the cleaned table and the report describing what was done.
    """
    rows_in = len(frame)
    findings: list[DataQualityFinding] = []

    cleaned, duplicate_finding = _remove_duplicates(frame)
    if duplicate_finding is not None:
        findings.append(duplicate_finding)

    cleaned, period_finding = _annualise_monthly_salaries(cleaned)
    if period_finding is not None:
        findings.append(period_finding)

    missing_finding = _flag_missing_ratings(cleaned)
    if missing_finding is not None:
        findings.append(missing_finding)

    report = DataQualityReport(rows_in=rows_in, rows_out=len(cleaned), findings=findings)
    return cleaned, report

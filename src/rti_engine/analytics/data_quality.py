"""Inject realistic data-quality defects into the finished workforce table.

Real HR extracts are imperfect: ratings go missing, rows get duplicated by
a bad join, and one country's payroll reports monthly rather than annual
figures. The system under test must detect these, normalise them, and log
what it did — never compute over them silently.

This runs last, after scenario injection, because these defects describe
the state of the extract rather than anything about pay itself. Nothing
may recompute derived pay columns afterwards: doing so would recalculate
the monthly rows back to annual values and erase the defect.
"""

import numpy as np
import pandas as pd
from numpy.random import Generator

from rti_engine.analytics.catalog import Catalog, Injection

MONTHS_PER_YEAR = 12

MONETARY_COLUMNS: list[str] = [
    "base_salary_fte_eur",
    "base_salary_actual_eur",
    "bonus_actual_eur",
    "total_comp_actual_eur",
]
"""Columns expressed in currency, all of which must move together."""


def _find_defect_spec(catalog: Catalog) -> tuple[Injection | None, list[str]]:
    """Locate the data-quality scenario and the countries it applies to."""
    for scenario in catalog.scenarios:
        injection = scenario.injection
        if injection is None or injection.mechanism != "data_quality_defects":
            continue

        countries: list[str] = []
        if scenario.population is not None:
            declared = scenario.population.country
            if isinstance(declared, str):
                countries = [declared]
            elif declared is not None:
                countries = list(declared)

        return injection, countries

    return None, []


def _select_rows(frame: pd.DataFrame, mask: pd.Series, count: int, rng: Generator) -> pd.Index:
    """Choose up to `count` row labels at random from the masked rows."""
    candidates = frame.index[mask]
    if len(candidates) == 0 or count <= 0:
        return pd.Index([])

    take = min(count, len(candidates))
    chosen = rng.permutation(len(candidates))[:take]
    return candidates[chosen]


def apply_data_quality_defects(
    workforce: pd.DataFrame, catalog: Catalog, rng: Generator
) -> pd.DataFrame:
    """Apply every declared data-quality defect to a copy of the table.

    Returns the table unchanged if the catalog declares no such scenario.
    """
    injection, countries = _find_defect_spec(catalog)
    if injection is None:
        return workforce

    frame = workforce.copy()
    in_scope_country = (
        frame["country"].isin(countries) if countries else pd.Series(True, index=frame.index)
    )
    # Defects are confined to background employees. Landing them on a
    # scenario's population would alter that scenario's measured gap and
    # invalidate its ground truth.
    scope = in_scope_country & frame["scenario_id"].isna()

    if injection.missing_performance_rating_share is not None:
        eligible = int(scope.sum())
        blank_count = int(round(injection.missing_performance_rating_share * eligible))
        blanked = _select_rows(frame, scope, blank_count, rng)
        frame["performance_rating"] = frame["performance_rating"].astype("float64")
        frame.loc[blanked, "performance_rating"] = np.nan

    if injection.monthly_salary_rows is not None:
        spec = injection.monthly_salary_rows
        # in_country = frame["country"] == spec.country
        # monthly = _select_rows(frame, in_country, spec.count, rng)
        in_country = (frame["country"] == spec.country) & frame["scenario_id"].isna()
        monthly = _select_rows(frame, in_country, spec.count, rng)
        for column in MONETARY_COLUMNS:
            frame.loc[monthly, column] = (frame.loc[monthly, column] / MONTHS_PER_YEAR).round(2)
        frame.loc[monthly, "salary_period"] = "monthly"

    if injection.duplicate_row_count is not None:
        duplicated = _select_rows(frame, scope, injection.duplicate_row_count, rng)
        frame = pd.concat([frame, frame.loc[duplicated]], ignore_index=True)

    return frame

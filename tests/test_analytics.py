"""Golden tests for the deterministic analytics layer.

Asserts that the statistics reach the verdict the catalog declares for
each scenario: gaps that must be flagged are found and significant, gaps
that must not be flagged are explained away or shown to be inconclusive.

These are the tests that would catch a regression in the maths before it
ever reached an agent or a generated document.
"""

import pandas as pd
import pytest

from rti_engine.analytics.catalog import Catalog, load_catalog
from rti_engine.analytics.cleaning import clean_workforce
from rti_engine.analytics.gap_metrics import compute_pay_gap
from rti_engine.analytics.generate import generate_workforce
from rti_engine.analytics.regression import estimate_adjusted_gap, estimate_interaction_gap

BASE_FTE = "base_salary_fte_eur"
BASE_ACTUAL = "base_salary_actual_eur"
TOTAL_COMP = "total_comp_actual_eur"


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture(scope="module")
def workforce(catalog: Catalog) -> pd.DataFrame:
    cleaned, _ = clean_workforce(generate_workforce(catalog))
    return cleaned


def scenario_rows(frame: pd.DataFrame, scenario_id: str, arm: str | None = None) -> pd.DataFrame:
    mask = frame["scenario_id"] == scenario_id
    if arm is not None:
        mask &= frame["sub_population"] == arm
    return frame[mask]


def test_cleaning_meets_s8_ground_truth(catalog: Catalog) -> None:
    """Every declared defect is detected, and every required fix is applied."""
    _, report = clean_workforce(generate_workforce(catalog))
    ground_truth = catalog.scenario("S8").ground_truth
    assert ground_truth is not None

    assert sorted(report.codes) == sorted(ground_truth.must_detect or [])
    assert sorted(report.actions) == sorted(ground_truth.must_normalize or [])
    assert report.rows_out < report.rows_in


# scenario id, column the verdict is measured on
SIGNIFICANCE_CHECKS = [
    ("S1", BASE_FTE),
    ("S2", BASE_FTE),
    ("S3", BASE_FTE),
    ("S4", TOTAL_COMP),
    ("S7", BASE_FTE),
]


@pytest.mark.parametrize(("scenario_id", "column"), SIGNIFICANCE_CHECKS)
def test_significance_matches_ground_truth(
    workforce: pd.DataFrame, catalog: Catalog, scenario_id: str, column: str
) -> None:
    scenario = catalog.scenario(scenario_id)
    assert scenario.ground_truth is not None
    expected = scenario.ground_truth.statistically_significant
    assert expected is not None, f"{scenario_id}: catalog declares no significance expectation"

    result = estimate_adjusted_gap(
        scenario_rows(workforce, scenario_id), column, catalog.thresholds
    )
    assert result.significant is expected, (
        f"{scenario_id}: adjusted gap {result.adjusted_gap_pct:.2f}% "
        f"p={result.p_value:.4f}, expected significant={expected}"
    )


def test_s2_gap_is_explained_by_controls(workforce: pd.DataFrame, catalog: Catalog) -> None:
    """A large raw gap must disappear once tenure is accounted for."""
    scenario = catalog.scenario("S2")
    assert scenario.ground_truth is not None

    rows = scenario_rows(workforce, "S2")
    raw = compute_pay_gap(rows, BASE_FTE, catalog.thresholds)
    adjusted = estimate_adjusted_gap(rows, BASE_FTE, catalog.thresholds)

    raw_band = scenario.ground_truth.expected_raw_gap_pct
    adjusted_band = scenario.ground_truth.expected_adjusted_gap_pct
    assert raw_band is not None and adjusted_band is not None

    assert raw_band[0] <= raw.mean_gap_pct <= raw_band[1]
    assert adjusted_band[0] <= adjusted.adjusted_gap_pct <= adjusted_band[1]
    assert "tenure_years" in adjusted.controls


def test_s3_is_not_reportable(workforce: pd.DataFrame, catalog: Catalog) -> None:
    """A wide gap in a tiny group must be blocked before it becomes a finding."""
    result = compute_pay_gap(scenario_rows(workforce, "S3"), BASE_FTE, catalog.thresholds)
    assert result.reportable is False
    assert result.summary.n_total < catalog.thresholds.min_reportable_group_size * 2


def test_s6_threshold_routing(workforce: pd.DataFrame, catalog: Catalog) -> None:
    """Two real gaps either side of the assessment trigger must route differently."""
    above = compute_pay_gap(
        scenario_rows(workforce, "S6", "above_threshold"), BASE_FTE, catalog.thresholds
    )
    below = compute_pay_gap(
        scenario_rows(workforce, "S6", "below_threshold"), BASE_FTE, catalog.thresholds
    )

    assert above.exceeds_jpa_threshold is True
    assert below.exceeds_jpa_threshold is False
    assert above.mean_gap_pct > below.mean_gap_pct


def test_s5_needs_an_interaction_model(workforce: pd.DataFrame, catalog: Catalog) -> None:
    """A gender-only model misses the effect; the interaction model finds it."""
    rows = scenario_rows(workforce, "S5")

    gender_only = estimate_adjusted_gap(rows, BASE_FTE, catalog.thresholds)
    assert gender_only.significant is False

    interaction = estimate_interaction_gap(rows, BASE_FTE, catalog.thresholds)
    assert interaction.interaction_significant is True
    assert interaction.older_significant is True
    assert interaction.older_gap_pct > gender_only.adjusted_gap_pct


def test_s7_requires_fte_normalisation(workforce: pd.DataFrame, catalog: Catalog) -> None:
    """The gap must exist in actual pay and vanish once normalised to full time."""
    rows = scenario_rows(workforce, "S7")

    unnormalised = estimate_adjusted_gap(rows, BASE_ACTUAL, catalog.thresholds)
    normalised = estimate_adjusted_gap(rows, BASE_FTE, catalog.thresholds)

    assert unnormalised.significant is True
    assert normalised.significant is False
    assert unnormalised.adjusted_gap_pct > normalised.adjusted_gap_pct


def test_gap_direction_is_male_minus_female(catalog: Catalog) -> None:
    """A positive gap must mean women are paid less."""
    frame = pd.DataFrame(
        {
            "gender": ["F", "F", "M", "M"],
            BASE_FTE: [90.0, 90.0, 100.0, 100.0],
        }
    )
    result = compute_pay_gap(frame, BASE_FTE, catalog.thresholds)
    assert result.mean_gap_pct == pytest.approx(10.0)

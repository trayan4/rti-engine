"""Golden tests for the synthetic dataset.

Every planted scenario must land inside the ground-truth band declared for
it in the catalog, and the declared data-quality defects must be present.
Bands are read from the catalog rather than hardcoded here, so the YAML
remains the single source of truth.
"""

import numpy as np
import pandas as pd
import pytest

from rti_engine.analytics.catalog import Catalog, load_catalog
from rti_engine.analytics.generate import generate_workforce

AGE_CUTOFF = 45


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture(scope="module")
def workforce(catalog: Catalog) -> pd.DataFrame:
    return generate_workforce(catalog)


def gender_gap_pct(frame: pd.DataFrame, column: str) -> float:
    """Percentage by which the male mean exceeds the female mean."""
    female = float(frame.loc[frame["gender"] == "F", column].mean())
    male = float(frame.loc[frame["gender"] == "M", column].mean())
    return 100.0 * (male - female) / male


# scenario id, arm label, column to measure, ground-truth field holding the band
GAP_CHECKS = [
    ("S1", None, "base_salary_fte_eur", "expected_gap_pct"),
    ("S2", None, "base_salary_fte_eur", "expected_raw_gap_pct"),
    ("S3", None, "base_salary_fte_eur", "expected_gap_pct"),
    ("S4", None, "base_salary_fte_eur", "expected_base_gap_pct"),
    ("S4", None, "total_comp_actual_eur", "expected_total_comp_gap_pct"),
    ("S5", None, "base_salary_fte_eur", "expected_naive_gap_pct"),
    ("S6", "above_threshold", "base_salary_fte_eur", "expected_gap_pct"),
    ("S6", "below_threshold", "base_salary_fte_eur", "expected_gap_pct"),
    ("S7", None, "base_salary_actual_eur", "expected_unnormalized_gap_pct"),
    ("S7", None, "base_salary_fte_eur", "expected_fte_normalized_gap_pct"),
]


@pytest.mark.parametrize(("scenario_id", "arm", "column", "field"), GAP_CHECKS)
def test_scenario_gap_within_band(
    workforce: pd.DataFrame,
    catalog: Catalog,
    scenario_id: str,
    arm: str | None,
    column: str,
    field: str,
) -> None:
    scenario = catalog.scenario(scenario_id)

    if arm is not None:
        assert scenario.sub_populations is not None
        matching = [sub for sub in scenario.sub_populations if sub.label == arm]
        assert len(matching) == 1
        ground_truth = matching[0].ground_truth
        rows = workforce[
            (workforce["scenario_id"] == scenario_id) & (workforce["sub_population"] == arm)
        ]
    else:
        assert scenario.ground_truth is not None
        ground_truth = scenario.ground_truth
        rows = workforce[workforce["scenario_id"] == scenario_id]

    band = getattr(ground_truth, field)
    assert band is not None, f"{scenario_id}: catalog declares no {field}"

    low, high = band
    measured = gender_gap_pct(rows, column)
    assert low <= measured <= high, (
        f"{scenario_id} {arm or ''} {column}: {measured:.2f}% outside [{low}, {high}]"
    )


def test_s5_interaction_effect_within_band(workforce: pd.DataFrame, catalog: Catalog) -> None:
    """The penalty is confined to women above the age cutoff."""
    scenario = catalog.scenario("S5")
    assert scenario.ground_truth is not None
    band = scenario.ground_truth.expected_interaction_gap_pct
    assert band is not None

    rows = workforce[workforce["scenario_id"] == "S5"]
    older = rows[rows["age"] >= AGE_CUTOFF]

    low, high = band
    measured = gender_gap_pct(older, "base_salary_fte_eur")
    assert low <= measured <= high


def test_row_count_matches_target_plus_duplicates(
    workforce: pd.DataFrame, catalog: Catalog
) -> None:
    scenario = catalog.scenario("S8")
    assert scenario.injection is not None
    duplicates = scenario.injection.duplicate_row_count
    assert duplicates is not None

    assert len(workforce) == catalog.generation.total_employees + duplicates
    assert int(workforce.duplicated().sum()) == duplicates


def test_monthly_salary_rows_present_and_isolated(
    workforce: pd.DataFrame, catalog: Catalog
) -> None:
    """Monthly rows exist, sit in one country, and miss every scenario population."""
    scenario = catalog.scenario("S8")
    assert scenario.injection is not None
    spec = scenario.injection.monthly_salary_rows
    assert spec is not None

    monthly = workforce[workforce["salary_period"] == "monthly"]
    assert len(monthly) == spec.count
    assert monthly["country"].unique().tolist() == [spec.country]
    assert monthly["scenario_id"].isna().all()


def test_missing_ratings_present_and_isolated(workforce: pd.DataFrame) -> None:
    missing = workforce[workforce["performance_rating"].isna()]
    assert len(missing) > 0
    assert set(missing["country"].unique()) <= {"FR", "ES"}
    assert missing["scenario_id"].isna().all()


def test_generation_is_reproducible(catalog: Catalog) -> None:
    """The same seed must produce a byte-identical dataset."""
    first = generate_workforce(catalog)
    second = generate_workforce(catalog)
    pd.testing.assert_frame_equal(first, second)


def test_no_negative_pay(workforce: pd.DataFrame) -> None:
    assert (workforce["base_salary_fte_eur"] > 0).all()
    assert (workforce["bonus_actual_eur"] >= 0).all()
    assert not np.isinf(workforce["total_comp_actual_eur"]).any()

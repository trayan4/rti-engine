"""Tests for the inference and remediation layers.

Covers the two properties that make the remediation output safe to put in
front of an employer: only genuinely unexplained gaps are remediated, and
the plan respects every constraint it claims to.
"""

import pandas as pd
import pytest

from rti_engine.analytics.catalog import Catalog, load_catalog
from rti_engine.analytics.cleaning import clean_workforce
from rti_engine.analytics.generate import generate_workforce
from rti_engine.analytics.inference import benjamini_hochberg, classify_gap
from rti_engine.analytics.remediation import optimise_remediation

BASE_FTE = "base_salary_fte_eur"


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


@pytest.fixture(scope="module")
def workforce(catalog: Catalog) -> pd.DataFrame:
    cleaned, _ = clean_workforce(generate_workforce(catalog))
    return cleaned


@pytest.fixture(scope="module")
def plan(workforce: pd.DataFrame, catalog: Catalog):  # type: ignore[no-untyped-def]
    return optimise_remediation(workforce, catalog.thresholds)


def test_benjamini_hochberg_is_never_smaller_than_the_p_value() -> None:
    p_values = [0.001, 0.01, 0.03, 0.2, 0.6, 0.99]
    q_values = benjamini_hochberg(p_values)
    assert all(q >= p for p, q in zip(p_values, q_values, strict=True))
    assert all(0.0 <= q <= 1.0 for q in q_values)


def test_benjamini_hochberg_is_monotonic_in_p() -> None:
    """A less significant test may never receive a smaller q-value."""
    p_values = [0.001, 0.01, 0.03, 0.2, 0.6, 0.99]
    q_values = benjamini_hochberg(p_values)
    assert q_values == sorted(q_values)


def test_benjamini_hochberg_preserves_input_order() -> None:
    shuffled = [0.6, 0.001, 0.2, 0.01]
    q_values = benjamini_hochberg(shuffled)
    assert q_values[1] < q_values[3] < q_values[2] < q_values[0]


def test_benjamini_hochberg_handles_empty_input() -> None:
    assert benjamini_hochberg([]) == []


def test_classify_gap_distinguishes_explained_from_inconclusive() -> None:
    """A vanished gap and a merely unproven one must not share a verdict."""
    explained = classify_gap(raw_gap_pct=8.6, adjusted_gap_pct=-0.02, q_value=0.99, alpha=0.05)
    inconclusive = classify_gap(raw_gap_pct=8.9, adjusted_gap_pct=8.3, q_value=0.63, alpha=0.05)
    unexplained = classify_gap(raw_gap_pct=7.0, adjusted_gap_pct=7.8, q_value=0.026, alpha=0.05)

    assert explained.verdict == "explained"
    assert inconclusive.verdict == "inconclusive"
    assert unexplained.verdict == "unexplained"

    assert explained.actionable is False
    assert inconclusive.actionable is False
    assert unexplained.actionable is True

    assert "attributable" in explained.note
    assert "monitored" in inconclusive.note


def test_s2_is_classified_as_explained(plan) -> None:  # type: ignore[no-untyped-def]
    """The tenure-explained gap must be described as explained, not merely unproven."""
    matching = [v for v in plan.verdicts if v.group == "FR/Engineering/L4"]
    assert len(matching) == 1
    assert matching[0].verdict == "explained"
    assert matching[0].remediated is False
    assert abs(matching[0].adjusted_gap_pct) < 2.0


def test_only_unexplained_gaps_are_remediated(plan) -> None:  # type: ignore[no-untyped-def]
    for verdict in plan.verdicts:
        if verdict.remediated:
            assert verdict.verdict == "unexplained"
            assert verdict.exceeds_target is True


def test_plan_is_optimal_and_closes_every_targeted_gap(plan) -> None:  # type: ignore[no-untyped-def]
    assert plan.status == "Optimal"
    assert plan.feasible is True
    assert plan.groups_remediated > 0
    for group in plan.groups:
        assert group.gap_after_pct <= plan.target_gap_pct + 1e-6


def test_no_award_exceeds_the_individual_cap(plan) -> None:  # type: ignore[no-untyped-def]
    for award in plan.awards:
        assert award.raise_eur > 0.0
        assert award.raise_pct <= plan.max_individual_raise_pct + 1e-6
        assert award.new_salary_eur > award.current_salary_eur


def test_cost_equals_the_sum_of_awards(plan) -> None:  # type: ignore[no-untyped-def]
    assert plan.total_cost_eur == pytest.approx(
        sum(award.raise_eur for award in plan.awards), abs=1.0
    )


def test_awards_reach_the_underpaid_side_without_a_gender_term(
    plan, workforce: pd.DataFrame
) -> None:  # type: ignore[no-untyped-def]
    """No gender appears in the objective, yet the optimum funds women."""
    gender = workforce.set_index("employee_id")["gender"]
    recipients = [gender.loc[award.employee_id] for award in plan.awards]
    assert len(recipients) > 0
    assert all(value == "F" for value in recipients)


def test_disabling_the_filter_remediates_more_groups(
    workforce: pd.DataFrame, catalog: Catalog, plan
) -> None:  # type: ignore[no-untyped-def]
    """The safeguard must be doing real work, not decorating the output."""
    unfiltered = optimise_remediation(workforce, catalog.thresholds, unexplained_only=False)
    assert unfiltered.groups_remediated > plan.groups_remediated
    assert unfiltered.total_cost_eur > plan.total_cost_eur


def test_every_assessed_group_receives_a_verdict(plan) -> None:  # type: ignore[no-untyped-def]
    assert len(plan.verdicts) == plan.groups_assessed
    assert plan.multiple_comparison_correction == "benjamini-hochberg"

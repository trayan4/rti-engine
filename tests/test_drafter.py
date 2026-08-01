"""Tests for the drafting layer.

Whether the letter reads well is judged by reading it, and measured
against the scenario catalog in the eval harness. What is asserted here is
the code around the model: that the verdict is decided deterministically
rather than inferred from figures, that figures reach the model already
rounded, and that a declared source naming a field which does not exist is
refused.

The verdict tests matter most. A gap of -0.02% was once described to an
employee as indistinguishable from chance rather than as explained by
length of service, and that is the class of error these prevent.
"""

import pytest

from rti_engine.agents.analyst import GroupAnalysis, RequesterRecord
from rti_engine.agents.drafter import (
    DRAFTER_PROMPT,
    DraftLetter,
    FigureUse,
    LetterSection,
    base_salary_verdict,
    build_fact_sheet,
    check_declared_sources,
    fact_sheet_fields,
)


def requester() -> RequesterRecord:
    return RequesterRecord(
        employee_id="EMP-00001",
        country="DE",
        job_family="Sales",
        level="L3",
        working_pattern="full_time",
        fte=1.0,
        tenure_years=4.123456,
        base_salary_fte_eur=70123.456789,
        base_salary_actual_eur=70123.456789,
        bonus_actual_eur=8000.987654,
        total_comp_actual_eur=78124.444443,
        currency="EUR",
    )


def analysis(
    raw_gap: float = 7.0,
    adjusted_gap: float = 7.7569,
    p_value: float = 0.000709,
    significant: bool = True,
) -> GroupAnalysis:
    """A group analysis with the fields a test does not care about fixed."""
    return GroupAnalysis(
        requester=requester(),
        group="DE/Sales/L3",
        n_total=250,
        n_female=109,
        n_male=141,
        mean_female_eur=71885.142857,
        mean_male_eur=77295.857142,
        median_female_eur=70189.74,
        median_male_eur=76703.53,
        reportable=True,
        reportability_note="both gender groups meet the minimum size of 10",
        base_raw_gap_pct=raw_gap,
        base_median_gap_pct=8.4922,
        base_adjusted_gap_pct=adjusted_gap,
        base_p_value=p_value,
        base_significant=significant,
        base_confidence_interval_pct=[3.3784, 11.937],
        controls=["tenure_years"],
        exceeds_jpa_threshold=True,
        jpa_threshold_pct=5.0,
        alpha=0.05,
        total_comp_raw_gap_pct=6.9712,
        total_comp_adjusted_gap_pct=7.7512,
        total_comp_significant=True,
        age_cutoff=45,
        younger_gap_pct=5.4321,
        older_gap_pct=12.1987,
        older_significant=False,
        interaction_p_value=0.1817,
        interaction_significant=False,
        tools_called=["get_own_pay_record"],
    )


# --- the verdict, decided in code ---


def test_a_significant_surviving_gap_is_unexplained() -> None:
    verdict, _ = base_salary_verdict(analysis())
    assert verdict == "unexplained"


def test_a_gap_that_vanishes_under_controls_is_explained() -> None:
    """The bug this prevents: describing an explained result as uncertain."""
    verdict, note = base_salary_verdict(
        analysis(raw_gap=8.6368, adjusted_gap=-0.016, p_value=0.994966, significant=False)
    )

    assert verdict == "explained"
    assert "attributable" in note


def test_a_surviving_but_unproven_gap_is_inconclusive() -> None:
    verdict, note = base_salary_verdict(
        analysis(raw_gap=8.96, adjusted_gap=8.30, p_value=0.0812, significant=False)
    )

    assert verdict == "inconclusive"
    assert "monitored" in note


def test_the_verdict_reaches_the_fact_sheet() -> None:
    """The model is given the finding, not the ingredients to derive one."""
    facts = build_fact_sheet(analysis())
    assert facts["base_salary_analysis"]["verdict"] == "unexplained"
    assert facts["base_salary_analysis"]["verdict_explanation"]


# --- precision ---


def test_currency_reaches_the_letter_rounded_to_cents() -> None:
    """A letter quoting EUR 130893.686758033 is not a letter."""
    facts = build_fact_sheet(analysis())

    assert facts["requester"]["base_salary_fte_eur"] == 70123.46
    assert facts["category"]["mean_female_eur"] == 71885.14


def test_percentages_reach_the_letter_at_one_decimal() -> None:
    facts = build_fact_sheet(analysis())

    assert facts["base_salary_analysis"]["raw_gap_pct"] == 7.0
    assert facts["base_salary_analysis"]["adjusted_gap_pct"] == 7.8
    assert facts["base_salary_analysis"]["confidence_interval_pct"] == [3.4, 11.9]


def test_p_values_are_shortened() -> None:
    facts = build_fact_sheet(analysis())
    assert facts["base_salary_analysis"]["p_value"] == 0.001


def test_full_precision_survives_in_the_analysis() -> None:
    """Rounding is for the letter; the audit bundle keeps the real values."""
    assert analysis().base_adjusted_gap_pct == 7.7569


# --- declared sources ---


def test_both_dotted_and_bare_field_names_are_accepted() -> None:
    fields = fact_sheet_fields(build_fact_sheet(analysis()))

    assert "base_salary_analysis.raw_gap_pct" in fields
    assert "raw_gap_pct" in fields


def letter(source_field: str) -> DraftLetter:
    return DraftLetter(
        subject="s",
        salutation="Dear colleague,",
        sections=[LetterSection(heading="h", body="b")],
        closing="Yours sincerely,",
        figures_used=[FigureUse(value="7.0%", source_field=source_field, meaning="gap")],
    )


def test_a_known_source_field_passes() -> None:
    facts = build_fact_sheet(analysis())
    assert check_declared_sources(letter("base_salary_analysis.raw_gap_pct"), facts) == []


def test_an_invented_source_field_is_reported() -> None:
    """A figure attributed to a field that does not exist was not quoted."""
    facts = build_fact_sheet(analysis())
    assert check_declared_sources(letter("industry_benchmark_pct"), facts) == [
        "industry_benchmark_pct"
    ]


# --- the letter ---


def test_a_letter_needs_at_least_one_section() -> None:
    with pytest.raises(ValueError):
        DraftLetter(subject="s", salutation="d", sections=[], closing="c")


def test_rendering_produces_the_letter_in_order() -> None:
    rendered = letter("raw_gap_pct").render()
    assert rendered.index("Dear colleague,") < rendered.index("Yours sincerely,")


# --- the prompt ---


def test_the_prompt_renders_within_its_budget() -> None:
    values = {
        "request_text": "What is the average pay by sex at my level?",
        "facts": "{}",
        "legal_position": "{}",
        "pay_setting_criteria": "[policy]\nPay is set by level and country.",
        "revision_feedback": "This is the first draft.",
    }

    assert DRAFTER_PROMPT.fits(**values)
    assert DRAFTER_PROMPT.identifier == "response_letter@v1"


def test_the_prompt_forbids_deriving_the_verdict() -> None:
    assert "Do not derive your own" in DRAFTER_PROMPT.template


def test_the_prompt_forbids_inventing_figures() -> None:
    template = DRAFTER_PROMPT.template
    assert "You do not calculate" in template
    assert "Do not round" in template

"""Tests for figure grounding.

The check that matters: a number in the letter that appears in none of
the sources was invented, rounded or recomputed. All three have happened
in practice — a fabricated industry benchmark, an adjusted gap written to
one decimal instead of four, a total the model worked out for itself.

Deterministic and model-free, so unlike the compliance reviewer this
cannot be argued with.
"""

from rti_engine.guardrails.numbers import (
    numbers_in,
    permitted_values,
    validate_numbers,
)

FACTS = {
    "category": {"mean_female_eur": 71885.14, "mean_male_eur": 77295.85, "n_total": 250},
    "base_salary_analysis": {
        "raw_gap_pct": 7.0,
        "adjusted_gap_pct": 7.8,
        "p_value": 0.001,
    },
    "thresholds": {"joint_pay_assessment_trigger_pct": 5.0},
}

LEGAL = {
    "jurisdiction": "DE",
    "national_position": "The right applies in establishments with more than 200 employees.",
    "caveats": ["Spain applies a 25 per cent justification threshold."],
}


# --- reading numbers from prose ---


def test_thousands_separators_are_read() -> None:
    """A letter writes 71,885.14; the fact sheet holds 71885.14."""
    values = [value for value, _, _ in numbers_in("The average is 71,885.14 EUR.")]
    assert float(values[0]) == 71885.14


def test_percentages_and_negatives_are_read() -> None:
    values = [value for value, _, _ in numbers_in("It moved from 7.8% to -0.4%.")]
    assert [float(value) for value in values] == [7.8, -0.4]


def test_text_without_numbers_yields_none() -> None:
    assert numbers_in("No figures appear in this sentence.") == []


# --- what counts as a source ---


def test_a_numeric_field_is_a_source() -> None:
    assert any(float(value) == 7.8 for value in permitted_values(FACTS))


def test_a_number_inside_prose_is_a_source() -> None:
    """A figure quoted from a caveat is as sourced as one from a field."""
    permitted = permitted_values(LEGAL)

    assert any(float(value) == 200 for value in permitted)
    assert any(float(value) == 25 for value in permitted)


def test_absent_sources_are_ignored() -> None:
    assert permitted_values(None, FACTS)


# --- grounded letters ---


def test_a_letter_quoting_its_sources_passes() -> None:
    letter = (
        "The average base pay for women is 71,885.14 EUR and for men "
        "77,295.85 EUR. A difference of 7.8% remains after the controls."
    )
    result = validate_numbers(letter, ["71,885.14", "77,295.85", "7.8%"], FACTS, LEGAL)

    assert result.grounded
    assert result.ungrounded == []
    assert result.numbers_checked > 0


def test_section_and_article_numbers_pass_unsourced() -> None:
    """Otherwise every compliant letter fails on its own citations."""
    letter = "See policy section 8 and Article 7 of the Directive."
    assert validate_numbers(letter, [], FACTS).grounded


def test_a_letter_with_no_figures_passes() -> None:
    """The degraded response contains none, and must not be rejected."""
    assert validate_numbers("A person will review your request.", [], FACTS).grounded


# --- ungrounded letters ---


def test_an_invented_benchmark_is_caught() -> None:
    """This exact failure occurred: a sector average from nowhere."""
    letter = "Industry benchmarks put the sector average at 14.5%."
    result = validate_numbers(letter, [], FACTS, LEGAL)

    assert not result.grounded
    assert result.ungrounded[0].value == "14.5"


def test_a_rounded_figure_is_caught() -> None:
    """7.8 is in the fact sheet; 7.76 is the model doing arithmetic."""
    result = validate_numbers("A difference of 7.76% remains.", [], FACTS)

    assert not result.grounded
    assert result.ungrounded[0].value == "7.76"


def test_a_recomputed_total_is_caught() -> None:
    """The difference between the two means was never given as a figure."""
    result = validate_numbers("Men earn 5,410.71 EUR more on average.", [], FACTS)
    assert not result.grounded


def test_the_finding_quotes_where_the_figure_appeared() -> None:
    """A reader has to be able to find it in the letter."""
    letter = "Your pay is within range. The sector average is 14.5% by comparison."
    result = validate_numbers(letter, [], FACTS)

    assert "sector average" in result.ungrounded[0].context


def test_declaring_a_figure_does_not_ground_it() -> None:
    """The declaration is the model's own claim, not a source."""
    result = validate_numbers("The average is 99,999.99 EUR.", ["99,999.99"], FACTS)
    assert not result.grounded


# --- declaration bookkeeping ---


def test_an_undeclared_but_sourced_figure_is_advisory() -> None:
    """A missed declaration is a lapse; the number is still traceable."""
    result = validate_numbers("The average is 71,885.14 EUR.", [], FACTS)

    assert result.grounded
    assert "71,885.14" in result.undeclared


# --- feedback ---


def test_the_feedback_names_every_failure() -> None:
    letter = "The sector average is 14.5% and the median is 88,888.88 EUR."
    feedback = validate_numbers(letter, [], FACTS).feedback()

    assert "14.5" in feedback
    assert "88,888.88" in feedback


def test_a_passing_check_produces_no_instructions() -> None:
    feedback = validate_numbers("Article 7 applies.", [], FACTS).feedback()
    assert "traceable" in feedback


# --- the audit summary ---


def test_the_summary_counts_without_repeating_the_figures() -> None:
    summary = validate_numbers("The sector average is 14.5%.", [], FACTS).summary()

    assert summary["grounded"] is False
    assert summary["ungrounded_count"] == 1
    assert "14.5" not in str(summary)

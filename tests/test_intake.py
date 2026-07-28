"""Tests for tier classification.

These assert the code guarantee, not the model's judgment. `_apply_floor`
is where the safety property lives: whatever the model returns, a request
touching comparator data or marked ambiguous becomes T2. That is testable
exhaustively, offline, and deterministically.

Whether the model classifies well is a different question, measured
against the scenario catalog in the eval harness. Putting live model
calls in the unit suite would make `make check` cost money, take a minute
and fail intermittently — none of which tells you your code is correct.
"""

import pytest

from rti_engine.agents.intake import (
    INTAKE_PROMPT,
    IntakeClassification,
    _apply_floor,
)
from rti_engine.db.models import AutonomyTier


def classification(
    category: str = "general_information",
    seeks_own_pay: bool = False,
    seeks_comparator_data: bool = False,
    ambiguous: bool = False,
) -> IntakeClassification:
    """Build a classification with the defaults tests do not care about."""
    return IntakeClassification(
        category=category,  # type: ignore[arg-type]
        seeks_own_pay=seeks_own_pay,
        seeks_comparator_data=seeks_comparator_data,
        ambiguous=ambiguous,
        rationale="test",
    )


def test_general_information_stays_at_t0() -> None:
    tier, reason = _apply_floor(classification())
    assert tier is AutonomyTier.T0
    assert reason is None


def test_own_pay_is_t1() -> None:
    tier, _ = _apply_floor(classification("own_pay", seeks_own_pay=True))
    assert tier is AutonomyTier.T1


def test_comparator_disclosure_is_t2() -> None:
    tier, _ = _apply_floor(classification("comparator_disclosure", seeks_comparator_data=True))
    assert tier is AutonomyTier.T2


@pytest.mark.parametrize(
    "category", ["general_information", "own_pay", "comparator_disclosure", "unclear"]
)
def test_comparator_data_forces_t2_from_any_category(category: str) -> None:
    """The floor that matters: a misclassification cannot release others' pay."""
    tier, reason = _apply_floor(classification(category, seeks_comparator_data=True))

    assert tier is AutonomyTier.T2
    assert reason is not None
    assert "other employees" in reason


@pytest.mark.parametrize(
    "category", ["general_information", "own_pay", "comparator_disclosure", "unclear"]
)
def test_ambiguity_forces_t2_from_any_category(category: str) -> None:
    """A human decides scope where the model could not."""
    tier, reason = _apply_floor(classification(category, ambiguous=True))

    assert tier is AutonomyTier.T2
    assert reason is not None
    assert "ambiguous" in reason


def test_unclear_is_t2_even_when_not_flagged_ambiguous() -> None:
    tier, _ = _apply_floor(classification("unclear"))
    assert tier is AutonomyTier.T2


def test_own_pay_flag_lifts_general_information_to_t1() -> None:
    """A request needing the requester's record is T1 whatever it was called."""
    tier, _ = _apply_floor(classification("general_information", seeks_own_pay=True))
    assert tier is AutonomyTier.T1


def test_the_floor_never_lowers_a_tier() -> None:
    """Exhaustive: no combination of flags produces a tier below the category."""
    minimum = {
        "general_information": AutonomyTier.T0,
        "own_pay": AutonomyTier.T1,
        "comparator_disclosure": AutonomyTier.T2,
        "unclear": AutonomyTier.T2,
    }
    order = {AutonomyTier.T0: 0, AutonomyTier.T1: 1, AutonomyTier.T2: 2}

    for category, floor in minimum.items():
        for own in (True, False):
            for comparator in (True, False):
                for ambiguous in (True, False):
                    tier, _ = _apply_floor(classification(category, own, comparator, ambiguous))
                    assert order[tier] >= order[floor], (
                        f"{category} own={own} comparator={comparator} "
                        f"ambiguous={ambiguous} produced {tier}"
                    )


def test_the_classification_schema_is_closed() -> None:
    """An unexpected field means the model returned something unmodelled."""
    with pytest.raises(ValueError):
        IntakeClassification(
            category="own_pay",
            seeks_own_pay=True,
            seeks_comparator_data=False,
            ambiguous=False,
            rationale="test",
            confidence=0.9,  # type: ignore[call-arg]
        )


def test_an_invalid_category_is_refused() -> None:
    with pytest.raises(ValueError):
        classification("something_else")


def test_the_prompt_renders_and_fits_its_budget() -> None:
    request = "Am I paid fairly compared to my colleagues?"
    rendered = INTAKE_PROMPT.render(request_text=request)

    assert request in rendered
    assert INTAKE_PROMPT.fits(request_text=request)
    assert INTAKE_PROMPT.identifier == "intake_classification@v1"


def test_the_prompt_states_the_asymmetry() -> None:
    """The reasoning behind escalation must survive prompt edits."""
    assert "not equivalent errors" in INTAKE_PROMPT.template
    assert "ambiguous to true" in INTAKE_PROMPT.template

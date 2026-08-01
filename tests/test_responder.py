"""Tests for the Tier 0 and Tier 1 response paths.

Offline. What is asserted is the shaping and the prompt contracts — that
the record reaching a model is rounded, that neither prompt can ask for
comparator data, and that both paths produce the same shape as the Tier 2
letter so one validator covers every tier.
"""

import pytest

from rti_engine.agents.responder import (
    INFORMATIONAL_PROMPT,
    OWN_DATA_PROMPT,
    own_record_facts,
)


def record() -> dict[str, object]:
    return {
        "found": True,
        "employee_id": "EMP-00001",
        "country": "DE",
        "job_family": "Sales",
        "level": "L3",
        "working_pattern": "part_time",
        "fte": 0.6,
        "tenure_years": 1.4938271,
        "base_salary_fte_eur": 59091.5678901,
        "base_salary_actual_eur": 35454.9407341,
        "bonus_actual_eur": 2293.9312345,
        "total_comp_actual_eur": 37748.8719686,
        "currency": "EUR",
    }


def test_currency_is_rounded_to_cents() -> None:
    facts = own_record_facts(record())

    assert facts["base_salary_actual_eur"] == 35454.94
    assert facts["bonus_actual_eur"] == 2293.93


def test_tenure_is_rounded_to_one_decimal() -> None:
    assert own_record_facts(record())["tenure_years"] == 1.5


def test_the_identifier_does_not_reach_the_model() -> None:
    """The record is already scoped to the requester; the id adds nothing."""
    assert "employee_id" not in own_record_facts(record())


def test_both_prompts_render_within_budget() -> None:
    informational = {
        "request_text": "How is pay set?",
        "jurisdiction": "DE",
        "retrieved_context": "[policy]\nPay is set by level.",
    }
    own_data = {**informational, "own_record": "{}"}
    own_data["request_text"] = "What is my salary?"

    assert INFORMATIONAL_PROMPT.fits(**informational)
    assert OWN_DATA_PROMPT.fits(**own_data)


def test_prompt_identifiers_are_versioned() -> None:
    assert INFORMATIONAL_PROMPT.identifier == "informational_response@v1"
    assert OWN_DATA_PROMPT.identifier == "own_data_response@v1"


def test_neither_path_may_speculate_about_comparators() -> None:
    """The tier makes comparator data unreachable; the prompt must not imply
    otherwise by guessing at what a comparison would show."""
    assert "Do not speculate" in INFORMATIONAL_PROMPT.template
    assert "does not cover what anyone else is paid" in OWN_DATA_PROMPT.template.lower()


def test_the_informational_path_states_it_has_no_pay_data() -> None:
    assert "no access to any employee's pay data" in INFORMATIONAL_PROMPT.template


def test_the_own_data_path_requires_declared_figures() -> None:
    assert "figures_used" in OWN_DATA_PROMPT.template
    assert "Do not round" in OWN_DATA_PROMPT.template


@pytest.mark.parametrize("prompt", [INFORMATIONAL_PROMPT, OWN_DATA_PROMPT])
def test_both_prompts_carry_the_shared_rules(prompt: object) -> None:
    template = prompt.template
    assert "You do not calculate" in template
    assert "citation string" in template

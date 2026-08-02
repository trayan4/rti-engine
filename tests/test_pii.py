"""Tests for personal data detection.

Two failure modes, and the second is the one that bit. Missing a name
puts personal data into a model's context and a stored transcript.
Over-detecting destroys the request: an earlier version classified
"Engineering" and "L3" as places, which are the job family and level
every request turns on, leaving nothing answerable behind.

No model is involved here, so these are deterministic.
"""

from rti_engine.guardrails.pii import (
    DETECTED_ENTITIES,
    scan,
)

# --- what must be caught ---


def test_a_colleagues_name_is_removed() -> None:
    """The case this exists for: "how much does Maria earn"."""
    result = scan("How much does Maria Fernandez in my team earn?")

    assert "PERSON" in result.entity_types
    assert "Maria Fernandez" not in result.redacted


def test_an_employee_id_is_removed() -> None:
    """As identifying as a name in a system keyed by one."""
    result = scan("My employee id is EMP-00042, please confirm my salary.")

    assert "EMPLOYEE_ID" in result.entity_types
    assert "EMP-00042" not in result.redacted


def test_an_email_address_is_removed() -> None:
    result = scan("Contact me at someone@example.com about this.")

    assert "EMAIL_ADDRESS" in result.entity_types
    assert "example.com" not in result.redacted


# --- what must survive ---


def test_an_ordinary_request_is_untouched() -> None:
    """A guardrail that mangles normal requests is an outage."""
    text = "What is the average pay for men and women at my level?"
    result = scan(text)

    assert result.clean
    assert result.redacted == text


def test_the_job_family_and_level_survive() -> None:
    """These were once classified as places, leaving nothing answerable."""
    text = "I work in Engineering at level L3 in the Berlin office."
    result = scan(text)

    assert "Engineering" in result.redacted
    assert "L3" in result.redacted


def test_locations_are_not_detected() -> None:
    """Excluded deliberately; see the job family and level above."""
    assert "LOCATION" not in DETECTED_ENTITIES


def test_empty_text_is_returned_unchanged() -> None:
    assert scan("").clean
    assert scan("   ").redacted == "   "


# --- how removal reads ---


def test_redaction_names_the_kind_that_was_removed() -> None:
    """An empty space reads as a typo; a label says what happened."""
    assert "[PERSON]" in scan("Ask Maria Fernandez about it.").redacted


def test_the_surrounding_text_is_preserved() -> None:
    result = scan("How much does Maria Fernandez in my team earn?")

    assert result.redacted.startswith("How much does")
    assert result.redacted.endswith("in my team earn?")


# --- the audit summary ---


def test_the_summary_records_kinds_not_values() -> None:
    """Recording the values would put the personal data in the audit trail."""
    result = scan("Maria Fernandez, EMP-00042, maria@example.com")
    summary = result.summary()

    assert summary["pii_found"] is True
    assert "Maria" not in str(summary)
    assert "EMP-00042" not in str(summary)


def test_a_clean_summary_says_nothing_was_found() -> None:
    summary = scan("What is my current salary?").summary()

    assert summary["pii_found"] is False
    assert summary["count"] == 0


def test_entity_types_are_deduplicated_and_sorted() -> None:
    result = scan("Maria Fernandez and Peter Schmidt both work here.")

    assert result.entity_types == sorted(set(result.entity_types))


# --- determinism ---


def test_the_same_text_scans_identically() -> None:
    """No model in the path, so this must not vary between runs."""
    text = "How much does Maria Fernandez earn? My id is EMP-00042."

    first, second = scan(text), scan(text)
    assert first.redacted == second.redacted
    assert first.entity_types == second.entity_types

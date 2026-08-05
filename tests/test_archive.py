"""Tests for audit bundle archival.

Offline. What is asserted is the shape of a bundle and that archival
never raises into a request that has already been told it settled —
losing the archive copy is an operational problem to notice, not one to
surface as a failure of something that actually succeeded.
"""

import pytest

from rti_engine.agents.drafter import DraftLetter, FigureUse, LetterSection
from rti_engine.agents.state import Actor, AuditEntry
from rti_engine.observability.archive import (
    BUNDLE_SCHEMA_VERSION,
    build_bundle,
)


def letter() -> DraftLetter:
    return DraftLetter(
        subject="Your pay information request",
        salutation="Dear colleague,",
        sections=[LetterSection(heading="Your pay", body="Your salary is ...")],
        closing="Yours sincerely,",
        figures_used=[FigureUse(value="7.8%", source_field="adjusted_gap_pct", meaning="gap")],
        citations=["Directive (EU) 2023/970, Article 7"],
    )


def audit() -> list[AuditEntry]:
    return [
        AuditEntry(actor=Actor.SYSTEM, action="request_received"),
        AuditEntry(actor=Actor.INTAKE, action="tier_assigned", detail={"tier": "T2"}),
    ]


def test_a_bundle_carries_the_letter_and_its_sources() -> None:
    bundle = build_bundle(
        request_id="req-1",
        requester_employee_id="EMP-00001",
        tier="T2",
        jurisdiction="DE",
        status="approved",
        request_text="What is my average pay compared to colleagues?",
        draft=letter(),
        number_check=None,
        review=None,
        approval_decision="approved",
        approved_by="hr-7",
        revision_count=1,
        tokens_used=50000,
        cost_usd=0.28,
        audit=audit(),
    )

    assert bundle.schema_version == BUNDLE_SCHEMA_VERSION
    assert bundle.letter is not None
    assert "Your salary is" in bundle.letter
    assert bundle.figures_used[0]["value"] == "7.8%"
    assert bundle.citations == ["Directive (EU) 2023/970, Article 7"]


def test_a_bundle_without_a_draft_is_still_complete() -> None:
    """A degraded or refused request has no letter, and must still archive."""
    bundle = build_bundle(
        request_id="req-2",
        requester_employee_id="EMP-00001",
        tier="T2",
        jurisdiction="ES",
        status="failed",
        request_text="What is my average pay?",
        draft=None,
        number_check=None,
        review=None,
        approval_decision=None,
        approved_by=None,
        revision_count=0,
        tokens_used=500,
        cost_usd=0.01,
        audit=audit(),
    )

    assert bundle.letter is None
    assert bundle.figures_used == []
    assert bundle.citations == []


def test_the_audit_trail_is_fully_rendered() -> None:
    """A reviewer reading this file needs the trail without a database."""
    bundle = build_bundle(
        request_id="req-3",
        requester_employee_id="EMP-00001",
        tier="T1",
        jurisdiction="FR",
        status="completed",
        request_text="What is my salary?",
        draft=None,
        number_check=None,
        review=None,
        approval_decision=None,
        approved_by=None,
        revision_count=0,
        tokens_used=3000,
        cost_usd=0.01,
        audit=audit(),
    )

    assert len(bundle.audit) == 2
    assert bundle.audit[1]["action"] == "tier_assigned"
    assert bundle.audit[1]["detail"] == {"tier": "T2"}


def test_a_bundle_is_serialisable_as_stored() -> None:
    """This is the literal file content; it must round-trip cleanly."""
    bundle = build_bundle(
        request_id="req-4",
        requester_employee_id="EMP-00001",
        tier="T2",
        jurisdiction="DE",
        status="approved",
        request_text="text",
        draft=letter(),
        number_check=None,
        review=None,
        approval_decision="approved",
        approved_by="hr-7",
        revision_count=0,
        tokens_used=0,
        cost_usd=0.0,
        audit=[],
    )

    restored = bundle.model_validate_json(bundle.model_dump_json())
    assert restored == bundle


async def test_archiving_without_configuration_does_nothing(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """No account configured is the normal local and CI state."""
    from rti_engine.config import settings as settings_module
    from rti_engine.observability.archive import archive_bundle

    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("ARCHIVE_ACCOUNT_NAME", "")

    bundle = build_bundle(
        request_id="req-5",
        requester_employee_id="EMP-00001",
        tier="T0",
        jurisdiction="DE",
        status="completed",
        request_text="How is pay set?",
        draft=None,
        number_check=None,
        review=None,
        approval_decision=None,
        approved_by=None,
        revision_count=0,
        tokens_used=0,
        cost_usd=0.0,
        audit=[],
    )

    result = archive_bundle(bundle)
    settings_module.get_settings.cache_clear()

    assert result is None

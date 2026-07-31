"""Tests for the compliance review layer.

Whether the reviewer finds the right defects is measured against the
scenario catalog in the eval harness, where a change can be compared
across runs. It cannot be established here: the reviewer is
non-deterministic, and the same draft has been both approved with no
findings and blocked with two.

What is asserted here is the code around it — that an approval cannot
coexist with a blocking finding, that severity partitions cleanly, and
that the audit report carries what a reviewer of a Tier 2 response needs.
"""

import pytest

from rti_engine.agents.drafter import DraftLetter, FigureUse, LetterSection
from rti_engine.agents.reviewer import (
    REVIEWER_PROMPT,
    Finding,
    ReviewResult,
    _declared_figures,
    enforce_approval_consistency,
    review_report,
)


def finding(kind: str = "ungrounded_figure", severity: str = "blocking") -> Finding:
    return Finding(
        kind=kind,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        quote="the remaining difference of 1.2%",
        problem="No figure of 1.2% appears in the fact sheet.",
        suggested_fix="State 7.8%, the adjusted gap.",
    )


def letter(figures: list[FigureUse] | None = None) -> DraftLetter:
    return DraftLetter(
        subject="Your pay information request",
        salutation="Dear colleague,",
        sections=[LetterSection(heading="Your pay", body="Your salary is ...")],
        closing="Yours sincerely,",
        figures_used=figures or [],
    )


# --- approval consistency ---


def test_an_approval_with_a_blocking_finding_is_withdrawn() -> None:
    """The findings are evidence; the flag is a conclusion drawn from them."""
    result = ReviewResult(approved=True, findings=[finding()], summary="looks fine")
    checked = enforce_approval_consistency(result)

    assert checked.approved is False
    assert len(checked.findings) == 1


def test_an_approval_with_only_advisory_findings_stands() -> None:
    result = ReviewResult(
        approved=True,
        findings=[finding(kind="tone", severity="advisory")],
        summary="minor wording",
    )
    assert enforce_approval_consistency(result).approved is True


def test_a_clean_approval_is_untouched() -> None:
    result = ReviewResult(approved=True, findings=[], summary="no defects")
    assert enforce_approval_consistency(result) is result


def test_a_rejection_stays_rejected() -> None:
    result = ReviewResult(approved=False, findings=[], summary="withheld")
    assert enforce_approval_consistency(result).approved is False


# --- severity partitioning ---


def test_findings_partition_by_severity() -> None:
    result = ReviewResult(
        approved=False,
        findings=[
            finding(),
            finding(kind="misleading_framing"),
            finding(kind="tone", severity="advisory"),
        ],
        summary="s",
    )

    assert len(result.blocking) == 2
    assert len(result.advisory) == 1
    assert len(result.blocking) + len(result.advisory) == len(result.findings)


# --- the schema ---


def test_an_unknown_finding_kind_is_refused() -> None:
    with pytest.raises(ValueError):
        finding(kind="seems_wrong")


def test_an_unknown_severity_is_refused() -> None:
    with pytest.raises(ValueError):
        finding(severity="critical")


def test_the_result_schema_is_closed() -> None:
    with pytest.raises(ValueError):
        ReviewResult(approved=True, findings=[], summary="s", confidence=0.8)  # type: ignore[call-arg]


def test_a_finding_must_quote_the_text_at_fault() -> None:
    """Located findings; a reviewer's opinion without a quote is not actionable."""
    with pytest.raises(ValueError):
        Finding(
            kind="tone",  # type: ignore[arg-type]
            severity="advisory",  # type: ignore[arg-type]
            problem="reads oddly",
            suggested_fix="rewrite",
        )  # type: ignore[call-arg]


# --- declared figures ---


def test_declared_figures_are_rendered_for_the_reviewer() -> None:
    rendered = _declared_figures(
        letter([FigureUse(value="7.8%", source_field="adjusted_gap_pct", meaning="gap")])
    )

    assert "7.8%" in rendered
    assert "adjusted_gap_pct" in rendered


def test_an_undeclared_letter_says_so() -> None:
    """Silence would read as "no figures", which is a different claim."""
    assert "declared no figures" in _declared_figures(letter())


# --- the audit report ---


def test_the_report_carries_what_a_human_reviewer_needs() -> None:
    result = ReviewResult(
        approved=False,
        findings=[finding(), finding(kind="tone", severity="advisory")],
        summary="one fabricated figure",
        prompt_identifier="compliance_review@v1",
    )
    report = review_report(result)

    assert report["approved"] is False
    assert report["blocking_count"] == 1
    assert report["advisory_count"] == 1
    assert report["prompt"] == "compliance_review@v1"
    assert report["findings"][0]["quote"].startswith("the remaining difference")


def test_the_report_omits_the_suggested_fix() -> None:
    """The audit trail records what was found, not what was proposed."""
    report = review_report(ReviewResult(approved=False, findings=[finding()], summary="s"))
    assert "suggested_fix" not in report["findings"][0]


# --- the prompt ---


def test_the_prompt_renders_within_its_budget() -> None:
    values = {
        "letter": "Dear colleague, ...",
        "declared_figures": "[]",
        "facts": "{}",
        "legal_position": "{}",
    }

    assert REVIEWER_PROMPT.fits(**values)
    assert REVIEWER_PROMPT.identifier == "compliance_review@v1"


def test_the_prompt_forbids_manufacturing_findings() -> None:
    """A false finding costs a human time and teaches them to stop reading."""
    assert "Do not invent a finding" in REVIEWER_PROMPT.template


def test_the_prompt_defines_a_boundary_for_each_kind() -> None:
    assert "Choosing a kind" in REVIEWER_PROMPT.template
    assert "quote the\nsentence that is wrong" in REVIEWER_PROMPT.template

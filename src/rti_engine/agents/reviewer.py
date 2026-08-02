"""Compliance Reviewer: check a draft against the facts it came from.

The Drafter declares which figures it used and where they came from. This
reads the letter as written and asks whether that declaration holds: is
every number in the prose actually in the fact sheet, does the
characterisation match the verdict computed in code, is every legal claim
attributed.

It runs on a different vendor from the Drafter. A reviewer drawn from the
same model family tends to accept the same mistakes — it finds the errors
it would not itself have made, which is the wrong set.

Its output is a structured verdict with located findings rather than
commentary, so the graph can route on it and a human reviewing a Tier 2
response sees a list rather than an essay.

This does not replace the deterministic validator. That one checks
figures against the fact sheet arithmetically and cannot be talked out of
a finding. This one catches what a checker cannot: a sentence that is
technically supported and still misleading.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rti_engine.agents.analyst import GroupAnalysis
from rti_engine.agents.drafter import DraftLetter, build_fact_sheet, build_legal_position
from rti_engine.agents.prompts import GROUNDING_RULES, Prompt
from rti_engine.agents.regulatory import RegulatoryPosition
from rti_engine.llm.factory import ModelRole, get_structured_model, with_recorder
from rti_engine.llm.served import ModelRecorder

Severity = Literal["blocking", "advisory"]

FindingKind = Literal[
    "ungrounded_figure",
    "mischaracterised_finding",
    "unsupported_legal_claim",
    "missing_required_content",
    "misleading_framing",
    "tone",
]

REQUIRED_CONTENT = (
    "the requester's own pay",
    "average pay by sex in the comparison category",
    "how the category is defined",
    "whether a difference remains after controls",
    "the criteria used to determine pay and progression",
    "the basis on which the employer is responding",
)


class ReviewError(RuntimeError):
    """Raised when a review cannot be produced."""


class Finding(BaseModel):
    """One defect found in the draft."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: FindingKind
    severity: Severity = Field(
        description=(
            "blocking if the letter would be wrong or misleading as sent; "
            "advisory if it could be better but is not incorrect."
        )
    )
    quote: str = Field(description="The exact text from the letter that is at fault, verbatim.")
    problem: str = Field(description="What is wrong with it, in one or two sentences.")
    suggested_fix: str = Field(description="What it should say instead.")


class ReviewResult(BaseModel):
    """The reviewer's verdict on a draft."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool = Field(
        description="False if any blocking finding exists. Never approve with one."
    )
    findings: list[Finding] = Field(default_factory=list)
    summary: str = Field(description="One or two sentences on the letter's overall state.")
    prompt_identifier: str = ""

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocking"]

    @property
    def advisory(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "advisory"]


REVIEWER_PROMPT = Prompt(
    name="compliance_review",
    version=1,
    description="Review a draft response against the facts it was built from.",
    inputs=("letter", "declared_figures", "facts", "legal_position"),
    max_tokens=12000,
    template=f"""\
You review a draft response before it is sent to an employee who asked for
pay information. Your job is to find what is wrong with it, not to praise
what is right.

You are reviewing work produced by a different system. Do not assume it is
correct. Do not repair it silently — report what you find.

{GROUNDING_RULES}

## What to check

**1. Every figure is grounded.** Read the letter and find every number in
it. Each one must appear in the fact sheet. A number that is close to a
fact-sheet value but not identical is ungrounded — it has been rounded,
recomputed or misread. Report it as `ungrounded_figure` with severity
`blocking`.

**2. The characterisation matches the verdict.** The fact sheet states
`base_salary_analysis.verdict`. The letter must say what that verdict
means and nothing else:

- `unexplained` — a real difference remains after the controls. The letter
  must say so plainly. Softening it is `mischaracterised_finding`.
- `explained` — the difference disappears under the controls. A letter
  calling this uncertain, inconclusive, or "not a finding of equal pay"
  is wrong: it was explained. That is `mischaracterised_finding`.
- `inconclusive` — a difference remains but cannot be distinguished from
  chance. A letter presenting this as reassurance, or as evidence of
  equal pay, is `mischaracterised_finding`.

All three are `blocking`.

**3. Legal claims are attributed.** Any statement about what the law or
the employer's policy requires must carry a citation that appears in the
legal position. An uncited claim is `unsupported_legal_claim`, blocking.

**4. Required content is present.** Use this kind only for a topic the
letter never covers. The list is: the requester's own pay; average pay by
sex in their category; how the category is defined; whether a difference
remains after controls; the criteria used to determine pay and
progression; and the basis on which the employer is responding. Report as
`missing_required_content`, blocking.

Do not use this kind for a sentence you disagree with, for a figure you
think should have been included, or for wording you would have phrased
differently. If the letter covers the topic at all, this kind does not
apply — choose the kind that describes the actual defect.

If the fact sheet genuinely does not contain something, the letter saying
so is correct and no finding applies.

**5. Framing.** A statement can be individually true and collectively
misleading: a real gap buried under qualifications, an explained result
written to sound like an exoneration, or a conclusion the employee did not
ask for. Report as `misleading_framing`, blocking if it changes what a
reader takes away.

**6. Tone.** Condescension, evasion, or an opening that leads with the
employer's obligations rather than the employee's answer. Advisory.

## Choosing a kind

Each finding gets the one kind that names its actual defect:

- a number not in the fact sheet → `ungrounded_figure`
- the verdict stated as something other than what it is →
  `mischaracterised_finding`
- a claim about law or policy with no supporting citation →
  `unsupported_legal_claim`
- a required topic entirely absent → `missing_required_content`
- true statements that mislead in combination → `misleading_framing`
- how it reads rather than what it says → `tone`

A finding whose `quote` is a heading is almost always misfiled: quote the
sentence that is wrong, not the section it sits in.

## Approval

Set approved to false if there is any blocking finding. Never approve a
letter you have raised a blocking finding against.

An empty findings list is a legitimate result. Do not invent a finding to
appear diligent — a false finding costs a human the time to dismiss it and
trains them to stop reading.

## The draft

{{letter}}

## Figures the drafter declared

{{declared_figures}}

## Fact sheet

{{facts}}

## Legal position

{{legal_position}}""",
)


def _declared_figures(letter: DraftLetter) -> str:
    """Render the drafter's own declaration of the figures it used."""
    if not letter.figures_used:
        return "The drafter declared no figures."

    return json.dumps(
        [
            {
                "value": figure.value,
                "source_field": figure.source_field,
                "meaning": figure.meaning,
            }
            for figure in letter.figures_used
        ],
        indent=2,
    )


def enforce_approval_consistency(result: ReviewResult) -> ReviewResult:
    """Refuse an approval that contradicts the reviewer's own findings.

    A model that raises a blocking finding and then approves anyway has
    contradicted itself, and the approval is the half that would be acted
    on. Rather than trusting whichever half is right, the approval is
    withdrawn: the findings are the evidence, and the flag is a conclusion
    drawn from them.
    """
    if result.approved and result.blocking:
        return result.model_copy(update={"approved": False})
    return result


async def review_draft(
    letter: DraftLetter,
    analysis: GroupAnalysis,
    position: RegulatoryPosition,
    recorder: ModelRecorder | None = None,
) -> ReviewResult:
    """Review one draft against the facts and legal position it came from."""
    rendered = REVIEWER_PROMPT.render(
        letter=letter.render(),
        declared_figures=_declared_figures(letter),
        facts=json.dumps(build_fact_sheet(analysis), indent=2),
        legal_position=json.dumps(build_legal_position(position), indent=2),
    )

    model = get_structured_model(ModelRole.REVIEW, ReviewResult)
    result = await model.ainvoke(rendered, config=with_recorder(recorder) if recorder else None)

    if not isinstance(result, ReviewResult):
        raise ReviewError("review model did not return a result")

    checked = enforce_approval_consistency(result)
    return checked.model_copy(update={"prompt_identifier": REVIEWER_PROMPT.identifier})


def review_report(result: ReviewResult) -> dict[str, Any]:
    """Summarise a review for the audit trail."""
    return {
        "approved": result.approved,
        "blocking_count": len(result.blocking),
        "advisory_count": len(result.advisory),
        "summary": result.summary,
        "prompt": result.prompt_identifier,
        "findings": [
            {
                "kind": finding.kind,
                "severity": finding.severity,
                "quote": finding.quote,
                "problem": finding.problem,
            }
            for finding in result.findings
        ],
    }


def revision_feedback(review: ReviewResult) -> str:
    """Render blocking findings as instructions for a redraft.

    Only blocking findings are sent back. Advisory ones are recorded for
    the human reviewer but are not worth another model call, and a long
    list of minor notes crowds out the defects that actually matter.
    """
    if not review.blocking:
        return "The previous draft raised no blocking findings."

    parts = [
        "A reviewer found the following defects in your previous draft. "
        "Rewrite the letter to address every one of them. Change nothing "
        "else: the rest of the draft was accepted.",
        "",
    ]

    for index, finding in enumerate(review.blocking, start=1):
        parts.extend(
            [
                f"{index}. [{finding.kind}]",
                f"   You wrote: {finding.quote}",
                f"   Problem:   {finding.problem}",
                f"   Instead:   {finding.suggested_fix}",
                "",
            ]
        )

    return "\n".join(parts)

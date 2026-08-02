"""Drafter: write the response an employee actually receives.

This is the first output a person reads, and the first place a correct
figure can become a misstated one. The Analyst's numbers arrive here as
typed fields and must leave unchanged: not rounded, not approximated, not
recomputed into a different form.

Two things support that. The facts are passed as JSON rather than prose,
so there is no narrative to paraphrase and no adjacent sentence to blend
a figure into. And the letter is returned as a structured object in which
every figure used must be declared alongside the field it came from — an
invented number then appears in a list rather than in the middle of a
paragraph, where the validator can catch it.

The model writes language. It does not decide what is true.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rti_engine.agents.analyst import GroupAnalysis
from rti_engine.agents.prompts import (
    CITATION_RULES,
    GROUNDING_RULES,
    Prompt,
)
from rti_engine.agents.regulatory import RegulatoryPosition
from rti_engine.agents.tools import ToolCallError, call_tool
from rti_engine.analytics.inference import GapVerdict, classify_gap
from rti_engine.llm.factory import ModelRole, get_structured_model, with_recorder
from rti_engine.llm.served import ModelRecorder
from rti_engine.mcp.client import KNOWLEDGE, tool_session


class DraftingError(RuntimeError):
    """Raised when a draft cannot be produced or fails its own declaration."""


class FigureUse(BaseModel):
    """One figure written into the letter, and where it came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(
        description="The figure exactly as written in the letter, e.g. '7.0%' or '71,885.14'."
    )
    source_field: str = Field(
        description="The fact-sheet field this value was taken from, e.g. 'base_raw_gap_pct'."
    )
    meaning: str = Field(description="What this figure represents, in a few words.")


class LetterSection(BaseModel):
    """One headed section of the response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    heading: str
    body: str


class DraftLetter(BaseModel):
    """A complete draft response, with its figures and sources declared."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str
    salutation: str
    sections: list[LetterSection] = Field(min_length=1)
    closing: str
    figures_used: list[FigureUse] = Field(
        default_factory=list,
        description="Every figure appearing anywhere in the letter.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Citation strings for every legal or policy claim made.",
    )

    def render(self) -> str:
        """Render the letter as plain text."""
        parts = [f"Subject: {self.subject}", "", self.salutation, ""]
        for section in self.sections:
            parts.extend([section.heading, "", section.body, ""])
        parts.append(self.closing)
        return "\n".join(parts)


DRAFTER_PROMPT = Prompt(
    name="response_letter",
    version=1,
    description="Write the response letter to an employee's pay-information request.",
    inputs=(
        "request_text",
        "facts",
        "legal_position",
        "pay_setting_criteria",
        "revision_feedback",
    ),
    max_tokens=10000,
    template=f"""\
You write an employer's written response to an employee who has asked for
pay information.

The reader is the employee. They are not a lawyer and not a statistician.
Write plainly, in complete sentences, without jargon and without
condescension. Do not tell them how to feel about the answer.

{GROUNDING_RULES}

{CITATION_RULES}

## Declaring figures

For every number that appears anywhere in the letter, add an entry to
figures_used giving the value as written, the fact-sheet field it came
from, and what it represents. A number in the letter with no entry, or an
entry naming a field that does not exist, is a defect.

Write figures exactly as they appear in the fact sheet. Do not round,
approximate, or say "about" or "roughly".

## What the letter must cover

- The employee's own pay, as recorded.
- The average pay of women and men in their category, and how the
  category is defined.
- Whether a difference remains once level, job family, country and
  length of service are accounted for — and what that means.
- The criteria used to determine pay and pay progression.
- The basis on which the employer is responding.
- What happens next, where anything follows.

## Characterising the difference

The fact sheet states the verdict under
`base_salary_analysis.verdict`. **Use it. Do not derive your own from the
figures.**

- **unexplained** — a real difference remains after level, job family,
  country and length of service are accounted for, and it is
  statistically significant. Say so plainly, without softening.
- **explained** — the difference disappears once those factors are
  accounted for. Say which factors account for it. Do not describe it as
  a gap that was justified, and do not say it is inconclusive or
  uncertain: it was explained.
- **inconclusive** — a difference remains after those factors, but cannot
  be distinguished from chance in a group of this size. Say that plainly,
  and say that it is not a finding of equal pay. Never present it as
  reassurance.

`base_salary_analysis.verdict_explanation` states the reasoning in one
sentence. You may draw on it, but write for the employee rather than
quoting it.

## How the comparison was controlled

Two different things account for factors here, and conflating them
overstates what the analysis did.

The **category** holds country, job family and level constant: everyone
compared is in the same one. See `category.definition`.

The **regression** additionally adjusts for the factors listed in
`base_salary_analysis.controls` — typically length of service.

Describe them distinctly. "The comparison is between employees in the same
country, job family and level, and the analysis additionally accounts for
length of service" is accurate. "After accounting for level, job family,
country and length of service" is not: it implies all four were regression
controls, which the fact sheet does not say.

Name only the controls actually listed.

## Stating the legal basis

The letter must say on what basis the employer is responding, and cite it.
Take this from the legal position:

- `legal_basis` and `obligation_summary` — where the duty comes from.
- `national_position` — what national law currently provides.
- `caveats` — every one that bears on this request.
- `citations` — the source for each of the above.

Where the basis is the employer's policy rather than statute, say so
directly. An employee told only that "the employer is responding under its
policy" has not been told that no statutory route currently exists, which
is a material fact about their position.

## Source-backed limitations, not invented ones

A limitation stated in the legal position or its caveats **must be stated
in the letter, with the citation given there**. That includes the absence
of a national right, an unmet threshold, or a pending law.

A limitation that appears in neither must not be written at all.
Sentences like "this does not amount to an individual finding" or "the
policy does not specify a further determination" assert something about
the sources that no source supports. They read as caution and function as
unsupported claims.

The test is simple: can you name the citation? Then state it. If not,
leave it out.

## Tone

The employee asked a fair question and is entitled to a straight answer.
Do not open with the employer's obligations, do not pad, and do not close
with an invitation to raise a grievance unless the fact sheet says one is
available.

## The request

{{request_text}}

## Fact sheet

{{facts}}

## Legal position

{{legal_position}}

## Pay-setting and progression criteria

State these in the letter, in your own words, with their citation. They
are the employer's own published criteria.

{{pay_setting_criteria}}

## Revision

{{revision_feedback}}""",
)


PERCENTAGE_PLACES = 1
CURRENCY_PLACES = 2
P_VALUE_PLACES = 3


def _pct(value: float) -> float:
    """Round a percentage to the precision a letter should state."""
    return round(value, PERCENTAGE_PLACES)


def _eur(value: float) -> float:
    """Round a currency amount to cents."""
    return round(value, CURRENCY_PLACES)


def base_salary_verdict(analysis: GroupAnalysis) -> tuple[GapVerdict, str]:
    """Classify the base-salary difference deterministically.

    The distinction between a difference that is explained, one that is
    unexplained, and one that cannot be established is the single most
    consequential sentence in the letter — and it follows from the figures
    by a fixed rule. Left to the model it was got wrong: a gap of -0.02%
    was described as indistinguishable from chance rather than as
    explained by length of service.

    So it is decided here and passed in as a verdict. The model chooses
    the words; it does not choose the finding.

    The p-value is used directly rather than a corrected one: a single
    requester's group is one comparison, not a family of them, so there is
    no multiplicity to correct for.
    """
    classification = classify_gap(
        raw_gap_pct=analysis.base_raw_gap_pct,
        adjusted_gap_pct=analysis.base_adjusted_gap_pct,
        q_value=analysis.base_p_value,
        alpha=analysis.alpha,
    )
    return classification.verdict, classification.note


def build_fact_sheet(analysis: GroupAnalysis) -> dict[str, Any]:
    """Flatten the analysis into the fields the letter may quote.

    Only these values may appear in the letter. Naming them explicitly
    gives the model a closed set to draw from and the validator a closed
    set to check against.
    """
    requester = analysis.requester
    verdict, verdict_note = base_salary_verdict(analysis)

    return {
        "requester": {
            "employee_id": requester.employee_id,
            "country": requester.country,
            "job_family": requester.job_family,
            "level": requester.level,
            "working_pattern": requester.working_pattern,
            "fte": requester.fte,
            "tenure_years": round(requester.tenure_years, 1),
            "base_salary_fte_eur": _eur(requester.base_salary_fte_eur),
            "base_salary_actual_eur": _eur(requester.base_salary_actual_eur),
            "bonus_actual_eur": _eur(requester.bonus_actual_eur),
            "total_comp_actual_eur": _eur(requester.total_comp_actual_eur),
        },
        "category": {
            "definition": "country, job family and level",
            "group": analysis.group,
            "n_total": analysis.n_total,
            "n_female": analysis.n_female,
            "n_male": analysis.n_male,
            "mean_female_eur": _eur(analysis.mean_female_eur),
            "mean_male_eur": _eur(analysis.mean_male_eur),
            "median_female_eur": _eur(analysis.median_female_eur),
            "median_male_eur": _eur(analysis.median_male_eur),
            "reportable": analysis.reportable,
            "reportability_note": analysis.reportability_note,
        },
        "base_salary_analysis": {
            "verdict": verdict,
            "verdict_explanation": verdict_note,
            "raw_gap_pct": _pct(analysis.base_raw_gap_pct),
            "median_gap_pct": _pct(analysis.base_median_gap_pct),
            "adjusted_gap_pct": _pct(analysis.base_adjusted_gap_pct),
            "p_value": round(analysis.base_p_value, P_VALUE_PLACES),
            "significant": analysis.base_significant,
            "confidence_interval_pct": [
                _pct(bound) for bound in analysis.base_confidence_interval_pct
            ],
            "controls": analysis.controls,
            "alpha": analysis.alpha,
        },
        "total_compensation_analysis": {
            "raw_gap_pct": _pct(analysis.total_comp_raw_gap_pct),
            "adjusted_gap_pct": _pct(analysis.total_comp_adjusted_gap_pct),
            "significant": analysis.total_comp_significant,
        },
        "age_analysis": {
            "age_cutoff": analysis.age_cutoff,
            "younger_gap_pct": _pct(analysis.younger_gap_pct),
            "older_gap_pct": _pct(analysis.older_gap_pct),
            "older_significant": analysis.older_significant,
            "interaction_p_value": round(analysis.interaction_p_value, P_VALUE_PLACES),
            "interaction_significant": analysis.interaction_significant,
        },
        "thresholds": {
            "joint_pay_assessment_trigger_pct": analysis.jpa_threshold_pct,
            "exceeds_joint_pay_assessment_trigger": analysis.exceeds_jpa_threshold,
        },
        "currency": "EUR",
    }


def build_legal_position(position: RegulatoryPosition) -> dict[str, Any]:
    """Render the regulatory position as data for the drafter."""
    return {
        "jurisdiction": position.jurisdiction,
        "transposed": position.transposed,
        "legal_basis": position.legal_basis,
        "obligation_summary": position.obligation_summary,
        "national_position": position.national_position,
        "policy_commitment": position.policy_commitment,
        "caveats": position.caveats,
        "citations": [{"citation": c.citation, "supports": c.supports} for c in position.citations],
    }


def _flatten(prefix: str, value: Any, into: dict[str, Any]) -> None:
    """Collect leaf fields under dotted names, for source-field checking.

    List entries are recorded both bare and indexed, so a model citing
    ``caveats`` and one citing ``caveats[1]`` are both being precise
    enough to check.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), nested, into)
        return

    if isinstance(value, list):
        into[prefix] = value
        for index, item in enumerate(value):
            _flatten(f"{prefix}[{index}]", item, into)
        return

    into[prefix] = value


def fact_sheet_fields(facts: dict[str, Any]) -> set[str]:
    """Return every field name a figure may legitimately cite.

    Both the dotted path and the bare leaf name are accepted: a model
    naming ``base_salary_analysis.raw_gap_pct`` and one naming
    ``raw_gap_pct`` are both being precise enough to check.
    """
    flat: dict[str, Any] = {}
    _flatten("", facts, flat)

    names = set(flat)
    names.update(path.rsplit(".", 1)[-1] for path in flat)
    return names


def permitted_source_fields(
    facts: dict[str, Any], legal_position: dict[str, Any] | None = None
) -> set[str]:
    """Every field a figure may be attributed to.

    The legal position is a source too. The drafter is required to state
    the legal basis and cite it, so a threshold or a headcount quoted from
    a national provision is properly sourced — it simply is not in the
    fact sheet, and checking against the fact sheet alone rejected a
    correct citation.
    """
    names = fact_sheet_fields(facts)
    if legal_position is not None:
        names.update(fact_sheet_fields({"legal_position": legal_position}))
    return names


def check_declared_sources(
    letter: DraftLetter,
    facts: dict[str, Any],
    legal_position: dict[str, Any] | None = None,
) -> list[str]:
    """Return the declared source fields that exist in neither source.

    A cheap check on the model's own declaration. It does not verify that
    the letter's prose matches — that is the validator's job, and it works
    on the text rather than on what the model says about the text.
    """
    permitted = permitted_source_fields(facts, legal_position)
    return [
        figure.source_field
        for figure in letter.figures_used
        if figure.source_field not in permitted
    ]


PAY_SETTING_QUERY = (
    "How is base pay determined, how does pay progress, and what criteria "
    "are used for pay-setting decisions?"
)

POLICY_KIND = "company_policy"


async def fetch_pay_setting_criteria(employee_id: str, tier: str, jurisdiction: str) -> str:
    """Retrieve the employer's stated pay-setting and progression criteria.

    The directive requires a response to set out the criteria used to
    determine pay and pay progression. They are in the employer's policy,
    so without this the letter can only say it does not have them — which
    is what it said, correctly and unhelpfully, before this existed.

    Returns the passages with their citations attached. An empty result
    is reported as such rather than silently omitted.
    """
    async with tool_session(employee_id, tier, servers=[KNOWLEDGE]) as tools:
        try:
            passages = await call_tool(
                tools,
                "search_regulatory_knowledge",
                query=PAY_SETTING_QUERY,
                jurisdiction=jurisdiction,
            )
        except ToolCallError as error:
            raise DraftingError(f"could not retrieve pay-setting criteria: {error}") from error

    if not isinstance(passages, list):
        raise DraftingError("retrieval did not return a list of passages")

    policy = [item for item in passages if item.get("document_kind") == POLICY_KIND]
    if not policy:
        return "The employer's pay-setting criteria were not found in the policy."

    return "\n\n".join(
        f"[{item.get('citation', 'uncited')}]\n{item.get('text', '')}" for item in policy
    )


NO_REVISION = "This is the first draft. There is no prior review to address."


async def draft_response(
    request_text: str,
    analysis: GroupAnalysis,
    position: RegulatoryPosition,
    pay_setting_criteria: str | None = None,
    revision_feedback: str | None = None,
    recorder: ModelRecorder | None = None,
) -> DraftLetter:
    """Draft the response to one request.

    The pay-setting criteria are retrieved if not supplied, so the caller
    may fetch them once and reuse them across drafts.
    """
    if not request_text.strip():
        raise DraftingError("request text is empty")

    criteria = (
        pay_setting_criteria
        if pay_setting_criteria is not None
        else await fetch_pay_setting_criteria(
            analysis.requester.employee_id, "T2", position.jurisdiction
        )
    )

    facts = build_fact_sheet(analysis)
    legal = build_legal_position(position)
    rendered = DRAFTER_PROMPT.render(
        request_text=request_text,
        facts=json.dumps(facts, indent=2),
        legal_position=json.dumps(legal, indent=2),
        pay_setting_criteria=criteria,
        revision_feedback=revision_feedback or NO_REVISION,
    )

    model = get_structured_model(ModelRole.REASONING, DraftLetter)
    letter = await model.ainvoke(rendered, config=with_recorder(recorder) if recorder else None)

    if not isinstance(letter, DraftLetter):
        raise DraftingError("drafting model did not return a letter")

    if unknown := check_declared_sources(letter, facts, legal):
        raise DraftingError(f"letter cites source fields that do not exist: {', '.join(unknown)}")

    return letter

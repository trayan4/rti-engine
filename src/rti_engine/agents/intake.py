"""Intake: classify a request into an autonomy tier.

The tier decides what the rest of the system may reach, so this is the
one classification whose errors are not symmetric. Routing a request for
comparator data down to own-data disclosure releases other people's pay
without review. Routing an own-data request up to comparator disclosure
costs a human five minutes.

The prompt therefore asks the model to escalate when uncertain, and a
deterministic floor is applied to its answer afterwards: a request that
touches comparator data, or that the model itself marks ambiguous, is
T2 regardless of the tier it returned. The prompt asks; the code
guarantees.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rti_engine.agents.prompts import Prompt
from rti_engine.db.models import AutonomyTier
from rti_engine.llm.factory import ModelRole, get_structured_model, with_recorder
from rti_engine.llm.served import ModelRecorder

RequestCategory = Literal[
    "not_a_pay_request",
    "general_information",
    "own_pay",
    "comparator_disclosure",
    "unclear",
]


class IntakeClassification(BaseModel):
    """What the model concluded about a request, before the floor is applied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: RequestCategory = Field(description="What the requester is asking for.")
    seeks_own_pay: bool = Field(
        description="Whether the request asks about the requester's own pay."
    )
    seeks_comparator_data: bool = Field(
        description=(
            "Whether the request asks about pay of others, averages by sex, "
            "or any comparison against colleagues."
        )
    )
    ambiguous: bool = Field(
        description="Whether the request is unclear enough that a human should decide."
    )
    rationale: str = Field(description="One or two sentences explaining the classification.")


class IntakeResult(BaseModel):
    """The final tier, with the model's answer preserved alongside it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: AutonomyTier | None
    """None for input that is not a pay-information request at all — there
    is no data-access level to assign, since nothing will be accessed."""

    classification: IntakeClassification
    escalated: bool
    """True where code raised the tier above what the model returned."""

    escalation_reason: str | None = None
    prompt_identifier: str
    served_by: str = ""
    """Which model answered. A fallback chain nobody can see is one nobody
    can tell is being used."""

    used_fallback: bool = False
    tokens_used: int = 0
    cost_usd: float = 0.0


INTAKE_PROMPT = Prompt(
    name="intake_classification",
    version=1,
    description="Classify a pay-information request into an autonomy tier.",
    inputs=("request_text",),
    max_tokens=1500,
    template="""\
You classify incoming pay-information requests from employees.

## Tiers

- **not_a_pay_request** — the message is not asking about pay at all: a
  greeting, small talk, or something unrelated to pay information.
- **general_information** — the requester asks how pay is set, what their
  rights are, or what the law requires. No employee data is needed.
- **own_pay** — the requester asks about their own pay, level, bonus or
  working pattern, and nothing about anyone else.
- **comparator_disclosure** — the request touches pay of others: average
  pay by sex, what colleagues earn, whether they are paid fairly relative
  to others, or any comparison against a group.
- **unclear** — the request could plausibly be more than one of the above.

## How to decide

Ask what data would be needed to answer. A question that can only be
answered by looking at other people's pay is comparator disclosure, even
if phrased as a question about the requester.

"Am I paid fairly?" is comparator disclosure: fairness is relative.
"What is my salary?" is own pay.
"How does the company set pay?" is general information.

## When uncertain

Choose the broader category and set ambiguous to true. A request wrongly
treated as comparator disclosure costs a human a short review. A request
wrongly treated as own pay releases other people's pay without any
review at all. These are not equivalent errors.

## Request

{request_text}""",
)


def _apply_floor(
    classification: IntakeClassification,
) -> tuple[AutonomyTier | None, str | None]:
    """Derive the tier from the classification, escalating where required.

    The model's category is advisory. Comparator data and self-declared
    ambiguity both force T2 here, so a misclassification cannot release
    data that a human has not seen.
    """
    if classification.category == "not_a_pay_request":
        return None, None

    if classification.seeks_comparator_data:
        return AutonomyTier.T2, "request touches pay data about other employees"

    if classification.ambiguous or classification.category == "unclear":
        return AutonomyTier.T2, "request is ambiguous; a human decides the scope"

    if classification.category == "comparator_disclosure":
        return AutonomyTier.T2, None

    if classification.category == "own_pay" or classification.seeks_own_pay:
        return AutonomyTier.T1, None

    return AutonomyTier.T0, None


async def classify_request(request_text: str) -> IntakeResult:
    """Classify one request and return the tier the system will act under."""
    if not request_text.strip():
        raise ValueError("request text is empty")

    model = get_structured_model(ModelRole.CLASSIFICATION, IntakeClassification)
    rendered = INTAKE_PROMPT.render(request_text=request_text)

    recorder = ModelRecorder()
    classification = await model.ainvoke(rendered, config=with_recorder(recorder))
    if not isinstance(classification, IntakeClassification):
        raise TypeError("intake model did not return a classification")

    tier, reason = _apply_floor(classification)
    naive_tier = (
        AutonomyTier.T2
        if classification.category == "comparator_disclosure"
        else AutonomyTier.T1
        if classification.category == "own_pay"
        else AutonomyTier.T0
    )

    return IntakeResult(
        tier=tier,
        classification=classification,
        escalated=tier != naive_tier,
        escalation_reason=reason,
        prompt_identifier=INTAKE_PROMPT.identifier,
        served_by=recorder.served_by,
        used_fallback=recorder.used_fallback,
        tokens_used=recorder.total_tokens,
        cost_usd=recorder.cost_usd,
    )

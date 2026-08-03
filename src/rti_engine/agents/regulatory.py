"""Regulatory: establish what the law requires for one requester.

The Analyst produces figures. This answers a different question: what is
actually required of this employer, in this country, today. Since the
transposition deadline passed with several member states not having
transposed, that is not the same as what the Directive requires.

This is the first agent reasoning over retrieved text rather than typed
fields, so it is the first that can invent a legal claim. Three
constraints apply. It retrieves before it reasons. Every claim carries
the citation the tool returned. And its output is a structured object
with citations as a required field, so a claim without a source cannot
be represented.
"""

import json
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from rti_engine.agents.prompts import (
    CITATION_RULES,
    GROUNDING_RULES,
    JURISDICTION_RULES,
    TOOL_FAILURE_RULES,
    Prompt,
)
from rti_engine.agents.tools import (
    ToolCallError,
    call_tool,
    format_passages,
    result_text,
)
from rti_engine.llm.factory import ModelRole, get_structured_model, with_recorder
from rti_engine.llm.served import ModelRecorder
from rti_engine.mcp.client import KNOWLEDGE, tool_session

RIGHT_TO_INFORMATION_ARTICLE = 7
REPORTING_ARTICLE = 9
JOINT_ASSESSMENT_ARTICLE = 10

LegalBasis = Literal[
    "national_law",
    "directive_direct_effect",
    "employer_policy_only",
    "no_current_obligation",
]


class RegulatoryError(RuntimeError):
    """Raised when the regulatory position cannot be established."""


class Citation(BaseModel):
    """One source supporting a statement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation: str = Field(description="The citation string exactly as the tool returned it.")
    supports: str = Field(description="The specific claim this source supports.")


class RegulatoryPosition(BaseModel):
    """What the law requires of this employer, for this requester."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    jurisdiction: str
    transposed: bool = Field(
        description="Whether the country has transposed the Directive into national law."
    )
    legal_basis: LegalBasis = Field(
        description=(
            "The basis for the employer's duty to respond to THIS request. "
            "Not whether national law covers pay transparency generally — "
            "whether this requester could compel this response today."
        )
    )
    obligation_summary: str = Field(
        description=(
            "What the employer must do in response to this request, in two or "
            "three sentences. State the basis, not only the requirement."
        )
    )
    national_position: str = Field(
        description=(
            "What national law currently provides on the right to pay "
            "information, including any threshold that differs from the "
            "Directive's."
        )
    )
    policy_commitment: str = Field(
        description=(
            "What the employer's own policy commits to, and whether that "
            "exceeds what national law compels."
        )
    )
    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Anything that qualifies the position: pending legislation, "
            "obligations with no national basis, thresholds that must not be "
            "conflated."
        ),
    )
    citations: list[Citation] = Field(
        min_length=1,
        description="A source for every claim made above. At least one is required.",
    )


REGULATORY_PROMPT = Prompt(
    name="regulatory_position",
    version=1,
    description="Establish the legal basis for responding to a pay-information request.",
    inputs=("jurisdiction", "request_text", "retrieved_context", "graph_context"),
    max_tokens=8000,
    template=f"""\
You establish what the law requires of an employer responding to an
employee's pay-information request.

Your answer must distinguish three things that are easily conflated:

1. What Directive (EU) 2023/970 requires.
2. What the law of the requester's country currently compels, which may
   be less, or different, or not yet in force.
3. What the employer's own policy commits to, which may exceed both.

{JURISDICTION_RULES}

{CITATION_RULES}

{GROUNDING_RULES}

{TOOL_FAILURE_RULES}

## Legal basis

This field records where the duty to answer **this specific request**
comes from. It is not a summary of whether the country regulates pay
transparency at all.

Apply this test: **if the employer refused to answer, could this
requester compel a response under the law in force today?**

- **national_law** — yes. National law gives this requester an
  enforceable route to this information, and it is available to them.
  A national regime that covers the subject matter but does not reach
  this requester is not this basis. Examples that are not national_law:
  a register accessible only through employee representatives rather
  than by individual request; a right conditional on a headcount
  threshold you cannot confirm is met.
- **directive_direct_effect** — no route under national law, but the
  employer is a public-sector body or emanation of the state, so the
  Directive may be relied on against it directly.
- **employer_policy_only** — no enforceable route under law, but the
  employer's own policy commits it to respond.
- **no_current_obligation** — neither law nor policy requires a
  response.

Where a national regime exists but does not compel a response to this
request, say so in the national position and the caveats. Do not record
it as the basis for a duty that does not exist.

## Requester's jurisdiction

{{jurisdiction}}

## The request

{{request_text}}

## Retrieved sources

{{retrieved_context}}

## Structured legal position

{{graph_context}}""",
)


_result_text = result_text
"""Kept as a module-level name so existing callers and tests are unaffected."""


async def _call(tools: dict[str, BaseTool], name: str, **arguments: Any) -> Any:
    """Call one tool, raising this module's error type on any failure."""
    try:
        return await call_tool(tools, name, **arguments)
    except ToolCallError as error:
        raise RegulatoryError(str(error)) from error


_format_passages = format_passages
"""Kept as a module-level name so existing tests are unaffected."""


def _format_graph(status: Any, provisions: Any, gaps: Any, policy: Any) -> str:
    """Render the graph findings as compact JSON.

    Passed as data rather than prose: these are facts the model must
    report, not passages it must interpret.
    """
    return json.dumps(
        {
            "jurisdiction_status": status,
            "national_provisions": provisions,
            "obligations_without_national_basis": gaps,
            "policy_sections_implementing_article_7": policy,
        },
        indent=2,
    )


class RegulatoryEvidence(BaseModel):
    """What the position was reasoned from.

    Returned so a caller can score the reasoning against the passages it
    actually used. Re-retrieving for an evaluation would score a second
    call rather than the one that produced the answer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    passages: list[dict[str, Any]]
    retrieved_context: str
    graph_context: str
    status: dict[str, Any]

    @property
    def passage_texts(self) -> list[str]:
        return [str(item.get("text", "")) for item in self.passages]


async def _gather_context(
    tools: dict[str, BaseTool], jurisdiction: str, request_text: str
) -> RegulatoryEvidence:
    """Retrieve everything the position depends on, before any reasoning."""
    passages = await _call(
        tools,
        "search_regulatory_knowledge",
        query=request_text,
        jurisdiction=jurisdiction,
    )
    status = await _call(tools, "get_jurisdiction_status", jurisdiction=jurisdiction)
    provisions = await _call(tools, "list_provisions_in_jurisdiction", jurisdiction=jurisdiction)
    gaps = await _call(tools, "list_articles_without_national_basis", jurisdiction=jurisdiction)
    policy = await _call(
        tools, "get_policy_sections_for_article", article=RIGHT_TO_INFORMATION_ARTICLE
    )

    if not isinstance(passages, list):
        raise RegulatoryError("retrieval did not return a list of passages")
    if not isinstance(status, dict):
        raise RegulatoryError("jurisdiction status did not return an object")

    return RegulatoryEvidence(
        passages=passages,
        retrieved_context=format_passages(passages),
        graph_context=_format_graph(status, provisions, gaps, policy),
        status=status,
    )


def check_transposition_agrees(
    position: RegulatoryPosition, status: dict[str, Any], jurisdiction: str
) -> None:
    """Refuse a position that contradicts the recorded transposition status.

    Whether a country has transposed is a fact held in the graph, not a
    judgment the model is asked to make. A model that restates it wrongly
    has misread its own sources, and every conclusion resting on it —
    which is all of them — is unsafe. Fail rather than pass it on.
    """
    recorded = bool(status.get("transposed"))
    if position.transposed != recorded:
        raise RegulatoryError(
            f"model reported transposed={position.transposed} but the graph "
            f"records {recorded} for {jurisdiction}"
        )


async def establish_position(
    employee_id: str,
    tier: str,
    jurisdiction: str,
    request_text: str,
    recorder: ModelRecorder | None = None,
) -> RegulatoryPosition:
    """Determine the legal basis for responding to one request."""
    if not request_text.strip():
        raise RegulatoryError("request text is empty")

    async with tool_session(employee_id, tier, servers=[KNOWLEDGE]) as tools:
        evidence = await _gather_context(tools, jurisdiction, request_text)

    rendered = REGULATORY_PROMPT.render(
        jurisdiction=jurisdiction,
        request_text=request_text,
        retrieved_context=evidence.retrieved_context,
        graph_context=evidence.graph_context,
    )

    model = get_structured_model(ModelRole.REASONING, RegulatoryPosition)
    position = await model.ainvoke(rendered, config=with_recorder(recorder) if recorder else None)

    if not isinstance(position, RegulatoryPosition):
        raise RegulatoryError("regulatory model did not return a position")

    # The jurisdiction was supplied by the caller and is not the model's to
    # decide. Left as returned it arrives as "France" or "Spain", which the
    # tool schemas reject downstream — and a wrong country here would send
    # the whole response to the wrong body of law.
    checked = position.model_copy(update={"jurisdiction": jurisdiction})
    check_transposition_agrees(checked, evidence.status, jurisdiction)
    return checked


async def gather_evidence(
    employee_id: str, tier: str, jurisdiction: str, request_text: str
) -> RegulatoryEvidence:
    """Retrieve what a position would be reasoned from, without reasoning.

    Used by the evaluation harness to score retrieval quality against the
    same passages the agent would receive.
    """
    async with tool_session(employee_id, tier, servers=[KNOWLEDGE]) as tools:
        return await _gather_context(tools, jurisdiction, request_text)

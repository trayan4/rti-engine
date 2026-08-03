"""Score the regulatory agent's retrieval and its use of it.

Two questions no deterministic check answers. Whether the passages
retrieved were relevant to what was asked, and whether the legal position
stayed within what those passages actually said.

Everything else in this system is checkable without judgment: a figure
either appears in the fact sheet or it does not, a tier either permits a
scope or it does not. Legal reasoning over retrieved text is the one part
where the question is a matter of degree, which is what an LLM judge is
for.

The judge runs on a different vendor from the model that produced the
position. A model scoring the output of its own family tends to find it
reasonable.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from openevals.llm import create_llm_as_judge
from openevals.prompts import RAG_GROUNDEDNESS_PROMPT, RAG_RETRIEVAL_RELEVANCE_PROMPT
from pydantic import BaseModel, ConfigDict

from rti_engine.agents.regulatory import establish_position, gather_evidence
from rti_engine.evals.cases import SCENARIO_CASES, ScenarioCase
from rti_engine.llm.factory import get_judge_model

GROUNDEDNESS = "groundedness"
RETRIEVAL_RELEVANCE = "retrieval_relevance"

PASSING_SCORE = 0.7
"""Below this, retrieval or reasoning is weak enough to investigate.

A judgment call rather than a standard. It is recorded here so a change
to it is a visible decision rather than a moved goalpost.
"""

LEGAL_QUESTION = (
    "What does the law require of this employer in {jurisdiction} in "
    "responding to an employee's request for pay information, and what does "
    "the employer's own policy commit to?"
)
"""What the regulatory retrieval is actually responsible for.

The employee's request also asks what the pay figures are, and retrieval
is not meant to answer that — the analytics do. Scoring the passages
against the whole request marks them down for lacking something they
should not contain.
"""

CONCURRENCY = 3


class QualityScore(BaseModel):
    """One judged dimension of one case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    score: float
    comment: str


class QualityOutcome(BaseModel):
    """What the judge concluded about one scenario's retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    jurisdiction: str
    passages_retrieved: int
    scores: list[QualityScore]
    error: str | None = None
    unscored: list[str] = []
    """Dimensions the judge failed to return a usable score for.

    Recorded rather than treated as a zero. The judge occasionally emits
    its verdict as prose instead of a structured field, and a parse
    failure is not the same finding as a low score.
    """

    @property
    def passed(self) -> bool:
        return (
            self.error is None
            and not self.unscored
            and all(score.score >= PASSING_SCORE for score in self.scores)
        )

    def score_for(self, key: str) -> float | None:
        for score in self.scores:
            if score.key == key:
                return score.score
        return None


class QualityReport(BaseModel):
    """The retrieval quality suite's result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcomes: list[QualityOutcome]

    @property
    def passed(self) -> bool:
        return all(outcome.passed for outcome in self.outcomes)

    def _mean(self, key: str) -> float | None:
        values = [
            score for outcome in self.outcomes if (score := outcome.score_for(key)) is not None
        ]
        return round(sum(values) / len(values), 3) if values else None

    def summary(self) -> dict[str, Any]:
        return {
            "cases": len(self.outcomes),
            "passed_cases": sum(outcome.passed for outcome in self.outcomes),
            "mean_groundedness": self._mean(GROUNDEDNESS),
            "mean_retrieval_relevance": self._mean(RETRIEVAL_RELEVANCE),
            "threshold": PASSING_SCORE,
            "passed": self.passed,
        }


def _position_text(position: Any) -> str:
    """Render the position as the prose a judge should score.

    The citations are excluded: whether a claim carries a source is
    already checked deterministically, and including them would let a
    well-cited but unsupported claim read as grounded.
    """
    parts = [
        position.obligation_summary,
        position.national_position,
        position.policy_commitment,
        *position.caveats,
    ]
    return "\n\n".join(part for part in parts if part)


def _graph_summary(evidence: Any) -> str:
    """Render the structured half of the evidence compactly.

    The raw graph context is several kilobytes of JSON, and a judge given
    all of it alongside ten passages produces long enough output to
    sometimes lose its own verdict. This keeps the facts and drops the
    formatting.
    """
    status = evidence.status
    return (
        "Structured legal position:\n"
        f"- jurisdiction: {status.get('jurisdiction')}\n"
        f"- transposed: {status.get('transposed')}\n"
        f"- status: {status.get('status')}\n"
        f"- expected: {status.get('expected')}\n"
        f"- direct effect from: {status.get('direct_effect_from')}\n"
        f"- policy sections implementing the right to information: "
        f"{_policy_sections(evidence)}"
    )


def _policy_sections(evidence: Any) -> str:
    """Name the policy sections the graph supplied, without the JSON.

    The full graph context is several kilobytes. A judge given that plus
    ten passages writes long enough to sometimes lose its own verdict in
    the prose, which arrives as a missing score rather than a low one.
    """
    import json

    try:
        parsed = json.loads(evidence.graph_context)
    except (TypeError, ValueError):
        return "not available"

    sections = parsed.get("policy_sections_implementing_article_7") or []
    named = [f"{item.get('section')} ({item.get('title')})" for item in sections]
    return ", ".join(named) if named else "none recorded"


def _evaluators() -> tuple[Any, Any]:
    """Build the two judges, both on the review vendor."""
    judge = get_judge_model()

    groundedness = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key=GROUNDEDNESS,
        judge=judge,
        continuous=True,
    )
    relevance = create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        feedback_key=RETRIEVAL_RELEVANCE,
        judge=judge,
        continuous=True,
    )
    return groundedness, relevance


def _as_score(result: Any, key: str) -> QualityScore:
    """Normalise an evaluator result into a score."""
    if isinstance(result, list):
        result = result[0]

    if isinstance(result, dict):
        return QualityScore(
            key=str(result.get("key", key)),
            score=float(result.get("score") or 0.0),
            comment=str(result.get("comment", "")),
        )

    return QualityScore(
        key=getattr(result, "key", key),
        score=float(getattr(result, "score", 0.0) or 0.0),
        comment=str(getattr(result, "comment", "")),
    )


async def _score_case(
    case: ScenarioCase, employee_id: str, limit: asyncio.Semaphore
) -> QualityOutcome:
    """Retrieve, reason, and score both dimensions for one case."""
    groundedness, relevance = _evaluators()

    async with limit:
        try:
            evidence = await gather_evidence(employee_id, "T2", case.country, case.request_text)
            position = await establish_position(employee_id, "T2", case.country, case.request_text)
        except Exception as error:
            return QualityOutcome(
                name=case.name,
                jurisdiction=case.country,
                passages_retrieved=0,
                scores=[],
                error=f"{type(error).__name__}: {error}",
            )

    # Both halves of what the position was reasoned from. Scoring against
    # the passages alone marks down claims grounded in the graph — the
    # policy section titles and the transposition dates live there, not in
    # any retrieved passage.
    # Both halves of what the position was reasoned from. Scoring against
    # the passages alone marks down claims grounded in the graph — the
    # policy section titles and the transposition dates live there, not in
    # any retrieved passage.
    context = [*evidence.passage_texts, _graph_summary(evidence)]
    answer = _position_text(position)

    scores: list[QualityScore] = []
    unscored: list[str] = []

    judgements: tuple[tuple[str, Callable[[], Any]], ...] = (
        (GROUNDEDNESS, lambda: groundedness(context=context, outputs=answer)),
        (
            RETRIEVAL_RELEVANCE,
            lambda: relevance(
                inputs=LEGAL_QUESTION.format(jurisdiction=case.country),
                context=context,
            ),
        ),
    )

    for key, call in judgements:
        try:
            scores.append(_as_score(call(), key))
        except Exception as error:  # noqa: BLE001 - a judge failure is data
            unscored.append(f"{key}: {type(error).__name__}: {error}")

    return QualityOutcome(
        name=case.name,
        jurisdiction=case.country,
        passages_retrieved=len(context),
        scores=scores,
        unscored=unscored,
    )


async def run_quality_suite(
    employee_id: str = "EMP-00001", names: list[str] | None = None
) -> QualityReport:
    """Score retrieval quality across the scenario catalog."""
    wanted = set(names) if names else None
    cases = [case for case in SCENARIO_CASES if wanted is None or case.name in wanted]

    limit = asyncio.Semaphore(CONCURRENCY)
    outcomes = await asyncio.gather(*(_score_case(case, employee_id, limit) for case in cases))
    return QualityReport(outcomes=list(outcomes))

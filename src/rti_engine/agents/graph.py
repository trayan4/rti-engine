"""The request graph: classify, route, respond.

Nodes do one thing each and return a partial state update. LangGraph
merges the updates and checkpoints the result, so a request that pauses
for human approval can resume later — in a different process, after a
restart.

Routing is a pure function of state. It reads the tier that intake
assigned and returns a node name, without a model in the path: the tier
was already decided by the classifier and then floored in code, and
re-deciding it here would give a second chance to get it wrong.

The Tier 2 path loops. A blocked draft returns to the drafter with the
reviewer's findings, but only a bounded number of times — a draft the
reviewer will not approve goes to the human with its findings attached,
which is a better outcome than two models arguing indefinitely.

No Tier 2 request completes on its own. Approved or not, it ends awaiting
a human decision.
"""

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from rti_engine.agents.analyst import analyse_requester_group
from rti_engine.agents.drafter import draft_response, fetch_pay_setting_criteria
from rti_engine.agents.intake import classify_request
from rti_engine.agents.regulatory import establish_position
from rti_engine.agents.responder import answer_informational, answer_own_data
from rti_engine.agents.retry import (
    FAST_NODE_TIMEOUT,
    NODE_RETRY_POLICY,
    NODE_TIMEOUT,
    reraise_if_transient,
)
from rti_engine.agents.reviewer import ReviewResult, review_draft, revision_feedback
from rti_engine.agents.state import (
    MAX_REVISIONS,
    Actor,
    RequestState,
    audited,
    failed,
)
from rti_engine.db.models import AutonomyTier, RequestStatus
from rti_engine.llm.served import ModelRecorder
from rti_engine.observability.tracing import enable_tracing

TierNode = Literal["respond_informational", "respond_own_data", "analyst"]

INTAKE = "intake"
RESPOND_INFORMATIONAL: TierNode = "respond_informational"
RESPOND_OWN_DATA: TierNode = "respond_own_data"
ANALYST: TierNode = "analyst"

REGULATORY = "regulatory"
DRAFTER = "drafter"
REVIEWER = "reviewer"
APPROVAL = "approval"


class HumanDecision(BaseModel):
    """A person's decision on a Tier 2 response.

    Supplied when the graph is resumed. Nothing an agent produces can
    construct one: the run stops until a caller outside the graph provides
    it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: str = Field(description="approved, rejected, or changes_requested")
    reviewer_id: str
    comment: str | None = None


APPROVED = "approved"
REJECTED = "rejected"
CHANGES_REQUESTED = "changes_requested"

DECISION_STATUS = {
    APPROVED: RequestStatus.APPROVED,
    REJECTED: RequestStatus.REJECTED,
    CHANGES_REQUESTED: RequestStatus.IN_PROGRESS,
}

TIER_NODES: dict[AutonomyTier, TierNode] = {
    AutonomyTier.T0: RESPOND_INFORMATIONAL,
    AutonomyTier.T1: RESPOND_OWN_DATA,
    AutonomyTier.T2: ANALYST,
}
"""Which path each tier takes. The single place routing is decided."""


async def intake_node(state: RequestState) -> dict[str, Any]:
    """Classify the request and record the tier it will be handled under."""
    try:
        result = await classify_request(state["request_text"])
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.INTAKE, "classification_failed", error)

    return {
        "tier": result.tier,
        "intake": result,
        "status": RequestStatus.IN_PROGRESS,
        **audited(
            Actor.INTAKE,
            "tier_assigned",
            tier=result.tier.value,
            category=result.classification.category,
            escalated=result.escalated,
            escalation_reason=result.escalation_reason,
            prompt=result.prompt_identifier,
            served_by=result.served_by,
            used_fallback=result.used_fallback,
        ),
    }


def route_by_tier(state: RequestState) -> str:
    """Choose the path for a classified request.

    A request that failed classification, or that somehow has no tier,
    ends rather than defaulting to a path. Defaulting would mean choosing
    a disclosure level for a request nobody classified.
    """
    if state.get("errors"):
        return END

    tier = state.get("tier")
    if tier is None:
        return END

    return TIER_NODES[tier]


async def respond_informational_node(state: RequestState) -> dict[str, Any]:
    """Answer a Tier 0 request from the corpus alone.

    Completes without human approval: no employee data is involved, and
    the tier makes none reachable.
    """
    recorder = ModelRecorder()
    try:
        letter = await answer_informational(
            state["requester_employee_id"],
            state["request_text"],
            state["jurisdiction"],
            recorder=recorder,
        )
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.SUPERVISOR, "informational_response_failed", error)

    return {
        "draft": letter,
        "status": RequestStatus.COMPLETED,
        **audited(
            Actor.SUPERVISOR,
            "response_written",
            path=RESPOND_INFORMATIONAL,
            sections=len(letter.sections),
            citations=len(letter.citations),
            **recorder.summary(),
        ),
    }


async def respond_own_data_node(state: RequestState) -> dict[str, Any]:
    """Answer a Tier 1 request from the requester's own record.

    Also completes without approval: the requester is being shown their
    own data, and the authorization layer permits nothing else.
    """
    recorder = ModelRecorder()
    try:
        letter = await answer_own_data(
            state["requester_employee_id"],
            state["request_text"],
            state["jurisdiction"],
            recorder=recorder,
        )
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.SUPERVISOR, "own_data_response_failed", error)

    return {
        "draft": letter,
        "status": RequestStatus.COMPLETED,
        **audited(
            Actor.SUPERVISOR,
            "response_written",
            path=RESPOND_OWN_DATA,
            sections=len(letter.sections),
            figures=len(letter.figures_used),
            **recorder.summary(),
        ),
    }


async def analyst_node(state: RequestState) -> dict[str, Any]:
    """Run the deterministic analytical protocol for the requester's group."""
    try:
        analysis = await analyse_requester_group(state["requester_employee_id"], AutonomyTier.T2)
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.ANALYST, "analysis_failed", error)

    return {
        "analysis": analysis,
        **audited(
            Actor.ANALYST,
            "group_analysed",
            group=analysis.group,
            n_total=analysis.n_total,
            raw_gap_pct=analysis.base_raw_gap_pct,
            adjusted_gap_pct=analysis.base_adjusted_gap_pct,
            significant=analysis.base_significant,
            tools=analysis.tools_called,
        ),
    }


async def regulatory_node(state: RequestState) -> dict[str, Any]:
    """Establish what the law requires of this employer, for this requester."""
    recorder = ModelRecorder()
    try:
        position = await establish_position(
            state["requester_employee_id"],
            AutonomyTier.T2.value,
            state["jurisdiction"],
            state["request_text"],
            recorder=recorder,
        )
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.REGULATORY, "regulatory_position_failed", error)

    return {
        "position": position,
        **audited(
            Actor.REGULATORY,
            "position_established",
            jurisdiction=position.jurisdiction,
            transposed=position.transposed,
            legal_basis=position.legal_basis,
            citations=len(position.citations),
            caveats=len(position.caveats),
            **recorder.summary(),
        ),
    }


async def drafter_node(state: RequestState) -> dict[str, Any]:
    """Write the response, addressing any findings from a previous review.

    The pay-setting criteria are retrieved once and carried in state, so a
    revision does not pay for the same retrieval again.
    """
    analysis = state.get("analysis")
    position = state.get("position")
    if analysis is None or position is None:
        return failed(
            Actor.DRAFTER,
            "draft_failed",
            RuntimeError("drafting requires both an analysis and a legal position"),
        )

    review = state.get("review")
    human = state.get("human_feedback")

    # A person's own words take precedence over the automated findings:
    # they saw the draft the findings were raised against and decided
    # something else mattered more.
    if human:
        feedback = f"A reviewer read this draft and asked for changes:\n\n{human}"
    elif review is not None:
        feedback = revision_feedback(review)
    else:
        feedback = None

    criteria = state.get("pay_setting_criteria")

    recorder = ModelRecorder()
    try:
        if criteria is None:
            criteria = await fetch_pay_setting_criteria(
                state["requester_employee_id"],
                AutonomyTier.T2.value,
                position.jurisdiction,
            )

        letter = await draft_response(
            state["request_text"],
            analysis,
            position,
            pay_setting_criteria=criteria,
            revision_feedback=feedback,
            recorder=recorder,
        )
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.DRAFTER, "draft_failed", error)

    revision = state.get("revision_count", 0)
    return {
        "draft": letter,
        "pay_setting_criteria": criteria,
        "human_feedback": None,
        "revision_count": revision + 1 if review is not None else revision,
        **audited(
            Actor.DRAFTER,
            "draft_written" if review is None else "draft_revised",
            revision=revision,
            sections=len(letter.sections),
            figures=len(letter.figures_used),
            citations=len(letter.citations),
            **recorder.summary(),
        ),
    }


def needs_revision(state: RequestState, review: ReviewResult) -> bool:
    """Whether a draft should go back to the drafter.

    Used by both the reviewer node and the edge after it, so the status
    recorded and the path taken cannot disagree.
    """
    if review.approved:
        return False
    return state.get("revision_count", 0) < MAX_REVISIONS


async def reviewer_node(state: RequestState) -> dict[str, Any]:
    """Check the draft against the facts and legal position it came from."""
    draft = state.get("draft")
    analysis = state.get("analysis")
    position = state.get("position")
    if draft is None or analysis is None or position is None:
        return failed(
            Actor.REVIEWER,
            "review_failed",
            RuntimeError("review requires a draft, an analysis and a legal position"),
        )

    recorder = ModelRecorder()
    try:
        review = await review_draft(draft, analysis, position, recorder=recorder)
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.REVIEWER, "review_failed", error)

    revising = needs_revision(state, review)
    exhausted = not review.approved and not revising

    return {
        "review": review,
        "status": (RequestStatus.IN_PROGRESS if revising else RequestStatus.AWAITING_APPROVAL),
        **audited(
            Actor.REVIEWER,
            "draft_reviewed",
            approved=review.approved,
            blocking=len(review.blocking),
            advisory=len(review.advisory),
            revising=revising,
            revisions_exhausted=exhausted,
            prompt=review.prompt_identifier,
            **recorder.summary(),
        ),
    }


def route_after_review(state: RequestState) -> str:
    """Send the draft back for revision, or stop for a human decision."""
    if state.get("errors"):
        return END

    review = state.get("review")
    if review is None:
        return END

    return DRAFTER if needs_revision(state, review) else APPROVAL


def approval_payload(state: RequestState) -> dict[str, Any]:
    """Build what a person needs in order to decide.

    The letter as it would be sent, plus what the reviewer found and
    whether it accepted the draft. A decision made without the findings is
    a rubber stamp.
    """
    draft = state.get("draft")
    review = state.get("review")
    if draft is None or review is None:
        raise ValueError("an approval payload requires both a draft and a review")

    return {
        "request_id": state["request_id"],
        "jurisdiction": state["jurisdiction"],
        "letter": draft.render(),
        "figures_used": [
            {"value": figure.value, "source": figure.source_field} for figure in draft.figures_used
        ],
        "citations": draft.citations,
        "reviewer_approved": review.approved,
        "revisions_used": state.get("revision_count", 0),
        "blocking_findings": [
            {"kind": finding.kind, "quote": finding.quote, "problem": finding.problem}
            for finding in review.blocking
        ],
        "advisory_findings": [
            {"kind": finding.kind, "problem": finding.problem} for finding in review.advisory
        ],
        "decisions": [APPROVED, REJECTED, CHANGES_REQUESTED],
    }


def parse_decision(raw: Any) -> HumanDecision:
    """Validate what a caller supplied on resume.

    An unrecognised decision is refused rather than treated as an
    approval: a malformed resume must not release a statutory disclosure.
    """
    decision = raw if isinstance(raw, HumanDecision) else HumanDecision.model_validate(raw)
    if decision.decision not in DECISION_STATUS:
        permitted = ", ".join(sorted(DECISION_STATUS))
        raise ValueError(f"unknown decision {decision.decision!r}; permitted: {permitted}")
    return decision


async def approval_node(state: RequestState) -> dict[str, Any]:
    """Pause until a person decides.

    The run stops here. The process may exit; the request resumes when the
    graph is invoked again with the same thread id and a decision.

    No agent reaches past this point. That is what Tier 2 means.
    """
    if state.get("draft") is None or state.get("review") is None:
        return failed(
            Actor.SUPERVISOR,
            "approval_failed",
            RuntimeError("approval requires a draft and a review"),
        )

    raw = interrupt(approval_payload(state))

    try:
        decision = parse_decision(raw)
    except Exception as error:
        return failed(Actor.SUPERVISOR, "approval_failed", error)

    approved = decision.decision == APPROVED
    return {
        "approval_decision": decision.decision,
        "approved_by": decision.reviewer_id if approved else None,
        "human_feedback": (decision.comment if decision.decision == CHANGES_REQUESTED else None),
        "status": DECISION_STATUS[decision.decision],
        **audited(
            Actor.HUMAN,
            "approval_decided",
            decision=decision.decision,
            reviewer_id=decision.reviewer_id,
            has_comment=bool(decision.comment),
        ),
    }


def route_after_approval(state: RequestState) -> str:
    """Redraft if changes were requested; otherwise the request is settled."""
    if state.get("errors"):
        return END

    return DRAFTER if state.get("approval_decision") == CHANGES_REQUESTED else END


def _continue_unless_failed(next_node: str) -> Callable[[RequestState], str]:
    """Build an edge that stops the run if the previous node failed."""

    def route(state: RequestState) -> str:
        return END if state.get("errors") else next_node

    return route


def build_graph(checkpointer: Any = None) -> CompiledStateGraph[Any, Any, Any]:
    """Assemble and compile the request graph.

    The checkpointer is optional so the graph can be exercised without a
    database. Persistence is wired in at the step that needs it.
    """
    # Exported before any chain runs: the tracer reads os.environ, and a
    # chain built before this sees tracing as off.
    enable_tracing()

    builder: StateGraph[RequestState, Any, Any, Any] = StateGraph(RequestState)

    # Every node below calls something outside this process, so each can
    # fail for reasons that pass. The approval node does not: it waits for
    # a person, and a decision that failed to parse will not parse on a
    # second attempt.
    for name, node, timeout in (
        (INTAKE, intake_node, FAST_NODE_TIMEOUT),
        (RESPOND_INFORMATIONAL, respond_informational_node, NODE_TIMEOUT),
        (RESPOND_OWN_DATA, respond_own_data_node, NODE_TIMEOUT),
        (ANALYST, analyst_node, NODE_TIMEOUT),
        (REGULATORY, regulatory_node, NODE_TIMEOUT),
        (DRAFTER, drafter_node, NODE_TIMEOUT),
        (REVIEWER, reviewer_node, NODE_TIMEOUT),
    ):
        builder.add_node(name, node, retry_policy=NODE_RETRY_POLICY, timeout=timeout)

    builder.add_node(APPROVAL, approval_node)

    builder.add_edge(START, INTAKE)
    builder.add_conditional_edges(
        INTAKE,
        route_by_tier,
        {
            RESPOND_INFORMATIONAL: RESPOND_INFORMATIONAL,
            RESPOND_OWN_DATA: RESPOND_OWN_DATA,
            ANALYST: ANALYST,
            END: END,
        },
    )

    builder.add_edge(RESPOND_INFORMATIONAL, END)
    builder.add_edge(RESPOND_OWN_DATA, END)

    builder.add_conditional_edges(
        ANALYST,
        _continue_unless_failed(REGULATORY),
        {REGULATORY: REGULATORY, END: END},
    )
    builder.add_conditional_edges(
        REGULATORY,
        _continue_unless_failed(DRAFTER),
        {DRAFTER: DRAFTER, END: END},
    )
    builder.add_conditional_edges(
        DRAFTER,
        _continue_unless_failed(REVIEWER),
        {REVIEWER: REVIEWER, END: END},
    )
    builder.add_conditional_edges(
        REVIEWER, route_after_review, {DRAFTER: DRAFTER, APPROVAL: APPROVAL, END: END}
    )
    builder.add_conditional_edges(APPROVAL, route_after_approval, {DRAFTER: DRAFTER, END: END})

    return builder.compile(checkpointer=checkpointer)

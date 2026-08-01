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

from rti_engine.agents.analyst import analyse_requester_group
from rti_engine.agents.drafter import draft_response, fetch_pay_setting_criteria
from rti_engine.agents.intake import classify_request
from rti_engine.agents.regulatory import establish_position
from rti_engine.agents.responder import answer_informational, answer_own_data
from rti_engine.agents.reviewer import ReviewResult, review_draft, revision_feedback
from rti_engine.agents.state import (
    MAX_REVISIONS,
    Actor,
    RequestState,
    audited,
    failed,
)
from rti_engine.db.models import AutonomyTier, RequestStatus

TierNode = Literal["respond_informational", "respond_own_data", "analyst"]

INTAKE = "intake"
RESPOND_INFORMATIONAL: TierNode = "respond_informational"
RESPOND_OWN_DATA: TierNode = "respond_own_data"
ANALYST: TierNode = "analyst"

REGULATORY = "regulatory"
DRAFTER = "drafter"
REVIEWER = "reviewer"

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
    try:
        letter = await answer_informational(
            state["requester_employee_id"],
            state["request_text"],
            state["jurisdiction"],
        )
    except Exception as error:
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
        ),
    }


async def respond_own_data_node(state: RequestState) -> dict[str, Any]:
    """Answer a Tier 1 request from the requester's own record.

    Also completes without approval: the requester is being shown their
    own data, and the authorization layer permits nothing else.
    """
    try:
        letter = await answer_own_data(
            state["requester_employee_id"],
            state["request_text"],
            state["jurisdiction"],
        )
    except Exception as error:
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
        ),
    }


async def analyst_node(state: RequestState) -> dict[str, Any]:
    """Run the deterministic analytical protocol for the requester's group."""
    try:
        analysis = await analyse_requester_group(state["requester_employee_id"], AutonomyTier.T2)
    except Exception as error:
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
    try:
        position = await establish_position(
            state["requester_employee_id"],
            AutonomyTier.T2.value,
            state["jurisdiction"],
            state["request_text"],
        )
    except Exception as error:
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
    feedback = revision_feedback(review) if review is not None else None
    criteria = state.get("pay_setting_criteria")

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
        )
    except Exception as error:
        return failed(Actor.DRAFTER, "draft_failed", error)

    revision = state.get("revision_count", 0)
    return {
        "draft": letter,
        "pay_setting_criteria": criteria,
        "revision_count": revision + 1 if review is not None else revision,
        **audited(
            Actor.DRAFTER,
            "draft_written" if review is None else "draft_revised",
            revision=revision,
            sections=len(letter.sections),
            figures=len(letter.figures_used),
            citations=len(letter.citations),
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

    try:
        review = await review_draft(draft, analysis, position)
    except Exception as error:
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
        ),
    }


def route_after_review(state: RequestState) -> str:
    """Send the draft back for revision, or stop for a human decision.

    A Tier 2 response never completes here. Approved or not, it ends
    awaiting approval — that is what the tier means.
    """
    if state.get("errors"):
        return END

    review = state.get("review")
    if review is None:
        return END

    return DRAFTER if needs_revision(state, review) else END


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
    builder: StateGraph[RequestState, Any, Any, Any] = StateGraph(RequestState)

    builder.add_node(INTAKE, intake_node)
    builder.add_node(RESPOND_INFORMATIONAL, respond_informational_node)
    builder.add_node(RESPOND_OWN_DATA, respond_own_data_node)
    builder.add_node(ANALYST, analyst_node)
    builder.add_node(REGULATORY, regulatory_node)
    builder.add_node(DRAFTER, drafter_node)
    builder.add_node(REVIEWER, reviewer_node)

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
    builder.add_conditional_edges(REVIEWER, route_after_review, {DRAFTER: DRAFTER, END: END})

    return builder.compile(checkpointer=checkpointer)

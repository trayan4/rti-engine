"""The request graph: classify, route, respond.

Nodes do one thing each and return a partial state update. LangGraph
merges the updates and checkpoints the result, so a request that pauses
for human approval can resume later — in a different process, after a
restart.

Routing is a pure function of state. It reads the tier that intake
assigned and returns a node name, without a model in the path: the tier
was already decided by the classifier and then floored in code, and
re-deciding it here would give a second chance to get it wrong.

The three tier handlers are placeholders at this stage. They record where
a request landed so routing can be verified before the agents behind them
are wired in.
"""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rti_engine.agents.intake import classify_request
from rti_engine.agents.responder import answer_informational, answer_own_data
from rti_engine.agents.state import Actor, RequestState, audited, failed
from rti_engine.db.models import AutonomyTier, RequestStatus

TierNode = Literal["respond_informational", "respond_own_data", "disclosure_pipeline"]

INTAKE = "intake"
RESPOND_INFORMATIONAL: TierNode = "respond_informational"
RESPOND_OWN_DATA: TierNode = "respond_own_data"
DISCLOSURE_PIPELINE: TierNode = "disclosure_pipeline"

TIER_NODES: dict[AutonomyTier, TierNode] = {
    AutonomyTier.T0: RESPOND_INFORMATIONAL,
    AutonomyTier.T1: RESPOND_OWN_DATA,
    AutonomyTier.T2: DISCLOSURE_PIPELINE,
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


async def disclosure_pipeline_node(state: RequestState) -> dict[str, Any]:
    """Handle a Tier 2 request. Placeholder."""
    return {
        "status": RequestStatus.IN_PROGRESS,
        **audited(Actor.SUPERVISOR, "routed", path=DISCLOSURE_PIPELINE),
    }


def build_graph(checkpointer: Any = None) -> CompiledStateGraph[Any, Any, Any]:
    """Assemble and compile the request graph.

    The checkpointer is optional so the graph can be exercised without a
    database. Persistence is wired in at the step that needs it.
    """
    builder: StateGraph[RequestState, Any, Any, Any] = StateGraph(RequestState)

    builder.add_node(INTAKE, intake_node)
    builder.add_node(RESPOND_INFORMATIONAL, respond_informational_node)
    builder.add_node(RESPOND_OWN_DATA, respond_own_data_node)
    builder.add_node(DISCLOSURE_PIPELINE, disclosure_pipeline_node)

    builder.add_edge(START, INTAKE)
    builder.add_conditional_edges(
        INTAKE,
        route_by_tier,
        {
            RESPOND_INFORMATIONAL: RESPOND_INFORMATIONAL,
            RESPOND_OWN_DATA: RESPOND_OWN_DATA,
            DISCLOSURE_PIPELINE: DISCLOSURE_PIPELINE,
            END: END,
        },
    )

    for node in (RESPOND_INFORMATIONAL, RESPOND_OWN_DATA, DISCLOSURE_PIPELINE):
        builder.add_edge(node, END)

    return builder.compile(checkpointer=checkpointer)

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
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from rti_engine.agents.analyst import analyse_requester_group
from rti_engine.agents.budget import degraded_detail, degraded_letter, over_budget
from rti_engine.agents.drafter import (
    build_fact_sheet,
    build_legal_position,
    draft_response,
    fetch_pay_setting_criteria,
)
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
    current_tier,
    failed,
)
from rti_engine.db.models import AutonomyTier, RequestStatus
from rti_engine.guardrails.numbers import validate_numbers
from rti_engine.guardrails.pii import scan
from rti_engine.llm.served import ModelRecorder
from rti_engine.observability.otel import span
from rti_engine.observability.tracing import enable_tracing

TierNode = Literal["respond_informational", "respond_own_data", "analyst"]

INTAKE = "intake"
RESPOND_INFORMATIONAL: TierNode = "respond_informational"
RESPOND_OWN_DATA: TierNode = "respond_own_data"
ANALYST: TierNode = "analyst"

RESPOND_NOT_APPLICABLE = "respond_not_applicable"
"""Not part of TIER_NODES: it is reached when there is no tier to route
on, not by looking one up."""

REGULATORY = "regulatory"
DRAFTER = "drafter"
REVIEWER = "reviewer"
VALIDATOR = "validator"
APPROVAL = "approval"
DEGRADED = "degraded"


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
    """Redact the request, then classify what remains.

    Redaction happens first and everything downstream uses the result. A
    request naming a colleague — "how much does Maria earn" — is still a
    comparator request without the name, so nothing about the tier
    decision needs it, and the name has no reason to reach a model's
    context or a stored transcript.
    """
    inbound = scan(state["request_text"])

    try:
        result = await classify_request(inbound.redacted)
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.INTAKE, "classification_failed", error)

    return {
        "redacted_request_text": inbound.redacted,
        "tier": result.tier,
        "intake": result,
        "status": RequestStatus.IN_PROGRESS,
        "tokens_used": result.tokens_used,
        "cost_usd": result.cost_usd,
        **audited(
            Actor.INTAKE,
            "tier_assigned",
            tier=result.tier.value if result.tier else "not_a_pay_request",
            category=result.classification.category,
            escalated=result.escalated,
            escalation_reason=result.escalation_reason,
            prompt=result.prompt_identifier,
            served_by=result.served_by,
            used_fallback=result.used_fallback,
            **inbound.summary(),
        ),
    }


def request_text(state: RequestState) -> str:
    """The request as agents should see it.

    The redacted form once intake has produced one. Falling back to the
    original matters for the tier 0 and tier 1 paths, which can be
    invoked directly in tests without passing through intake.
    """
    return state.get("redacted_request_text") or state["request_text"]


def route_by_tier(state: RequestState) -> str:
    """Choose the path for a classified request.

    A request that failed classification ends by falling through to the
    degraded response. One that was classified but is not a pay request
    at all — a greeting, small talk — gets a short, immediate answer
    rather than running the disclosure pipeline on input that was never a
    real request.
    """
    if state.get("errors"):
        return DEGRADED

    tier = state.get("tier")
    if tier is None:
        return RESPOND_NOT_APPLICABLE

    return TIER_NODES[tier]


NOT_APPLICABLE_MESSAGE = (
    "This system handles requests for pay information — your own pay, "
    "how pay is set, or how your pay compares with colleagues doing work "
    "of equal value. Send a request like that and it will be answered."
)


async def respond_not_applicable_node(state: RequestState) -> dict[str, Any]:
    """Answer input that was never a pay-information request.

    No model call: the classifier has already decided this is not a real
    request, and composing an answer to it would be spending a call on
    something with nothing to answer.
    """
    return {
        "status": RequestStatus.COMPLETED,
        **audited(Actor.SYSTEM, "not_applicable_response_issued"),
    }


async def respond_informational_node(state: RequestState) -> dict[str, Any]:
    """Answer a Tier 0 request from the corpus alone.

    Completes without human approval: no employee data is involved, and
    the tier makes none reachable.
    """
    recorder = ModelRecorder()
    try:
        letter = await answer_informational(
            state["requester_employee_id"],
            request_text(state),
            state["jurisdiction"],
            recorder=recorder,
        )
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.SUPERVISOR, "informational_response_failed", error)

    return {
        "draft": letter,
        "status": RequestStatus.COMPLETED,
        "tokens_used": recorder.total_tokens,
        "cost_usd": recorder.cost_usd,
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
            request_text(state),
            state["jurisdiction"],
            recorder=recorder,
        )
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.SUPERVISOR, "own_data_response_failed", error)

    return {
        "draft": letter,
        "status": RequestStatus.COMPLETED,
        "tokens_used": recorder.total_tokens,
        "cost_usd": recorder.cost_usd,
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
            request_text(state),
            recorder=recorder,
        )
    except Exception as error:
        reraise_if_transient(error)
        return failed(Actor.REGULATORY, "regulatory_position_failed", error)

    return {
        "position": position,
        "tokens_used": recorder.total_tokens,
        "cost_usd": recorder.cost_usd,
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
    check = state.get("number_check")

    # A person's own words take precedence over the automated findings:
    # they saw the draft the findings were raised against and decided
    # something else mattered more.
    if human:
        feedback = f"A reviewer read this draft and asked for changes:\n\n{human}"
    elif check is not None and not check.grounded:
        feedback = check.feedback()
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
            request_text(state),
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
        "number_check": None,
        "revision_count": (revision + 1 if review is not None or check is not None else revision),
        "tokens_used": recorder.total_tokens,
        "cost_usd": recorder.cost_usd,
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


async def validator_node(state: RequestState) -> dict[str, Any]:
    """Check every figure in the draft against its sources.

    Runs before the reviewer: it is deterministic, costs nothing, and
    removes the mechanical failures from what a model is asked to judge.
    A reviewer spending its attention on a fabricated number has less
    left for the things only a reader can catch.
    """
    draft = state.get("draft")
    analysis = state.get("analysis")
    position = state.get("position")
    if draft is None or analysis is None or position is None:
        return failed(
            Actor.SYSTEM,
            "validation_failed",
            RuntimeError("validation requires a draft, an analysis and a position"),
        )

    result = validate_numbers(
        draft.render(),
        [figure.value for figure in draft.figures_used],
        build_fact_sheet(analysis),
        build_legal_position(position),
    )

    return {
        "number_check": result,
        **audited(Actor.SYSTEM, "figures_validated", **result.summary()),
    }


def route_after_validation(state: RequestState) -> str:
    """Send an ungrounded draft back, or pass it to the reviewer."""
    if state.get("errors"):
        return DEGRADED

    check = state.get("number_check")
    if check is None:
        return DEGRADED

    if check.grounded:
        return REVIEWER

    # An ungrounded figure is not a matter of judgment, so a revision is
    # worth spending even when the reviewer might have approved. Once the
    # revisions are used up the draft goes forward with the finding
    # attached, for a person to see.
    return DRAFTER if state.get("revision_count", 0) < MAX_REVISIONS else REVIEWER


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
        "tokens_used": recorder.total_tokens,
        "cost_usd": recorder.cost_usd,
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
        return DEGRADED

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
        return DEGRADED

    return DRAFTER if state.get("approval_decision") == CHANGES_REQUESTED else END


def _guarded(next_node: str) -> Callable[[RequestState], str]:
    """Build an edge that diverts a request that cannot safely continue.

    Two ways a request stops early: a node failed, or the request has
    spent more than it is allowed. Both go to the degraded response
    rather than ending, because an employee who receives nothing has been
    told less than one who receives an acknowledgement.
    """

    def route(state: RequestState) -> str:
        if state.get("errors"):
            return DEGRADED
        if over_budget(state.get("tokens_used", 0), state.get("cost_usd", 0.0)):
            return DEGRADED
        return next_node

    return route


async def degraded_node(state: RequestState) -> dict[str, Any]:
    """Acknowledge a request the pipeline could not answer.

    No model is called. The circumstances that reach this node are ones
    where model calls are the thing failing, so composing this response
    with one would be asking the broken part to explain itself.

    The status is failure, because the automated pipeline did fail. The
    employee still receives a response, and the request is queued for a
    person.
    """
    errors = state.get("errors", [])
    reason = over_budget(state.get("tokens_used", 0), state.get("cost_usd", 0.0)) or (
        errors[0] if errors else "the request could not be completed"
    )

    return {
        "draft": degraded_letter(reason),
        "status": RequestStatus.FAILED,
        **audited(Actor.SYSTEM, "degraded_response_issued", **degraded_detail(reason, errors)),
    }


def _traced[NodeT: Callable[..., Any]](name: str, node: NodeT) -> NodeT:
    """Wrap a node so its work appears as a span.

    Applied at assembly rather than inside each node, so a node added
    later is traced without anyone remembering to instrument it.
    """

    async def wrapped(state: RequestState) -> dict[str, Any]:
        tier = current_tier(state)
        with span(
            f"node.{name}",
            **{
                "rti.request_id": state.get("request_id", ""),
                "rti.tier": tier.value if tier else "unclassified",
                "rti.revision": state.get("revision_count", 0),
            },
        ) as current:
            update: dict[str, Any] = await node(state)
            if errors := update.get("errors"):
                current.set_attribute("rti.error", str(errors[0])[:200])
            return update

    # Returned as the type it was given, so the graph's node protocol sees
    # what it saw before the wrapper existed.
    return cast(NodeT, wrapped)


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
        builder.add_node(name, _traced(name, node), retry_policy=NODE_RETRY_POLICY, timeout=timeout)

    builder.add_node(APPROVAL, _traced(APPROVAL, approval_node))
    builder.add_node(VALIDATOR, _traced(VALIDATOR, validator_node))
    builder.add_node(DEGRADED, _traced(DEGRADED, degraded_node))
    builder.add_node(
        RESPOND_NOT_APPLICABLE,
        _traced(RESPOND_NOT_APPLICABLE, respond_not_applicable_node),
    )

    builder.add_edge(START, INTAKE)
    builder.add_conditional_edges(
        INTAKE,
        route_by_tier,
        {
            RESPOND_INFORMATIONAL: RESPOND_INFORMATIONAL,
            RESPOND_OWN_DATA: RESPOND_OWN_DATA,
            ANALYST: ANALYST,
            RESPOND_NOT_APPLICABLE: RESPOND_NOT_APPLICABLE,
            DEGRADED: DEGRADED,
            END: END,
        },
    )

    for source, following in (
        (RESPOND_INFORMATIONAL, END),
        (RESPOND_OWN_DATA, END),
        (RESPOND_NOT_APPLICABLE, END),
        (ANALYST, REGULATORY),
        (REGULATORY, DRAFTER),
        (DRAFTER, VALIDATOR),
    ):
        builder.add_conditional_edges(
            source, _guarded(following), {following: following, DEGRADED: DEGRADED}
        )

    builder.add_conditional_edges(
        VALIDATOR,
        route_after_validation,
        {DRAFTER: DRAFTER, REVIEWER: REVIEWER, DEGRADED: DEGRADED},
    )

    builder.add_conditional_edges(
        REVIEWER,
        route_after_review,
        {DRAFTER: DRAFTER, APPROVAL: APPROVAL, DEGRADED: DEGRADED, END: END},
    )
    builder.add_conditional_edges(
        APPROVAL,
        route_after_approval,
        {DRAFTER: DRAFTER, DEGRADED: DEGRADED, END: END},
    )
    builder.add_edge(DEGRADED, END)

    return builder.compile(checkpointer=checkpointer)

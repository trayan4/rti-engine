"""Between HTTP and the graph.

Two stores, each doing what the other cannot. The checkpointer holds
working state keyed by thread id and resumes a paused request exactly
where it stopped; it cannot answer "what is awaiting approval?". The
requests table can, and holds nothing an agent needs.

Keeping them consistent is this module's job, so an endpoint never has to
know there are two.

Requests run in the background. A Tier 2 request takes minutes and pauses
for a human, so a caller that waited for it would hold a connection open
across a decision that may take days.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sqlalchemy import select

from rti_engine.agents.checkpointing import checkpointer
from rti_engine.agents.graph import build_graph
from rti_engine.agents.state import (
    Jurisdiction,
    RequestState,
    current_status,
    current_tier,
    initial_state,
)
from rti_engine.db.models import AutonomyTier, Request, RequestStatus
from rti_engine.db.session import session_scope
from rti_engine.observability.tracing import run_config

RECURSION_LIMIT = 30

OPEN_STATUSES = (
    RequestStatus.RECEIVED,
    RequestStatus.CLASSIFYING,
    RequestStatus.IN_PROGRESS,
)


class RequestNotFoundError(LookupError):
    """Raised when no request exists under an id, or none the caller may see.

    One error covers both cases deliberately: a distinct "not yours" would
    confirm that a given request exists.
    """


def _config(request_id: str, tier: str | None = None) -> RunnableConfig:
    """Build the graph config for one request.

    The thread id is the request id, so resuming a paused request is a
    matter of invoking the graph with the same config rather than
    reconstructing anything.
    """
    labels = run_config(request_id, tier=tier)
    return RunnableConfig(
        configurable={"thread_id": request_id},
        recursion_limit=RECURSION_LIMIT,
        run_name=labels["run_name"],
        tags=labels["tags"],
        metadata=labels["metadata"],
    )


def create_request(employee_id: str, request_text: str) -> str:
    """Record a new request and return its id.

    Written before the graph runs, so a request that fails mid-pipeline is
    still visible rather than lost.
    """
    request_id = str(uuid.uuid4())

    with session_scope() as session:
        session.add(
            Request(
                id=uuid.UUID(request_id),
                requester_employee_id=employee_id,
                request_text=request_text,
                status=RequestStatus.RECEIVED,
                thread_id=request_id,
            )
        )

    return request_id


def _sync_record(request_id: str, values: RequestState) -> None:
    """Copy the graph's outcome onto the queryable record."""
    status = current_status(values)
    tier = current_tier(values)

    with session_scope() as session:
        record = session.get(Request, uuid.UUID(request_id))
        if record is None:
            return

        record.status = status
        record.tier = tier
        if status in (RequestStatus.COMPLETED, RequestStatus.REJECTED):
            record.completed_at = datetime.now(UTC)


async def run_request(request_id: str, employee_id: str, jurisdiction: Jurisdiction) -> None:
    """Run a request through the graph until it finishes or pauses."""
    with session_scope() as session:
        record = session.get(Request, uuid.UUID(request_id))
        if record is None:
            raise RequestNotFoundError(request_id)
        request_text = record.request_text

    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        result = await graph.ainvoke(
            initial_state(request_id, employee_id, request_text, jurisdiction),
            config=_config(request_id),
        )

    _sync_record(request_id, cast(RequestState, result))


async def resume_request(
    request_id: str, decision: str, reviewer_id: str, comment: str | None
) -> None:
    """Resume a paused request with a person's decision."""
    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        result = await graph.ainvoke(
            Command(
                resume={
                    "decision": decision,
                    "reviewer_id": reviewer_id,
                    "comment": comment,
                }
            ),
            config=_config(request_id),
        )

    _sync_record(request_id, cast(RequestState, result))


def get_record(request_id: str, employee_id: str | None = None) -> Request:
    """Return one request record, refusing another employee's.

    A caller who is not the requester gets the same error as one asking
    about a request that does not exist. Distinguishing the two would
    confirm that a given request exists.
    """
    try:
        key = uuid.UUID(request_id)
    except ValueError as error:
        raise RequestNotFoundError(request_id) from error

    with session_scope() as session:
        record = session.get(Request, key)
        if record is None:
            raise RequestNotFoundError(request_id)
        if employee_id is not None and record.requester_employee_id != employee_id:
            raise RequestNotFoundError(request_id)
        return record


def list_requests(employee_id: str) -> list[Request]:
    """Return one employee's own requests, most recent first."""
    with session_scope() as session:
        return list(
            session.scalars(
                select(Request)
                .where(Request.requester_employee_id == employee_id)
                .order_by(Request.created_at.desc())
            )
        )


def list_awaiting_approval() -> list[Request]:
    """Return every request paused for a decision, oldest first.

    Oldest first because these carry a statutory response deadline, and
    the one waiting longest is the one closest to missing it.
    """
    with session_scope() as session:
        return list(
            session.scalars(
                select(Request)
                .where(Request.status == RequestStatus.AWAITING_APPROVAL)
                .order_by(Request.created_at)
            )
        )


async def get_state(request_id: str) -> dict[str, Any]:
    """Return the graph's working state for a request."""
    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        snapshot = await graph.aget_state(_config(request_id))

    if not snapshot.values:
        raise RequestNotFoundError(request_id)
    return dict(snapshot.values)


async def get_pending_approval(request_id: str) -> dict[str, Any]:
    """Return what a reviewer needs to decide on a paused request."""
    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        snapshot = await graph.aget_state(_config(request_id))

    if not snapshot.values:
        raise RequestNotFoundError(request_id)

    for task in snapshot.tasks:
        for item in task.interrupts:
            return dict(item.value)

    raise RequestNotFoundError(f"{request_id} is not awaiting approval")


def tier_name(record: Request) -> str | None:
    """Return a record's tier as a string, if it has one."""
    return AutonomyTier(record.tier).value if record.tier else None

"""The HTTP interface.

Four things a caller can do: submit a request, check one, see what is
awaiting a decision, and decide.

Submitting returns immediately. A Tier 2 request takes minutes and then
waits for a person, so a caller that held a connection open would time
out long before a decision arrived. The id comes back at once and the work
happens behind it.

Every route takes its identity from the session. Nothing accepts an
employee id in a body, so a caller cannot act as someone else — the rule
the agents operate under, applied at the edge.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from rti_engine.api import service
from rti_engine.api.schemas import (
    ApprovalItem,
    DecisionRequest,
    RequestDetail,
    RequestSummary,
    SubmitRequest,
)
from rti_engine.api.security import CurrentPrincipal, CurrentReviewer
from rti_engine.observability.otel import configure_tracing
from rti_engine.observability.tracing import enable_tracing, tracing_status

DEFAULT_JURISDICTION: Literal["DE", "FR", "ES"] = "DE"
"""Where a requester's country is not yet resolved from their record.

Temporary: the analytics server already knows each employee's country,
and this should read from there rather than assume.
"""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Turn tracing on before any request runs."""
    enable_tracing()
    if configure_tracing().enabled:
        FastAPIInstrumentor.instrument_app(app)
    yield


app = FastAPI(
    title="Pay Information Request Engine",
    description=(
        "Handles employee pay-information requests under Directive (EU) "
        "2023/970. Statutory disclosures pause for human approval."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Report that the service is up, and whether tracing is on."""
    return {"status": "ok", "tracing": tracing_status().enabled}


@app.post("/requests", status_code=status.HTTP_202_ACCEPTED)
async def submit_request(
    body: SubmitRequest,
    principal: CurrentPrincipal,
    background: BackgroundTasks,
) -> dict[str, str]:
    """Accept a request and start handling it.

    Returns 202 rather than 200: the request has been accepted, not
    answered. Its id is what the caller polls.
    """
    request_id = service.create_request(principal.employee_id, body.request_text)

    background.add_task(
        service.run_request,
        request_id,
        principal.employee_id,
        DEFAULT_JURISDICTION,
    )

    return {"request_id": request_id, "status": "accepted"}


@app.get("/requests")
async def list_requests(principal: CurrentPrincipal) -> list[RequestSummary]:
    """List the caller's own requests, most recent first."""
    return [
        RequestSummary(
            request_id=str(record.id),
            requester_employee_id=record.requester_employee_id,
            tier=service.tier_name(record),
            status=record.status.value,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        for record in service.list_requests(principal.employee_id)
    ]


@app.get("/requests/{request_id}")
async def get_request(request_id: str, principal: CurrentPrincipal) -> RequestDetail:
    """Return one of the caller's own requests, with its response if ready.

    A request belonging to someone else is reported as not found rather
    than forbidden: a distinct error would confirm it exists.
    """
    try:
        record = service.get_record(request_id, principal.employee_id)
    except service.RequestNotFoundError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "request not found") from error

    detail = RequestDetail(
        request_id=str(record.id),
        tier=service.tier_name(record),
        status=record.status.value,
    )

    try:
        state = await service.get_state(request_id)
    except service.RequestNotFoundError:
        return detail

    draft = state.get("draft")
    return detail.model_copy(
        update={
            "letter": draft.render() if draft else None,
            "citations": draft.citations if draft else [],
            "audit": [
                {"actor": entry.actor.value, "action": entry.action}
                for entry in state.get("audit", [])
            ],
            "errors": state.get("errors", []),
        }
    )


@app.get("/approvals")
async def list_approvals(reviewer: CurrentReviewer) -> list[RequestSummary]:
    """List requests awaiting a decision, longest-waiting first."""
    return [
        RequestSummary(
            request_id=str(record.id),
            requester_employee_id=record.requester_employee_id,
            tier=service.tier_name(record),
            status=record.status.value,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        for record in service.list_awaiting_approval()
    ]


@app.get("/approvals/{request_id}")
async def get_approval(request_id: str, reviewer: CurrentReviewer) -> ApprovalItem:
    """Return everything a reviewer needs to decide on one request."""
    try:
        payload = await service.get_pending_approval(request_id)
    except service.RequestNotFoundError as error:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no request awaiting approval under that id"
        ) from error

    return ApprovalItem.model_validate(payload)


@app.post("/approvals/{request_id}/decision")
async def decide(
    request_id: str,
    body: DecisionRequest,
    reviewer: CurrentReviewer,
    background: BackgroundTasks,
) -> dict[str, str]:
    """Record a decision and resume the request.

    Approving completes it, rejecting ends it, and requesting changes
    sends it back to be redrafted against the reviewer's comment.

    The reviewer's identity comes from the session, so an approval is
    always attributable to whoever was authenticated.
    """
    try:
        await service.get_pending_approval(request_id)
    except service.RequestNotFoundError as error:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no request awaiting approval under that id"
        ) from error

    background.add_task(
        service.resume_request,
        request_id,
        body.decision,
        reviewer.employee_id,
        body.comment,
    )

    return {"request_id": request_id, "decision": body.decision, "status": "accepted"}

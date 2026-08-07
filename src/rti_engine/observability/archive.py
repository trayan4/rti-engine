"""Audit bundle archival to Blob Storage.

A bundle is written once a request is settled — approved, rejected, or
completed autonomously — because that is the point at which the record
becomes final. Each is one JSON file, independent of Postgres and the
checkpoint store: the compliance question a bundle answers ("what did we
send, and why") should be answerable even if the database that produced
it is gone.

Authentication is DefaultAzureCredential throughout: the managed identity
in a deployment, a local Azure CLI login otherwise. No key or connection
string is held by the application at any point, consistent with how Key
Vault access already works.

Archival failing must never fail the request it is archiving. A person
has already been told their request is settled; losing the archive copy
is a problem to notice and fix, not one to surface to them.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from pydantic import BaseModel, ConfigDict

from rti_engine.agents.drafter import DraftLetter
from rti_engine.agents.reviewer import ReviewResult, review_report
from rti_engine.agents.state import AuditEntry, RequestState
from rti_engine.config.settings import get_settings
from rti_engine.guardrails.numbers import ValidationResult
from rti_engine.observability.otel import span

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = 1


class ArchiveError(RuntimeError):
    """Raised when a bundle cannot be written.

    Callers in the request path should catch this rather than let it
    propagate: archival is a record of what happened, not a condition of
    it happening.
    """


class AuditBundle(BaseModel):
    """Everything behind one settled request, as it will be stored.

    A flat, self-contained document. A reviewer or an auditor reading this
    file six months from now should need nothing else to understand what
    was sent and why.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = BUNDLE_SCHEMA_VERSION
    request_id: str
    requester_employee_id: str
    tier: str
    jurisdiction: str
    status: str
    request_text: str

    letter: str | None
    figures_used: list[dict[str, str]]
    citations: list[str]

    number_check: dict[str, Any] | None
    review: dict[str, Any] | None

    approval_decision: str | None
    approved_by: str | None

    revision_count: int
    tokens_used: int
    cost_usd: float

    audit: list[dict[str, Any]]
    archived_at: str


def build_bundle(
    *,
    request_id: str,
    requester_employee_id: str,
    tier: str,
    jurisdiction: str,
    status: str,
    request_text: str,
    draft: DraftLetter | None,
    number_check: ValidationResult | None,
    review: ReviewResult | None,
    approval_decision: str | None,
    approved_by: str | None,
    revision_count: int,
    tokens_used: int,
    cost_usd: float,
    audit: list[AuditEntry],
) -> AuditBundle:
    """Assemble a bundle from a settled request's state.

    Takes fields rather than the graph state directly, so this has no
    dependency on the state schema's internal shape and can be called
    from a test without constructing one.
    """
    return AuditBundle(
        request_id=request_id,
        requester_employee_id=requester_employee_id,
        tier=tier,
        jurisdiction=jurisdiction,
        status=status,
        request_text=request_text,
        letter=draft.render() if draft else None,
        figures_used=[
            {"value": f.value, "source_field": f.source_field, "meaning": f.meaning}
            for f in (draft.figures_used if draft else [])
        ],
        citations=draft.citations if draft else [],
        number_check=number_check.model_dump() if number_check else None,
        review=review_report(review) if review else None,
        approval_decision=approval_decision,
        approved_by=approved_by,
        revision_count=revision_count,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        audit=[
            {
                "actor": entry.actor.value,
                "action": entry.action,
                "detail": entry.detail,
                "occurred_at": entry.occurred_at.isoformat(),
            }
            for entry in audit
        ],
        archived_at=datetime.now(UTC).isoformat(),
    )


def _blob_path(bundle: AuditBundle) -> str:
    """Where a bundle lives in the container.

    Partitioned by date so a container listing stays browsable as the
    number of requests grows, and by request id so a lookup from the
    application database is a direct path rather than a search.
    """
    date = bundle.archived_at[:10]
    return f"{date}/{bundle.request_id}.json"


def archive_bundle(bundle: AuditBundle) -> str | None:
    """Write a bundle to Blob Storage. Returns its path, or None if archival
    is not configured.

    Never raises for a reason outside the caller's control — a missing
    configuration is normal locally, and the caller decides whether an
    actual write failure should be surfaced or only logged.
    """
    settings = get_settings()
    if not settings.archive_account_name:
        return None

    path = _blob_path(bundle)

    with span("archive.write", **{"rti.request_id": bundle.request_id}):
        try:
            client = BlobServiceClient(
                account_url=f"https://{settings.archive_account_name}.blob.core.windows.net",
                credential=DefaultAzureCredential(),
            )
            blob = client.get_blob_client(container=settings.archive_container_name, blob=path)
            blob.upload_blob(
                bundle.model_dump_json(indent=2),
                overwrite=True,
                content_settings=_json_content_settings(),
            )
        except AzureError as error:
            raise ArchiveError(f"could not write bundle {bundle.request_id}: {error}") from error

    return path


def _json_content_settings() -> Any:
    """Content type for a stored bundle, imported lazily.

    Deferred for the same reason the Application Insights import is: this
    pulls in azure-storage-blob's own dependency chain, and nothing in
    this module should fail to import just because that chain has an
    issue somewhere no one is currently using.
    """
    from azure.storage.blob import ContentSettings

    return ContentSettings(content_type="application/json")


async def archive_settled_request(state: "RequestState") -> str | None:
    """Archive a request's state once it has settled.

    Failure is logged, not raised: a person has already been told their
    request is settled, and an archival failure is an operational problem
    to notice, not something that should appear to them as an error in a
    process that actually succeeded.
    """
    from rti_engine.agents.state import current_status, current_tier

    tier = current_tier(state)
    bundle = build_bundle(
        request_id=state["request_id"],
        requester_employee_id=state["requester_employee_id"],
        tier=tier.value if tier else "unclassified",
        jurisdiction=state["jurisdiction"],
        status=current_status(state).value,
        request_text=state["request_text"],
        draft=state.get("draft"),
        number_check=state.get("number_check"),
        review=state.get("review"),
        approval_decision=state.get("approval_decision"),
        approved_by=state.get("approved_by"),
        revision_count=state.get("revision_count", 0),
        tokens_used=state.get("tokens_used", 0),
        cost_usd=state.get("cost_usd", 0.0),
        audit=state.get("audit", []),
    )

    try:
        path = await asyncio.to_thread(archive_bundle, bundle)
    except Exception:
        logger.exception("archival failed for request %s", bundle.request_id)
        return None

    if path is None:
        logger.info("archival skipped for request %s: not configured", bundle.request_id)
    else:
        logger.info("archived request %s to %s", bundle.request_id, path)
    return path

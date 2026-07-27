"""Application tables: request lifecycle, approvals, audit trail, documents.

These hold what the system must remember between turns and what a
reviewer may later require it to produce. The analytical dataset is not
stored here — it is read-only input. What lives here is the record of
what was asked, what was decided, by whom, and what was sent.

Timestamps are timezone-aware throughout: statutory deadlines computed
against naive timestamps are a defect waiting to happen.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base shared by every application table."""


def enum_values(enum_class: type[enum.Enum]) -> list[str]:
    """Store enum values in the database rather than Python member names.

    Without this, SQLAlchemy persists ``RECEIVED`` while the application and
    its API use ``received``. One spelling everywhere is worth the helper.
    """
    return [member.value for member in enum_class]


class AutonomyTier(enum.StrEnum):
    """How much independence the system has for a given request.

    T0 informational and T1 own-data requests are handled autonomously.
    T2 covers statutory disclosure and always requires human approval.
    """

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"


class RequestStatus(enum.StrEnum):
    """Where a request currently sits in its lifecycle."""

    RECEIVED = "received"
    CLASSIFYING = "classifying"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalDecision(enum.StrEnum):
    """The outcome of a human review."""

    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class ActorType(enum.StrEnum):
    """Whether an audited action was taken by software or a person."""

    AGENT = "agent"
    HUMAN = "human"
    SYSTEM = "system"


class DocumentKind(enum.StrEnum):
    """What a stored document is."""

    RESPONSE_LETTER = "response_letter"
    AUDIT_BUNDLE = "audit_bundle"


class Request(Base):
    """One pay-information request and its current state."""

    __tablename__ = "requests"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    requester_employee_id: Mapped[str] = mapped_column(String(32), index=True)
    """The employee making the request. Also the authorization scope for T1."""

    request_text: Mapped[str] = mapped_column(Text)
    """The request as written, kept verbatim. Never overwritten."""

    tier: Mapped[AutonomyTier | None] = mapped_column(
        Enum(AutonomyTier, name="autonomy_tier", values_callable=enum_values),
        nullable=True,
        index=True,
    )
    """Null until the intake agent has classified the request."""

    status: Mapped[RequestStatus] = mapped_column(
        Enum(RequestStatus, name="request_status", values_callable=enum_values),
        default=RequestStatus.RECEIVED,
        index=True,
    )

    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    """LangGraph checkpoint thread, so an interrupted run can be resumed."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    approvals: Mapped[list["Approval"]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )
    audit_events: Mapped[list["AuditEvent"]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )
    documents: Mapped[list["GeneratedDocument"]] = relationship(
        back_populates="request", cascade="all, delete-orphan"
    )


class Approval(Base):
    """A human decision on a request that could not proceed autonomously."""

    __tablename__ = "approvals"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"), index=True
    )

    decision: Mapped[ApprovalDecision] = mapped_column(
        Enum(ApprovalDecision, name="approval_decision", values_callable=enum_values)
    )
    reviewer_id: Mapped[str] = mapped_column(String(64))
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    request: Mapped["Request"] = relationship(back_populates="approvals")


class AuditEvent(Base):
    """One append-only record of something the system or a person did.

    Written once and never modified. This is the table that answers "why
    did the system reach that conclusion" after the fact.
    """

    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"), index=True
    )

    actor_type: Mapped[ActorType] = mapped_column(
        Enum(ActorType, name="actor_type", values_callable=enum_values)
    )

    actor: Mapped[str] = mapped_column(String(64))
    """Agent name, human identifier, or subsystem name."""

    action: Mapped[str] = mapped_column(String(64), index=True)
    """Short machine-readable verb, for example tier_assigned or tool_called."""

    detail: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    """Structured payload: tool inputs and outputs, statistics, reasons."""

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    request: Mapped["Request"] = relationship(back_populates="audit_events")

    __table_args__ = (Index("ix_audit_events_request_time", "request_id", "occurred_at"),)


class GeneratedDocument(Base):
    """A document produced for a request, stored exactly as issued."""

    __tablename__ = "generated_documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"), index=True
    )

    kind: Mapped[DocumentKind] = mapped_column(
        Enum(DocumentKind, name="document_kind", values_callable=enum_values)
    )
    content: Mapped[str] = mapped_column(Text)

    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    """SHA-256 of the content, so an issued document can be proven unaltered."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    request: Mapped["Request"] = relationship(back_populates="documents")

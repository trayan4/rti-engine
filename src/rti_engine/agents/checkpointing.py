"""Graph state persistence.

Without a checkpointer the graph holds state in memory, so a run that
ends takes its state with it. With one, state is written to Postgres
after every node — which is what lets a Tier 2 request stop for human
approval, let the process exit, and resume days later somewhere else.

A run is identified by its thread id. The request id serves as that, so
resuming a request means invoking the graph again with the same id rather
than reconstructing anything.

The DSN needs care: SQLAlchemy requires the driver named in the URL
scheme, and psycopg's own parser rejects that form. The two consumers
need different spellings of the same connection.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from rti_engine.db.session import PLAIN_PREFIX, PSYCOPG_PREFIX, resolve_dsn

ALLOWED_TYPES: list[tuple[str, str]] = [
    ("rti_engine.agents.state", "Actor"),
    ("rti_engine.agents.state", "AuditEntry"),
    ("rti_engine.agents.intake", "IntakeClassification"),
    ("rti_engine.agents.intake", "IntakeResult"),
    ("rti_engine.agents.analyst", "GroupAnalysis"),
    ("rti_engine.agents.analyst", "RequesterRecord"),
    ("rti_engine.agents.regulatory", "Citation"),
    ("rti_engine.agents.regulatory", "RegulatoryPosition"),
    ("rti_engine.agents.drafter", "DraftLetter"),
    ("rti_engine.agents.drafter", "FigureUse"),
    ("rti_engine.agents.drafter", "LetterSection"),
    ("rti_engine.agents.reviewer", "Finding"),
    ("rti_engine.agents.reviewer", "ReviewResult"),
    ("rti_engine.db.models", "AutonomyTier"),
    ("rti_engine.db.models", "RequestStatus"),
]
"""Types the checkpointer may reconstruct from stored state.

Without an allowlist the serializer will rebuild any Python type named in
a checkpoint, which makes write access to the checkpoint database into
code execution. Listing the types explicitly means a checkpoint naming
anything else is refused rather than instantiated.

Adding a typed field to the state means adding it here. That is friction,
and it is the point.
"""


def serializer() -> JsonPlusSerializer:
    """Return a serializer restricted to this application's own types."""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)


def checkpointer_dsn() -> str:
    """Return the connection string in the form psycopg accepts.

    ``resolve_dsn`` names the driver for SQLAlchemy's benefit. psycopg
    parses the URL itself and treats ``postgresql+psycopg://`` as an
    unknown scheme, so the driver suffix comes back off here.
    """
    dsn = resolve_dsn()
    if dsn.startswith(PSYCOPG_PREFIX):
        return PLAIN_PREFIX + dsn[len(PSYCOPG_PREFIX) :]
    return dsn


@asynccontextmanager
async def checkpointer() -> AsyncGenerator[AsyncPostgresSaver]:
    """Open a checkpointer for the duration of a caller's work.

    The connection is closed on exit. Callers that run many requests
    should hold one open across all of them rather than reopening per
    request.
    """
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn(), serde=serializer()) as saver:
        yield saver


async def setup_checkpointer() -> None:
    """Create the checkpoint tables if they do not exist.

    Idempotent, and separate from opening a connection: schema creation is
    a deployment action, not something every process should attempt at
    startup.
    """
    async with checkpointer() as saver:
        await saver.setup()

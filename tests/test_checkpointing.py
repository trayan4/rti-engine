"""Tests for graph state persistence.

The DSN and allowlist tests are pure. The round-trip test needs Postgres
and is skipped without it — what it proves is the property the tier 2
design depends on: that a request can be written by one process and read
back by another that never ran it.
"""

import pytest

from rti_engine.agents.checkpointing import (
    ALLOWED_TYPES,
    checkpointer,
    checkpointer_dsn,
    serializer,
)
from rti_engine.agents.graph import INTAKE, build_graph
from rti_engine.agents.state import Actor, AuditEntry, current_tier, initial_state
from rti_engine.config.settings import get_settings
from rti_engine.db.models import AutonomyTier, RequestStatus

needs_postgres = pytest.mark.skipif(
    not get_settings().postgres_dsn, reason="Postgres is not configured"
)


# --- the connection string ---


@needs_postgres
def test_the_driver_suffix_is_removed_for_psycopg() -> None:
    """SQLAlchemy needs the driver named; psycopg rejects that form."""
    dsn = checkpointer_dsn()
    assert dsn.startswith("postgresql://")
    assert "+psycopg" not in dsn


# --- the allowlist ---


def test_every_state_type_is_allowlisted() -> None:
    """A type missing here is refused at load, not silently reconstructed."""
    names = {name for _, name in ALLOWED_TYPES}

    assert {
        "AuditEntry",
        "IntakeResult",
        "GroupAnalysis",
        "RegulatoryPosition",
        "DraftLetter",
        "ReviewResult",
        "AutonomyTier",
        "RequestStatus",
    } <= names


def test_the_allowlist_holds_only_this_application() -> None:
    """Widening it to a third-party module would reopen what it closes."""
    assert all(module.startswith("rti_engine.") for module, _ in ALLOWED_TYPES)


def test_the_serializer_round_trips_an_allowlisted_type() -> None:
    """An omission here makes every existing checkpoint unreadable."""
    serde = serializer()
    entry = AuditEntry(actor=Actor.INTAKE, action="tier_assigned", detail={"tier": "T2"})

    restored = serde.loads_typed(serde.dumps_typed(entry))

    assert isinstance(restored, AuditEntry)
    assert restored.actor is Actor.INTAKE
    assert restored.detail == {"tier": "T2"}


# --- tier coercion ---


def test_a_stored_tier_string_becomes_an_enum() -> None:
    """msgpack returns a string enum as a plain string; .value then fails."""
    state = initial_state("req-1", "EMP-00001", "text", "DE")
    state["tier"] = "T2"  # type: ignore[typeddict-item]

    assert current_tier(state) is AutonomyTier.T2


def test_an_enum_tier_passes_through() -> None:
    state = initial_state("req-1", "EMP-00001", "text", "DE")
    state["tier"] = AutonomyTier.T1

    assert current_tier(state) is AutonomyTier.T1


def test_an_unclassified_request_has_no_tier() -> None:
    assert current_tier(initial_state("req-1", "EMP-00001", "text", "DE")) is None


# --- the round trip ---


@needs_postgres
async def test_state_survives_the_process_that_wrote_it() -> None:
    """The property tier 2 approval rests on: pause, exit, resume."""
    thread = "test-round-trip"
    config = {"configurable": {"thread_id": thread}}

    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        await graph.aupdate_state(
            config,
            {
                "request_id": thread,
                "requester_employee_id": "EMP-00001",
                "request_text": "What is my salary?",
                "jurisdiction": "DE",
                "tier": AutonomyTier.T2,
                "status": RequestStatus.AWAITING_APPROVAL,
            },
            as_node=INTAKE,
        )

    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        recovered = (await graph.aget_state(config)).values

    assert recovered["request_text"] == "What is my salary?"
    assert current_tier(recovered) is AutonomyTier.T2
    assert recovered["status"] == RequestStatus.AWAITING_APPROVAL


@needs_postgres
async def test_an_unknown_thread_recovers_nothing() -> None:
    """A missing thread is empty, not an error — nothing to resume."""
    async with checkpointer() as saver:
        graph = build_graph(checkpointer=saver)
        snapshot = await graph.aget_state({"configurable": {"thread_id": "no-such-thread"}})

    assert not snapshot.values

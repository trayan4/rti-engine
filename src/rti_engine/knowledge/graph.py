"""Neo4j connection and schema for the knowledge graph.

The graph holds what the vector store cannot represent: which national
provision corresponds to which article of the directive, which company
policy section implements which obligation, and which articles reference
each other. Similarity search finds text resembling a query; only a
traversal answers "what does this correspond to".

The graph is authored, not extracted. Every node and edge is written by
hand from the source documents, for the same reason the statistics are
pure Python: a relationship inferred by a model can be wrong without
anything downstream noticing, and the value of the graph lies in its
claims being checkable.

Constraints are applied idempotently so ingestion can be re-run after a
change without duplicating nodes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from neo4j import Driver, GraphDatabase, Session

from rti_engine.config.settings import get_settings

ARTICLE = "Article"
JURISDICTION = "Jurisdiction"
NATIONAL_PROVISION = "NationalProvision"
POLICY_SECTION = "PolicySection"

UNIQUE_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    (ARTICLE, "number"),
    (JURISDICTION, "code"),
    (NATIONAL_PROVISION, "provision_id"),
    (POLICY_SECTION, "number"),
)
"""Node label and the property that identifies it uniquely.

Uniqueness is what makes ingestion idempotent: a MERGE on a constrained
property updates the existing node rather than creating a second one.
"""


class GraphConfigurationError(RuntimeError):
    """Raised when the graph is used without being configured."""


def _require(value: str | None, name: str) -> str:
    """Return a required setting, or fail with a message naming it."""
    if not value:
        raise GraphConfigurationError(f"{name} is not set; check your .env file")
    return value


@lru_cache
def get_driver() -> Driver:
    """Return the process-wide driver, created on first use.

    The driver owns a connection pool and is thread-safe; one per process
    is both sufficient and correct.
    """
    settings = get_settings()
    return GraphDatabase.driver(
        _require(settings.neo4j_uri, "NEO4J_URI"),
        auth=(
            _require(settings.neo4j_username, "NEO4J_USERNAME"),
            _require(settings.neo4j_password, "NEO4J_PASSWORD"),
        ),
    )


@contextmanager
def graph_session() -> Iterator[Session]:
    """Provide a session, closed on exit."""
    session = get_driver().session()
    try:
        yield session
    finally:
        session.close()


def verify_connectivity() -> None:
    """Raise if the database is unreachable or the credentials are wrong."""
    get_driver().verify_connectivity()


def apply_schema() -> None:
    """Create the uniqueness constraints, if they do not already exist."""
    with graph_session() as session:
        for label, prop in UNIQUE_CONSTRAINTS:
            name = f"unique_{label.lower()}_{prop}"
            session.run(
                f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )


def clear_graph() -> None:
    """Delete every node and relationship.

    Used before a full re-ingest. Constraints survive: they are schema, not
    data.
    """
    with graph_session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def graph_summary() -> dict[str, Any]:
    """Return node and relationship counts by type, for verification."""
    with graph_session() as session:
        nodes = {
            str(record["label"]): int(record["count"])
            for record in session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label"
            )
        }
        relationships = {
            str(record["type"]): int(record["count"])
            for record in session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
            )
        }

    return {"nodes": nodes, "relationships": relationships}

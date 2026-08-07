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

Reached over HTTP rather than the Bolt driver's binary protocol. Bolt is
raw TCP, and this deployment's environment cannot route TCP-transport
ingress between container apps reliably — confirmed a platform-level gap,
not a configuration one, after eliminating every configuration cause.
HTTP rides the same ingress path already proven for the API and both MCP
servers.

Constraints are applied idempotently so ingestion can be re-run after a
change without duplicating nodes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import httpx

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

DEFAULT_DATABASE = "neo4j"

CONNECT_TIMEOUT_SECONDS = 20.0
"""Azure's internal HTTP ingress has occasionally taken longer than a
single connection attempt should reasonably need — a platform behaviour
seen during deployment, not a property of this database. Generous enough
to absorb that without masking a genuine outage."""

READ_TIMEOUT_SECONDS = 15.0
"""These are six hand-authored templates against a small graph. A read
this slow means something is actually wrong, not that the query needs
more time."""


class GraphConfigurationError(RuntimeError):
    """Raised when the graph is used without being configured."""


class GraphQueryError(RuntimeError):
    """Raised when Neo4j's HTTP endpoint reports a query error.

    Kept distinct from a connectivity failure: this means the request
    reached the server and was rejected — a syntax error or a constraint
    violation — not that the server could not be reached at all.
    """


def _require(value: str | None, name: str) -> str:
    """Return a required setting, or fail with a message naming it."""
    if not value:
        raise GraphConfigurationError(f"{name} is not set; check your .env file")
    return value


class GraphSession:
    """A thin client over Neo4j's HTTP Cypher transaction endpoint.

    One statement per call, auto-committed. Nothing here holds a
    transaction open across calls, so there is no session state that
    could leak between agents sharing a process — a property the Bolt
    driver's session object had to be closed to guarantee, and this does
    not need to.
    """

    def __init__(self, client: httpx.Client, endpoint: str) -> None:
        self._client = client
        self._endpoint = endpoint

    def run(self, cypher: str, **parameters: Any) -> list[dict[str, Any]]:
        """Run one statement and return its rows as plain dictionaries."""
        response = self._client.post(
            self._endpoint,
            json={"statements": [{"statement": cypher, "parameters": parameters}]},
        )
        response.raise_for_status()
        body = response.json()

        if errors := body.get("errors"):
            raise GraphQueryError("; ".join(e.get("message", str(e)) for e in errors))

        rows: list[dict[str, Any]] = []
        for result in body.get("results", []):
            columns = result.get("columns", [])
            for record in result.get("data", []):
                rows.append(dict(zip(columns, record.get("row", []), strict=True)))
        return rows


@lru_cache
def _get_client() -> httpx.Client:
    """Return the process-wide HTTP client, created on first use.

    One client per process is both sufficient and correct: httpx.Client
    pools connections internally the same way the Bolt driver pooled its
    own.
    """
    settings = get_settings()
    return httpx.Client(
        auth=(
            _require(settings.neo4j_username, "NEO4J_USERNAME"),
            _require(settings.neo4j_password, "NEO4J_PASSWORD"),
        ),
        timeout=httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS, read=READ_TIMEOUT_SECONDS, write=10.0, pool=10.0
        ),
    )


def _endpoint() -> str:
    """Return the Cypher transaction endpoint for the configured server."""
    settings = get_settings()
    base = _require(settings.neo4j_uri, "NEO4J_URI").rstrip("/")
    return f"{base}/db/{DEFAULT_DATABASE}/tx/commit"


@contextmanager
def graph_session() -> Iterator[GraphSession]:
    """Provide a session. Nothing to close: each call is its own request."""
    yield GraphSession(_get_client(), _endpoint())


def verify_connectivity() -> None:
    """Raise if the database is unreachable or the credentials are wrong."""
    with graph_session() as session:
        session.run("RETURN 1")


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

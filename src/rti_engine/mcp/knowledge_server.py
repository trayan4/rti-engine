"""MCP server exposing the regulatory knowledge layer as tools.

Two kinds of access, deliberately separated.

Retrieval finds passages resembling a question. It is quota-balanced, so
the national status notes are always represented alongside the directive
and the company policy — without that, a question phrased in legal
language returns only the directive, and the system would answer as
though it were in force everywhere.

Graph queries answer questions similarity cannot: what a national
provision corresponds to, what an article is qualified by, which policy
section implements an obligation. Each is a fixed template. Agents choose
from a menu and cannot compose Cypher, for the same reason they cannot
compose SQL.

Neither surface returns employee data. Personal pay information lives
behind the analytics server and its authorization checks.
"""

from typing import Any, Literal

from fastmcp import FastMCP

from rti_engine.knowledge.graph_queries import run_query
from rti_engine.knowledge.vectorstore import retrieve

KNOWN_JURISDICTIONS: frozenset[str] = frozenset({"DE", "FR", "ES"})
"""Countries the employer operates in, and the only ones with status notes."""

MIN_ARTICLE = 1
MAX_ARTICLE = 37

Jurisdiction = Literal["DE", "FR", "ES"]
"""Constrains the tool schema so a model cannot emit an invalid country.

Given a bare string type, a model asked about Spain sends "Spain". The
runtime check still stands as defence in depth, but a value the schema
forbids never reaches it.
"""

mcp: FastMCP[Any] = FastMCP("rti-knowledge")


class KnowledgeRequestError(ValueError):
    """Raised when a request names something outside the known corpus."""


def _jurisdiction(code: str) -> str:
    """Validate a country code against the corpus.

    Refused rather than returned empty: a model reading an empty result
    would conclude no national law exists, which is a different and much
    more damaging claim than "that country is not covered here".
    """
    normalised = code.strip().upper()
    if normalised not in KNOWN_JURISDICTIONS:
        permitted = ", ".join(sorted(KNOWN_JURISDICTIONS))
        raise KnowledgeRequestError(f"unknown jurisdiction {code!r}; covered: {permitted}")
    return normalised


def _article(number: int) -> int:
    """Validate an article number against the instrument."""
    if not MIN_ARTICLE <= number <= MAX_ARTICLE:
        raise KnowledgeRequestError(
            f"article {number} does not exist; the directive has {MIN_ARTICLE}-{MAX_ARTICLE}"
        )
    return number


@mcp.tool()
def search_regulatory_knowledge(query: str, jurisdiction: Jurisdiction) -> list[dict[str, Any]]:
    """Find passages relevant to a question, scoped to one country.

    Returns the directive, the company policy, and that country's national
    status note. Other countries' law is excluded: it does not apply to
    this requester and citing it would be wrong.

    Every result carries a citation. Any statement drawn from these
    passages must be attributed to the citation given, not to the corpus
    generally.
    """
    if not query.strip():
        raise KnowledgeRequestError("query is empty")

    results = retrieve(query, jurisdiction=_jurisdiction(jurisdiction))

    return [
        {
            "citation": chunk.citation,
            "text": chunk.text,
            "document_kind": chunk.document_kind,
            "jurisdiction": chunk.jurisdiction,
            "score": round(chunk.score, 4),
        }
        for chunk in results
    ]


@mcp.tool()
def get_jurisdiction_status(jurisdiction: Jurisdiction) -> dict[str, Any]:
    """Report whether a country has transposed the directive.

    The first question any regulatory reasoning must settle: an obligation
    under the directive is not necessarily an obligation under the law a
    given employer is subject to today.
    """
    rows = run_query("jurisdiction_status", jurisdiction=_jurisdiction(jurisdiction))
    if not rows:
        raise KnowledgeRequestError(f"no status recorded for {jurisdiction}")
    return rows[0]


@mcp.tool()
def get_national_position_on_article(
    jurisdiction: Jurisdiction, article: int
) -> list[dict[str, Any]]:
    """Report what national law currently provides in respect of one article.

    An empty result means the country has no corresponding provision,
    which is a finding in itself rather than a failure to look.
    """
    return run_query(
        "national_position_on_article",
        jurisdiction=_jurisdiction(jurisdiction),
        article=_article(article),
    )


@mcp.tool()
def get_article_context(article: int) -> dict[str, Any]:
    """Return an article with the articles it references and is referenced by.

    A provision read alone can be read wrongly: the articles citing it are
    often what qualify or condition it.
    """
    rows = run_query("article_context", article=_article(article))
    if not rows:
        raise KnowledgeRequestError(f"article {article} not found in the graph")
    return rows[0]


@mcp.tool()
def get_policy_sections_for_article(article: int) -> list[dict[str, Any]]:
    """Return the employer's policy sections implementing an article.

    Lets a response point at the employer's own commitment rather than
    only at the law, which matters where national law does not yet compel
    the obligation.
    """
    return run_query("policy_sections_for_article", article=_article(article))


@mcp.tool()
def list_articles_without_national_basis(jurisdiction: Jurisdiction) -> list[dict[str, Any]]:
    """List transparency obligations with no counterpart in a country's law.

    The compliance gap: what the directive requires but national law does
    not yet compel in that country.
    """
    return run_query("articles_without_national_basis", jurisdiction=_jurisdiction(jurisdiction))


@mcp.tool()
def list_provisions_in_jurisdiction(jurisdiction: Jurisdiction) -> list[dict[str, Any]]:
    """List every national provision recorded for a country.

    Each carries its own instrument and threshold. National thresholds are
    not the directive's and must never be presented as such.
    """
    return run_query("provisions_in_jurisdiction", jurisdiction=_jurisdiction(jurisdiction))


if __name__ == "__main__":
    mcp.run()

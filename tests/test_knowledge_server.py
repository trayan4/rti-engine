"""Tests for the knowledge MCP server.

Two properties matter at this boundary. An agent must never be able to
reach another country's law, since both national notes discuss the same
directive and a model given the wrong one would answer fluently and
wrongly. And a request naming something outside the corpus must be
refused rather than answered emptily: an empty result reads as "no such
law exists", which is a far stronger claim than "not covered here".
"""

from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from rti_engine.config.settings import get_settings
from rti_engine.mcp.knowledge_server import KNOWN_JURISDICTIONS, mcp

settings = get_settings()

needs_graph = pytest.mark.skipif(
    not (settings.neo4j_uri and settings.neo4j_password),
    reason="Neo4j is not configured",
)

needs_vectors = pytest.mark.skipif(
    not (settings.pinecone_api_key and settings.pinecone_index),
    reason="Pinecone is not configured",
)

NATIONAL_QUERY = "Has this country transposed the directive into national law yet?"


async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Call one tool through an MCP client and return its structured result."""
    async with Client(mcp) as client:
        result = await client.call_tool(name, arguments)
    return result.data


async def test_every_tool_is_registered() -> None:
    async with Client(mcp) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert names == {
        "search_regulatory_knowledge",
        "get_jurisdiction_status",
        "get_national_position_on_article",
        "get_article_context",
        "get_policy_sections_for_article",
        "list_articles_without_national_basis",
        "list_provisions_in_jurisdiction",
    }


@needs_graph
@pytest.mark.parametrize("code", sorted(KNOWN_JURISDICTIONS))
async def test_no_country_is_reported_as_transposed(code: str) -> None:
    data = await call_tool("get_jurisdiction_status", {"jurisdiction": code})
    assert data["transposed"] is False
    assert data["status"]


@needs_graph
async def test_spanish_position_on_the_right_to_information() -> None:
    rows = await call_tool("get_national_position_on_article", {"jurisdiction": "ES", "article": 7})
    assert rows
    assert any("registro" in str(row["provision"]).lower() for row in rows)


@needs_graph
async def test_national_thresholds_stay_attached_to_their_instrument() -> None:
    """Spain's 25% trigger must never be presented as the directive's 5%."""
    rows = await call_tool("list_provisions_in_jurisdiction", {"jurisdiction": "ES"})
    with_threshold = [row for row in rows if row["threshold"]]

    assert with_threshold
    assert all(row["instrument"] for row in with_threshold)
    assert any("25" in str(row["threshold"]) for row in with_threshold)


@needs_graph
async def test_the_compliance_gap_is_reportable() -> None:
    """Obligations with no national counterpart are a finding, not an absence."""
    rows = await call_tool("list_articles_without_national_basis", {"jurisdiction": "ES"})
    numbers = {row["article"] for row in rows}

    assert 5 in numbers
    assert 7 not in numbers


@needs_graph
async def test_article_context_returns_both_directions() -> None:
    data = await call_tool("get_article_context", {"article": 7})
    assert {8, 12, 18} <= set(data["referenced_by"])


@needs_graph
async def test_policy_sections_are_traceable_to_an_article() -> None:
    rows = await call_tool("get_policy_sections_for_article", {"article": 7})
    assert 8 in {row["section"] for row in rows}


@needs_vectors
@pytest.mark.parametrize("code", sorted(KNOWN_JURISDICTIONS))
async def test_retrieval_reaches_the_national_note(code: str) -> None:
    hits = await call_tool(
        "search_regulatory_knowledge", {"query": NATIONAL_QUERY, "jurisdiction": code}
    )
    assert any(hit["jurisdiction"] == code for hit in hits)


@needs_vectors
@pytest.mark.parametrize("code", sorted(KNOWN_JURISDICTIONS))
async def test_retrieval_never_returns_another_country(code: str) -> None:
    foreign = KNOWN_JURISDICTIONS - {code}
    hits = await call_tool(
        "search_regulatory_knowledge", {"query": NATIONAL_QUERY, "jurisdiction": code}
    )

    assert hits
    assert not (foreign & {hit["jurisdiction"] for hit in hits})


@needs_vectors
async def test_every_passage_carries_a_citation() -> None:
    """A statement drawn from a passage must be attributable to a source."""
    hits = await call_tool(
        "search_regulatory_knowledge",
        {"query": "right to request pay information", "jurisdiction": "DE"},
    )
    assert hits
    assert all(hit["citation"].strip() for hit in hits)


async def test_an_unknown_jurisdiction_is_refused() -> None:
    """Refused, not empty: an empty result reads as "no such law exists"."""
    with pytest.raises(ToolError, match="Input should be"):
        await call_tool("get_jurisdiction_status", {"jurisdiction": "XX"})


async def test_an_article_outside_the_instrument_is_refused() -> None:
    with pytest.raises(ToolError, match="does not exist"):
        await call_tool("get_article_context", {"article": 99})

    with pytest.raises(ToolError, match="does not exist"):
        await call_tool("get_policy_sections_for_article", {"article": 0})


@needs_vectors
async def test_an_empty_query_is_refused() -> None:
    with pytest.raises(ToolError, match="query is empty"):
        await call_tool("search_regulatory_knowledge", {"query": "   ", "jurisdiction": "DE"})


async def test_the_schema_admits_only_exact_country_codes() -> None:
    """Casing is no longer normalised at the tool surface.

    The schema enumerates the permitted codes, so a model is shown exactly
    what to send and cannot send anything else. Accepting variants would
    widen the contract for no benefit, since the caller is never guessing.
    """
    with pytest.raises(ToolError, match="Input should be"):
        await call_tool("get_jurisdiction_status", {"jurisdiction": "es"})

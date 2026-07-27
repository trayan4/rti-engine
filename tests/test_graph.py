"""Tests for the knowledge graph and its query templates.

Reference extraction is tested offline: it is pure text parsing, and the
enumeration bug it guards against was invisible until the graph was
inspected by hand.

The query tests need a live database and are skipped without one. What
they assert is that agents reach the graph only through named templates,
and that anything else is refused rather than ignored.
"""

import pytest

from rti_engine.config.settings import get_settings
from rti_engine.knowledge.graph import graph_summary
from rti_engine.knowledge.graph_data import (
    JURISDICTIONS,
    NATIONAL_PROVISIONS,
    POLICY_SECTIONS,
)
from rti_engine.knowledge.graph_ingest import _extract_references, _numbers_in_list
from rti_engine.knowledge.graph_queries import (
    TEMPLATES,
    GraphQueryError,
    available_queries,
    run_query,
)

settings = get_settings()

needs_graph = pytest.mark.skipif(
    not (settings.neo4j_uri and settings.neo4j_password),
    reason="Neo4j is not configured",
)

EXPECTED_ARTICLE_COUNT = 37


# --- reference extraction (offline) ---


def test_enumerated_citations_are_all_captured() -> None:
    """The bug this guards: a pattern anchored on the singular "Article"
    misses every enumerated reference, which is how legislation usually
    cross-references."""
    text = "information shared pursuant to Articles 5, 6 and 7 in a format"
    assert _extract_references(text, source_number=8) == {5, 6, 7}


def test_singular_citations_still_work() -> None:
    assert _extract_references("as set out in Article 9 of this Directive", 10) == {9}


def test_inclusive_ranges_expand() -> None:
    assert _numbers_in_list("5 to 8") == {5, 6, 7, 8}


def test_paragraph_qualifiers_are_not_read_as_articles() -> None:
    """In "Article 7(2)" the 2 is a paragraph, not a reference to Article 2."""
    assert _numbers_in_list("7(2)") == {7}
    assert _extract_references("referred to in Article 7(2) and 9", 12) == {7, 9}


def test_citations_to_other_instruments_are_excluded() -> None:
    """The directive cites the Treaty and other directives by article number."""
    assert _extract_references("Article 258 TFEU applies", 5) == set()
    assert _extract_references("Articles 18(2) and 30 of Directive 2014/24/EU", 24) == set()


def test_an_article_does_not_reference_itself() -> None:
    assert _extract_references("as provided in Article 7 above", 7) == set()


# --- graph content (live) ---


@needs_graph
def test_the_graph_holds_every_article_and_authored_record() -> None:
    summary = graph_summary()
    nodes = summary["nodes"]

    assert nodes["Article"] == EXPECTED_ARTICLE_COUNT
    assert nodes["Jurisdiction"] == len(JURISDICTIONS)
    assert nodes["NationalProvision"] == len(NATIONAL_PROVISIONS)
    assert nodes["PolicySection"] == len(POLICY_SECTIONS)


@needs_graph
def test_article_8_references_the_articles_it_cites() -> None:
    """A regression test for the enumeration bug, asserted against the graph."""
    rows = run_query("article_context", article=8)
    assert set(rows[0]["references"]) == {5, 6, 7}


@needs_graph
def test_article_7_is_reached_by_the_provisions_that_qualify_it() -> None:
    rows = run_query("article_context", article=7)
    assert {8, 12, 18} <= set(rows[0]["referenced_by"])


@needs_graph
@pytest.mark.parametrize("code", ["DE", "FR", "ES"])
def test_no_country_is_recorded_as_transposed(code: str) -> None:
    """All three missed the deadline; a change here is a corpus update."""
    rows = run_query("jurisdiction_status", jurisdiction=code)
    assert len(rows) == 1
    assert rows[0]["transposed"] is False
    assert rows[0]["direct_effect_from"]


@needs_graph
def test_spanish_position_on_the_right_to_information() -> None:
    """The traversal behind any statement about what applies in Spain today."""
    rows = run_query("national_position_on_article", jurisdiction="ES", article=7)
    assert rows
    assert all(row["transposed"] is False for row in rows)
    assert any("registro" in str(row["provision"]).lower() for row in rows)


@needs_graph
def test_spanish_justification_threshold_is_attributed_to_spain() -> None:
    """Spain's 25% trigger must never be presented as the directive's 5%."""
    rows = run_query("provisions_in_jurisdiction", jurisdiction="ES")
    thresholds = " ".join(str(row["threshold"]) for row in rows)

    assert "25" in thresholds
    assert all(row["instrument"] for row in rows)


@needs_graph
def test_policy_sections_map_to_the_articles_they_implement() -> None:
    rows = run_query("policy_sections_for_article", article=7)
    assert 8 in {row["section"] for row in rows}


@needs_graph
def test_gaps_exclude_articles_that_do_have_a_national_basis() -> None:
    """Spain has a provision for Article 7, so Article 7 is not a gap there."""
    rows = run_query("articles_without_national_basis", jurisdiction="ES")
    gaps = {row["article"] for row in rows}
    assert 7 not in gaps
    assert 5 in gaps


# --- the template boundary ---


def test_every_template_declares_its_parameters() -> None:
    described = available_queries()
    assert len(described) == len(TEMPLATES)
    assert all(entry["description"] and entry["parameters"] is not None for entry in described)


def test_an_unknown_query_is_refused() -> None:
    """Agents choose from a menu; they cannot invent one."""
    with pytest.raises(GraphQueryError, match="unknown query"):
        run_query("MATCH (n) DETACH DELETE n")


def test_missing_parameters_are_refused() -> None:
    with pytest.raises(GraphQueryError, match="missing parameters"):
        run_query("jurisdiction_status")


def test_unexpected_parameters_are_refused() -> None:
    """An extra parameter is a smuggling attempt, not a refinement."""
    with pytest.raises(GraphQueryError, match="unexpected parameters"):
        run_query("article_context", article=7, limit=999)


def test_no_template_interpolates_a_string() -> None:
    """Every value must arrive as a bound parameter, never formatted in."""
    for template in TEMPLATES:
        assert "%" not in template.cypher
        assert "format(" not in template.cypher
        assert all(f"${name}" in template.cypher for name in template.parameters)

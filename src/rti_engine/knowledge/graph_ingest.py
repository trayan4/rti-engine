"""Build the knowledge graph from the directive and the authored records.

Three sources, with different degrees of judgment involved:

Article nodes are derived mechanically from the parsed directive.

Cross-references between articles are extracted from the text, because the
directive states them literally. References to other instruments are
excluded — the text cites the Treaty and the Charter by article number
too, and treating those as internal references would fabricate edges.

National provisions and policy mappings come from the authored records,
where the legal judgment involved has been made by a person.

Every write uses MERGE against a constrained property, so ingestion is
idempotent and can be re-run after any change.
"""

import re

from rti_engine.knowledge.directive import Section, parse_directive
from rti_engine.knowledge.graph import apply_schema, clear_graph, graph_session
from rti_engine.knowledge.graph_data import (
    JURISDICTIONS,
    NATIONAL_PROVISIONS,
    POLICY_SECTIONS,
)

ARTICLE_REFERENCE = re.compile(
    r"Articles?\s+(\d{1,3}(?:\([^)]*\))?(?:\s*(?:,|and|to)\s*\d{1,3}(?:\([^)]*\))?)*)"
)
"""Matches a citation and the whole list that follows it.

Legislation cross-references in enumerations — "pursuant to Articles 5, 6
and 7" — so a pattern anchored on the singular "Article" misses most of
them entirely, including every reference into Article 7.
"""

PARENTHETICAL = re.compile(r"\([^)]*\)")
"""Paragraph qualifiers such as the (2) in "Article 7(2)".

Stripped before numbers are read, or the paragraph number would be
mistaken for a reference to another article.
"""

LIST_TOKEN = re.compile(r"\d{1,3}|to")

FOREIGN_INSTRUMENT_MARKERS: tuple[str, ...] = (
    "TFEU",
    "of the Treaty",
    "of the Charter",
    "of Directive",
    "of Regulation",
    "of Council Directive",
    "of the United Nations",
    "of the European Convention",
)
"""Phrases that mark a citation to another instrument rather than to this one.

The directive cites the Treaty and the Charter by article number, so an
unfiltered extraction would create edges to articles that do not exist in
this instrument and imply relationships that are not there.
"""

REFERENCE_LOOKAHEAD = 60
"""Characters after a citation to inspect for a foreign-instrument marker."""

MAX_ARTICLE_NUMBER = 37


def _numbers_in_list(list_text: str) -> set[int]:
    """Read every article number from a citation list.

    Handles enumerations and inclusive ranges: "Articles 5, 6 and 7" gives
    three numbers, and "Articles 5 to 8" gives four.
    """
    tokens = LIST_TOKEN.findall(PARENTHETICAL.sub("", list_text))

    numbers: set[int] = set()
    previous: int | None = None
    pending_range = False

    for token in tokens:
        if token == "to":
            pending_range = True
            continue

        value = int(token)
        if pending_range and previous is not None and previous < value:
            numbers.update(range(previous, value + 1))
        else:
            numbers.add(value)

        previous = value
        pending_range = False

    return numbers


def _extract_references(text: str, source_number: int) -> set[int]:
    """Find articles of this directive referenced from a passage.

    The foreign-instrument check applies to the citation as a whole: in
    "Articles 18(2) and 30 of Directive 2014/24/EU" the marker follows the
    complete list, and every number in that list belongs to the other
    instrument.
    """
    found: set[int] = set()

    for match in ARTICLE_REFERENCE.finditer(text):
        following = text[match.end() : match.end() + REFERENCE_LOOKAHEAD]
        if any(marker in following for marker in FOREIGN_INSTRUMENT_MARKERS):
            continue

        for number in _numbers_in_list(match.group(1)):
            if number == source_number or not 1 <= number <= MAX_ARTICLE_NUMBER:
                continue
            found.add(number)

    return found


def _load_articles(sections: list[Section]) -> list[dict[str, object]]:
    """Shape article sections for the graph.

    The full text is not stored: it lives in the vector store, and holding
    two copies invites them to diverge. The graph holds structure.
    """
    return [
        {
            "number": section.number,
            "heading": section.heading or "",
            "citation": section.citation,
        }
        for section in sections
        if section.section_type == "article"
    ]


def ingest_graph(replace: bool = True) -> dict[str, int]:
    """Build the whole graph. Returns counts of what was written."""
    sections = parse_directive()
    articles = _load_articles(sections)

    references = [
        {"source": section.number, "target": target}
        for section in sections
        if section.section_type == "article"
        for target in sorted(_extract_references(section.text, section.number))
    ]

    if replace:
        clear_graph()
    apply_schema()

    with graph_session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (a:Article {number: row.number})
            SET a.heading = row.heading, a.citation = row.citation
            """,
            rows=articles,
        )

        session.run(
            """
            UNWIND $rows AS row
            MATCH (source:Article {number: row.source})
            MATCH (target:Article {number: row.target})
            MERGE (source)-[:REFERENCES]->(target)
            """,
            rows=references,
        )

        session.run(
            """
            UNWIND $rows AS row
            MERGE (j:Jurisdiction {code: row.code})
            SET j.name = row.name,
                j.transposed = row.transposed,
                j.status = row.status,
                j.expected = row.expected,
                j.direct_effect_from = row.direct_effect_from
            """,
            rows=[record.model_dump() for record in JURISDICTIONS],
        )

        session.run(
            """
            UNWIND $rows AS row
            MERGE (p:NationalProvision {provision_id: row.provision_id})
            SET p.instrument = row.instrument,
                p.title = row.title,
                p.summary = row.summary,
                p.threshold = row.threshold
            WITH p, row
            MATCH (j:Jurisdiction {code: row.jurisdiction})
            MERGE (p)-[:IN_JURISDICTION]->(j)
            WITH p, row
            UNWIND row.corresponds_to AS article_number
            MATCH (a:Article {number: article_number})
            MERGE (p)-[:CORRESPONDS_TO]->(a)
            """,
            rows=[record.model_dump() for record in NATIONAL_PROVISIONS],
        )

        session.run(
            """
            UNWIND $rows AS row
            MERGE (s:PolicySection {number: row.number})
            SET s.title = row.title
            WITH s, row
            UNWIND row.implements AS article_number
            MATCH (a:Article {number: article_number})
            MERGE (s)-[:IMPLEMENTS]->(a)
            """,
            rows=[record.model_dump() for record in POLICY_SECTIONS],
        )

    return {
        "articles": len(articles),
        "references": len(references),
        "jurisdictions": len(JURISDICTIONS),
        "provisions": len(NATIONAL_PROVISIONS),
        "policy_sections": len(POLICY_SECTIONS),
    }


def main() -> None:
    """Build the graph and report what was written."""
    counts = ingest_graph()
    for name, count in counts.items():
        print(f"{name:16s} {count}")


if __name__ == "__main__":
    main()

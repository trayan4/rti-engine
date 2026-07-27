"""Tests for the full knowledge corpus.

The property that matters most here is jurisdictional isolation. A Spanish
requester answered with German law would be given a confidently wrong
answer, and since both documents discuss the same directive, embedding
similarity alone will not separate them. The jurisdiction tag is what
makes that filterable, so it has to be correct on every chunk.
"""

import pytest

from rti_engine.knowledge.chunking import MAX_CHUNK_TOKENS, Chunk
from rti_engine.knowledge.corpus import (
    MARKDOWN_DOCUMENTS,
    chunk_corpus,
    chunk_document,
)
from rti_engine.knowledge.directive import CELEX_ID

EXPECTED_DOCUMENT_COUNT = 5


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return chunk_corpus()


def test_every_corpus_file_exists() -> None:
    """A missing file would silently shrink the corpus rather than fail."""
    for document in MARKDOWN_DOCUMENTS:
        assert document.path.is_file(), f"missing corpus file: {document.path}"


def test_all_five_documents_are_represented(chunks: list[Chunk]) -> None:
    document_ids = {c.document_id for c in chunks}
    assert len(document_ids) == EXPECTED_DOCUMENT_COUNT
    assert CELEX_ID in document_ids


def test_no_chunk_exceeds_the_budget(chunks: list[Chunk]) -> None:
    assert all(c.token_count <= MAX_CHUNK_TOKENS for c in chunks)


def test_chunk_ids_are_unique_across_documents(chunks: list[Chunk]) -> None:
    """Ids are vector store keys; a collision across documents loses content."""
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_national_notes_carry_their_jurisdiction(chunks: list[Chunk]) -> None:
    for code in ("DE", "FR", "ES"):
        matching = [c for c in chunks if c.jurisdiction == code]
        assert matching, f"no chunks tagged {code}"
        assert all(c.document_kind == "national_status" for c in matching)


def test_the_directive_is_not_tagged_to_one_country(chunks: list[Chunk]) -> None:
    """It applies across the Union; tagging it would hide it from filters."""
    directive = [c for c in chunks if c.document_id == CELEX_ID]
    assert all(c.jurisdiction is None for c in directive)
    assert all(c.document_kind == "legislation" for c in directive)


def test_filtering_by_jurisdiction_excludes_other_countries(
    chunks: list[Chunk],
) -> None:
    """The isolation property: Spanish retrieval must not surface German law."""
    spanish_view = [c for c in chunks if c.jurisdiction in (None, "ES")]

    assert not any(c.jurisdiction == "DE" for c in spanish_view)
    assert not any(c.jurisdiction == "FR" for c in spanish_view)
    assert any(c.jurisdiction == "ES" for c in spanish_view)
    assert any(c.document_id == CELEX_ID for c in spanish_view)


def test_spanish_note_contains_its_own_threshold(chunks: list[Chunk]) -> None:
    """Spain's 25% justification trigger must be retrievable and attributed."""
    spanish = [c for c in chunks if c.jurisdiction == "ES"]
    matching = [c for c in spanish if "25%" in c.text]

    assert matching, "the 25% justification threshold is not in any chunk"
    assert all("Spain" in c.citation for c in matching)


def test_company_policy_defines_its_comparator_categories(
    chunks: list[Chunk],
) -> None:
    """The policy's category definition must match the analytics grouping."""
    policy = [c for c in chunks if c.document_kind == "company_policy"]
    text = "\n".join(c.text for c in policy).lower()

    assert "work of equal value" in text
    assert "job family" in text
    assert "full-time equivalent" in text


def test_every_chunk_is_citable(chunks: list[Chunk]) -> None:
    assert all(c.citation.strip() for c in chunks)
    assert all(c.section_number > 0 for c in chunks)


def test_document_preamble_is_not_discarded() -> None:
    """The status date sits above the first heading and must survive chunking."""
    germany = next(d for d in MARKDOWN_DOCUMENTS if d.jurisdiction == "DE")
    produced = chunk_document(germany)

    assert produced
    assert "Status as at" in produced[0].text


def test_section_headings_reach_the_citation() -> None:
    policy = next(d for d in MARKDOWN_DOCUMENTS if d.kind == "company_policy")
    produced = chunk_document(policy)
    citations = {c.citation for c in produced}

    assert any("Pay gap monitoring" in citation for citation in citations)

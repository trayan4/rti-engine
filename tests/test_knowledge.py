"""Tests for directive parsing and chunking.

Two properties matter here. The parse must recover the instrument's legal
structure intact, because a missing or merged article is a hole in the
system's knowledge that nothing downstream would notice. And every chunk
must remain citable, because a retrieved fragment that cannot name its
article is unusable in a document that has to be checkable.

These run against the real corpus file, which is committed to the repo.
"""

import pytest

from rti_engine.knowledge.chunking import (
    MAX_CHUNK_TOKENS,
    Chunk,
    chunk_directive,
    chunk_section,
    count_tokens,
)
from rti_engine.knowledge.directive import (
    CELEX_ID,
    INSTRUMENT,
    Section,
    parse_directive,
)

EXPECTED_ARTICLE_COUNT = 37
LAST_ARTICLE = 37


@pytest.fixture(scope="module")
def sections() -> list[Section]:
    return parse_directive()


@pytest.fixture(scope="module")
def articles(sections: list[Section]) -> list[Section]:
    return [s for s in sections if s.section_type == "article"]


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return chunk_directive()


def test_every_article_is_present_and_in_sequence(articles: list[Section]) -> None:
    numbers = [a.number for a in articles]
    assert len(articles) == EXPECTED_ARTICLE_COUNT
    assert numbers == list(range(1, LAST_ARTICLE + 1))


def test_recitals_are_parsed_from_the_preamble(sections: list[Section]) -> None:
    recitals = [s for s in sections if s.section_type == "recital"]
    assert len(recitals) > 50
    assert all(r.number > 0 for r in recitals)


def test_articles_carry_their_headings(articles: list[Section]) -> None:
    """Headings use non-breaking spaces; a failure here means normalisation broke."""
    by_number = {a.number: a for a in articles}
    assert by_number[7].heading == "Right to information"
    assert by_number[10].heading == "Joint pay assessment"
    assert by_number[34].heading == "Transposition"


def test_article_7_contains_the_right_to_request_pay_information(
    articles: list[Section],
) -> None:
    """The provision every request in this system ultimately rests on."""
    seven = next(a for a in articles if a.number == 7)
    assert "right to request" in seven.text.lower()
    assert "broken down by sex" in seven.text.lower()


def test_the_signature_block_is_not_treated_as_law(articles: list[Section]) -> None:
    """Left in place, the signature and footnotes land inside the last article."""
    last = next(a for a in articles if a.number == LAST_ARTICLE)
    assert len(last.text) < 200
    assert "Done at Strasbourg" not in last.text
    assert "METSOLA" not in last.text


def test_no_section_is_suspiciously_empty(sections: list[Section]) -> None:
    """A near-empty section means a split landed in the wrong place."""
    assert all(len(s.text) >= 40 for s in sections)


def test_citations_name_the_instrument(articles: list[Section]) -> None:
    seven = next(a for a in articles if a.number == 7)
    assert seven.citation == f"{INSTRUMENT}, Article 7"


def test_no_chunk_exceeds_the_token_budget(chunks: list[Chunk]) -> None:
    assert all(c.token_count <= MAX_CHUNK_TOKENS for c in chunks)


def test_chunk_ids_are_unique(chunks: list[Chunk]) -> None:
    """Ids become vector store keys; a collision silently overwrites content."""
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_every_chunk_is_citable(chunks: list[Chunk]) -> None:
    assert all(c.citation and c.section_number > 0 for c in chunks)
    assert all(c.document_id == CELEX_ID for c in chunks)


def test_long_articles_split_and_short_ones_do_not(chunks: list[Chunk]) -> None:
    counts: dict[int, int] = {}
    for chunk in chunks:
        if chunk.section_type == "article":
            counts[chunk.section_number] = counts.get(chunk.section_number, 0) + 1

    assert counts[3] > 1, "the definitions article is too long for one chunk"
    assert counts[36] == 1, "a one-sentence article should not be split"


def test_a_short_section_is_returned_whole() -> None:
    section = Section(
        section_type="article", number=1, heading="Subject matter", text="Short text."
    )
    produced = chunk_section(section, CELEX_ID)
    assert len(produced) == 1
    assert produced[0].text.startswith("Subject matter")


def test_a_long_section_splits_with_overlap() -> None:
    """A provision spanning a boundary must be retrievable from either side."""
    paragraphs = [f"{n}. " + "word " * 40 for n in range(1, 9)]
    section = Section(
        section_type="article", number=9, heading="Reporting", text="\n".join(paragraphs)
    )

    produced = chunk_section(section, CELEX_ID, max_tokens=150)
    assert len(produced) > 1
    assert all(c.token_count <= 150 for c in produced)

    tail = produced[0].text.split("\n")[-1]
    assert tail in produced[1].text


def test_budget_holds_when_the_overlap_would_not_fit() -> None:
    """Overlap is dropped rather than pushing a chunk past the hard limit."""
    paragraphs = [f"{n}. " + "word " * 200 for n in range(1, 6)]
    section = Section(
        section_type="article", number=9, heading="Reporting", text="\n".join(paragraphs)
    )

    produced = chunk_section(section, CELEX_ID, max_tokens=300)
    assert len(produced) > 1
    assert all(c.token_count <= 300 for c in produced)


def test_chunk_ids_encode_their_position() -> None:
    section = Section(section_type="article", number=7, heading="H", text="Body text here.")
    produced = chunk_section(section, CELEX_ID)
    assert produced[0].chunk_id == f"{CELEX_ID}:article:7:0"


def test_token_counting_is_not_a_character_count() -> None:
    """Legal text tokenises poorly; characters would under-estimate."""
    assert count_tokens("hello world") < len("hello world")
    assert count_tokens("") == 0

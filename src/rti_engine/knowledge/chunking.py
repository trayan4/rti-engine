"""Split parsed sections into embeddable chunks that remain citable.

A section is the unit of legal meaning; a chunk is the unit of retrieval.
They differ: some articles are pages long, others a single sentence. This
module packs each section's paragraphs up to a token budget, so that short
articles stay whole and long ones become several chunks.

Every chunk carries its section's citation. Retrieval that cannot say
which article a fragment came from is unusable here — the system's
regulatory claims must be checkable against the instrument.

Splitting never crosses a sentence boundary if it can be avoided, and
prefers paragraph boundaries. A provision cut in half states a rule
without its qualifying condition, which is worse than not retrieving it.
"""

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rti_engine.knowledge.directive import CELEX_ID, Section, parse_directive
from rti_engine.llm.tokens import count_tokens, get_encoding

MAX_CHUNK_TOKENS = 500
"""Upper bound per chunk. Comfortably inside the model's limit, and small
enough that a chunk represents one idea rather than an average of many."""

OVERLAP_PARAGRAPHS = 1
"""Trailing paragraphs repeated at the start of the next chunk."""

SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:])\s+")


class Chunk(BaseModel):
    """One embeddable fragment, with everything needed to cite it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    document_kind: str = "legislation"
    jurisdiction: str | None = None
    citation: str
    section_type: str
    section_number: int
    heading: str | None
    chunk_index: int
    text: str
    token_count: int

    @property
    def chunk_id(self) -> str:
        """Stable identifier, used as the vector store's primary key."""
        return f"{self.document_id}:{self.section_type}:{self.section_number}:{self.chunk_index}"


def _split_paragraphs(text: str) -> list[str]:
    """Split a section into paragraphs, discarding blank lines."""
    return [line.strip() for line in text.split("\n") if line.strip()]


def _split_oversized(paragraph: str, max_tokens: int) -> list[str]:
    """Break a paragraph that exceeds the budget on its own.

    Sentence boundaries are tried first. A single sentence longer than the
    budget is split on tokens as a last resort, which is lossy but bounded
    and rare in practice.
    """
    pieces: list[str] = []
    buffer: list[str] = []

    for sentence in SENTENCE_BOUNDARY.split(paragraph):
        candidate = " ".join([*buffer, sentence])
        if buffer and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(buffer))
            buffer = [sentence]
        else:
            buffer = [*buffer, sentence]

    if buffer:
        pieces.append(" ".join(buffer))

    encoding = get_encoding()
    bounded: list[str] = []
    for piece in pieces:
        tokens = encoding.encode(piece)
        if len(tokens) <= max_tokens:
            bounded.append(piece)
            continue
        for start in range(0, len(tokens), max_tokens):
            bounded.append(encoding.decode(tokens[start : start + max_tokens]))

    return bounded


def _pack(paragraphs: list[str], max_tokens: int) -> list[str]:
    """Group paragraphs into chunks no larger than the budget."""
    units: list[str] = []
    for paragraph in paragraphs:
        if count_tokens(paragraph) > max_tokens:
            units.extend(_split_oversized(paragraph, max_tokens))
        else:
            units.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        candidate = [*current, unit]
        if current and count_tokens("\n".join(candidate)) > max_tokens:
            chunks.append("\n".join(current))
            # Carry the tail forward so a provision spanning the boundary can
            # be retrieved from either side — but only when it fits. Overlap
            # is a retrieval convenience; the token budget is a hard limit,
            # and an oversized chunk is rejected by the embedding model.
            overlap = [*current[-OVERLAP_PARAGRAPHS:], unit]
            fits = count_tokens("\n".join(overlap)) <= max_tokens
            current = overlap if fits else [unit]
        else:
            current = candidate

    if current:
        chunks.append("\n".join(current))

    return chunks


def chunk_passage(
    *,
    document_id: str,
    citation: str,
    section_type: str,
    section_number: int,
    heading: str | None,
    text: str,
    document_kind: str = "legislation",
    jurisdiction: str | None = None,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[Chunk]:
    """Split one passage of any document into citable chunks.

    Shared by legislation and by the prose documents in the corpus, so
    that both produce chunks with identical structure and a retrieval
    result never has to care which kind of source it came from.
    """
    heading_prefix = f"{heading}\n" if heading else ""
    body = f"{heading_prefix}{text}".strip()
    if not body:
        return []

    return [
        Chunk(
            document_id=document_id,
            document_kind=document_kind,
            jurisdiction=jurisdiction,
            citation=citation,
            section_type=section_type,
            section_number=section_number,
            heading=heading,
            chunk_index=index,
            text=chunk_text,
            token_count=count_tokens(chunk_text),
        )
        for index, chunk_text in enumerate(_pack(_split_paragraphs(body), max_tokens))
    ]


def chunk_section(
    section: Section, document_id: str, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[Chunk]:
    """Split one section of the directive into chunks."""
    return chunk_passage(
        document_id=document_id,
        citation=section.citation,
        section_type=section.section_type,
        section_number=section.number,
        heading=section.heading,
        text=section.text,
        document_kind="legislation",
        max_tokens=max_tokens,
    )


def chunk_directive(path: Path | None = None, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    """Parse and chunk the directive in one call."""
    chunks: list[Chunk] = []
    for section in parse_directive(path):
        chunks.extend(chunk_section(section, CELEX_ID, max_tokens))
    return chunks

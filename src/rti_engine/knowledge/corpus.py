"""The knowledge corpus: what the system is allowed to reason from.

Five documents, of three kinds. The directive is the instrument itself.
Three national status notes record what is actually in force in each
country the employer operates in, which since the transposition deadline
passed is not the same thing as what the directive requires. The company
policy is the employer's own rules.

Every chunk carries the jurisdiction it applies to, so retrieval for a
Spanish requester is not answered with German law. The directive itself
carries no jurisdiction: it applies everywhere in scope.
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rti_engine.knowledge.chunking import MAX_CHUNK_TOKENS, Chunk, chunk_passage
from rti_engine.knowledge.directive import (
    CELEX_ID,
    DEFAULT_DIRECTIVE_PATH,
    parse_directive,
)

CORPUS_DIRECTORY = Path("data/corpus")

DocumentKind = Literal["legislation", "national_status", "company_policy"]

TITLE_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)
SECTION_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)


class CorpusDocument(BaseModel):
    """One source document and everything needed to cite and filter it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    title: str
    kind: DocumentKind
    filename: str
    jurisdiction: str | None = None
    """Two-letter country code, or None where the document applies broadly."""

    @property
    def path(self) -> Path:
        return CORPUS_DIRECTORY / self.filename


MARKDOWN_DOCUMENTS: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        document_id="status-de",
        title="National implementation status — Germany",
        kind="national_status",
        filename="national-status-germany.md",
        jurisdiction="DE",
    ),
    CorpusDocument(
        document_id="status-fr",
        title="National implementation status — France",
        kind="national_status",
        filename="national-status-france.md",
        jurisdiction="FR",
    ),
    CorpusDocument(
        document_id="status-es",
        title="National implementation status — Spain",
        kind="national_status",
        filename="national-status-spain.md",
        jurisdiction="ES",
    ),
    CorpusDocument(
        document_id="policy-compensation",
        title="Meridian Group Compensation Policy",
        kind="company_policy",
        filename="company-compensation-policy.md",
    ),
)
"""The prose corpus. The directive is handled separately: it has legal
structure worth parsing, where these are ordinary documents."""


def _split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    """Split a markdown document on its second-level headings.

    Anything before the first heading is returned as an unheaded preamble,
    which is where the status date and scope of these documents live —
    material a reader needs in order to interpret anything below it.
    """
    matches = list(SECTION_PATTERN.finditer(text))

    sections: list[tuple[str | None, str]] = []
    preamble = text[: matches[0].start()] if matches else text
    stripped = TITLE_PATTERN.sub("", preamble).strip()
    if stripped:
        sections.append((None, stripped))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            sections.append((match.group(1).strip(), body))

    return sections


def chunk_document(document: CorpusDocument, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    """Split one markdown corpus document into citable chunks."""
    text = document.path.read_text(encoding="utf-8")

    chunks: list[Chunk] = []
    for number, (heading, body) in enumerate(_split_markdown_sections(text), start=1):
        citation = f"{document.title}, {heading}" if heading else document.title
        chunks.extend(
            chunk_passage(
                document_id=document.document_id,
                citation=citation,
                section_type="section",
                section_number=number,
                heading=heading,
                text=body,
                document_kind=document.kind,
                jurisdiction=document.jurisdiction,
                max_tokens=max_tokens,
            )
        )

    return chunks


def chunk_corpus(
    directive_path: Path | None = None, max_tokens: int = MAX_CHUNK_TOKENS
) -> list[Chunk]:
    """Chunk every document in the corpus, ready for embedding."""
    chunks: list[Chunk] = []

    for section in parse_directive(directive_path or DEFAULT_DIRECTIVE_PATH):
        chunks.extend(
            chunk_passage(
                document_id=CELEX_ID,
                citation=section.citation,
                section_type=section.section_type,
                section_number=section.number,
                heading=section.heading,
                text=section.text,
                document_kind="legislation",
                max_tokens=max_tokens,
            )
        )

    for document in MARKDOWN_DOCUMENTS:
        chunks.extend(chunk_document(document, max_tokens))

    return chunks

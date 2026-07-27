"""Parse the directive into article-level sections with citable metadata.

Retrieval returns fragments. A fragment that cannot say which article it
came from is unusable here: the system's regulatory claims have to be
checkable against the instrument, and "somewhere in the directive" is not
a citation.

So the document is split on legal boundaries first — recitals, then
articles — and every chunk carries its article number and heading. Only
then is each section split further to fit the embedding model.

Two properties of the EUR-Lex XHTML drive the parsing. Headings separate
"Article" from its number with a non-breaking space, which must be
normalised or nothing matches. And numbered paragraphs are laid out as
table rows rather than paragraphs, so extraction must pass through tables.
"""

import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict

CELEX_ID = "32023L0970"
INSTRUMENT = "Directive (EU) 2023/970"
DEFAULT_DIRECTIVE_PATH = Path("data/corpus/directive-2023-970.html")

ARTICLE_HEADING = re.compile(r"^Article (\d{1,2})$", re.MULTILINE)
"""Matches only a heading on its own line, so inline citations are excluded."""

RECITAL_MARKER = re.compile(r"^\((\d{1,3})\)$", re.MULTILINE)

SIGNATURE_MARKER = "Done at Strasbourg"
"""Where the enacting terms end and the signature and footnotes begin."""


class Section(BaseModel):
    """One citable unit of the instrument: a recital or an article."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_type: str
    """Either 'recital' or 'article'."""

    number: int
    heading: str | None
    text: str

    @property
    def citation(self) -> str:
        """How this section should be referred to in generated text."""
        if self.section_type == "article":
            return f"{INSTRUMENT}, Article {self.number}"
        return f"{INSTRUMENT}, recital ({self.number})"


def _normalise(text: str) -> str:
    """Collapse Unicode spacing so headings match and text reads cleanly.

    NFKC folds the non-breaking spaces EUR-Lex uses between a heading and
    its number into ordinary spaces. Without this step no article heading
    is ever matched.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace("\xa0", " ")
    folded = re.sub(r"[ \t]+", " ", folded)
    return re.sub(r"\n{3,}", "\n\n", folded)


def extract_text(path: Path | None = None) -> str:
    """Read the directive and return its normalised plain text."""
    source = path if path is not None else DEFAULT_DIRECTIVE_PATH
    html = source.read_text(encoding="utf-8", errors="ignore")
    return _normalise(BeautifulSoup(html, "lxml").get_text("\n", strip=True))


def _split_articles(text: str) -> list[Section]:
    """Split the enacting terms into one section per article."""
    matches = list(ARTICLE_HEADING.finditer(text))
    sections: list[Section] = []

    for index, match in enumerate(matches):
        number = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()

        # The line immediately after the heading is the article's title.
        lines = body.split("\n", 1)
        heading = lines[0].strip() if lines else None
        remainder = lines[1].strip() if len(lines) > 1 else ""

        sections.append(
            Section(
                section_type="article",
                number=number,
                heading=heading or None,
                text=remainder,
            )
        )

    return sections


def _split_recitals(text: str) -> list[Section]:
    """Split the preamble into one section per numbered recital."""
    matches = list(RECITAL_MARKER.finditer(text))
    sections: list[Section] = []

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if not body:
            continue

        sections.append(
            Section(
                section_type="recital",
                number=int(match.group(1)),
                heading=None,
                text=body,
            )
        )

    return sections


def parse_directive(path: Path | None = None) -> list[Section]:
    """Return every recital and article of the directive, in document order.

    The preamble is everything before the first article heading; recitals
    are taken from there so that numbered markers inside the enacting terms
    are not mistaken for recital numbers.
    """
    text = extract_text(path)

    first_article = ARTICLE_HEADING.search(text)
    if first_article is None:
        raise ValueError("no article headings found; the source file may be incomplete")

    preamble = text[: first_article.start()]
    enacting_terms = text[first_article.start() :]

    # The signature block and the footnotes after it are not enacting terms.
    # Left in place they are absorbed into the final article and returned as
    # though they were law.
    signature = enacting_terms.find(SIGNATURE_MARKER)
    if signature != -1:
        enacting_terms = enacting_terms[:signature]

    return _split_recitals(preamble) + _split_articles(enacting_terms)

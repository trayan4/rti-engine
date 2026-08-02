"""Check that every number in a letter came from somewhere.

The Drafter declares which figures it used. This reads what it actually
wrote and checks the declaration held: a number in the prose that appears
nowhere in the source material was invented, rounded or recomputed, and
none of those are permitted.

What is checked is traceability, not correctness — whether a figure came
from somewhere, not whether it is the right figure for that sentence. The
second needs to understand the sentence; the first catches the failures
that actually occur, which are a fabricated benchmark, a rounded
percentage, and a total the model worked out for itself.

Deterministic and model-free, which is the point. The compliance reviewer
can be persuaded; this cannot.
"""

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict

NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
"""Matches a number as written in prose, including thousands separators."""

CONTEXT_CHARACTERS = 60
"""How much surrounding text to quote when reporting an ungrounded figure."""

CITATION_PATTERN = re.compile(
    r"\b\d{1,4}/\d{1,4}(?:/[A-Z]{2,4})?\b"  # 2023/970, 902/2020, 2006/54/EC
    r"|\(EU\)\s*\d{4}/\d{1,4}"  # (EU) 2023/970
    r"|\b(?:Article|Recital|section|Section)\s+\d{1,3}\b"
)
"""Where a number identifies an instrument rather than a quantity.

The digits in "Directive (EU) 2023/970" are part of a name. They are
still checked for grounding — a letter citing law it was not given should
be flagged — but reporting them as undeclared figures fills the audit
bundle with noise a reviewer has to read past.
"""

STRUCTURAL_MAXIMUM = Decimal("100")
"""Small integers below this are permitted without a source.

Section and article numbers appear throughout a compliant letter —
"policy section 8", "Article 7" — and every one of them would otherwise be
reported. They carry no pay information, so the cost of permitting them
is far lower than the cost of a finding on every paragraph.
"""


class UngroundedFigure(BaseModel):
    """A number in the letter that no source accounts for."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str
    context: str
    """The surrounding sentence fragment, so a reader can find it."""


class ValidationResult(BaseModel):
    """What the letter's figures were checked against, and what failed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grounded: bool
    ungrounded: list[UngroundedFigure]
    numbers_checked: int
    undeclared: list[str]
    """Numbers in the prose the drafter did not declare in figures_used.

    Advisory. A declaration that misses a figure is a bookkeeping lapse,
    not a false statement — the number is still traceable to a source.
    """

    def summary(self) -> dict[str, Any]:
        """Describe the check for the audit trail."""
        return {
            "grounded": self.grounded,
            "numbers_checked": self.numbers_checked,
            "ungrounded_count": len(self.ungrounded),
            "undeclared_count": len(self.undeclared),
        }

    def feedback(self) -> str:
        """Render the failures as instructions for a redraft."""
        if self.grounded:
            return "Every figure in the previous draft was traceable to a source."

        parts = [
            "The previous draft contained figures that appear in none of the "
            "sources you were given. Every number in the letter must be one "
            "you were handed, written exactly as it appeared. Rewrite the "
            "letter without these:",
            "",
        ]
        for index, figure in enumerate(self.ungrounded, start=1):
            parts.extend([f"{index}. {figure.value}", f"   in: {figure.context}", ""])

        return "\n".join(parts)


def _citation_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges occupied by instrument and section references."""
    return [(match.start(), match.end()) for match in CITATION_PATTERN.finditer(text)]


def _within(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _as_decimal(text: str) -> Decimal | None:
    """Parse a number as written, or None if it cannot be read."""
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def numbers_in(text: str) -> list[tuple[Decimal, str, int]]:
    """Return every number in a text, with how it was written and where."""
    found: list[tuple[Decimal, str, int]] = []

    for match in NUMBER_PATTERN.finditer(text):
        value = _as_decimal(match.group())
        if value is not None:
            found.append((value, match.group(), match.start()))

    return found


def permitted_values(*sources: Any) -> set[Decimal]:
    """Every number that appears anywhere in the source material.

    Sources are serialised whole rather than walked field by field: a
    figure quoted from a prose caveat is as legitimately sourced as one
    read from a numeric field, and the distinction is not worth drawing
    when the question is only whether the number came from somewhere.
    """
    permitted: set[Decimal] = set()

    for source in sources:
        if source is None:
            continue
        text = source if isinstance(source, str) else json.dumps(source, default=str)
        permitted.update(value for value, _, _ in numbers_in(text))

    return permitted


def _context(text: str, position: int, written: str) -> str:
    """Quote the text around a figure so a reader can locate it."""
    start = max(0, position - CONTEXT_CHARACTERS)
    end = min(len(text), position + len(written) + CONTEXT_CHARACTERS)

    fragment = text[start:end].replace("\n", " ").strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{fragment}{suffix}"


def validate_numbers(
    letter_text: str,
    declared: list[str],
    *sources: Any,
) -> ValidationResult:
    """Check every number in a letter against the sources it came from."""
    permitted = permitted_values(*sources)
    declared_values = {value for item in declared for value, _, _ in numbers_in(item)}

    citations = _citation_spans(letter_text)
    ungrounded: list[UngroundedFigure] = []
    undeclared: list[str] = []
    checked = 0

    for value, written, position in numbers_in(letter_text):
        checked += 1

        if value in permitted:
            if (
                value not in declared_values
                and abs(value) >= STRUCTURAL_MAXIMUM
                and not _within(position, citations)
            ):
                undeclared.append(written)
            continue

        if abs(value) < STRUCTURAL_MAXIMUM and value == value.to_integral_value():
            continue

        ungrounded.append(
            UngroundedFigure(value=written, context=_context(letter_text, position, written))
        )

    return ValidationResult(
        grounded=not ungrounded,
        ungrounded=ungrounded,
        numbers_checked=checked,
        undeclared=sorted(set(undeclared)),
    )

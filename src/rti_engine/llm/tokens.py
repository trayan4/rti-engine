"""Token counting and budgeting.

Shared by the corpus chunker and the prompt layer. Both need to know how
large a piece of text is in the units a model actually charges for, and
characters are a poor proxy: legal citations, long compound words and
non-breaking spaces all tokenise worse than ordinary prose, so a
character count under-estimates exactly where it matters most.

The encoding is loaded once per process. Constructing one downloads and
parses a vocabulary file.
"""

from functools import lru_cache

import tiktoken

ENCODING_NAME = "cl100k_base"
"""Tokeniser used by the text-embedding-3 and GPT-4 family."""

TRUNCATION_MARKER = "\n[truncated]"


@lru_cache
def get_encoding() -> tiktoken.Encoding:
    """Return the tokeniser, loaded once per process."""
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Count tokens the way the model will."""
    return len(get_encoding().encode(text))


def fits_within(text: str, max_tokens: int) -> bool:
    """Report whether text is within a budget."""
    return count_tokens(text) <= max_tokens


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut text to a token budget, marking that it was cut.

    The marker matters: a model given silently truncated input reasons
    confidently over a fragment, where one told the text was cut can say
    so. Reserving room for the marker means the result always fits.
    """
    encoding = get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    marker_size = len(encoding.encode(TRUNCATION_MARKER))
    keep = max(max_tokens - marker_size, 0)
    return encoding.decode(tokens[:keep]) + TRUNCATION_MARKER

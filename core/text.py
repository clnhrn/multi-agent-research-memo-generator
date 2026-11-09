"""Word counting, citation extraction and memo rendering.

Shared by the Reviewer's deterministic checks and the evaluator, so both
judge length and citations the same way.
"""

import re
from collections.abc import Mapping

SOURCE_ID = re.compile(r"P\d{2}")
_BRACKET = re.compile(r"\[([^\]]*)\]")
_WORD = re.compile(r"\b[\w'-]+\b")
FOOTNOTES_HEADING = "Footnotes:"


def extract_citations(text: str) -> list[str]:
    """Source IDs cited in brackets, in order, including [P01, P02] forms."""
    return [sid for group in _BRACKET.findall(text) for sid in SOURCE_ID.findall(group)]


def strip_citations(text: str) -> str:
    return _BRACKET.sub(lambda m: "" if SOURCE_ID.search(m.group(1)) else m.group(0), text)


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def body_word_count(body: str) -> int:
    """Words in the memo body, not counting citation brackets."""
    return word_count(strip_citations(body))


def render_memo(memo: Mapping) -> str:
    """Plain-text memo: title, body, then footnotes."""
    footnotes = "\n".join(memo["footnotes"])
    return f"{memo['title']}\n\n{memo['body']}\n\n{FOOTNOTES_HEADING}\n{footnotes}"

"""Word counting, citation extraction and memo rendering.

Shared by the Reviewer's deterministic checks and the evaluator, so both
judge length and citations the same way.
"""

import re
from collections.abc import Mapping

from core.config import MAX_WORDS, MIN_WORDS

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


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LEADING_CITES = re.compile(r"^((?:\[[^\]]*\]\s*)+)")
_DIGIT = re.compile(r"\d")


def sentences(body: str) -> list[str]:
    """Split prose into sentences; a citation written after the full stop stays with
    the sentence it closes ("... 2024. [P01] Next" -> "... 2024. [P01]", "Next")."""
    result: list[str] = []
    for part in _SENTENCE_END.split(body.strip()):
        lead = _LEADING_CITES.match(part)
        if lead and result:
            result[-1] += " " + lead.group(1).strip()
            part = part[lead.end() :]
        if part.strip():
            result.append(part.strip())
    return result


def has_citation(text: str) -> bool:
    return bool(extract_citations(text))


def has_figure(text: str) -> bool:
    return bool(_DIGIT.search(strip_citations(text)))


_MISSING_DATA = re.compile(
    r"\b(?:data|figures|evidence|survey|surveys|report|corpus)\b[^.]*?"
    r"\b(?:does not|do not|did not|doesn't|don't|lacks?|cannot|omits?|excludes?)\b"
    r"|\bno (?:data|information|figures)\b"
    r"|\b(?:absence of|without) (?:this|these|such|that|the)? ?(?:data|information|figures)\b",
    re.IGNORECASE,
)


def states_missing_data(sentence: str) -> bool:
    """True for sentences about what the data does not cover ("the data does not report...")."""
    return bool(_MISSING_DATA.search(sentence))


def best_sentence(body: str, quote: str) -> tuple[str | None, float]:
    """The body sentence that best contains `quote`, with the share of quote words found."""
    quote_words = set(_WORD.findall(quote.lower()))
    if not quote_words:
        return None, 0.0
    scored = [
        (len(quote_words & set(_WORD.findall(s.lower()))) / len(quote_words), s)
        for s in sentences(body)
    ]
    overlap, sentence = max(scored, default=(0.0, None))
    return sentence, overlap


def check_memo(memo: Mapping) -> list[str]:
    """Deterministic rubric checks on the memo alone; returns problems, empty if none."""
    body = memo["body"]
    problems = []
    words = body_word_count(body)
    if not MIN_WORDS <= words <= MAX_WORDS:
        problems.append(f"Body is {words} words; it must be {MIN_WORDS}-{MAX_WORDS}.")
    cited = set(extract_citations(body))
    if not cited:
        problems.append("The body has no [Pxx] citations.")
    if any(len(SOURCE_ID.findall(group)) > 1 for group in _BRACKET.findall(body)):
        problems.append("Use one source ID per bracket, e.g. [P01][P02], not [P01, P02].")
    footnoted = {sid for note in memo["footnotes"] for sid in extract_citations(note)}
    if missing := sorted(cited - footnoted):
        problems.append(f"Cited without a footnote: {', '.join(missing)}.")
    if extra := sorted(footnoted - cited):
        problems.append(f"Footnotes for IDs not cited in the body: {', '.join(extra)}.")
    return problems


def render_memo(memo: Mapping) -> str:
    """Plain-text memo: title, body, then footnotes."""
    footnotes = "\n".join(memo["footnotes"])
    return f"{memo['title']}\n\n{memo['body']}\n\n{FOOTNOTES_HEADING}\n{footnotes}"

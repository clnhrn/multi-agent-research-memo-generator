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
    """Find the source IDs cited in brackets, including [P01, P02] forms.

    Args:
        text: Text to search.

    Returns:
        Source IDs in order of appearance, with repeats.
    """
    return [sid for group in _BRACKET.findall(text) for sid in SOURCE_ID.findall(group)]


def strip_citations(text: str) -> str:
    """Return the text without its citation brackets."""
    return _BRACKET.sub(lambda m: "" if SOURCE_ID.search(m.group(1)) else m.group(0), text)


def word_count(text: str) -> int:
    """Return the number of words in the text."""
    return len(_WORD.findall(text))


def body_word_count(body: str) -> int:
    """Return the number of words in the memo body, without citation brackets."""
    return word_count(strip_citations(body))


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LEADING_CITES = re.compile(r"^((?:\[[^\]]*\]\s*)+)")
_DIGIT = re.compile(r"\d")


def sentences(body: str) -> list[str]:
    """Split prose into sentences.

    A citation written after the full stop stays with the sentence it closes
    ("... 2024. [P01] Next" -> "... 2024. [P01]", "Next").

    Args:
        body: The memo body.

    Returns:
        The sentences in order.
    """
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
    """Return whether the text cites at least one source."""
    return bool(extract_citations(text))


def has_figure(text: str) -> bool:
    """Return whether the text contains a digit outside its citations."""
    return bool(_DIGIT.search(strip_citations(text)))


# Phrases that name a gap or limitation in the evidence. Tested offline against the
# Reviewer findings on the saved Phase 3-7 drafts.
_MISSING_DATA = re.compile(
    "|".join(
        [
            r"\b(?:data|figures|evidence|survey|surveys|report|corpus)\b[^.]*?"
            r"\b(?:does not|do not|did not|doesn't|don't|lacks?|cannot|omits?|excludes?)\b",
            r"\bno (?:data|information|figures)\b",
            r"\b(?:absence|lack|exclusion) of\b",
            r"\bmissing\b",
            r"\bwithout (?:this|these|such|that|the|additional)\b",
            r"\b(?:is|are) not (?:provided|specified|available|reported|included|given)\b",
            r"\b(?:does|do) not (?:provide|include|capture|cover|report|show|represent|account"
            r"|specify)\b",
            r"\b(?:data|information|research|studies|analysis)\b[^.]*?\b(?:is|are) (?:needed"
            r"|required)\b",
            r"\bwould (?:also )?(?:be (?:valuable|beneficial|needed|essential|useful|helpful)"
            r"|provide|enhance|help)\b",
            r"\b(?:limits?|limiting|hinders?) (?:the |a )?(?:ability|understanding|insights?"
            r"|complete|evaluation)\b",
            r"\b(?:challenging|difficult) to\b",
            r"\bremains? (?:unclear|uncertain|unknown)\b",
            r"\blimitations?\b",
            r"\b(?:these|the) gaps\b",
            r"\brequires further\b",
        ]
    ),
    re.IGNORECASE,
)

# Words that add a cause, motive, prediction, recommendation or judgement. A sentence
# with one of these is not a plain statement of missing data, even if it names a gap
# ("insights are missing, which could influence future sales").
_CLAIM_CUE = re.compile(
    "|".join(
        [
            r"\b(?:should|must|will|likely|expected to|because|due to|driven by|attributed to)\b",
            r"\b(?:suggests?|reflects|indicates?)\b",
            r"\b(?:may|might|could) (?:reflect|indicate|suggest|lead|impact|influence|affect"
            r"|drive|result|make)\b",
            r"\b(?:promising|favorable|positive|traction|strategic)\b",
        ]
    ),
    re.IGNORECASE,
)


def states_missing_data(sentence: str) -> bool:
    """Check whether a sentence only says what the evidence does not cover.

    A sentence that names a gap but also adds a cause, prediction, recommendation or
    judgement is not a plain statement of missing data.

    Args:
        sentence: The sentence, e.g. "The data does not report generation."

    Returns:
        True if the sentence names missing data or a limitation and makes no claim.
    """
    return bool(_MISSING_DATA.search(sentence)) and not _CLAIM_CUE.search(sentence)


def best_sentence(body: str, quote: str) -> tuple[str | None, float]:
    """Find the body sentence that best contains a quote.

    Args:
        body: The memo body.
        quote: The text to look for.

    Returns:
        The sentence and the share of quote words found in it. The sentence is None if the
        quote or the body is empty.
    """
    quote_words = set(_WORD.findall(quote.lower()))
    if not quote_words:
        return None, 0.0
    scored = [
        (len(quote_words & set(_WORD.findall(s.lower()))) / len(quote_words), s)
        for s in sentences(body)
    ]
    overlap, sentence = max(scored, default=(0.0, None))
    return sentence, overlap


def length_ok(body: str) -> bool:
    """Return whether the memo body is within the required length range."""
    return MIN_WORDS <= body_word_count(body) <= MAX_WORDS


def check_memo(memo: Mapping) -> list[str]:
    """Run the deterministic rubric checks on the memo alone.

    Args:
        memo: A dumped `Memo`.

    Returns:
        The problems found, empty if none.
    """
    body = memo["body"]
    problems = []
    if not length_ok(body):
        words = body_word_count(body)
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
    """Render a memo as plain text: title, body, then footnotes.

    Args:
        memo: A dumped `Memo`.

    Returns:
        The memo text.
    """
    footnotes = "\n".join(memo["footnotes"])
    return f"{memo['title']}\n\n{memo['body']}\n\n{FOOTNOTES_HEADING}\n{footnotes}"


_FOOTNOTES_LINE = re.compile(
    rf"^[\W_]*{re.escape(FOOTNOTES_HEADING.rstrip(':'))}[\W_]*$", re.IGNORECASE | re.MULTILINE
)


def parse_memo(text: str) -> dict | None:
    """Split a plain-text memo, the reverse of `render_memo`.

    Used on both systems' memos so they are checked the same way. Markdown around the
    heading, such as "**Footnotes:**", is accepted.

    Args:
        text: The memo as plain text.

    Returns:
        The memo as a dict with `title`, `body` and `footnotes`, or None if it has no
        title line, body and footnotes heading.
    """
    headings = list(_FOOTNOTES_LINE.finditer(text))
    if not headings:
        return None
    title, _, body = text[: headings[-1].start()].strip().partition("\n")
    if not body.strip():
        return None
    notes = text[headings[-1].end() :].splitlines()
    return {
        "title": title.strip(),
        "body": body.strip(),
        "footnotes": [line.strip() for line in notes if line.strip()],
    }

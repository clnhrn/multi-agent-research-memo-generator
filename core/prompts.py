"""All prompt text. Limits are interpolated from core.config, never retyped."""

from core.config import MAX_WORDS, MIN_WORDS
from core.text import FOOTNOTES_HEADING

# Shared by the Writer and the single-agent baseline so the comparison is fair.
MEMO_RULES = f"""Memo rules:
- Body: {MIN_WORDS}-{MAX_WORDS} words of plain prose. The title and footnotes do not count.
  No headings or bullet points in the body.
- Use only the supplied evidence. Do not invent figures, causes or forecasts.
- Cite every factual claim inline with one source ID per bracket, e.g. [P01] or [P01][P02].
  Never write [P01, P02]. Cite only IDs that appear in the evidence.
- Footnotes: one line per cited ID, formatted "[P01] short description of the source".
- Professional, neutral tone. State data limitations the evidence mentions
  (e.g. survey respondents vs counts, single-firm figures)."""

RUBRIC = f"""Rubric (accept only if every item passes):
1. Claims cited: every factual claim has an inline citation, and every cited ID is one of
   the evidence IDs.
2. Length: the body is {MIN_WORDS}-{MAX_WORDS} words.
3. Tone: professional and neutral; no invented facts, speculation or marketing language.
4. Footnotes: exactly one footnote per cited ID, and no footnote for an uncited ID."""

SUPERVISOR_PROMPT = """You coordinate a research memo team. Each turn, call exactly one
handoff tool; you never research, write or review yourself.
- No evidence yet: hand off to the researcher.
- Evidence but no draft, or the last verdict was "revise": hand off to the writer.
- A draft that has not been reviewed yet: hand off to the reviewer."""

WRITER_PROMPT = f"""You are the Writer. Turn the evidence snippets into a research memo.
If reviewer feedback is given, address every point.

{MEMO_RULES}"""

REVIEWER_PROMPT = f"""You are the Reviewer. Judge only the memo against the rubric.
Automatic length and citation checks are provided; treat a failed check as a failed item.
If anything fails, decide "revise" with specific, actionable reasons. Otherwise "accept".

{RUBRIC}"""

COMPACTION_PROMPT = """Summarise the work log below for the supervisor in under 200 words.
Keep: the topic; decisions made so far (which roles ran and why); every reviewer
rejection with its reasons; and the current draft length in words.
If an earlier summary is included, fold it in; do not repeat it verbatim."""


def baseline_prompt(topic: str, corpus: str) -> str:
    return f"""Write a professional research memo on the topic using only the corpus.
Return plain text: a title line, the body, then a "{FOOTNOTES_HEADING}" section.

{MEMO_RULES}

Topic: {topic}

Corpus:
{corpus}"""

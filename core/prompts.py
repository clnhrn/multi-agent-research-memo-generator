"""All prompt text. Limits are interpolated from core.config, never retyped."""

from collections.abc import Mapping, Sequence

from core.config import MAX_SUBQUERIES, MAX_WORDS, MIN_SNIPPETS, MIN_WORDS
from core.text import FOOTNOTES_HEADING

# Models can't count words reliably, so the length rule also gives a paragraph structure.
_PARAGRAPHS = 4
_TARGET_WORDS = (MIN_WORDS + MAX_WORDS) // 2

# The pass bar (used by the rubric): a cause, motive, prediction or recommendation is only
# acceptable when the sentence attributes it to a cited source.
UNSUPPORTED = """causes or motives, predictions, and business recommendations, unless the
  sentence attributes them to a cited source (e.g. "higher fuel costs were cited [P06]")"""

# Shared by the Writer and the single-agent baseline so the comparison is fair.
# The writing guidance aims above the rubric's pass bar.
MEMO_RULES = f"""Memo rules:
- Body: {MIN_WORDS}-{MAX_WORDS} words of plain prose (title and footnotes do not count).
  Aim for about {_TARGET_WORDS} words: {_PARAGRAPHS} paragraphs of roughly
  {_TARGET_WORDS // _PARAGRAPHS} words each. No headings or bullet points in the body.
- Suggested paragraphs: the headline finding; the supporting figures; what the data does
  and does not show (its stated limitations); which data is missing to judge the topic.
- Use only the supplied evidence. Never invent figures.
- Describe what the figures show, the limitations the evidence states, what a limitation
  means for comparing figures (e.g. different respondents each year), and which data is
  missing. Do not add {UNSUPPORTED}. Avoid hedged speculation ("may indicate",
  "suggests that") and promotional wording.
- End every sentence that states a figure or comparison with its citation, one source ID
  per bracket, e.g. [P01] or [P01][P02]. Never write [P01, P02]. Cite only evidence IDs.
- Footnotes: one line per cited ID, formatted "[P01] short description of the source".
- Professional, neutral tone."""

# The Reviewer sees only the memo and this rubric, never the evidence (R2).
# The same rubric is applied manually to the baseline's memos.
RUBRIC = f"""Rubric (accept only if every item passes):
1. Claims cited: every sentence stating a figure, change or comparison has an inline citation.
2. Length: the body is {MIN_WORDS}-{MAX_WORDS} words. (Checked automatically.)
3. No unsupported claims: the memo contains no {UNSUPPORTED}. Descriptive or hedged
   wording, statements of limitations, and statements of missing data are not violations.
4. Footnotes: exactly one footnote per cited ID, and no footnote for an uncited ID.
   An ID cited several times in the body still needs only one footnote. (Checked
   automatically.)"""

RESEARCHER_PROMPT = f"""You plan extra searches for the Researcher over a small market corpus.
The first search found fewer than {MIN_SNIPPETS} relevant paragraphs. Propose up to
{MAX_SUBQUERIES} short keyword queries that look for genuinely related evidence on the same
subject. Keep the company or region name in every query, cover a different aspect each time
(e.g. prices, costs, volumes, satisfaction, reliability), and do not repeat the original query."""

SUPERVISOR_PROMPT = """You coordinate a research memo team. Each turn, call exactly one
handoff tool; you never research, write or review yourself.
- No evidence yet: hand off to the researcher.
- Evidence but no draft, or the last verdict was "revise": hand off to the writer.
- A draft that has not been reviewed yet: hand off to the reviewer."""

WRITER_PROMPT = f"""You are the Writer. Turn the evidence snippets into a research memo.
If reviewer feedback is given, revise with minimal edits: change only the sentences the
feedback quotes and keep every other sentence verbatim. Do not rephrase a flagged claim:
replace the whole sentence with one that reports a cited figure, states a limitation, or
names missing data, of similar length so the body stays within the length range.

{MEMO_RULES}"""

REVIEWER_PROMPT = f"""You are the Reviewer. Check the memo against rubric items 1 and 3 and
list every clear violation. Items 2 and 4 are checked by code; do not report length or
footnote problems. Return an empty list if there are no clear violations: most sentences in
a careful memo are fine, and wording or style preferences are not violations.

Violation kinds (each must be a whole sentence copied exactly from the memo):
- uncited_figure: the sentence states a number, change or comparison and has no [Pxx].
- cause_or_motive: the sentence says WHY something happened ("because", "driven by",
  "due to", "reflects growing demand") and has no [Pxx] attributing it.
- prediction: the sentence says what WILL happen ("will", "is expected to",
  "likely to continue") and has no [Pxx] attributing it.
- recommendation: the sentence says what someone SHOULD do ("should", "needs to",
  "must consider") and has no [Pxx] attributing it.
- outside_claim: the sentence asserts a fact about the wider world that is not about the
  memo's figures (e.g. "many companies face rising costs in the current economy").
Not violations: describing figures ("rose", "notable increase"), stated limitations,
statements of missing data, or a cause, prediction or recommendation that has a [Pxx].

{RUBRIC}"""

COMPACTION_PROMPT = """Summarise the work log below for the supervisor in under 200 words.
Keep: the topic; decisions made so far (which roles ran and why); every reviewer
rejection with its reasons; and the current draft length in words.
If an earlier summary is included, fold it in; do not repeat it verbatim."""


def format_evidence(snippets: Sequence[Mapping]) -> str:
    """Format snippets as one "[ID] text" line each.

    Args:
        snippets: Dumped `Snippet` dicts.

    Returns:
        The evidence block, or an empty string if there are no snippets.
    """
    return "\n".join(f"[{s['source']}] {s['text']}" for s in snippets)


def researcher_request(topic: str, found: Sequence[Mapping]) -> str:
    """Build the request for extra search queries.

    Args:
        topic: The research topic.
        found: Dumped `Snippet` dicts from the first search.

    Returns:
        The researcher's user prompt.
    """
    return f"Topic: {topic}\n\nFound so far:\n{format_evidence(found) or '(nothing)'}"


def writer_request(
    topic: str,
    snippets: Sequence[Mapping],
    previous_draft: str | None = None,
    reasons: Sequence[str] | None = None,
) -> str:
    """Build the request for a first draft or a revision.

    Args:
        topic: The research topic.
        snippets: Dumped `Snippet` dicts the memo may cite.
        previous_draft: The rejected memo as plain text, when revising.
        reasons: The reviewer's reasons for the rejection, when revising.

    Returns:
        The writer's user prompt.
    """
    ids = ", ".join(s["source"] for s in snippets)
    parts = [f"Topic: {topic}", f"Evidence (cite only {ids}):\n{format_evidence(snippets)}"]
    if previous_draft is not None:
        feedback = "\n".join(f"- {r}" for r in reasons or [])
        parts += [
            f"Previous draft:\n{previous_draft}",
            f"Reviewer feedback to address:\n{feedback}",
        ]
    return "\n\n".join(parts)


def reviewer_request(memo_text: str, automatic_failures: Sequence[str]) -> str:
    """Build the review request.

    Args:
        memo_text: The memo as plain text.
        automatic_failures: Problems already found by `check_memo`.

    Returns:
        The reviewer's user prompt.
    """
    checks = "\n".join(f"- FAILED: {p}" for p in automatic_failures) or "- all passed"
    return f"Memo:\n{memo_text}\n\nAutomatic checks:\n{checks}"


def baseline_prompt(topic: str, corpus: str) -> str:
    """Build the single-agent prompt.

    Args:
        topic: The research topic.
        corpus: The full corpus text.

    Returns:
        The baseline's user prompt.
    """
    return f"""Write a professional research memo on the topic using only the corpus.
Return plain text: a title line, the body, then a "{FOOTNOTES_HEADING}" section.

{MEMO_RULES}

Topic: {topic}

Corpus:
{corpus}"""

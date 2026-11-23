"""Word counts, citations, sentences and the deterministic memo checks."""

import pytest

from core.config import MIN_WORDS
from core.text import (
    FOOTNOTES_HEADING,
    best_sentence,
    body_word_count,
    check_memo,
    extract_citations,
    has_figure,
    render_memo,
    sentences,
    states_missing_data,
    word_count,
)


def test_sentences_keep_trailing_citation_with_its_sentence():
    body = "Sales rose to 4,000. [P03] Prices fell [P04]. Data is limited."
    assert sentences(body) == [
        "Sales rose to 4,000. [P03]",
        "Prices fell [P04].",
        "Data is limited.",
    ]


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("The data does not report generation.", True),
        ("The survey lacks demographic detail.", True),
        ("There is no information on costs.", True),
        ("The absence of this data means segments cannot be compared.", True),
        ("Without such information, adoption by age is unknown.", True),
        # Phrasings from saved Phase 3-7 drafts that the first version missed.
        ("Information on the demographics of the installations is not provided.", True),
        ("This growth is notable, but the reasons behind it are not specified.", True),
        ("The lack of context regarding motivations limits the understanding.", True),
        ("Additionally, there is missing data regarding customer satisfaction.", True),
        ("Data on maintenance costs would also be valuable in assessing the offer.", True),
        ("To fully assess the trends in Northstar, further data is needed.", True),
        ("Without this data, it is challenging to draw definitive conclusions.", True),
        ("The survey did not ask respondents to rank every factor, which limits it.", True),
        # A gap plus a cause, prediction, recommendation or judgement is still a claim.
        ("Insights are missing, which could influence future sales trends.", False),
        ("The sales increase suggests a growing interest, but reasons are not provided.", False),
        ("This price change reflects a strategic adjustment but omits maintenance.", False),
        ("Future research should include comprehensive market data.", False),
        ("Adoption is expected to cut costs, although figures are not provided.", False),
        ("While the increase is promising, a more complete dataset is needed.", False),
        ("Many companies face rising costs in the current economy.", False),
        ("Costs rose to 5.40 credits.", False),
    ],
)
def test_states_missing_data(sentence, expected):
    assert states_missing_data(sentence) is expected


def test_has_figure_ignores_citation_ids():
    assert has_figure("Sales rose to 4,000 units [P03].")
    assert not has_figure("Sales rose [P03].")


def test_best_sentence_tolerates_small_quote_differences():
    body = "Costs rose to 5.40 credits [P06]. This reflects strong demand."
    sentence, overlap = best_sentence(body, "this reflects strong demand")
    assert sentence == "This reflects strong demand." and overlap == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Installs rose [P01].", ["P01"]),
        ("Both [P01][P02] agree.", ["P01", "P02"]),
        ("Grouped [P01, P02] and [P03; P04].", ["P01", "P02", "P03", "P04"]),
        ("Repeated [P01] then [P01].", ["P01", "P01"]),
        ("No citations [see note] here.", []),
    ],
)
def test_extract_citations(text, expected):
    assert extract_citations(text) == expected


def test_body_word_count_ignores_citation_brackets():
    assert word_count("Sales rose [P03] sharply [P04].") == 5
    assert body_word_count("Sales rose [P03] sharply [P04].") == 3


def test_body_word_count_keeps_non_citation_brackets():
    assert body_word_count("Costs [in credits] rose.") == 4


def _memo(body_words=MIN_WORDS, citations="[P01][P02]", footnotes=("[P01] a", "[P02] b")):
    return {"title": "T", "body": "word " * body_words + citations, "footnotes": list(footnotes)}


def test_check_memo_passes_clean_memo():
    assert check_memo(_memo()) == []


@pytest.mark.parametrize(
    ("memo", "fragment"),
    [
        (_memo(body_words=MIN_WORDS - 1), "words"),
        (_memo(citations="", footnotes=()), "no [Pxx] citations"),
        (_memo(citations="[P01, P02]"), "one source ID per bracket"),
        (_memo(footnotes=("[P01] a",)), "Cited without a footnote: P02"),
        (_memo(footnotes=("[P01] a", "[P02] b", "[P03] c")), "not cited in the body: P03"),
    ],
)
def test_check_memo_reports_each_problem(memo, fragment):
    assert any(fragment in problem for problem in check_memo(memo))


def test_render_memo_layout():
    memo = {"title": "Solar", "body": "Up [P01].", "footnotes": ["[P01] Capacity"]}
    assert render_memo(memo) == f"Solar\n\nUp [P01].\n\n{FOOTNOTES_HEADING}\n[P01] Capacity"

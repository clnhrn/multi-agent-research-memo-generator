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

import pytest

from core.text import (
    FOOTNOTES_HEADING,
    body_word_count,
    extract_citations,
    render_memo,
    word_count,
)


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


def test_render_memo_layout():
    memo = {"title": "Solar", "body": "Up [P01].", "footnotes": ["[P01] Capacity"]}
    assert render_memo(memo) == f"Solar\n\nUp [P01].\n\n{FOOTNOTES_HEADING}\n[P01] Capacity"

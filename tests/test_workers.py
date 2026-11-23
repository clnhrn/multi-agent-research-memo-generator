"""Workers run standalone with a faked model; the Researcher uses the real MCP server."""

import pytest

import team
from core.config import MAX_WORDS, MIN_WORDS
from core.schemas import Memo, ReviewFindings, SubQueries, Violation, new_state
from core.text import body_word_count

SOLAR = [
    {"text": "Northstar installed 120 MW of rooftop solar in 2025.", "source": "P01"},
    {"text": "Residential installs rose from 40 to 55 MW.", "source": "P02"},
]


def _memo(words=260, cite="P01", footnotes=("[P01] Solar capacity",)):
    return Memo(title="Solar", body="word " * words + f"[{cite}].", footnotes=list(footnotes))


def _kinds(update):
    return [e["kind"] for e in update["trace"]]


# --- Researcher -------------------------------------------------------------------


def test_researcher_enough_evidence_skips_subqueries(fake_call):
    llm = fake_call(None)
    state = new_state("Assess SwiftCart delivery growth, reliability and costs.", "T03")

    update = team.researcher(state).update

    assert [s["source"] for s in update["snippets"]] == ["P06", "P05", "P08"]
    assert update["mcp_calls"] == 1
    assert llm.calls == []
    assert update["trace"][-1]["data"]["shortfall"] is False


def test_researcher_shortfall_tries_subqueries_and_never_pads(fake_call):
    queries = ["CareBridge patient satisfaction", "CareBridge consultation volumes"]
    llm = fake_call(SubQueries(queries=queries))
    state = new_state("Assess CareBridge remote consultations and patient satisfaction.", "T05")

    update = team.researcher(state).update

    assert {s["source"] for s in update["snippets"]} == {"P09", "P10"}
    assert update["mcp_calls"] == 3
    assert [role for role, _ in llm.calls] == ["researcher"]
    assert _kinds(update) == ["mcp_search", "llm_call", "mcp_search", "mcp_search", "research_done"]
    assert update["trace"][-1]["data"]["shortfall"] is True


# --- Writer -----------------------------------------------------------------------


def test_writer_drafts_from_state_and_flags_stray_citations(fake_call):
    fake_call(_memo(cite="P11", footnotes=["[P11] Buses"]))
    state = {**new_state("solar topic", "T01"), "snippets": SOLAR}

    update = team.writer(state).update

    assert update["draft_memo"]["title"] == "Solar"
    assert update["review_verdict"] is None  # new draft awaits review
    draft = update["trace"][-1]
    assert draft["data"]["revision"] is False
    assert draft["data"]["stray_citations"] == ["P11"]


def test_writer_revision_sees_previous_draft_and_reasons(fake_call):
    llm = fake_call(_memo())
    state = {
        **new_state("solar topic", "T01"),
        "snippets": SOLAR,
        "draft_memo": _memo(words=100).model_dump(),
        "review_verdict": {"decision": "revise", "reasons": ["Body is too short."]},
    }

    update = team.writer(state).update

    _, prompt = llm.calls[0]
    assert "Previous draft" in prompt and "Body is too short." in prompt
    assert update["trace"][-1]["data"]["revision"] is True


def _writes(fake_call, *words):
    """Make the writer produce drafts of these lengths in turn; returns the fake and update."""
    drafts = iter([_memo(words=n) for n in words])
    llm = fake_call(lambda state, role: next(drafts))
    state = {**new_state("solar topic", "T01"), "snippets": SOLAR}
    return llm, team.writer(state).update


def _body_words(update):
    return body_word_count(update["draft_memo"]["body"])


@pytest.mark.parametrize("words", [MIN_WORDS, MAX_WORDS])
def test_writer_leaves_a_draft_in_range_alone(fake_call, words):
    llm, update = _writes(fake_call, words)
    assert _body_words(update) == words
    assert len(llm.calls) == 1
    assert "length_fix" not in _kinds(update)


@pytest.mark.parametrize(
    ("draft", "action"), [(MIN_WORDS - 1, "Add two or three"), (MAX_WORDS + 1, "Cut about")]
)
def test_writer_fixes_its_own_length_before_handing_over(fake_call, draft, action):
    llm, update = _writes(fake_call, draft, 300)
    assert _body_words(update) == 300
    prompt = llm.calls[1][1]
    assert f"body is {draft} words" in prompt and action in prompt
    assert SOLAR[0]["text"] in prompt  # the evidence is still available
    event = update["trace"][-2]
    assert event["data"] == {"before": draft, "after": 300, "kept": True}
    # Both calls are counted and logged.
    assert update["total_tokens"] == 20 and len(update["messages"]) == 2
    assert _kinds(update) == ["llm_call", "llm_call", "length_fix", "draft"]


def test_writer_tries_once_and_keeps_its_draft_if_the_fix_fails(fake_call):
    llm, update = _writes(fake_call, 240, 245)
    assert _body_words(update) == 240
    assert len(llm.calls) == 2
    assert update["trace"][-2]["data"]["kept"] is False


# --- Reviewer ---------------------------------------------------------------------


NO_FINDINGS = ReviewFindings(violations=[])
BODY = (
    "Installations rose from 100 to 120 megawatts [P01]. "
    "Residential capacity reached 55 megawatts. "
    "This growth reflects rising demand for clean energy. "
    "Higher fuel costs were cited by the operator [P06]. "
    "The data does not report generation. "
    "Without this data, it is challenging to assess demand. "
    "Insights are missing, which could influence future sales."
)


def _violation(kind, quote):
    return Violation(kind=kind, quote=quote, explanation="test")


def test_reviewer_accepts_when_model_finds_nothing(fake_call):
    fake_call(NO_FINDINGS)
    state = {**new_state("solar topic", "T01"), "draft_memo": _memo().model_dump()}

    update = team.reviewer(state).update

    assert update["review_verdict"] == {"decision": "accept", "reasons": []}
    assert update["rejections"] == 0


def test_reviewer_automatic_failures_force_revise(fake_call):
    fake_call(NO_FINDINGS)
    state = {**new_state("solar topic", "T01"), "draft_memo": _memo(words=100).model_dump()}

    update = team.reviewer(state).update

    assert update["review_verdict"]["decision"] == "revise"
    assert any("words" in r for r in update["review_verdict"]["reasons"])
    assert update["rejections"] == 1


@pytest.mark.parametrize(
    ("kind", "quote", "kept"),
    [
        ("uncited_figure", "Residential capacity reached 55 megawatts.", True),
        ("cause_or_motive", "This growth reflects rising demand for clean energy.", True),
        ("cause_or_motive", "Higher fuel costs were cited by the operator.", False),  # attributed
        ("uncited_figure", "The data does not report generation.", False),  # no figure
        ("prediction", "Solar will keep growing next year.", False),  # not in memo
        ("outside_claim", "The data does not report generation.", False),  # missing data
        # Missing data is allowed whatever kind the model gives it.
        ("prediction", "Without this data, it is challenging to assess demand.", False),
        ("recommendation", "Without this data, it is challenging to assess demand.", False),
        ("cause_or_motive", "Without this data, it is challenging to assess demand.", False),
        # A gap plus a claim in the same sentence still counts.
        ("prediction", "Insights are missing, which could influence future sales.", True),
        ("outside_claim", "Higher fuel costs were cited by the operator.", False),  # cited
    ],
)
def test_verify_keeps_only_supported_findings(kind, quote, kept):
    result, dropped = team.verify(BODY, [_violation(kind, quote)])
    assert bool(result) is kept
    assert bool(dropped) is not kept


def test_reviewer_decision_follows_verified_findings_only(fake_call):
    fake_call(
        ReviewFindings(
            violations=[
                _violation("cause_or_motive", "Higher fuel costs were cited by the operator"),
            ]
        )
    )
    memo = Memo(title="T", body=BODY + " word" * 230, footnotes=["[P01] a", "[P06] b"])
    state = {**new_state("solar topic", "T01"), "draft_memo": memo.model_dump()}

    update = team.reviewer(state).update

    assert update["review_verdict"]["decision"] == "accept"
    assert update["trace"][-1]["data"]["dropped_findings"][0]["why"].startswith("sentence is")


def test_reviewer_never_sees_the_evidence(fake_call):
    llm = fake_call(NO_FINDINGS)
    state = {
        **new_state("solar topic", "T01"),
        "snippets": SOLAR,
        "draft_memo": _memo().model_dump(),
    }

    team.reviewer(state)

    _, prompt = llm.calls[0]
    assert not any(s["text"] in prompt for s in SOLAR)

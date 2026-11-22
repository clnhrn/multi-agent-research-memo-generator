"""Workers run standalone with a faked model; the Researcher uses the real MCP server."""

import pytest

import team
from core.schemas import Memo, ReviewFindings, SubQueries, Violation, new_state

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


# --- Reviewer ---------------------------------------------------------------------


NO_FINDINGS = ReviewFindings(violations=[])
BODY = (
    "Installations rose from 100 to 120 megawatts [P01]. "
    "Residential capacity reached 55 megawatts. "
    "This growth reflects rising demand for clean energy. "
    "Higher fuel costs were cited by the operator [P06]. "
    "The data does not report generation."
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

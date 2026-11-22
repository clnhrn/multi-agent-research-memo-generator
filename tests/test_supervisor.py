"""Supervisor routing, hard stops and guards; the model is faked, the graph is real."""

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Command

import team
from core.config import MAX_REJECTIONS, MAX_TURNS
from core.prompts import supervisor_request
from core.schemas import Memo, ReviewFindings, SubQueries, Violation, new_state

SNIPPETS = [{"text": "Northstar installed 120 MW.", "source": "P01"}]
MEMO = {"title": "T", "body": "Secret body [P01].", "footnotes": ["[P01] a"]}
ACCEPT = {"decision": "accept", "reasons": []}
REVISE = {"decision": "revise", "reasons": ["Too short."]}


def _state(**fields):
    return {**new_state("solar topic", "T01"), **fields}


def _pick(*roles):
    calls = [
        {"name": f"hand_off_to_{role}", "args": {}, "id": str(i)} for i, role in enumerate(roles)
    ]
    return AIMessage("", tool_calls=calls)


# --- Tools and prompt -------------------------------------------------------------


def test_the_only_tools_are_three_handoffs_without_arguments():
    assert set(team.HANDOFFS) == {
        "hand_off_to_researcher",
        "hand_off_to_writer",
        "hand_off_to_reviewer",
    }
    for name, handoff in team.HANDOFFS.items():
        assert handoff.args == {}
        assert handoff.description
        command = handoff.invoke({})
        assert isinstance(command, Command)
        assert name == f"hand_off_to_{command.goto}"


def test_supervisor_sees_progress_but_no_evidence_or_memo_text():
    prompt = supervisor_request(
        _state(snippets=SNIPPETS, draft_memo=MEMO, review_verdict=REVISE, rejections=1)
    )
    assert "Evidence: P01" in prompt and "Last verdict: revise" in prompt
    assert f"Rejections: 1 of {MAX_REJECTIONS}" in prompt
    assert SNIPPETS[0]["text"] not in prompt and "Secret body" not in prompt


# --- Hard stops -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "status"),
    [
        ({"review_verdict": ACCEPT}, "completed"),
        ({"review_verdict": REVISE, "rejections": MAX_REJECTIONS}, "escalated"),
        ({"mcp_calls": 3, "snippets": []}, "escalated"),
        ({"turn_count": MAX_TURNS}, "max_turns"),
        # Accepted on the last turn, or after earlier rejections, is still completed.
        ({"review_verdict": ACCEPT, "turn_count": MAX_TURNS}, "completed"),
        ({"review_verdict": ACCEPT, "rejections": MAX_REJECTIONS - 1}, "completed"),
        ({"review_verdict": REVISE, "rejections": MAX_REJECTIONS, "turn_count": MAX_TURNS},
         "escalated"),
    ],
)  # fmt: skip
def test_hard_stops_end_the_run_without_a_model_call(fake_call, fields, status):
    llm = fake_call(_pick("researcher"))

    command = team.supervisor(_state(**fields))

    assert command.goto == END
    assert command.update["status"] == status
    event = command.update["trace"][-1]
    assert event["kind"] == "stop" and event["data"]["reason"]
    assert llm.calls == []


def test_no_stop_while_work_remains():
    assert team.stop_reason(_state()) is None
    assert (
        team.stop_reason(_state(snippets=SNIPPETS, mcp_calls=1, turn_count=MAX_TURNS - 1)) is None
    )


# --- Handoffs and guards ----------------------------------------------------------


DRAFTED = {"snippets": SNIPPETS, "draft_memo": MEMO}


@pytest.mark.parametrize(
    ("fields", "reply", "goto", "redirected"),
    [
        ({}, _pick("researcher"), "researcher", False),
        ({"snippets": SNIPPETS}, _pick("writer"), "writer", False),
        (DRAFTED, _pick("reviewer"), "reviewer", False),
        ({**DRAFTED, "review_verdict": REVISE}, _pick("writer"), "writer", False),
        # Wasteful but possible: the model's choice stands and costs a turn.
        (DRAFTED, _pick("researcher"), "researcher", False),
        # Impossible or malformed: redirected.
        ({}, _pick("writer"), "researcher", True),
        ({}, _pick("reviewer"), "researcher", True),
        ({"snippets": SNIPPETS}, _pick("reviewer"), "writer", True),
        ({**DRAFTED, "review_verdict": REVISE}, _pick("reviewer"), "writer", True),
        (DRAFTED, _pick("writer", "reviewer"), "reviewer", True),
        (DRAFTED, _pick(), "reviewer", True),
        ({}, _pick("intern"), "researcher", True),
    ],
)
def test_handoff_follows_the_model_unless_it_is_impossible(
    fake_call, fields, reply, goto, redirected
):
    fake_call(reply)

    command = team.supervisor(_state(**fields))

    assert command.goto == goto
    assert command.update["turn_count"] == 1
    event = command.update["trace"][-1]
    assert event["kind"] == "supervisor_decision" and event["turn"] == 1
    assert event["data"]["handoff"] == f"hand_off_to_{goto}"
    assert bool(event["data"]["redirected"]) is redirected


# --- Whole graph ------------------------------------------------------------------


GOOD = Memo(title="T", body="word " * 260 + "[P01].", footnotes=["[P01] Solar"])
CLAIM = "Demand will surely rise."
BAD = GOOD.model_copy(update={"body": f"{GOOD.body} {CLAIM}"})
FINDING = Violation(kind="prediction", quote=CLAIM, explanation="unsupported")


def _run(fake_call, *, rejections=0, pick=None):
    """Run the real graph and MCP server; the reviewer rejects the first `rejections` drafts."""

    def reply(state, role):
        if role == "supervisor":
            return pick(state) if pick else _pick(team.fallback_handoff(state).invoke({}).goto)
        if role == "researcher":
            return SubQueries(queries=["Northstar solar prices"])
        rejecting = state["rejections"] < rejections
        if role == "writer":
            return BAD if rejecting else GOOD
        return ReviewFindings(violations=[FINDING] if rejecting else [])

    fake_call(reply)
    return team.build_graph().invoke(
        new_state("Assess rooftop solar installation trends in Northstar.", "T01")
    )


def _route(final):
    return [
        e["data"]["handoff"].removeprefix("hand_off_to_")
        for e in final["trace"]
        if e["kind"] == "supervisor_decision"
    ]


def test_graph_accepts_first_draft(fake_call):
    final = _run(fake_call)
    assert _route(final) == ["researcher", "writer", "reviewer"]
    assert (final["status"], final["turn_count"], final["rejections"]) == ("completed", 3, 0)
    assert final["mcp_calls"] >= 1


def test_graph_revises_then_accepts(fake_call):
    final = _run(fake_call, rejections=2)
    assert _route(final) == ["researcher"] + ["writer", "reviewer"] * 3
    assert (final["status"], final["turn_count"], final["rejections"]) == ("completed", 7, 2)


def test_graph_escalates_after_three_rejections(fake_call):
    final = _run(fake_call, rejections=MAX_REJECTIONS)
    assert (final["status"], final["rejections"]) == ("escalated", MAX_REJECTIONS)
    assert final["turn_count"] == 7  # within the 8-turn cap
    assert final["draft_memo"] is not None  # kept for the human


def test_graph_stops_at_the_turn_cap(fake_call):
    final = _run(fake_call, pick=lambda state: _pick("researcher"))
    assert (final["status"], final["turn_count"]) == ("max_turns", MAX_TURNS)
    assert final["trace"][-1]["kind"] == "stop"


def test_graph_escalates_when_nothing_is_found(fake_call, monkeypatch):
    async def nothing(session, query):
        return []

    monkeypatch.setattr(team, "search", nothing)
    final = _run(fake_call)
    assert _route(final) == ["researcher"]
    assert final["status"] == "escalated"

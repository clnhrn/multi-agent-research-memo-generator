"""Atomic saves, validated loads, and resuming a run killed at any stage (R4)."""

import json
import sys

import pytest
from helpers import TOPIC, pick, route, scripted
from langchain_core.messages import AIMessage, HumanMessage

import baseline
import core.scratchpad
import evaluate
import team
from core.schemas import TERMINAL_STATUSES, apply_update, new_state
from core.scratchpad import ScratchpadError, load_state, save_state

REQUIRED = {
    "topic", "snippets", "draft_memo", "review_verdict", "turn_count", "summary", "status",
}  # fmt: skip


class Killed(Exception):
    """Stands in for the process dying."""


# --- Saving and loading -----------------------------------------------------------


def test_round_trip_keeps_every_field_and_message_ids(state, tmp_scratchpad):
    messages = [HumanMessage("ask", name="writer"), AIMessage("draft", name="writer")]
    state = apply_update(
        state, {"messages": messages, "trace": [{"kind": "x"}], "total_tokens": None}
    )
    save_state(state)
    assert load_state("T01") == state
    assert all(m.id for m in load_state("T01")["messages"])
    saved = json.loads((tmp_scratchpad / "run-T01.json").read_text())
    assert REQUIRED <= saved.keys()


def test_missing_run_loads_as_none():
    assert load_state("T01") is None


@pytest.mark.parametrize("step", ["fsync", "replace"])
def test_a_failed_write_leaves_the_last_save_intact(state, monkeypatch, step):
    save_state({**state, "turn_count": 1})

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(core.scratchpad.os, step, fail)
    with pytest.raises(OSError, match="disk full"):
        save_state({**state, "turn_count": 2})
    assert load_state("T01")["turn_count"] == 1


@pytest.mark.parametrize("run_id", ["", "../x", "a/b", "a.b", "T01\n", "a" * 65])
def test_unsafe_run_ids_never_become_file_names(run_id, tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        load_state(run_id)
    with pytest.raises(ValueError, match="run_id"):
        save_state({**new_state(TOPIC, "T01"), "run_id": run_id})
    assert list(tmp_path.rglob("run-*")) == []


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda text: text[: len(text) // 2],
        lambda text: text.replace('"running"', '"winning"'),
        lambda text: text.replace('"pending": null', '"pending": "intern"'),
        lambda text: text.replace('"run_id": "T01"', '"run_id": "T02"'),
    ],
    ids=["truncated", "bad status", "bad worker", "other run"],
)
def test_unusable_files_are_reported_and_never_overwritten(state, tmp_scratchpad, corrupt):
    save_state(state)
    path = tmp_scratchpad / "run-T01.json"
    path.write_text(corrupt(path.read_text()))
    before = path.read_text()
    with pytest.raises(ScratchpadError):
        team.start_state(TOPIC, "T01")
    assert path.read_text() == before


# --- Persistence during a run -----------------------------------------------------


def test_every_turn_is_saved_exactly_as_the_graph_holds_it(fake_call, monkeypatch):
    saves = []
    monkeypatch.setattr(team, "save_state", lambda s: (saves.append(s), save_state(s)))
    fake_call(scripted(rejections=1))
    final = team.run_team(TOPIC, "T01")
    ends = [(s["turn_count"], s["pending"]) for s in saves if s["pending"]]
    assert ends == list(enumerate(route(final), start=1))
    assert load_state("T01") == final
    assert final["status"] == "completed"


# --- Resume -----------------------------------------------------------------------


SEARCHES = 2  # the topic, then the one sub-query the scripted researcher proposes


def _killed_at(role, nth):
    """Build a model that dies on the nth call by a role."""
    inner, seen = scripted(rejections=1), []

    def reply(state, caller):
        if caller == role:
            seen.append(caller)
            if len(seen) == nth:
                raise Killed
        return inner(state, caller)

    return reply


@pytest.mark.parametrize(
    ("role", "nth", "searches_after"),
    [
        ("supervisor", 1, SEARCHES),  # nothing done yet
        ("researcher", 1, SEARCHES),  # in flight: only the unfinished worker is repeated
        ("supervisor", 2, 0),  # research finished, next decision not made
        ("writer", 1, 0),
        ("reviewer", 1, 0),
        ("supervisor", 4, 0),  # rejected, revision not yet chosen
        ("writer", 2, 0),
        ("reviewer", 2, 0),
    ],
)
def test_killed_run_resumes_without_repeating_finished_work(
    fake_call, monkeypatch, role, nth, searches_after
):
    fake_call(_killed_at(role, nth))
    # MCP's task group wraps errors raised inside the researcher.
    with pytest.raises((Killed, ExceptionGroup)):
        team.run_team(TOPIC, "T01")
    assert load_state("T01")["status"] not in TERMINAL_STATUSES

    searches, real_search = [], team.search

    async def counting(session, query):
        searches.append(query)
        return await real_search(session, query)

    monkeypatch.setattr(team, "search", counting)
    fake_call(scripted(rejections=1))
    final = team.run_team(TOPIC, "T01")

    assert len(searches) == searches_after
    # Same outcome and turn count as a run that was never killed.
    assert route(final) == ["researcher", "writer", "reviewer", "writer", "reviewer"]
    assert (final["status"], final["turn_count"], final["rejections"]) == ("completed", 5, 1)
    assert final["mcp_calls"] == SEARCHES
    assert [e["kind"] for e in final["trace"]].count("resumed") == 1


def test_resume_goes_straight_to_the_pending_worker(fake_call):
    fake_call(_killed_at("writer", 1))
    with pytest.raises(Killed):
        team.run_team(TOPIC, "T01")
    assert load_state("T01")["pending"] == "writer"
    llm = fake_call(scripted())
    team.run_team(TOPIC, "T01")
    assert [role for role, _ in llm.calls][0] == "writer"


@pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
def test_finished_runs_start_fresh(state, status):
    save_state({**state, "status": status, "turn_count": 5})
    started = team.start_state(TOPIC, "T01")
    assert started["turn_count"] == 0
    assert started["trace"][-1]["kind"] == "run_started"


def test_fresh_ignores_an_unfinished_run(state):
    save_state({**state, "turn_count": 2})
    assert team.start_state(TOPIC, "T01")["turn_count"] == 2
    assert team.start_state(TOPIC, "T01", fresh=True)["turn_count"] == 0


def test_resume_refuses_another_topic(state):
    save_state(state)
    with pytest.raises(ScratchpadError, match="another topic"):
        team.start_state("Assess something else.", "T01")


def test_turn_cap_survives_a_resume(state, fake_call):
    llm = fake_call(pick("researcher"))
    save_state({**state, "turn_count": team.MAX_TURNS})
    assert team.run_team(TOPIC, "T01")["status"] == "max_turns"
    assert llm.calls == []


# --- Evaluator switch -------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "fresh"), [([], True), (["--fresh"], True), (["--resume"], False)]
)
def test_evaluator_starts_clean_unless_asked_to_resume(monkeypatch, tmp_path, flags, fresh):
    seen = []

    def run(topic, topic_id, *, fresh=None):
        seen.append(fresh)
        return {"memo": "text"}

    monkeypatch.setattr(team, "run_topic", run)
    monkeypatch.setattr(baseline, "run_topic", lambda topic, topic_id: {"memo": "text"})
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--output", str(tmp_path / "e.json"), *flags])
    evaluate.main()
    assert set(seen) == {fresh}

"""Rolling-summary memory: trigger, preserved facts, idempotency and resume (R5)."""

import pytest
from helpers import CLAIM, TOPIC, route, scripted
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

import core.memory
import team
from core.config import KEEP_RECENT_MESSAGES
from core.memory import SUMMARY_NAME, facts, maybe_compact
from core.prompts import supervisor_request
from core.schemas import apply_update, new_state
from core.scratchpad import load_state

# The real function: the offline fixture replaces the module attribute, not this name.
REAL_COUNT_TOKENS = core.memory.count_tokens


class Killed(Exception):
    """Stands in for the process dying."""


@pytest.fixture
def threshold(monkeypatch):
    """Return a setter for the compaction threshold."""
    return lambda tokens: monkeypatch.setattr(core.memory, "COMPACTION_TOKENS", tokens)


def _long_state(n=10):
    """Build a state with a long log, one rejection and a draft."""
    messages = [HumanMessage(f"message {i} " + "word " * 50, name="writer") for i in range(n)]
    trace = [
        {"kind": "supervisor_decision", "data": {"handoff": "hand_off_to_researcher"}},
        {"kind": "supervisor_decision", "data": {"handoff": "hand_off_to_writer"}},
        {"kind": "verdict", "data": {"decision": "revise", "reasons": ["Too short.", "Claim."]}},
        {"kind": "verdict", "data": {"decision": "accept", "reasons": []}},
    ]
    memo = {"title": "T", "body": "word " * 260 + "[P01].", "footnotes": []}
    return apply_update(
        new_state(TOPIC, "T01"), {"messages": messages, "trace": trace, "draft_memo": memo}
    )


def test_nothing_happens_at_or_below_the_threshold(fake_call, threshold):
    llm = fake_call(AIMessage("notes"))
    state = _long_state()
    threshold(core.memory.count_tokens(state["messages"])[0])
    assert maybe_compact(state) == {}
    assert llm.calls == []


def test_compaction_replaces_older_messages_and_keeps_recent_ones(fake_call, threshold):
    threshold(100)
    llm = fake_call(AIMessage("The notes."))
    state = _long_state()
    update = maybe_compact(state)
    after = apply_update(state, update)

    assert [m.name for m in after["messages"]][0] == SUMMARY_NAME
    assert after["messages"][1:] == state["messages"][-KEEP_RECENT_MESSAGES:]
    assert after["messages"][0].content == after["summary"]
    event = update["trace"][-1]
    assert event["kind"] == "compaction"
    assert event["data"]["removed"] == 10 - KEEP_RECENT_MESSAGES
    assert event["data"]["after"] < event["data"]["before"]
    assert after["total_tokens"] == 10  # the summary call is paid for
    role, prompt = llm.calls[0]
    assert role == "memory" and "message 0" in prompt and "message 9" not in prompt


def test_summary_preserves_the_required_facts_whatever_the_model_writes(fake_call, threshold):
    threshold(100)
    fake_call(AIMessage(""))
    summary = maybe_compact(_long_state())["summary"]
    assert f"Topic: {TOPIC}" in summary
    assert "Decisions (handoffs): researcher, writer" in summary
    assert "Current draft: 260 words" in summary
    assert "Reviewer rejections: 1 of 3" in summary
    assert "- Rejection 1: Too short. | Claim." in summary


def test_facts_before_any_work():
    assert "Current draft: none" in facts(new_state(TOPIC, "T01"))


def test_compaction_is_idempotent(fake_call, threshold):
    threshold(100)  # so low that the compacted log is still above it
    llm = fake_call(AIMessage("The notes."))
    once = apply_update(_long_state(), maybe_compact(_long_state()))
    calls = len(llm.calls)
    assert core.memory.count_tokens(once["messages"])[0] > 100
    assert maybe_compact(once) == {}
    assert len(llm.calls) == calls


def test_a_later_compaction_folds_the_earlier_summary_and_never_doubles(fake_call, threshold):
    threshold(100)
    llm = fake_call(AIMessage("The notes."))
    state = apply_update(_long_state(), maybe_compact(_long_state()))
    more = [HumanMessage("later " + "word " * 50, name="writer") for _ in range(6)]
    state = apply_update(state, {"messages": more})
    state = apply_update(state, maybe_compact(state))

    assert [m.name for m in state["messages"]].count(SUMMARY_NAME) == 1
    assert "The notes." in llm.calls[-1][1]  # the earlier summary was folded in
    assert state["summary"].count("Rejection 1:") == 1
    assert state["summary"].count("Topic:") == 1


def test_supervisor_sees_the_summary_only_once_there_is_one():
    state = new_state(TOPIC, "T01")
    assert "Memory" not in supervisor_request(state)
    assert "Memory of earlier work:\nNOTES" in supervisor_request({**state, "summary": "NOTES"})


@pytest.mark.parametrize("error", [NotImplementedError("model"), ConnectionError("blocked")])
def test_token_count_falls_back_to_an_estimate(monkeypatch, error):
    class Llm:
        def get_num_tokens_from_messages(self, messages):
            raise error

    monkeypatch.setattr(core.memory, "make_llm", Llm)
    count, counter = REAL_COUNT_TOKENS([HumanMessage("hello")])
    assert count > 0 and counter == "estimate"


# --- In a whole run ---------------------------------------------------------------


def _script(**options):
    """Build the scripted model with a summary writer."""
    inner = scripted(**options)
    return lambda state, role: AIMessage("Notes.") if role == "memory" else inner(state, role)


@pytest.mark.parametrize("tokens", [1, 200, 1000, 8000])
@pytest.mark.parametrize("rejections", [0, 2, 3])
def test_compaction_never_changes_the_outcome(fake_call, threshold, tokens, rejections):
    threshold(10**9)
    fake_call(_script(rejections=rejections))
    plain = team.run_team(TOPIC, "T01", "plain")
    threshold(tokens)
    llm = fake_call(_script(rejections=rejections))
    final = team.run_team(TOPIC, "T01", "compacted")

    assert route(final) == route(plain)
    for field in ("status", "turn_count", "rejections", "draft_memo", "mcp_calls"):
        assert final[field] == plain[field]
    assert load_state("compacted") == final
    assert not any(isinstance(m, RemoveMessage) for m in final["messages"])
    compactions = [e for e in final["trace"] if e["kind"] == "compaction"]
    assert bool(compactions) is (tokens <= 1000)
    # Every compaction reached the graph: none was lost and paid for again.
    assert len(compactions) == [role for role, _ in llm.calls].count("memory")
    assert final["total_tokens"] == 10 * len(llm.calls)
    assert bool(final["summary"]) is bool(compactions)
    assert [m.name for m in final["messages"]].count(SUMMARY_NAME) == bool(compactions)
    if rejections and compactions:
        assert CLAIM in final["summary"]


def test_a_kill_during_compaction_loses_nothing_and_never_compacts_twice(fake_call, threshold):
    threshold(200)
    inner, died = _script(rejections=1), []

    def dies_once(state, role):
        if role == "memory" and not died:
            died.append(role)
            raise Killed
        return inner(state, role)

    fake_call(dies_once)
    with pytest.raises(Killed):
        team.run_team(TOPIC, "T01")
    assert load_state("T01")["summary"] == ""

    fake_call(inner)
    final = team.run_team(TOPIC, "T01")
    assert (final["status"], final["turn_count"]) == ("completed", 5)
    assert [m.name for m in final["messages"]].count(SUMMARY_NAME) == 1
    assert final["mcp_calls"] == 2

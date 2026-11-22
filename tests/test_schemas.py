"""Schema validation and the state reducers."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from core.config import MAX_SNIPPETS
from core.schemas import Memo, ResearchResult, Snippet, State, Verdict, new_state


def _snippets(n):
    return [{"text": f"t{i}", "source": f"P{i:02d}"} for i in range(1, n + 1)]


def test_snippet_rejects_bad_source_format():
    with pytest.raises(ValidationError):
        Snippet(text="x", source="paragraph 1")


@pytest.mark.parametrize("n", [0, MAX_SNIPPETS + 1])
def test_research_result_bounds(n):
    with pytest.raises(ValidationError):
        ResearchResult(snippets=_snippets(n))


def test_research_result_rejects_duplicate_sources():
    with pytest.raises(ValidationError):
        ResearchResult(snippets=[{"text": "a", "source": "P01"}, {"text": "b", "source": "P01"}])


def test_overlong_memo_is_still_a_valid_memo():
    """Length is the Reviewer's job; the schema must not crash the run."""
    Memo(title="T", body="word " * 1000, footnotes=[])


def test_revise_verdict_needs_reasons():
    with pytest.raises(ValidationError):
        Verdict(decision="revise", reasons=[])
    assert Verdict(decision="accept", reasons=[]).decision == "accept"


def test_new_state_has_required_scratchpad_fields():
    required = {
        "topic",
        "snippets",
        "draft_memo",
        "review_verdict",
        "turn_count",
        "summary",
        "status",
    }
    assert required <= new_state("topic", "T01").keys()


def test_state_reducers_append_logs():
    """The messages and trace fields accumulate across nodes via LangGraph reducers."""

    def node(state):
        return {"messages": [AIMessage("reply")], "trace": [{"kind": "node"}]}

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    start = new_state("topic", "T01")
    start["messages"] = [HumanMessage("hello")]
    start["trace"] = [{"kind": "start"}]

    out = graph.compile().invoke(start)

    assert [m.content for m in out["messages"]] == ["hello", "reply"]
    assert [e["kind"] for e in out["trace"]] == ["start", "node"]

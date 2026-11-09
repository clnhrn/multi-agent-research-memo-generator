import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import core.llm
from core.llm import add_tokens, call, tokens_of


def test_add_tokens_becomes_unknown_once_usage_is_missing():
    assert add_tokens(10, 5) == 15
    assert add_tokens(10, None) is None
    assert add_tokens(None, 5) is None


def test_tokens_of_reads_provider_usage():
    usage = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert tokens_of(AIMessage("x", usage_metadata=usage)) == 10
    assert tokens_of(AIMessage("x")) is None


def test_call_builds_update_without_mutating_state(state, monkeypatch, tmp_logs):
    usage = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    fake = GenericFakeChatModel(messages=iter([AIMessage("draft text", usage_metadata=usage)]))
    monkeypatch.setattr(core.llm, "make_llm", lambda: fake)

    output, update = call(state, "writer", "system prompt", "user prompt")

    assert output.content == "draft text"
    assert update["total_tokens"] == 10
    assert [(m.name, m.content) for m in update["messages"]] == [
        ("writer", "user prompt"),
        ("writer", "draft text"),
    ]
    assert update["trace"][0]["kind"] == "llm_call"
    assert state["messages"] == [] and state["trace"] == []  # nodes return updates only
    logged = json.loads((tmp_logs / "run-T01.jsonl").read_text().splitlines()[0])
    assert logged["data"] == {"role": "writer", "tokens": 10}

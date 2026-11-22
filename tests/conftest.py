"""Shared fixtures: temporary logs and scratchpad, a fresh state and a faked model call."""

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

import core.memory
import core.scratchpad
import core.trace
import team
from core.schemas import new_state


@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    """Keep test traces out of the real logs/ directory."""
    monkeypatch.setattr(core.trace, "LOG_DIR", tmp_path / "logs")
    return tmp_path / "logs"


@pytest.fixture(autouse=True)
def tmp_scratchpad(tmp_path, monkeypatch):
    """Keep test runs out of the real scratchpad/ directory."""
    monkeypatch.setattr(core.scratchpad, "SCRATCHPAD_DIR", tmp_path / "scratchpad")
    return tmp_path / "scratchpad"


@pytest.fixture(autouse=True)
def offline_token_count(monkeypatch):
    """Count tokens without downloading a tokenizer or reading model settings."""
    monkeypatch.setattr(
        core.memory, "count_tokens", lambda messages: (count_tokens_approximately(messages), "test")
    )


@pytest.fixture
def state():
    """Return a fresh state for T01."""
    return new_state("Assess rooftop solar installation trends in Northstar.", "T01")


@pytest.fixture
def fake_call(monkeypatch):
    """Replace the model call with a fake that records (role, user prompt).

    Returns:
        A function that installs the fake. Its `output` argument is returned as is, or
        called with (state, role) to vary the reply.
    """

    def install(output):
        def _call(state, role, system, user, *, schema=None, tools=None):
            _call.calls.append((role, user))
            update = {
                "total_tokens": state["total_tokens"] + 10,
                "messages": [HumanMessage(user, name=role)],
                "trace": [{"kind": "llm_call", "data": {"role": role}}],
            }
            return output(state, role) if callable(output) else output, update

        _call.calls = []
        monkeypatch.setattr(team, "call", _call)
        monkeypatch.setattr(core.memory, "call", _call)
        return _call

    return install

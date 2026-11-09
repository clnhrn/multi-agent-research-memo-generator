import pytest

import core.trace
from core.schemas import new_state


@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    """Keep test traces out of the real logs/ directory."""
    monkeypatch.setattr(core.trace, "LOG_DIR", tmp_path / "logs")
    return tmp_path / "logs"


@pytest.fixture
def state():
    return new_state("Assess rooftop solar installation trends in Northstar.", "T01")

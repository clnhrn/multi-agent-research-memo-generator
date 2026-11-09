"""Static guardrails for the assignment's anti-goals."""

import ast
from pathlib import Path

import pytest

from core.config import ROOT

# Team code must reach the corpus only through MCP (anti-goal: no inline corpus).
TEAM_FILES = [ROOT / "team.py", *sorted((ROOT / "core").glob("*.py"))]
ALLOWED_DIRECT_CORPUS = {ROOT / "core" / "corpus.py", ROOT / "core" / "config.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
    return names


@pytest.mark.parametrize("path", TEAM_FILES, ids=lambda p: p.name)
def test_team_code_does_not_bypass_mcp(path):
    if path in ALLOWED_DIRECT_CORPUS:
        return
    imports = _imports(path)
    assert "core.corpus" not in imports
    assert not {"core.config.CORPUS_CSV", "core.config.MARKET_CORPUS_MD"} & imports


def _used_in(function: str) -> set[str]:
    """Names and string literals used inside a top-level function of team.py."""
    tree = ast.parse((ROOT / "team.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    used = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            used.add(child.id)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            used.add(child.value)
    return used


MCP_NAMES = {"mcp_session", "search", "_research"}


@pytest.mark.parametrize("role", ["writer", "reviewer"])
def test_only_the_researcher_touches_mcp(role):
    """R2: crossing role boundaries (e.g. the writer calling MCP) is a fail."""
    assert not MCP_NAMES & _used_in(role)


def test_reviewer_reads_only_the_memo():
    """R2: the reviewer gets the writer's output and the static rubric, nothing else."""
    assert not {"snippets", "topic", "messages"} & _used_in("reviewer")


@pytest.mark.parametrize("role", ["researcher", "writer", "reviewer"])
def test_workers_never_read_the_memory_log(role):
    """Anti-goal: no free-form chat; workers use typed fields, not `messages`."""
    assert "messages" not in _used_in(role)

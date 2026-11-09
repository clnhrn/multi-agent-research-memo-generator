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

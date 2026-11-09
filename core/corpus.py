"""Direct corpus access for the MCP server and the evaluator only.

Team code must never import this module: the Researcher reaches the corpus
solely through the MCP `search_corpus` tool.
"""

import csv
from functools import lru_cache

from core.config import CORPUS_CSV


@lru_cache(maxsize=1)
def load_rows() -> tuple[dict[str, str], ...]:
    with CORPUS_CSV.open(encoding="utf-8", newline="") as f:
        return tuple(csv.DictReader(f))


def valid_ids() -> frozenset[str]:
    return frozenset(row["id"] for row in load_rows())

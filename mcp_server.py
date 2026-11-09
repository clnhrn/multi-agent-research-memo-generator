"""MCP stdio server for the fictional market corpus.

Exposes the corpus CSV and glossary as resources and `search_corpus` as a tool.
Retrieval is a small lexical ranker: light suffix stripping so word forms
match, IDF weighting so rare words (company names) outweigh common ones
("Northstar"), an extra weight for names in the query, and a relative cut-off
so weak matches are dropped instead of padding the results.
"""

import math
import re
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from core.config import CORPUS_CSV, GLOSSARY_MD
from core.corpus import load_rows

NAME_WEIGHT = 2.0  # capitalised query words (e.g. SwiftCart) count double
MIN_RELATIVE_SCORE = 0.5  # keep results scoring at least half the best match
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s", "y", "e")
_STOPWORDS = frozenset(
    "a an and as assess at by for from in is its of on the than to trend trends with".split()
)
_WORD = re.compile(r"[a-z0-9]+")
_NAME = re.compile(r"(?<!^)\b[A-Z][a-zA-Z]+")  # capitalised, not the first word

server = FastMCP("Week 18 fictional market corpus")


def _stem(word: str) -> str:
    """Strip one common suffix, e.g. installations/installed/install -> install."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, set[str]], dict[str, float]]:
    """Terms per paragraph and smoothed IDF per term, built once."""
    docs = {row["id"]: _terms(row["text"]) for row in load_rows()}
    n = len(docs)
    doc_freq: dict[str, int] = {}
    for terms in docs.values():
        for term in terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1
    idf = {term: math.log((n + 1) / (df + 1)) for term, df in doc_freq.items()}
    return docs, idf


def rank(query: str) -> list[tuple[float, str]]:
    """(score, source ID) for paragraphs above the cut-off, best first."""
    docs, idf = _index()
    query_terms = _terms(query)
    names = {_stem(w.lower()) for w in _NAME.findall(query)}
    scored = [
        (sum(idf[t] * (NAME_WEIGHT if t in names else 1.0) for t in query_terms & terms), sid)
        for sid, terms in docs.items()
    ]
    scored = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))
    if not scored:
        return []
    cutoff = MIN_RELATIVE_SCORE * scored[0][0]
    return [s for s in scored if s[0] >= cutoff]


@server.resource("corpus://market/corpus.csv", mime_type="text/csv")
def corpus() -> str:
    return CORPUS_CSV.read_text(encoding="utf-8")


@server.resource("corpus://market/glossary.md", mime_type="text/markdown")
def glossary() -> str:
    return GLOSSARY_MD.read_text(encoding="utf-8")


@server.tool()
def search_corpus(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search the market corpus. Returns up to `limit` relevant paragraphs, best
    first, each as {"text", "source"} where source is a citation ID like P01.
    Returns an empty list when nothing is relevant."""
    rows = {row["id"]: row["text"] for row in load_rows()}
    if not 1 <= limit <= len(rows):
        raise ValueError(f"limit must be between 1 and {len(rows)}")
    return [{"text": rows[sid], "source": sid} for _, sid in rank(query)[:limit]]


if __name__ == "__main__":
    server.run(transport="stdio")

"""Retrieval ranking, and a real MCP round trip over stdio."""

import asyncio
import json

import pytest

import mcp_server
from core.config import ROOT
from core.mcp_client import mcp_session, search
from core.schemas import Snippet

TOPICS = {t["id"]: t["topic"] for t in json.loads((ROOT / "topics.json").read_text())}

# The two paragraphs each topic is actually about. The corpus has no others,
# which is why the 3-snippet minimum needs a policy in Phase 3.
ON_TOPIC = {
    "T01": {"P01", "P02"},
    "T02": {"P03", "P04"},
    "T03": {"P05", "P06"},
    "T04": {"P07", "P08"},
    "T05": {"P09", "P10"},
}
RELATED_CONTEXT = {"T03": {"P08"}}  # grocery survey on delivery reliability


@pytest.mark.parametrize("topic_id", sorted(ON_TOPIC))
def test_on_topic_paragraphs_rank_first(topic_id):
    sources = [r["source"] for r in mcp_server.search_corpus(TOPICS[topic_id])]
    assert set(sources[:2]) == ON_TOPIC[topic_id]
    assert set(sources) <= ON_TOPIC[topic_id] | RELATED_CONTEXT.get(topic_id, set())


@pytest.mark.parametrize(
    ("words", "stem"),
    [
        ("installation installations installed", "install"),
        ("price prices pricing", "pric"),
        ("consultation consultations", "consult"),
        ("delivery deliveries", "deliver"),
    ],
)
def test_word_forms_share_a_stem(words, stem):
    assert mcp_server._terms(words) == {stem}


def test_lowercase_query_still_finds_the_company():
    sources = [r["source"] for r in mcp_server.search_corpus("swiftcart delivery costs")]
    assert sources[:2] == ["P06", "P05"]


def test_name_at_start_of_query_still_counts_as_a_name():
    """Researcher sub-queries start with the subject; 'volumes' must not pull in P07."""
    sources = {r["source"] for r in mcp_server.search_corpus("CareBridge consultation volumes")}
    assert sources == {"P09", "P10"}


def test_irrelevant_query_returns_nothing():
    assert mcp_server.search_corpus("quantum computing") == []


def test_limit_is_respected_and_validated():
    assert len(mcp_server.search_corpus(TOPICS["T03"], limit=1)) == 1
    with pytest.raises(ValueError):
        mcp_server.search_corpus("solar", limit=0)


def test_stdio_round_trip():
    """Real MCP over stdio: discovery, resource schema and a validated search."""

    async def run():
        async with mcp_session() as session:
            resources = (await session.list_resources()).resources
            tools = (await session.list_tools()).tools
            snippets = await search(session, TOPICS["T02"])
        return resources, tools, snippets

    resources, tools, snippets = asyncio.run(run())

    assert {(str(r.uri), r.mimeType) for r in resources} == {
        ("corpus://market/corpus.csv", "text/csv"),
        ("corpus://market/glossary.md", "text/markdown"),
    }
    assert [t.name for t in tools] == ["search_corpus"]
    assert all(isinstance(s, Snippet) for s in snippets)
    assert {s.source for s in snippets} == ON_TOPIC["T02"]

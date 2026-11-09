"""Research memo team: a LangGraph supervisor coordinating three bounded workers.

Role boundaries (R2):
- Researcher: topic in, MCP `search_corpus` only, ResearchResult out.
- Writer: topic + snippets (+ previous draft and reviewer reasons) in, Memo out. No MCP.
- Reviewer: memo + static rubric in, Verdict out. Never sees the evidence. Its model
  proposes violations; code verifies them and computes the decision.
Workers read typed state fields only, never the `messages` memory log.
"""

import asyncio
from typing import Any, Literal

from langgraph.graph import START, StateGraph
from langgraph.types import Command

from core.config import MAX_SNIPPETS, MIN_SNIPPETS
from core.llm import call
from core.mcp_client import mcp_session, search
from core.prompts import (
    RESEARCHER_PROMPT,
    REVIEWER_PROMPT,
    WRITER_PROMPT,
    researcher_request,
    reviewer_request,
    writer_request,
)
from core.schemas import (
    Memo,
    ResearchResult,
    ReviewFindings,
    Snippet,
    State,
    SubQueries,
    Verdict,
    Violation,
)
from core.text import (
    best_sentence,
    body_word_count,
    check_memo,
    extract_citations,
    has_citation,
    has_figure,
    render_memo,
    states_missing_data,
)
from core.trace import log_event


def hand_off_to_researcher() -> Command:
    return Command(goto="researcher")


def hand_off_to_writer() -> Command:
    return Command(goto="writer")


def hand_off_to_reviewer() -> Command:
    return Command(goto="reviewer")


def supervisor(state: State) -> Command[Literal["researcher", "writer", "reviewer", "__end__"]]:
    # TODO (Phase 4): select only typed handoff tools, enforce 8 turns and 3 rejections,
    # update/persist state after each supervisor turn, handle completion/escalation.
    raise NotImplementedError("Implement supervisor routing and persistence")


# --- Researcher -----------------------------------------------------------------


def sources(snippets: list[Snippet]) -> list[str]:
    return [s.source for s in snippets]


async def _research(state: State) -> dict[str, Any]:
    """Search the topic via MCP; if fewer than MIN_SNIPPETS, try typed sub-queries.

    Never pads with irrelevant text: a shortfall is recorded, not hidden.
    """
    topic, update, trace = state["topic"], {}, []
    async with mcp_session() as session:

        async def run(query: str) -> list[Snippet]:
            found = await search(session, query)
            trace.append(log_event(state, "mcp_search", query=query, sources=sources(found)))
            return found

        snippets = await run(topic)
        if len(snippets) < MIN_SNIPPETS:
            plan, update = call(
                state,
                "researcher",
                RESEARCHER_PROMPT,
                researcher_request(topic, [s.model_dump() for s in snippets]),
                schema=SubQueries,
            )
            trace += update.pop("trace")
            for query in plan.queries:
                seen = set(sources(snippets))
                snippets += [s for s in await run(query) if s.source not in seen]

    snippets = snippets[:MAX_SNIPPETS]
    if snippets:
        snippets = ResearchResult(snippets=snippets).snippets
    searches = sum(e["kind"] == "mcp_search" for e in trace)
    trace.append(
        log_event(
            state,
            "research_done",
            sources=sources(snippets),
            shortfall=len(snippets) < MIN_SNIPPETS,
        )
    )
    return {
        **update,
        "snippets": [s.model_dump() for s in snippets],
        "mcp_calls": state["mcp_calls"] + searches,
        "trace": trace,
    }


def researcher(state: State) -> Command[Literal["supervisor"]]:
    return Command(goto="supervisor", update=asyncio.run(_research(state)))


# --- Writer ---------------------------------------------------------------------


def writer(state: State) -> Command[Literal["supervisor"]]:
    verdict = state["review_verdict"]
    revising = verdict is not None and verdict["decision"] == "revise"
    request = writer_request(
        state["topic"],
        state["snippets"],
        previous_draft=render_memo(state["draft_memo"]) if revising else None,
        reasons=verdict["reasons"] if revising else None,
    )
    memo, update = call(state, "writer", WRITER_PROMPT, request, schema=Memo)
    evidence = {s["source"] for s in state["snippets"]}
    event = log_event(
        state,
        "draft",
        revision=revising,
        body_words=body_word_count(memo.body),
        cited=sorted(set(extract_citations(memo.body))),
        # The Reviewer never sees the evidence, so stray IDs are logged here.
        stray_citations=sorted(set(extract_citations(memo.body)) - evidence),
    )
    return Command(
        goto="supervisor",
        update={
            **update,
            "draft_memo": memo.model_dump(),
            "review_verdict": None,  # the new draft has not been reviewed yet
            "trace": [*update["trace"], event],
        },
    )


# --- Reviewer -------------------------------------------------------------------


QUOTE_MATCH = 0.8  # share of a quote's words that must appear in one memo sentence
# A cited sentence is attributed to the evidence, so it cannot be one of these.
_NEEDS_ATTRIBUTION = {"cause_or_motive", "prediction", "recommendation", "outside_claim"}


def verify(body: str, violations: list[Violation]) -> tuple[list[Violation], list[dict]]:
    """Keep only violations the memo text supports; return (kept, dropped with reasons).

    The model proposes, code verifies: a quote must exist in the memo, an attributed
    (cited) sentence cannot be an unsupported claim, and an uncited figure needs a figure.
    """
    kept, dropped = [], []
    for v in violations:
        sentence, overlap = best_sentence(body, v.quote)
        if sentence is None or overlap < QUOTE_MATCH:
            why = "quote not found in memo"
        elif v.kind in _NEEDS_ATTRIBUTION and has_citation(sentence):
            why = "sentence is attributed to a cited source"
        elif v.kind == "outside_claim" and states_missing_data(sentence):
            why = "sentence states missing data, which the rubric allows"
        elif v.kind == "uncited_figure" and (has_citation(sentence) or not has_figure(sentence)):
            why = "sentence has a citation or no figure"
        else:
            kept.append(v.model_copy(update={"quote": sentence}))
            continue
        dropped.append({"kind": v.kind, "quote": v.quote, "why": why})
    return kept, dropped


def reviewer(state: State) -> Command[Literal["supervisor"]]:
    memo = state["draft_memo"]
    failures = check_memo(memo)
    findings, update = call(
        state,
        "reviewer",
        REVIEWER_PROMPT,
        reviewer_request(render_memo(memo), failures),
        schema=ReviewFindings,
    )
    kept, dropped = verify(memo["body"], findings.violations)
    reasons = [*failures, *(f'{v.kind}: "{v.quote}" ({v.explanation})' for v in kept)]
    verdict = Verdict(decision="revise" if reasons else "accept", reasons=reasons)
    event = log_event(
        state,
        "verdict",
        decision=verdict.decision,
        reasons=verdict.reasons,
        automatic_failures=failures,
        dropped_findings=dropped,
    )
    return Command(
        goto="supervisor",
        update={
            **update,
            "review_verdict": verdict.model_dump(),
            "rejections": state["rejections"] + (verdict.decision == "revise"),
            "trace": [*update["trace"], event],
        },
    )


# --- Graph and adapter ----------------------------------------------------------


def build_graph():
    graph = StateGraph(State)
    for name, node in [
        ("supervisor", supervisor),
        ("researcher", researcher),
        ("writer", writer),
        ("reviewer", reviewer),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "supervisor")
    return graph.compile()


def run_topic(topic: str, topic_id: str) -> dict:
    """Implement adapter contract documented in EVALUATION.md."""
    # TODO (Phase 7): initialise/resume state, run graph, capture metrics and MCP evidence.
    raise NotImplementedError("Implement run_topic adapter")

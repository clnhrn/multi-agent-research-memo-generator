"""Research memo team: a LangGraph supervisor coordinating three bounded workers.

Role boundaries (R1, R2):
- Supervisor: a status line from typed fields in, one handoff tool out. Stops are code.
- Researcher: topic in, MCP `search_corpus` only, ResearchResult out.
- Writer: topic + snippets (+ previous draft and reviewer reasons) in, Memo out. No MCP.
  Checks its own draft's length and fixes it once before handing over.
- Reviewer: memo + static rubric in, Verdict out. Never sees the evidence. Its model
  proposes violations; code verifies them and computes the decision.
Workers read typed state fields only, never the `messages` memory log.
"""

import argparse
import asyncio
import json
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from core.config import (
    MAX_REJECTIONS,
    MAX_SNIPPETS,
    MAX_TURNS,
    MIN_SNIPPETS,
    TOPICS_JSON,
    check_run_id,
)
from core.llm import call
from core.mcp_client import mcp_session, search
from core.prompts import (
    RESEARCHER_PROMPT,
    REVIEWER_PROMPT,
    SUPERVISOR_PROMPT,
    WRITER_PROMPT,
    length_fix_request,
    researcher_request,
    reviewer_request,
    supervisor_request,
    writer_request,
)
from core.schemas import (
    TERMINAL_STATUSES,
    Memo,
    ResearchResult,
    ReviewFindings,
    Snippet,
    State,
    Status,
    SubQueries,
    Verdict,
    Violation,
    apply_update,
    new_state,
)
from core.scratchpad import ScratchpadError, load_state, save_state
from core.text import (
    best_sentence,
    body_word_count,
    check_memo,
    extract_citations,
    has_citation,
    has_figure,
    length_ok,
    render_memo,
    states_missing_data,
)
from core.trace import log_event

# --- Supervisor -----------------------------------------------------------------


@tool
def hand_off_to_researcher() -> Command:
    """Hand off to the researcher to gather evidence for the topic."""
    return Command(goto="researcher")


@tool
def hand_off_to_writer() -> Command:
    """Hand off to the writer to draft the memo, or to revise a rejected draft."""
    return Command(goto="writer")


@tool
def hand_off_to_reviewer() -> Command:
    """Hand off to the reviewer to check a draft that has not been reviewed yet."""
    return Command(goto="reviewer")


# The supervisor's whole tool list (R1). No free-text arguments: handoffs dispatch, never chat.
HANDOFFS = {t.name: t for t in (hand_off_to_researcher, hand_off_to_writer, hand_off_to_reviewer)}


def stop_reason(state: State) -> tuple[Status, str] | None:
    """Check the hard stops, in code.

    Accept is first so a memo accepted on the last turn counts as completed.

    Args:
        state: Current run state.

    Returns:
        The final status and the reason, or None if the run should continue.
    """
    verdict = state["review_verdict"]
    if verdict is not None and verdict["decision"] == "accept":
        return "completed", "reviewer accepted the memo"
    if state["rejections"] >= MAX_REJECTIONS:
        return "escalated", f"{state['rejections']} reviewer rejections: needs a human"
    if state["mcp_calls"] > 0 and not state["snippets"]:
        return "escalated", "research found no evidence: needs a human"
    if state["turn_count"] >= MAX_TURNS:
        return "max_turns", f"reached the cap of {MAX_TURNS} supervisor turns"
    return None


def awaiting_review(state: State) -> bool:
    """Return whether there is a draft that has not been reviewed yet."""
    return state["draft_memo"] is not None and state["review_verdict"] is None


def blocked(state: State, names: list[str]) -> str | None:
    """Check whether the model's handoff can run now.

    Args:
        state: Current run state.
        names: Names of the tools the model called.

    Returns:
        Why the handoff cannot run, or None if it can.
    """
    if len(names) != 1 or names[0] not in HANDOFFS:
        return "expected exactly one known handoff"
    if names[0] == hand_off_to_writer.name and not state["snippets"]:
        return "writer needs evidence"
    if names[0] == hand_off_to_reviewer.name and not awaiting_review(state):
        return "reviewer needs an unreviewed draft"
    return None


def fallback_handoff(state: State) -> BaseTool:
    """Choose the handoff to use when the model's choice is blocked.

    Args:
        state: Current run state.

    Returns:
        The handoff tool for the next missing step.
    """
    if not state["snippets"]:
        return hand_off_to_researcher
    return hand_off_to_reviewer if awaiting_review(state) else hand_off_to_writer


def supervisor(state: State) -> Command[Literal["researcher", "writer", "reviewer", "__end__"]]:
    """Take one turn: code checks the stops, the model picks a handoff, code checks it.

    The scratchpad is saved twice: on entry, so the worker that just finished is not
    repeated after a kill, and at the end of the turn with the pending handoff.

    Args:
        state: Current run state.

    Returns:
        A command to the chosen worker, or to END with the final status.
    """
    state = {**state, "pending": None}
    save_state(state)

    stop = stop_reason(state)
    if stop is not None:
        status, reason = stop
        event = log_event(state, "stop", status=status, reason=reason)
        update = {"status": status, "pending": None, "trace": [event]}
        save_state(apply_update(state, update))
        return Command(goto=END, update=update)

    # A turn is one handoff decision; the stop check above makes no decision and is not counted.
    state = {**state, "turn_count": state["turn_count"] + 1}
    reply, update = call(
        state,
        "supervisor",
        SUPERVISOR_PROMPT,
        supervisor_request(state),
        tools=list(HANDOFFS.values()),
    )
    chosen = [c["name"] for c in reply.tool_calls]
    problem = blocked(state, chosen)
    handoff = fallback_handoff(state) if problem else HANDOFFS[chosen[0]]
    event = log_event(
        state, "supervisor_decision", chosen=chosen, handoff=handoff.name, redirected=problem
    )
    worker = handoff.invoke({}).goto
    update = {
        **update,
        "turn_count": state["turn_count"],
        "pending": worker,
        "trace": [*update["trace"], event],
    }
    save_state(apply_update(state, update))
    return Command(goto=worker, update=update)


# --- Researcher -----------------------------------------------------------------


def sources(snippets: list[Snippet]) -> list[str]:
    """Return the source ID of each snippet."""
    return [s.source for s in snippets]


async def _research(state: State) -> dict[str, Any]:
    """Search the topic via MCP; if fewer than MIN_SNIPPETS, try typed sub-queries.

    Never pads with irrelevant text: a shortfall is recorded, not hidden.

    Args:
        state: Current run state; only the topic and counters are used.

    Returns:
        The state update with snippets, MCP call count and trace.
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
    """Gather evidence through MCP `search_corpus`, the only role that may.

    Args:
        state: Current run state.

    Returns:
        A command to the supervisor with the snippets.
    """
    return Command(goto="supervisor", update=asyncio.run(_research(state)))


# --- Writer ---------------------------------------------------------------------


def fix_length(
    state: State, request: str, memo: Memo, update: dict[str, Any]
) -> tuple[Memo, dict[str, Any]]:
    """Ask the writer once to bring its own draft into the length range.

    Cheaper than a rejection, which costs two turns and one of the allowed rejections.

    Args:
        state: Current run state.
        request: The request the draft answered.
        memo: The draft that is too short or too long.
        update: The state update of the call that wrote the draft.

    Returns:
        The fixed memo, or the draft if the fix is out of range too, and the update
        covering both calls.
    """
    words = body_word_count(memo.body)
    fixed, fix_update = call(
        apply_update(state, update),
        "writer",
        WRITER_PROMPT,
        length_fix_request(request, render_memo(memo.model_dump()), words),
        schema=Memo,
    )
    kept = length_ok(fixed.body)
    event = log_event(
        state, "length_fix", before=words, after=body_word_count(fixed.body), kept=kept
    )
    return fixed if kept else memo, {
        **fix_update,
        "messages": [*update["messages"], *fix_update["messages"]],
        "trace": [*update["trace"], *fix_update["trace"], event],
    }


def writer(state: State) -> Command[Literal["supervisor"]]:
    """Draft the memo from the snippets, or revise a rejected draft.

    Args:
        state: Current run state.

    Returns:
        A command to the supervisor with the new, unreviewed draft.
    """
    verdict = state["review_verdict"]
    revising = verdict is not None and verdict["decision"] == "revise"
    request = writer_request(
        state["topic"],
        state["snippets"],
        previous_draft=render_memo(state["draft_memo"]) if revising else None,
        reasons=verdict["reasons"] if revising else None,
    )
    memo, update = call(state, "writer", WRITER_PROMPT, request, schema=Memo)
    if not length_ok(memo.body):
        memo, update = fix_length(state, request, memo, update)
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
    """Keep only the violations the memo text supports.

    The model proposes, code verifies: a quote must exist in the memo, an attributed
    (cited) sentence cannot be an unsupported claim, and an uncited figure needs a figure.

    Args:
        body: The memo body.
        violations: Violations proposed by the Reviewer model.

    Returns:
        The violations kept, and the dropped ones with the reason for each.
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
    """Review the draft against the static rubric, without seeing the evidence.

    Args:
        state: Current run state; only the draft and counters are used.

    Returns:
        A command to the supervisor with the verdict.
    """
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


def entry_point(state: State) -> str:
    """Return the first node: the pending worker of a killed run, otherwise the supervisor."""
    return state["pending"] or "supervisor"


def build_graph():
    """Return the compiled graph; a fresh run starts at the supervisor."""
    graph = StateGraph(State)
    nodes = {
        "supervisor": supervisor,
        "researcher": researcher,
        "writer": writer,
        "reviewer": reviewer,
    }
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_conditional_edges(START, entry_point, list(nodes))
    return graph.compile()


def start_state(topic: str, topic_id: str, run_id: str | None = None, fresh: bool = False) -> State:
    """Load an unfinished run from the scratchpad, or start a new one.

    Args:
        topic: The research topic.
        topic_id: The topic's ID, e.g. T01.
        run_id: The run's ID; defaults to `topic_id`.
        fresh: Ignore any saved run.

    Returns:
        The state to run the graph from.

    Raises:
        ScratchpadError: If the saved run is corrupt or is about another topic.
        ValueError: If the run ID is not a safe file name.
    """
    saved = None if fresh else load_state(run_id or topic_id)
    if saved is None or saved["status"] in TERMINAL_STATUSES:
        state = new_state(topic, topic_id, run_id)
        event = log_event(state, "run_started", topic=topic)
    else:
        if saved["topic"] != topic:
            raise ScratchpadError(f"run {saved['run_id']} is about another topic; use --fresh")
        state = saved
        event = log_event(state, "resumed", pending=state["pending"])
    return apply_update(state, {"trace": [event]})


def run_team(topic: str, topic_id: str, run_id: str | None = None, fresh: bool = False) -> State:
    """Run the team on one topic, resuming an unfinished run if there is one.

    Args:
        topic: The research topic.
        topic_id: The topic's ID, e.g. T01.
        run_id: The run's ID; defaults to `topic_id`.
        fresh: Ignore any saved run.

    Returns:
        The final state.

    Raises:
        ScratchpadError: If the saved run is corrupt or is about another topic.
        ValueError: If the run ID is not a safe file name.
    """
    return build_graph().invoke(start_state(topic, topic_id, run_id, fresh))


def run_topic(topic: str, topic_id: str, *, fresh: bool = False) -> dict:
    """Run the team on one topic, following the adapter contract in EVALUATION.md.

    Args:
        topic: The research topic.
        topic_id: The topic's ID, e.g. T01.
        fresh: Ignore any saved run.

    Returns:
        The result dict described in EVALUATION.md.

    Raises:
        NotImplementedError: Always, until Phase 7.
    """
    # TODO (Phase 7): call run_team and build the result from the final state.
    raise NotImplementedError("Implement run_topic adapter")


def main() -> None:
    """Run one topic from the command line, resuming by default."""
    topics = {t["id"]: t["topic"] for t in json.loads(TOPICS_JSON.read_text(encoding="utf-8"))}
    parser = argparse.ArgumentParser(description="Run the research memo team on one topic.")
    parser.add_argument("topic_id", choices=sorted(topics))
    parser.add_argument(
        "--run-id", type=check_run_id, help="scratchpad and log name (default: the topic ID)"
    )
    parser.add_argument("--fresh", action="store_true", help="ignore any saved run")
    args = parser.parse_args()
    try:
        final = run_team(topics[args.topic_id], args.topic_id, args.run_id, args.fresh)
    except ScratchpadError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"{final['status']}: {final['turn_count']} turns, {final['rejections']} rejections")


if __name__ == "__main__":
    main()

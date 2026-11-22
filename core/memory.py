"""Rolling-summary memory (R5): compact the message log once it grows past a threshold.

Code writes the facts the summary must preserve, from typed state; the model only adds a
short narrative. The facts are rebuilt from the trace every time, so they never double.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from core.config import COMPACTION_TOKENS, KEEP_RECENT_MESSAGES, MAX_REJECTIONS, make_llm
from core.llm import call
from core.prompts import COMPACTION_PROMPT, compaction_request
from core.schemas import State
from core.text import body_word_count
from core.trace import log_event

SUMMARY_NAME = "summary"


def count_tokens(messages: Sequence[BaseMessage]) -> tuple[int, str]:
    """Count the tokens of the message log; used for the trigger only, never reported as usage.

    Args:
        messages: The message log.

    Returns:
        The count and the counter used. The estimate is the fallback when the tokenizer
        cannot be downloaded or does not know the model.
    """
    try:
        return make_llm().get_num_tokens_from_messages(messages), "exact"
    except (NotImplementedError, OSError):
        return count_tokens_approximately(messages), "estimate"


def facts(state: State) -> str:
    """Write the facts every summary must preserve.

    Args:
        state: Current run state.

    Returns:
        The topic, handoffs, current draft length and every rejection with its reasons.
    """
    events = state["trace"]
    handoffs = [
        e["data"]["handoff"].removeprefix("hand_off_to_")
        for e in events
        if e["kind"] == "supervisor_decision"
    ]
    rejections = [
        e["data"]["reasons"]
        for e in events
        if e["kind"] == "verdict" and e["data"]["decision"] == "revise"
    ]
    memo = state["draft_memo"]
    lines = [
        f"Topic: {state['topic']}",
        f"Decisions (handoffs): {', '.join(handoffs) or 'none'}",
        f"Current draft: {body_word_count(memo['body'])} words" if memo else "Current draft: none",
        f"Reviewer rejections: {len(rejections)} of {MAX_REJECTIONS}",
        *(f"- Rejection {i}: {' | '.join(reasons)}" for i, reasons in enumerate(rejections, 1)),
    ]
    return "\n".join(lines)


def maybe_compact(state: State) -> dict[str, Any]:
    """Replace older messages with one summary when the log exceeds the threshold.

    Idempotent: after a compaction the only older message is the summary itself, so a
    second call changes nothing and makes no model call.

    Args:
        state: Current run state.

    Returns:
        The state update, empty if there is nothing to compact.
    """
    before, counter = count_tokens(state["messages"])
    if before <= COMPACTION_TOKENS:
        return {}
    older = state["messages"][:-KEEP_RECENT_MESSAGES]
    recent = state["messages"][-KEEP_RECENT_MESSAGES:]
    if all(m.name == SUMMARY_NAME for m in older):
        return {}

    reply, update = call(state, "memory", COMPACTION_PROMPT, compaction_request(older))
    summary = f"{facts(state)}\nNotes: {str(reply.content).strip()}"
    messages = [AIMessage(summary, name=SUMMARY_NAME), *recent]
    event = log_event(
        state,
        "compaction",
        before=before,
        after=count_tokens(messages)[0],
        removed=len(older),
        threshold=COMPACTION_TOKENS,
        counter=counter,
    )
    return {
        "total_tokens": update["total_tokens"],
        "summary": summary,
        # The call's own messages hold the whole old log, so they are not kept.
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages],
        "trace": [*update["trace"], event],
    }

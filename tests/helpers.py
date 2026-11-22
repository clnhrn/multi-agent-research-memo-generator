"""A scripted model for whole-graph tests."""

from langchain_core.messages import AIMessage

import team
from core.schemas import Memo, ReviewFindings, SubQueries, Violation

TOPIC = "Assess rooftop solar installation trends in Northstar."
GOOD = Memo(title="T", body="word " * 260 + "[P01].", footnotes=["[P01] Solar"])
CLAIM = "Demand will surely rise."
BAD = GOOD.model_copy(update={"body": f"{GOOD.body} {CLAIM}"})
FINDING = Violation(kind="prediction", quote=CLAIM, explanation="unsupported")


def pick(*roles):
    """Build a supervisor reply that calls the handoff tool of each role."""
    calls = [
        {"name": f"hand_off_to_{role}", "args": {}, "id": str(i)} for i, role in enumerate(roles)
    ]
    return AIMessage("", tool_calls=calls)


def scripted(*, rejections=0, choose=None):
    """Build a model that rejects the first `rejections` drafts.

    The supervisor makes the sensible handoff unless `choose(state)` is given.
    """

    def reply(state, role):
        if role == "supervisor":
            return choose(state) if choose else pick(team.fallback_handoff(state).invoke({}).goto)
        if role == "researcher":
            return SubQueries(queries=["Northstar solar prices"])
        rejecting = state["rejections"] < rejections
        if role == "writer":
            return BAD if rejecting else GOOD
        return ReviewFindings(violations=[FINDING] if rejecting else [])

    return reply


def route(final):
    """Return the workers the supervisor handed off to, in order."""
    return [
        e["data"]["handoff"].removeprefix("hand_off_to_")
        for e in final["trace"]
        if e["kind"] == "supervisor_decision"
    ]

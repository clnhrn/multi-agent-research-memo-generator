"""Typed outputs for each role and the shared scratchpad state.

Pydantic models validate worker outputs at the boundary; the graph state
stores their `.model_dump()` so the scratchpad is plain JSON.

Schemas check *shape*, not quality: an over-long memo is still a valid Memo
that the Reviewer sends back, rather than an exception that ends the run.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator

from core.config import MAX_SNIPPETS, MAX_SUBQUERIES
from core.text import SOURCE_ID


class Snippet(BaseModel):
    text: str
    source: str = Field(description="Corpus source ID, e.g. P01")

    @field_validator("source")
    @classmethod
    def _source_format(cls, value: str) -> str:
        if not SOURCE_ID.fullmatch(value):
            raise ValueError(f"source must look like P01, got {value!r}")
        return value


class ResearchResult(BaseModel):
    snippets: list[Snippet]

    @field_validator("snippets")
    @classmethod
    def _bounded_and_unique(cls, snippets: list[Snippet]) -> list[Snippet]:
        if not 1 <= len(snippets) <= MAX_SNIPPETS:
            raise ValueError(f"expected 1-{MAX_SNIPPETS} snippets, got {len(snippets)}")
        sources = [s.source for s in snippets]
        if len(set(sources)) != len(sources):
            raise ValueError("duplicate snippet sources")
        return snippets


class SubQueries(BaseModel):
    """The Researcher's plan for extra MCP searches."""

    queries: list[str] = Field(description="Short keyword queries, each naming the subject")

    @field_validator("queries")
    @classmethod
    def _cap(cls, queries: list[str]) -> list[str]:
        return [q.strip() for q in queries if q.strip()][:MAX_SUBQUERIES]


class Memo(BaseModel):
    title: str
    body: str = Field(description="Plain prose with inline citations like [P01]")
    footnotes: list[str] = Field(description='One per cited ID, e.g. "[P01] Solar capacity"')


ViolationKind = Literal[
    "uncited_figure", "cause_or_motive", "prediction", "recommendation", "outside_claim"
]


class Violation(BaseModel):
    """One rubric violation the Reviewer model proposes; code verifies it before it counts."""

    kind: ViolationKind
    quote: str = Field(description="The full offending sentence, copied exactly from the memo")
    explanation: str


class ReviewFindings(BaseModel):
    """The Reviewer model's raw output. The Verdict is computed from it in code."""

    violations: list[Violation] = Field(description="Empty if the memo has no violations")


class Verdict(BaseModel):
    decision: Literal["accept", "revise"]
    reasons: list[str]

    @model_validator(mode="after")
    def _revise_needs_reasons(self) -> "Verdict":
        if self.decision == "revise" and not self.reasons:
            raise ValueError("a 'revise' verdict must give at least one reason")
        return self


Status = Literal["running", "completed", "escalated", "max_turns", "error"]
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "escalated", "max_turns", "error"})


class State(TypedDict):
    # Required scratchpad fields (R4)
    topic: str
    snippets: list[dict[str, Any]]  # ResearchResult.snippets, dumped
    draft_memo: dict[str, Any] | None  # Memo, dumped
    review_verdict: dict[str, Any] | None  # Verdict, dumped
    turn_count: int
    summary: str
    status: Status
    # Bookkeeping for metrics, memory and evidence
    run_id: str
    topic_id: str
    rejections: int
    mcp_calls: int
    total_tokens: int | None  # provider-reported; None once any call lacks usage
    messages: Annotated[list[AnyMessage], add_messages]  # memory log, never read by workers
    trace: Annotated[list[dict[str, Any]], operator.add]  # append-only event log


def new_state(topic: str, topic_id: str, run_id: str | None = None) -> State:
    return State(
        topic=topic,
        snippets=[],
        draft_memo=None,
        review_verdict=None,
        turn_count=0,
        summary="",
        status="running",
        run_id=run_id or topic_id,
        topic_id=topic_id,
        rejections=0,
        mcp_calls=0,
        total_tokens=0,
        messages=[],
        trace=[],
    )

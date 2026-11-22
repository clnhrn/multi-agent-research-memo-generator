"""Thin helper around one model call: invoke, then build the state update.

Uses LangChain primitives (with_structured_output, bind_tools); this is
bookkeeping for tokens, memory log and trace, not an agent abstraction.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from core.config import make_llm
from core.trace import log_event


def tokens_of(message: BaseMessage) -> int | None:
    """Read the provider-reported token count of a model reply.

    Args:
        message: The model's reply.

    Returns:
        Total tokens, or None if the provider sent no usage.
    """
    usage = getattr(message, "usage_metadata", None)
    return usage.get("total_tokens") if usage else None


def add_tokens(total: int | None, new: int | None) -> int | None:
    """Add to a running total that becomes unknown once any call lacks usage.

    Args:
        total: Tokens so far, or None if already unknown.
        new: Tokens of the latest call, or None if not reported.

    Returns:
        The new total, or None if either value is unknown.
    """
    return None if total is None or new is None else total + new


def _log_text(output: Any, message: AIMessage) -> str:
    # Check messages first: LangChain messages are themselves Pydantic models.
    if isinstance(output, BaseMessage):
        if message.tool_calls:
            return "called " + ", ".join(call["name"] for call in message.tool_calls)
        return str(message.content)
    return output.model_dump_json()


def call(
    state: dict[str, Any],
    role: str,
    system: str,
    user: str,
    *,
    schema: type[BaseModel] | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run one model call for a role.

    Args:
        state: Current run state; read for the token total and trace fields, never modified.
        role: Name of the calling role, used in the memory log and trace.
        system: Static system prompt.
        user: Dynamic request text.
        schema: If given, parse the reply into this Pydantic model.
        tools: If given (and no schema), require exactly one call to one of these tools.

    Returns:
        A pair (output, update). The output is the parsed schema instance, or the AIMessage
        when no schema is given. The update holds the token total, memory log and trace.

    Raises:
        Exception: The parsing error, if the reply does not match `schema`.
    """
    llm = make_llm()
    prompt = [SystemMessage(system), HumanMessage(user)]
    if schema is not None:
        result = llm.with_structured_output(
            schema, method="function_calling", include_raw=True
        ).invoke(prompt)
        if result["parsing_error"] is not None:
            raise result["parsing_error"]
        output, message = result["parsed"], result["raw"]
    elif tools:
        output = message = llm.bind_tools(
            tools, tool_choice="required", parallel_tool_calls=False
        ).invoke(prompt)
    else:
        output = message = llm.invoke(prompt)

    tokens = tokens_of(message)
    update = {
        "total_tokens": add_tokens(state["total_tokens"], tokens),
        # Only the dynamic user content and the reply are logged; system prompts are static.
        "messages": [
            HumanMessage(user, name=role),
            AIMessage(_log_text(output, message), name=role),
        ],
        "trace": [log_event(state, "llm_call", role=role, tokens=tokens)],
    }
    return output, update

"""Single-agent comparator: one model call over the whole corpus.

Shares the model, settings and memo rules with the team (core.config,
core.prompts) so the comparison measures architecture, not prompt wording.
It reads the corpus file directly by design; only the team must use MCP.
"""

from langchain_core.messages import HumanMessage

from core.config import MARKET_CORPUS_MD, make_llm
from core.llm import tokens_of
from core.prompts import baseline_prompt


def run_topic(topic: str, topic_id: str) -> dict:
    """Write one memo with a single model call over the whole corpus.

    Args:
        topic: The research topic.
        topic_id: The topic's ID, e.g. T01.

    Returns:
        The result dict described in EVALUATION.md.

    Raises:
        ValueError: If the provider returns no memo text.
    """
    corpus = MARKET_CORPUS_MD.read_text(encoding="utf-8")
    message = make_llm().invoke([HumanMessage(baseline_prompt(topic, corpus))])
    memo = message.content
    if not isinstance(memo, str) or not memo.strip():
        raise ValueError("Provider returned no memo text")
    return {
        "topic_id": topic_id,
        "memo": memo,
        "total_tokens": tokens_of(message),
        "turns": 1,
        "rejections": 0,
        "mcp_calls": 0,
        "status": "completed",
    }

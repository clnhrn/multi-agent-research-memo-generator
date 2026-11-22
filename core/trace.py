"""Minimal run trace: one JSON line per event in logs/run-<id>.jsonl.

Evidence for the report (handoffs, MCP calls, revisions, stops), nothing more.
"""

import json
from datetime import UTC, datetime
from typing import Any

from core.config import LOG_DIR, run_file


def log_event(state: dict[str, Any], kind: str, **data: Any) -> dict[str, Any]:
    """Append one event to the run's log file.

    Args:
        state: Current run state; gives the run ID and turn.
        kind: Event type, e.g. "mcp_search".
        **data: JSON-serialisable event details.

    Returns:
        The event, for the caller to add to the state's trace.
    """
    event = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": state["run_id"],
        "turn": state["turn_count"],
        "kind": kind,
        "data": data,
    }
    LOG_DIR.mkdir(exist_ok=True)
    with run_file(LOG_DIR, state["run_id"], ".jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event

"""Minimal run trace: one JSON line per event in logs/run-<id>.jsonl.

Evidence for the report (handoffs, MCP calls, revisions, stops), nothing more.
"""

import json
from datetime import UTC, datetime
from typing import Any

from core.config import LOG_DIR


def log_event(state: dict[str, Any], kind: str, **data: Any) -> dict[str, Any]:
    """Write one event to the run's log file and return it for the state's trace."""
    event = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": state["run_id"],
        "turn": state["turn_count"],
        "kind": kind,
        "data": data,
    }
    LOG_DIR.mkdir(exist_ok=True)
    with (LOG_DIR / f"run-{state['run_id']}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event

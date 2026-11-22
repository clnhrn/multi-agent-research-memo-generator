"""The shared scratchpad: the run state as one JSON file per run (R4).

Writes are atomic, so a run killed at any moment leaves the last complete
save behind. Loading validates every field against the `State` schema.
"""

import os

from pydantic import TypeAdapter, ValidationError

from core.config import SCRATCHPAD_DIR, run_file
from core.schemas import State

_STATE = TypeAdapter(State)


class ScratchpadError(RuntimeError):
    """A saved run exists but cannot be used."""


def save_state(state: State) -> None:
    """Write the state atomically: a temporary file, then a rename over the old one.

    Args:
        state: The complete run state.
    """
    path = run_file(SCRATCHPAD_DIR, state["run_id"], ".json")
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("wb") as f:
        f.write(_STATE.dump_json(state, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_state(run_id: str) -> State | None:
    """Load a saved run.

    Args:
        run_id: The run's ID.

    Returns:
        The saved state, or None if the run was never saved.

    Raises:
        ScratchpadError: If the file is corrupt or belongs to another run.
        ValueError: If the run ID is not a safe file name.
    """
    path = run_file(SCRATCHPAD_DIR, run_id, ".json")
    try:
        state = _STATE.validate_json(path.read_bytes())
    except FileNotFoundError:
        return None
    except ValidationError as exc:
        raise ScratchpadError(f"{path.name} is not a valid scratchpad; use --fresh") from exc
    if state["run_id"] != run_id:
        raise ScratchpadError(f"{path.name} belongs to run {state['run_id']!r}; use --fresh")
    return state

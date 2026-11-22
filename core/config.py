"""Settings, limits and paths: the only place these values are defined."""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:  # imported lazily so the MCP server process stays light
    from langchain_openai import ChatOpenAI

# Variables already set in the shell (e.g. by Vocareum) win over .env.
load_dotenv(override=False)

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_CSV = DATA_DIR / "corpus.csv"
GLOSSARY_MD = DATA_DIR / "glossary.md"
MARKET_CORPUS_MD = DATA_DIR / "market_corpus.md"
TOPICS_JSON = ROOT / "topics.json"
MCP_SERVER_SCRIPT = ROOT / "mcp_server.py"
SCRATCHPAD_DIR = ROOT / "scratchpad"
LOG_DIR = ROOT / "logs"
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def check_run_id(run_id: str) -> str:
    """Validate a run ID against an allow-list, because it becomes a file name.

    Args:
        run_id: The run's ID; letters, digits, "_" and "-" only.

    Returns:
        The run ID, unchanged.

    Raises:
        ValueError: If the run ID is not a safe file name.
    """
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be 1-64 letters, digits, '_' or '-'")
    return run_id


def run_file(directory: Path, run_id: str, suffix: str) -> Path:
    """Build the path of a per-run file, the only place run IDs become file names.

    Args:
        directory: Folder holding the files.
        run_id: The run's ID.
        suffix: File extension, e.g. ".json".

    Returns:
        The path run-<id><suffix> inside the folder.

    Raises:
        ValueError: If the run ID is not a safe file name.
    """
    return directory / f"run-{check_run_id(run_id)}{suffix}"


# --- Limits from the assignment ------------------------------------------------
MAX_TURNS = 8  # supervisor turns per topic (R1)
MAX_REJECTIONS = 3  # reviewer rejections before escalating to a human
MIN_WORDS, MAX_WORDS = 250, 350  # memo body length
MIN_SNIPPETS, MAX_SNIPPETS = 3, 5  # researcher output
MAX_SUBQUERIES = 2  # extra MCP searches when the first one finds < MIN_SNIPPETS
COMPACTION_TOKENS = int(os.getenv("COMPACTION_TOKENS", "8000"))  # R5 trigger
KEEP_RECENT_MESSAGES = 4  # the last two exchanges stay verbatim after a compaction

# --- Model settings (identical for baseline and team) --------------------------
TEMPERATURE = 0
REQUEST_TIMEOUT_S = 120  # same as the starter baseline
MODEL_ENV_VARS = ("MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME")


class ConfigError(RuntimeError):
    """Raised when required model settings are missing."""


@dataclass(frozen=True)
class Settings:
    """Model connection settings.

    Attributes:
        api_key: Secret key for the model endpoint.
        base_url: URL of the OpenAI-compatible endpoint.
        model: Name of the chat model.
    """

    api_key: str
    base_url: str
    model: str

    def __repr__(self) -> str:  # never expose the key in logs or tracebacks
        """Return a representation without the API key, for safe logs and tracebacks."""
        return f"Settings(base_url={self.base_url!r}, model={self.model!r})"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read the model settings from the environment, once.

    Returns:
        The model connection settings.

    Raises:
        ConfigError: If a required variable is missing or empty.
    """
    missing = [name for name in MODEL_ENV_VARS if not os.getenv(name)]
    if missing:
        raise ConfigError(
            f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in."
        )
    return Settings(*(os.environ[name] for name in MODEL_ENV_VARS))


@lru_cache(maxsize=1)
def make_llm() -> "ChatOpenAI":
    """Build the chat model, the only place one is constructed.

    Returns:
        The chat model shared by the baseline and every agent.

    Raises:
        ConfigError: If a required variable is missing or empty.
    """
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT_S,
    )

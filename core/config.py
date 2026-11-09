"""Settings, limits and paths: the only place these values are defined."""

import os
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
MCP_SERVER_SCRIPT = ROOT / "mcp_server.py"
SCRATCHPAD_DIR = ROOT / "scratchpad"
LOG_DIR = ROOT / "logs"

# --- Limits from the assignment ------------------------------------------------
MAX_TURNS = 8  # supervisor turns per topic (R1)
MAX_REJECTIONS = 3  # reviewer rejections before escalating to a human
MIN_WORDS, MAX_WORDS = 250, 350  # memo body length
MIN_SNIPPETS, MAX_SNIPPETS = 3, 5  # researcher output
COMPACTION_TOKENS = int(os.getenv("COMPACTION_TOKENS", "8000"))  # R5 trigger

# --- Model settings (identical for baseline and team) --------------------------
TEMPERATURE = 0
REQUEST_TIMEOUT_S = 120  # same as the starter baseline
MODEL_ENV_VARS = ("MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME")


class ConfigError(RuntimeError):
    """Raised when required model settings are missing."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str

    def __repr__(self) -> str:  # never expose the key in logs or tracebacks
        return f"Settings(base_url={self.base_url!r}, model={self.model!r})"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    missing = [name for name in MODEL_ENV_VARS if not os.getenv(name)]
    if missing:
        raise ConfigError(
            f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in."
        )
    return Settings(*(os.environ[name] for name in MODEL_ENV_VARS))


@lru_cache(maxsize=1)
def make_llm() -> "ChatOpenAI":
    """The only place a chat model is constructed."""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT_S,
    )

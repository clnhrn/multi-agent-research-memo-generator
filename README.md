# Multi-Agent Research Memo Generator

[![ci](https://github.com/clnhrn/multi-agent-research-memo-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/clnhrn/multi-agent-research-memo-generator/actions/workflows/ci.yml)

A supervisor-led team (Researcher, Writer, Reviewer) writes cited 250–350-word research memos from a fictional market corpus, and is evaluated against a single-agent baseline on five fixed topics. The supervisor only routes, using typed handoff tools; the Researcher reaches the corpus only through an MCP server; state persists to a typed scratchpad after every turn and can resume after a kill; a rolling summary compacts the message log above 8,000 tokens. Design details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Setup

Python 3.11. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
cp .env.example .env          # then fill in the three values; never commit .env
uv run python check_server.py # MCP server check, no API key needed
```

Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`, then run the same commands with `python` instead of `uv run python` (for the tests, also `pip install pytest`).

**Model settings.** The code reads `MODEL_API_KEY`, `MODEL_BASE_URL` and `MODEL_NAME`, from the shell or from `.env` (shell values win). On Vocareum you can set `export MODEL_API_KEY="$OPENAI_API_KEY"` instead of writing the key to a file. Use the endpoint and model confirmed by the course team; the results here used `gpt-4o-mini` at temperature 0.

## Run

```bash
uv run python team.py T03                    # one topic; trace in logs/, state in scratchpad/
uv run python team.py T03 --fresh            # ignore a saved run and start clean
uv run python evaluate.py --output results/my-run.json              # both systems, all five topics
uv run python evaluate.py --output results/my-run.json --topics T01 --system team
uv run python evaluate.py --summary results/evaluation.reviewed.json  # rates and error-amplification ratio
uv run pytest                                # offline tests, no API calls
```

Model runs use course quota (a full evaluation was about 56,000 tokens). Killing `team.py` and running the same command again resumes from `scratchpad/run-<id>.json`. `evaluate.py` starts team runs fresh by default (`--resume` to continue) and will not overwrite an existing results file without `--overwrite`.

## Results

Evaluation of 2026-11-23, manually reviewed against the same rubric for both systems:

| | Single agent | Multi-agent |
|---|---|---|
| Rubric passes | 0 / 5 | 3 / 5 |
| Total tokens | 6,983 | 48,562 |

Error-amplification ratio: **0.4** (goal ≤ 1.0). Reasons for each verdict are in [`results/MANUAL_REVIEW.md`](results/MANUAL_REVIEW.md), and [`results/README.md`](results/README.md) maps each piece of required evidence (traces, MCP calls, resume, compaction) to its file.

## Layout

| Path | Contents |
|---|---|
| `team.py` | Supervisor, workers, handoff tools, graph, `run_topic()` |
| `core/` | Shared config, schemas, prompts, text checks, LLM calls, MCP client, scratchpad, memory, trace |
| `mcp_server.py`, `check_server.py` | MCP server (corpus resources, `search_corpus` tool) and a stdio client check |
| `baseline.py`, `evaluate.py` | Single-agent comparator and the evaluation harness |
| `data/`, `topics.json` | Fictional corpus, glossary and the five seed topics |
| `tests/` | Offline tests, including role-boundary guards |
| `results/` | Evaluation results, manual review, traces and scratchpad evidence |
| `EVALUATION.md` | Course-provided evaluation contract |

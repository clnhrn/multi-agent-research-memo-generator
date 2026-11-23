# Results and evidence

Full evaluation run on 2026-11-23 (`gpt-4o-mini`, temperature 0, five seed topics, both systems).

| Evidence | File | What to look for |
|---|---|---|
| Evaluation results, as produced | `evaluation.raw.json` | Untouched evaluator output; `rubric_pass` is null |
| Evaluation results, reviewed | `evaluation.reviewed.json` | `rubric_pass` and `review.notes` per memo; summarise with `python evaluate.py --summary results/evaluation.reviewed.json` |
| Reasons for each verdict | `MANUAL_REVIEW.md` | Review rules, item-by-item evidence, sensitivity to the rule |
| One complete run per topic | `traces/run-T01.jsonl` … `traces/run-T05.jsonl` | `supervisor_decision` (handoffs), `draft`, `verdict`, `stop` events |
| MCP invoked | the same traces | `mcp_search` events with the query and returned source IDs |
| Rolling-summary memory | `traces/run-T03.jsonl` | `compaction` event: 8,316 → 1,234 tokens at the 8,000 threshold |
| Scratchpad persistence | `scratchpad/run-T01.json` … `run-T05.json` | Final typed state of each run |
| Resume after a kill | `scratchpad/run-live-resume-T03.after-kill.json`, `scratchpad/run-live-resume-T03.json`, `traces/run-live-resume-T03.jsonl` | State after a SIGKILL following turn 2, the finished state, and the `resumed` event (`pending: writer`) |

The resume run is from the Phase 5 demo (2026-11-22); all other files are from the full evaluation run.

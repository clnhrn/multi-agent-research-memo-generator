"""Run the baseline and the team on the same topics, and summarise reviewed results.

Automatic checks are the same for both systems. They cannot judge factual support or tone,
so `rubric_pass` stays None until a human reviews the memo.
"""

import argparse
import json
import os
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import baseline
import team
from core.config import (
    LOG_DIR,
    MAX_WORDS,
    MIN_WORDS,
    TEMPERATURE,
    TOPICS_JSON,
    ConfigError,
    get_settings,
)
from core.corpus import valid_ids
from core.text import (
    body_word_count,
    check_memo,
    extract_citations,
    length_ok,
    parse_memo,
    word_count,
)

Result = dict[str, Any]
NOTE = "Review each memo, set rubric_pass to true or false, then run --summary on the file."
# The adapter contract in EVALUATION.md.
CONTRACT = {"topic_id", "memo", "total_tokens", "turns", "rejections", "mcp_calls", "status"}
# Result keys, as in the starter's output, by command-line name.
SYSTEMS = {"baseline": "baseline", "team": "multi_agent"}


def assess(memo: str) -> Result:
    """Run the automatic checks on a memo, the same for both systems.

    Args:
        memo: The memo as plain text.

    Returns:
        The check results, with `rubric_pass` left as None for manual review. A memo that
        cannot be split into title, body and footnotes fails, and all its words are counted.
        The starter's whole-memo count is recorded too, but does not affect the result.
    """
    parsed = parse_memo(memo)
    body = parsed["body"] if parsed else memo
    problems = check_memo(parsed) if parsed else ["No title, body and footnotes section found."]
    invalid = sorted(set(extract_citations(memo)) - valid_ids())
    whole = word_count(memo)
    return {
        "word_count": body_word_count(body),
        "counted": "body" if parsed else "whole memo",
        "length_pass": length_ok(body),
        # The starter's measure: every word, with title, footnotes and citations.
        "whole_memo_word_count": whole,
        "whole_memo_length_pass": MIN_WORDS <= whole <= MAX_WORDS,
        "citation_present": bool(extract_citations(body)),
        "invalid_citations": invalid,
        "problems": problems,
        "automatic_checks_pass": not problems and not invalid,
        "rubric_pass": None,
    }


def root_causes(exc: BaseException) -> Iterator[BaseException]:
    """Yield the errors inside nested exception groups, or the error itself.

    MCP's task group wraps anything raised inside the researcher.

    Args:
        exc: The caught error.

    Yields:
        Each error that is not a group.
    """
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            yield from root_causes(inner)
    else:
        yield exc


def log_error(system: str, topic_id: str, exc: BaseException) -> None:
    """Append the full traceback to the local error log, without the API key.

    Args:
        system: Result key of the system that failed.
        topic_id: The topic's ID.
        exc: The caught error.
    """
    text = "".join(traceback.format_exception(exc))
    if key := os.getenv("MODEL_API_KEY"):
        text = text.replace(key, "[redacted]")
    LOG_DIR.mkdir(exist_ok=True)
    with (LOG_DIR / "evaluate-errors.log").open("a", encoding="utf-8") as f:
        f.write(f"--- {_now()} {system} {topic_id}\n{text}\n")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_one(system: str, fn: Callable[[str, str], Result], topic: dict[str, str]) -> Result:
    """Run one system on one topic, reporting any exception as an error result.

    Args:
        system: Result key of the system.
        fn: The system's `run_topic` function.
        topic: A topic dict with `id` and `topic`.

    Returns:
        The system's result with automatic checks and timing, or an error entry. Error
        details stay in the local log because they may contain private data.
    """
    timing = {"started_at": _now()}
    start = time.perf_counter()
    try:
        result = fn(topic["topic"], topic["id"])
        if not isinstance(result, dict) or not CONTRACT <= result.keys():
            raise ValueError(f"Adapter must return a dict with {', '.join(sorted(CONTRACT))}")
        if not isinstance(result["memo"], str):
            raise ValueError("Adapter must return the memo as text")
        result = {**result, "checks": assess(result["memo"])}
    except Exception as exc:
        log_error(system, topic["id"], exc)
        causes = sorted({type(cause).__name__ for cause in root_causes(exc)})
        result = {
            "topic_id": topic["id"],
            "status": "error",
            "error_type": ", ".join(causes),
            "checks": {"rubric_pass": None},
        }
    return {**result, **timing, "seconds": round(time.perf_counter() - start, 1)}


def failed(result: Result) -> bool | None:
    """Return whether a run fails the rubric, or None if it has not been reviewed.

    An execution error has no memo to review, so it counts as a failure.
    """
    if result["status"] == "error":
        return True
    passed = result["checks"]["rubric_pass"]
    if passed not in (True, False, None):
        raise ValueError(f"{result['topic_id']}: rubric_pass must be true, false or null")
    return None if passed is None else not passed


def totals(results: list[Result]) -> Result:
    """Summarise one system's runs.

    Args:
        results: The system's results.

    Returns:
        Counts, totals and the rubric-failure rate. The rate is None while any run is
        unreviewed; token totals are None if any run lacks provider usage.
    """
    outcomes = [failed(r) for r in results]
    tokens = [r.get("total_tokens") for r in results]
    failures = sum(o is True for o in outcomes)
    unreviewed = outcomes.count(None)
    return {
        "topics": [r["topic_id"] for r in results],
        "statuses": dict(Counter(r["status"] for r in results)),
        "automatic_passes": sum(bool(r["checks"].get("automatic_checks_pass")) for r in results),
        "rubric_failures": failures,
        "unreviewed": unreviewed,
        "failure_rate": failures / len(results) if results and not unreviewed else None,
        "total_tokens": None if None in tokens else sum(tokens),
        "turns": sum(r.get("turns", 0) for r in results),
        "rejections": sum(r.get("rejections", 0) for r in results),
    }


def summarise(output: Result) -> Result:
    """Compare the two systems.

    Args:
        output: The evaluation file's content, with reviewed `rubric_pass` values.

    Returns:
        Each system's totals and the error-amplification ratio. The ratio is None, with
        the reason, when it is undefined.

    Raises:
        ValueError: If a `rubric_pass` value is not true, false or null.
    """
    base, multi = (totals(output.get(key, [])) for key in SYSTEMS.values())
    ratio, reason = None, None
    if not base["topics"] or sorted(base["topics"]) != sorted(multi["topics"]):
        reason = "the systems were not run on the same topics"
    elif base["unreviewed"] or multi["unreviewed"]:
        reason = "some memos have not been reviewed"
    elif base["failure_rate"] == 0:
        reason = "the baseline failure rate is zero; report both rates instead"
    else:
        ratio = multi["failure_rate"] / base["failure_rate"]
    return {
        "baseline": base,
        "multi_agent": multi,
        "error_amplification": ratio,
        "ratio_undefined_because": reason,
        "errors_counted_as": "rubric failures",
    }


def evaluate(systems: list[str], topics: list[dict[str, str]], path: Path, fresh: bool) -> Result:
    """Run the systems on the topics, saving after every run.

    Args:
        systems: Command-line names from `SYSTEMS`.
        topics: Topic dicts with `id` and `topic`.
        path: The output file.
        fresh: Start every team run clean.

    Returns:
        The content written to the output file.
    """
    runners = {"baseline": baseline.run_topic, "team": partial(team.run_topic, fresh=fresh)}
    settings = {"model": get_settings().model, "temperature": TEMPERATURE, "started_at": _now()}
    output: Result = {"note": NOTE, "settings": settings, **{SYSTEMS[s]: [] for s in systems}}
    path.parent.mkdir(parents=True, exist_ok=True)
    for system in systems:
        for topic in topics:
            result = run_one(SYSTEMS[system], runners[system], topic)
            output[SYSTEMS[system]].append(result)
            path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"{system} {topic['id']}: {result['status']} in {result['seconds']}s")
    return output


def main() -> None:
    """Run the evaluation, or summarise a reviewed one, from the command line."""
    topics = json.loads(TOPICS_JSON.read_text(encoding="utf-8"))
    known = [t["id"] for t in topics]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--system", choices=[*SYSTEMS, "both"], default="both")
    parser.add_argument("--topics", default=",".join(known), help="e.g. T01,T05 (default: all)")
    parser.add_argument("--summary", type=Path, metavar="FILE", help="summarise a results file")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--fresh", action="store_true", help="start every team run clean (default)")
    start.add_argument("--resume", action="store_true", help="continue unfinished team runs")
    args = parser.parse_args()

    if args.summary:
        try:
            reviewed = json.loads(args.summary.read_text(encoding="utf-8"))
            print(json.dumps(summarise(reviewed), indent=2))
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"cannot summarise {args.summary}: {exc}")
        return

    chosen = args.topics.split(",")
    if unknown := sorted(set(chosen) - set(known)):
        parser.error(f"unknown topics: {', '.join(unknown)}; choose from {', '.join(known)}")
    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} exists; choose another --output or pass --overwrite")
    systems = list(SYSTEMS) if args.system == "both" else [args.system]
    try:
        evaluate(
            systems, [t for t in topics if t["id"] in chosen], args.output, fresh=not args.resume
        )
    except ConfigError as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}. {NOTE}")


if __name__ == "__main__":
    main()

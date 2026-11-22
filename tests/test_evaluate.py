"""The evaluator: equal checks for both systems, safe errors, and an honest summary (R6)."""

import json
import sys

import pytest
from helpers import GOOD, TOPIC, pick, scripted

import baseline
import evaluate
import team
from core.config import ConfigError, Settings
from core.text import parse_memo, render_memo

KEY = "sk-test-placeholder"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_logs):
    """Use fake model settings and a temporary error log."""
    monkeypatch.setattr(evaluate, "LOG_DIR", tmp_logs)
    monkeypatch.setattr(evaluate, "get_settings", lambda: Settings(KEY, "url", "test-model"))
    monkeypatch.setenv("MODEL_API_KEY", KEY)


def _result(topic_id="T01", **fields):
    return {
        "topic_id": topic_id, "memo": render_memo(GOOD.model_dump()), "total_tokens": 100,
        "turns": 3, "rejections": 1, "mcp_calls": 1, "status": "completed", **fields,
    }  # fmt: skip


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return a function that runs evaluate.main with fake systems and reads the output."""
    seen = []

    def fake(topic, topic_id, *, fresh=None):
        seen.append((topic_id, fresh))
        return _result(topic_id)

    def go(*flags, output="e.json"):
        monkeypatch.setattr(
            sys, "argv", ["evaluate.py", "--output", str(tmp_path / output), *flags]
        )
        evaluate.main()
        return json.loads((tmp_path / output).read_text())

    go.seen = seen
    monkeypatch.setattr(team, "run_topic", fake)
    monkeypatch.setattr(baseline, "run_topic", fake)
    return go


# --- Parsing and checks -----------------------------------------------------------


def test_parse_reverses_render():
    memo = GOOD.model_dump()
    assert parse_memo(render_memo(memo)) == memo


@pytest.mark.parametrize(
    "heading", ["Footnotes:", "Footnotes", "**Footnotes:**", "## Footnotes", "FOOTNOTES:"]
)
def test_parse_accepts_heading_variants(heading):
    parsed = parse_memo(f"Title\n\nBody [P01].\n\n{heading}\n[P01] Solar  \n\n")
    assert parsed == {"title": "Title", "body": "Body [P01].", "footnotes": ["[P01] Solar"]}


@pytest.mark.parametrize("text", ["", "Title\n\nBody only.", "Title\n\nFootnotes:\n[P01] x"])
def test_parse_returns_none_without_all_three_parts(text):
    assert parse_memo(text) is None


def test_mentioning_footnotes_in_the_body_is_not_a_heading():
    assert parse_memo("Title\n\nThe footnotes: see below [P01].") is None


def test_title_and_footnotes_do_not_count_as_body_words():
    checks = evaluate.assess(render_memo(GOOD.model_dump()))
    assert (checks["word_count"], checks["counted"]) == (260, "body")
    assert checks["automatic_checks_pass"] is True and checks["rubric_pass"] is None


@pytest.mark.parametrize(("body_words", "whole_pass"), [(260, True), (349, False)])
def test_the_starter_count_is_recorded_but_does_not_decide(body_words, whole_pass):
    memo = GOOD.model_copy(update={"body": "word " * body_words + "[P01]."}).model_dump()
    checks = evaluate.assess(render_memo(memo))
    # Title, citation, heading, footnote ID and footnote text add to the body.
    assert checks["whole_memo_word_count"] == body_words + 5
    assert checks["whole_memo_length_pass"] is whole_pass
    assert checks["length_pass"] and checks["automatic_checks_pass"]


@pytest.mark.parametrize(
    ("memo", "problem"),
    [
        (GOOD.model_copy(update={"body": "Too short [P01]."}), "words"),
        (GOOD.model_copy(update={"footnotes": []}), "footnote"),
        (GOOD.model_copy(update={"body": GOOD.body.replace("[P01]", "[P01, P01]")}), "bracket"),
    ],
)
def test_checks_report_problems(memo, problem):
    checks = evaluate.assess(render_memo(memo.model_dump()))
    assert checks["automatic_checks_pass"] is False
    assert any(problem in p.lower() for p in checks["problems"])


def test_ids_outside_the_corpus_fail():
    body = GOOD.body.replace("P01", "P99")
    memo = render_memo({"title": "T", "body": body, "footnotes": ["[P99] x"]})
    checks = evaluate.assess(memo)
    assert checks["invalid_citations"] == ["P99"] and not checks["automatic_checks_pass"]


@pytest.mark.parametrize("memo", ["", "word " * 300])
def test_unparsed_memos_fail_and_say_so(memo):
    checks = evaluate.assess(memo)
    assert checks["counted"] == "whole memo" and not checks["automatic_checks_pass"]


# --- The team adapter -------------------------------------------------------------


def test_run_topic_meets_the_contract(fake_call):
    fake_call(scripted())
    result = team.run_topic(TOPIC, "T01")
    assert result.keys() == evaluate.CONTRACT
    assert result["status"] == "completed"
    assert parse_memo(result["memo"]) == GOOD.model_dump()
    assert result["turns"] == 3 and result["mcp_calls"] == 2


@pytest.mark.parametrize(
    ("script", "status"),
    [({"rejections": 3}, "escalated"), ({"choose": lambda s: pick("researcher")}, "max_turns")],
)
def test_run_topic_never_reports_an_unaccepted_run_as_completed(fake_call, script, status):
    fake_call(scripted(**script))
    result = team.run_topic(TOPIC, "T01")
    assert result["status"] == status
    assert bool(result["memo"]) is (status == "escalated")


def test_run_topic_keeps_unknown_token_totals_unknown(monkeypatch):
    final = {**team.new_state(TOPIC, "T01"), "total_tokens": None, "status": "escalated"}
    monkeypatch.setattr(team, "run_team", lambda *args, **kwargs: final)
    assert team.run_topic(TOPIC, "T01")["total_tokens"] is None


# --- Errors -----------------------------------------------------------------------


def test_errors_show_the_inner_type_and_log_the_traceback_without_the_key(tmp_logs):
    def broken(topic, topic_id):
        raise ExceptionGroup("outer", [ExceptionGroup("inner", [KeyError(f"bad {KEY}")])])

    result = evaluate.run_one("multi_agent", broken, {"id": "T01", "topic": TOPIC})
    assert result["status"] == "error" and result["error_type"] == "KeyError"
    assert result["seconds"] >= 0 and result["started_at"]
    assert KEY not in json.dumps(result) and "bad" not in json.dumps(result)
    log = (tmp_logs / "evaluate-errors.log").read_text()
    assert "KeyError" in log and "[redacted]" in log and KEY not in log


@pytest.mark.parametrize("bad", [None, "text", {"memo": "x"}, {**_result(), "memo": None}])
def test_a_broken_adapter_is_an_error_not_a_crash(bad):
    result = evaluate.run_one("baseline", lambda t, i: bad, {"id": "T01", "topic": TOPIC})
    assert (result["status"], result["error_type"]) == ("error", "ValueError")


# --- Command line -----------------------------------------------------------------


def test_default_runs_both_systems_on_all_topics(run):
    output = run()
    ids = ["T01", "T02", "T03", "T04", "T05"]
    for key in ("baseline", "multi_agent"):
        assert [r["topic_id"] for r in output[key]] == ids
    assert output["settings"]["model"] == "test-model" and output["settings"]["temperature"] == 0
    assert KEY not in json.dumps(output)


def test_one_system_and_chosen_topics(run):
    output = run("--system", "team", "--topics", "T05,T01")
    assert "baseline" not in output
    assert [r["topic_id"] for r in output["multi_agent"]] == ["T01", "T05"]


@pytest.mark.parametrize(
    ("flags", "fresh"), [([], True), (["--fresh"], True), (["--resume"], False)]
)
def test_team_runs_start_clean_unless_asked_to_resume(run, flags, fresh):
    run("--system", "team", *flags)
    assert {f for _, f in run.seen} == {fresh}


def test_existing_results_are_not_overwritten_by_accident(run, capsys):
    run("--topics", "T01")
    with pytest.raises(SystemExit):
        run("--topics", "T02")
    assert "--overwrite" in capsys.readouterr().err
    assert [t for t, _ in run.seen] == ["T01", "T01"]  # no model calls were made
    assert run("--topics", "T02", "--overwrite")["baseline"][0]["topic_id"] == "T02"


@pytest.mark.parametrize("topics", ["T99", "T01,T99", ""])
def test_unknown_topics_are_refused(run, topics):
    with pytest.raises(SystemExit):
        run("--topics", topics)
    assert run.seen == []


def test_missing_settings_stop_before_any_run(run, monkeypatch):
    def missing():
        raise ConfigError("Missing MODEL_API_KEY")

    monkeypatch.setattr(evaluate, "get_settings", missing)
    with pytest.raises(SystemExit):
        run()
    assert run.seen == []


def test_finished_runs_are_saved_if_a_later_one_is_interrupted(run, monkeypatch, tmp_path):
    def interrupted(topic, topic_id):
        if topic_id == "T02":
            raise KeyboardInterrupt
        return _result(topic_id)

    monkeypatch.setattr(baseline, "run_topic", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run()
    saved = json.loads((tmp_path / "e.json").read_text())
    assert [r["topic_id"] for r in saved["baseline"]] == ["T01"]


# --- Summary ----------------------------------------------------------------------


def _reviewed(base, multi):
    """Build an output file; each value is a rubric_pass, or "error"."""

    def results(values):
        for i, value in enumerate(values, 1):
            if value == "error":
                yield {"topic_id": f"T0{i}", "status": "error", "checks": {"rubric_pass": None}}
            else:
                yield _result(f"T0{i}", checks={"rubric_pass": value})

    return {"baseline": list(results(base)), "multi_agent": list(results(multi))}


@pytest.mark.parametrize(
    ("base", "multi", "ratio"),
    [
        ([False, False, True, True], [False, True, True, True], 0.5),
        ([False, True], [False, True], 1.0),
        ([False, True], [True, True], 0.0),
        ([False, True], ["error", False], 2.0),  # errors count as failures
    ],
)
def test_ratio_is_team_failure_rate_over_baseline_failure_rate(base, multi, ratio):
    summary = evaluate.summarise(_reviewed(base, multi))
    assert summary["error_amplification"] == ratio
    assert summary["ratio_undefined_because"] is None


@pytest.mark.parametrize(
    ("base", "multi", "reason"),
    [
        ([True, True], [False, True], "zero"),
        ([True, True], [True, True], "zero"),  # 0/0 is not a success
        ([None, False], [True, True], "reviewed"),
        ([False, True], [None, True], "reviewed"),
        ([False, True], [True], "same topics"),
        ([], [], "same topics"),
    ],
)
def test_ratio_is_undefined_when_it_would_mislead(base, multi, reason):
    summary = evaluate.summarise(_reviewed(base, multi))
    assert summary["error_amplification"] is None
    assert reason in summary["ratio_undefined_because"]


def test_summary_reports_counts_and_unknown_tokens():
    output = _reviewed([True, False], [True, "error"])
    output["baseline"][0]["total_tokens"] = None
    summary = evaluate.summarise(output)
    assert summary["baseline"]["total_tokens"] is None
    assert summary["multi_agent"]["statuses"] == {"completed": 1, "error": 1}
    assert summary["multi_agent"]["rubric_failures"] == 1
    assert summary["baseline"]["rejections"] == 2


def test_summary_rejects_unclear_review_values():
    with pytest.raises(ValueError, match="rubric_pass"):
        evaluate.summarise(_reviewed(["yes"], [True]))


def test_summary_from_the_command_line(run, monkeypatch, tmp_path, capsys):
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(_reviewed([False], [True])))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--summary", str(path)])
    evaluate.main()
    assert json.loads(capsys.readouterr().out)["error_amplification"] == 0.0
    assert run.seen == []

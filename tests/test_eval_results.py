import json
from pathlib import Path

from rag_sql.agent.prompts import prompt_version
from rag_sql.evaluation.results import (
    CaseResult,
    compare_runs,
    format_report,
    summarize_results,
    write_results,
)


def _case_result(
    case_id: str,
    outcome: str,
    *,
    run: int = 1,
    ms: int = 2000,
    expected: str = "sql",
    intent: str | None = None,
) -> CaseResult:
    metrics = {
        "total_ms": ms,
        "attempts": 1,
        "llm_ms": ms,
        "load_ms": 0,
        "input_tokens": 900,
        "output_tokens": 100,
        "nodes": [],
    }
    return CaseResult(
        case_id, ["imba"], run, outcome, None, None, "", 1, ms / 1000, metrics, expected, intent
    )


def test_summary_and_report() -> None:
    results = [
        _case_result("a", "correct", ms=1000),
        _case_result("b", "correct_extra_columns", ms=3000),
        _case_result("c", "sql_failed", ms=9000),
    ]
    summary = summarize_results(results)
    assert (summary["correct"], summary["exact"], summary["runs"]) == (2, 1, 3)
    assert (summary["p50_s"], summary["p95_s"], summary["avg_tokens"]) == (3.0, 9.0, 1000)
    assert summary["by_tag"] == {"imba": {"correct": 2, "runs": 3}}
    report = format_report("gemma4:e4b", results)
    assert report.startswith("Model gemma4:e4b: 2/3 correct (67%), 1 exact")
    assert "  FAIL c: sql_failed" in report


def test_results_file_and_compare(tmp_path: Path, settings) -> None:
    old = [_case_result("a", "correct"), _case_result("b", "wrong_result")]
    new = [_case_result("a", "wrong_result"), _case_result("b", "correct")]
    paths = [
        write_results(
            tmp_path,
            "qwen3:14b",
            results,
            settings=settings,
            cases_path=Path("cases.yaml"),
            with_history=False,
            repeat=1,
        )
        for results in (old, new)
    ]
    old_run, new_run = (json.loads(p.read_text(encoding="utf-8")) for p in paths)
    assert paths[0].name.endswith("-qwen3-14b.json")
    assert paths[0] != paths[1]  # same model in the same second: both kept
    assert old_run["summary"]["correct"] == 1 and old_run["settings"]["query_history_k"] == 0
    assert old_run["results"][0]["case_id"] == "a"

    diff = compare_runs(old_run, new_run)
    assert "  - a: correct (1/1) -> wrong_result (0/1)" in diff
    assert "  + b: wrong_result (0/1) -> correct (1/1)" in diff
    assert compare_runs(old_run, old_run).endswith("no case changed")


def test_summary_by_kind_router_and_flaky_cases() -> None:
    results = [
        _case_result("a", "correct", expected="sql", intent="data"),
        _case_result("b", "unneeded_clarify", expected="sql", intent="clarify"),
        _case_result("c", "correct", expected="clarify", intent="clarify"),
        _case_result("d", "correct", expected="chat", intent="chat"),
        _case_result("d", "unneeded_sql", run=2, expected="chat", intent="data"),
    ]
    s = summarize_results(results)
    assert s["by_kind"] == {
        "chat": {"correct": 1, "runs": 2},
        "clarify": {"correct": 1, "runs": 1},
        "sql": {"correct": 1, "runs": 2},
    }
    assert s["route"] == {"right": 3, "runs": 5}
    assert s["flaky"] == 1  # d passed once and failed once

    report = format_report("gemma4:e4b", results)
    assert "  router: 3/5 right intent" in report
    assert "  by kind: chat 1/2, clarify 1/1, sql 1/2" in report
    assert "  flaky: 1 case(s)" in report


def test_runs_without_a_router_have_no_router_score() -> None:
    s = summarize_results([_case_result("a", "correct")])
    assert s["route"] == {"right": 0, "runs": 0}
    assert "router:" not in format_report("m", [_case_result("a", "correct")])


def test_router_only_report_lists_the_wrong_intents() -> None:
    results = [
        _case_result("a", "correct", expected="sql", intent="data"),
        _case_result("b", "guessed", expected="clarify", intent="data"),
    ]
    report = format_report("gemma4:e4b", results, router_only=True)
    assert report.startswith("Router of gemma4:e4b: 1/2 right intent (50%)")
    assert "  FAIL b: guessed\n       chose data, expected clarify" in report


def test_results_file_records_prompts_and_router_model(tmp_path: Path, settings) -> None:
    path = write_results(
        tmp_path,
        "gemma4:e4b",
        [_case_result("a", "correct", intent="data")],
        settings=settings.model_copy(update={"ollama_router_model": "qwen3:4b"}),
        cases_path=Path("cases.yaml"),
        with_history=False,
        repeat=1,
        router_only=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["router_only"] is True
    assert payload["prompt_version"] == prompt_version()
    assert payload["settings"]["ollama_router_model"] == "qwen3:4b"
    assert payload["results"][0]["intent"] == "data"


def test_compare_shows_prompts_and_router_score(tmp_path: Path, settings) -> None:
    def run(outcomes: list[str], intent: str) -> dict:
        results = [_case_result(f"c{i}", o, intent=intent) for i, o in enumerate(outcomes)]
        path = write_results(
            tmp_path,
            "m",
            results,
            settings=settings,
            cases_path=Path("c.yaml"),
            with_history=False,
            repeat=1,
        )
        return json.loads(path.read_text(encoding="utf-8"))

    diff = compare_runs(run(["correct"], "chat"), run(["correct"], "data"))
    assert f"prompts {prompt_version()}" in diff
    assert "router 0/1" in diff and "router 1/1" in diff

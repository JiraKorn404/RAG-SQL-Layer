import json
from pathlib import Path

from rag_sql.evaluation.results import (
    CaseResult,
    compare_runs,
    format_report,
    summarize_results,
    write_results,
)


def _case_result(case_id: str, outcome: str, *, run: int = 1, ms: int = 2000) -> CaseResult:
    metrics = {
        "total_ms": ms,
        "attempts": 1,
        "llm_ms": ms,
        "load_ms": 0,
        "input_tokens": 900,
        "output_tokens": 100,
        "nodes": [],
    }
    return CaseResult(case_id, ["imba"], run, outcome, None, None, "", 1, ms / 1000, metrics)


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

import json
from pathlib import Path

import pytest

from rag_sql.agent.graph import build_graph
from rag_sql.db.query import QueryResult
from rag_sql.evaluation.cases import EarlierTurn, EvalCase
from rag_sql.evaluation.runner import (
    CaseResult,
    EvalError,
    compare_runs,
    format_report,
    reference_results,
    run_case,
    run_cases,
    summarize_results,
    write_results,
)
from tests.conftest import fake_llm

CASE = EvalCase(
    "top-earner", "Who earns the most?", "SELECT emp_name FROM employees", ("employees",)
)
EXPECTED = QueryResult(columns=["emp_name"], rows=[("Employee_1",)])


def _graph(settings, retriever, ok_runner, stores, *replies):
    return build_graph(
        llm=fake_llm(*replies),
        retriever=retriever,
        query_runner=ok_runner,
        **stores,
        settings=settings,
    )


def test_correct_case_with_extra_column(settings, retriever, ok_runner, stores, chat_store) -> None:
    # ok_runner returns (emp_name, salary): the reference only needs emp_name.
    graph = _graph(settings, retriever, ok_runner, stores, "```sql\nSELECT 1\n```", "Employee_1.")
    result = run_case(graph, CASE, EXPECTED)

    assert (result.outcome, result.correct, result.attempts) == ("correct_extra_columns", True, 1)
    assert result.sql.startswith("SELECT") and result.answer == "Employee_1."
    assert result.metrics["attempts"] == 1
    assert chat_store.threads() == []  # no thread: nothing saved


def test_wrong_and_failed_cases(settings, retriever, ok_runner, stores) -> None:
    graph = _graph(settings, retriever, ok_runner, stores, "SELECT 1", "Someone.")
    expected = QueryResult(columns=["emp_name"], rows=[("Employee_9",)])
    assert run_case(graph, CASE, expected).outcome == "wrong_result"

    graph = _graph(settings, retriever, ok_runner, stores, *["DROP TABLE employees"] * 3)
    failed = run_case(graph, CASE, EXPECTED)
    assert (failed.outcome, failed.sql, failed.attempts) == ("sql_failed", None, 3)
    assert "Only SELECT" in failed.error


class RecordingGraph:
    """Records the graph input; raises when asked to."""

    def __init__(self, error: Exception | None = None) -> None:
        self.inputs: list[dict] = []
        self.error = error

    def invoke(self, graph_input: dict) -> dict:
        self.inputs.append(graph_input)
        if self.error:
            raise self.error
        return {"result": EXPECTED, "sql": "SELECT 1", "attempts": 1, "answer": "A."}


def test_follow_up_history_is_passed_without_a_thread() -> None:
    case = EvalCase(
        "followup", "And the lowest?", "SELECT 1", history=(EarlierTurn("Highest?", "SELECT 2"),)
    )
    graph = RecordingGraph()
    assert run_case(graph, case, EXPECTED).outcome == "correct"
    [graph_input] = graph.inputs
    assert graph_input["question"] == "And the lowest?"
    assert [t["question"] for t in graph_input["history"]] == ["Highest?"]
    assert "thread_id" not in graph_input


def test_agent_error_is_recorded_and_the_run_goes_on() -> None:
    graph = RecordingGraph(error=RuntimeError("Ollama is down"))
    results = run_cases(graph, [CASE, CASE], {CASE.id: EXPECTED}, repeat=2)
    assert [(r.outcome, r.error, r.run) for r in results] == [
        ("agent_error", "Ollama is down", 1),
        ("agent_error", "Ollama is down", 1),
        ("agent_error", "Ollama is down", 2),
        ("agent_error", "Ollama is down", 2),
    ]


def test_reference_results_must_run_and_fit_the_row_limit() -> None:
    assert reference_results([CASE], lambda _sql: EXPECTED, 10) == {CASE.id: EXPECTED}
    truncated = QueryResult(columns=["a"], rows=[(1,)], truncated=True)
    with pytest.raises(EvalError, match="more than 10 rows"):
        reference_results([CASE], lambda _sql: truncated, 10)

    def failing(_sql: str) -> QueryResult:
        raise RuntimeError("boom")

    with pytest.raises(EvalError, match="top-earner.*boom"):
        reference_results([CASE], failing, 10)


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
            cases_path=Path("eval.yaml"),
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

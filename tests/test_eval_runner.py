import pytest

from rag_sql.agent.graph import build_graph
from rag_sql.db.query import QueryResult
from rag_sql.evaluation.cases import EarlierTurn, EvalCase
from rag_sql.evaluation.runner import (
    EvalError,
    reference_results,
    run_case,
    run_cases,
    run_router_cases,
)
from tests.conftest import ROUTE_DATA, fake_llm, route_reply

CASE = EvalCase(
    "top-earner", "Who earns the most?", "SELECT emp_name FROM employees", ("employees",)
)
EXPECTED = QueryResult(columns=["emp_name"], rows=[("Employee_1",)])


def _graph(settings, retriever, ok_runner, stores, *replies):
    """A graph whose model gives `replies` in order (the router's reply first)."""
    return build_graph(
        llm=fake_llm(*replies),
        retriever=retriever,
        query_runner=ok_runner,
        **stores,
        settings=settings,
    )


def test_correct_case_with_extra_column(settings, retriever, ok_runner, stores, chat_store) -> None:
    # ok_runner returns (emp_name, salary): the reference only needs emp_name.
    graph = _graph(
        settings, retriever, ok_runner, stores, ROUTE_DATA, "```sql\nSELECT 1\n```", "Employee_1."
    )
    result = run_case(graph, CASE, EXPECTED)

    assert (result.outcome, result.correct, result.attempts) == ("correct_extra_columns", True, 1)
    assert result.sql.startswith("SELECT") and result.answer == "Employee_1."
    assert result.metrics["attempts"] == 1
    assert chat_store.threads() == []  # no thread: nothing saved


def test_wrong_and_failed_cases(settings, retriever, ok_runner, stores) -> None:
    graph = _graph(settings, retriever, ok_runner, stores, ROUTE_DATA, "SELECT 1", "Someone.")
    expected = QueryResult(columns=["emp_name"], rows=[("Employee_9",)])
    assert run_case(graph, CASE, expected).outcome == "wrong_result"

    graph = _graph(
        settings, retriever, ok_runner, stores, ROUTE_DATA, *["DROP TABLE employees"] * 3
    )
    failed = run_case(graph, CASE, EXPECTED)
    assert (failed.outcome, failed.sql, failed.attempts) == ("sql_failed", None, 3)
    assert "Only SELECT" in failed.error


GREETING = EvalCase("greeting", "Hello, who are you?", None, ("no-sql",))


def test_no_sql_cases(settings, retriever, ok_runner, stores) -> None:
    graph = _graph(settings, retriever, ok_runner, stores, route_reply("chat"), "Hi!")
    result = run_case(graph, GREETING, None)
    assert (result.outcome, result.correct, result.sql) == ("correct", True, None)

    # A placeholder query for a greeting is wrong, even though it runs.
    graph = _graph(
        settings, retriever, ok_runner, stores, ROUTE_DATA, "SELECT 1", "There is 1 row."
    )
    assert run_case(graph, GREETING, None).outcome == "unneeded_sql"

    # A chat reply to a question about the data is wrong too.
    graph = _graph(settings, retriever, ok_runner, stores, route_reply("chat"), "Hi!")
    assert run_case(graph, CASE, EXPECTED).outcome == "declined"


def test_no_sql_cases_have_no_reference(settings, retriever, ok_runner, stores) -> None:
    assert reference_results([CASE, GREETING], lambda _sql: EXPECTED, 10) == {CASE.id: EXPECTED}
    graph = _graph(settings, retriever, ok_runner, stores, route_reply("chat"), "Hi!")
    [result] = run_cases(graph, [GREETING], {})
    assert result.outcome == "correct"


class RecordingGraph:
    """Records the graph input; raises when asked to."""

    def __init__(self, error: Exception | None = None) -> None:
        self.inputs: list[dict] = []
        self.error = error

    def invoke(self, graph_input: dict, config: dict | None = None) -> dict:
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


VAGUE = EvalCase("vague", "Who are the best employees?", None, ("vague",), clarify=True)


def test_vague_cases(settings, retriever, ok_runner, stores) -> None:
    graph = _graph(settings, retriever, ok_runner, stores, route_reply("clarify"), "By what?")
    result = run_case(graph, VAGUE, None)
    assert (result.outcome, result.correct, result.intent) == ("correct", True, "clarify")
    assert (result.expected, result.routed_right) == ("clarify", True)

    # Guessing a query for a vague question is wrong, even when it runs.
    graph = _graph(settings, retriever, ok_runner, stores, ROUTE_DATA, "SELECT 1", "One.")
    guessed = run_case(graph, VAGUE, None)
    assert (guessed.outcome, guessed.routed_right) == ("guessed", False)

    graph = _graph(settings, retriever, ok_runner, stores, route_reply("chat"), "Hi!")
    assert run_case(graph, VAGUE, None).outcome == "declined"


def test_asking_back_on_a_clear_question_is_its_own_outcome(
    settings, retriever, ok_runner, stores
) -> None:
    graph = _graph(settings, retriever, ok_runner, stores, route_reply("clarify"), "By what?")
    result = run_case(graph, CASE, EXPECTED)
    assert (result.outcome, result.correct, result.sql) == ("unneeded_clarify", False, None)


def test_result_records_the_routers_rewrite(settings, retriever, ok_runner, stores) -> None:
    reply = route_reply(standalone="Who earns the most money?")
    graph = _graph(settings, retriever, ok_runner, stores, reply, "SELECT 1", "Employee_1.")
    result = run_case(graph, CASE, EXPECTED)
    assert (result.intent, result.standalone) == ("data", "Who earns the most money?")


def test_router_only_run_needs_no_sql_model_or_database() -> None:
    cases = [CASE, GREETING, VAGUE]
    router = fake_llm(route_reply("data"), route_reply("chat"), route_reply("data"))
    results = run_router_cases(router, cases)

    assert [(r.case_id, r.intent, r.outcome) for r in results] == [
        ("top-earner", "data", "correct"),  # the SQL isn't run: routing was right
        ("greeting", "chat", "correct"),
        ("vague", "data", "guessed"),
    ]
    assert all(r.sql is None and r.metrics is None and r.attempts == 0 for r in results)


def test_router_only_run_keeps_going_after_a_router_error() -> None:
    router = fake_llm()  # no replies left: the fake raises
    [result] = run_router_cases(router, [CASE])
    assert result.outcome == "agent_error" and result.error

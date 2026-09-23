import pytest
from sqlalchemy.exc import ProgrammingError

from rag_sql.agent.graph import build_graph, route_after_execute, route_after_validate
from rag_sql.db.query import QueryResult
from tests.conftest import fake_llm


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"error": None, "attempts": 1}, "execute_sql"),
        ({"error": "bad", "attempts": 1}, "generate_sql"),
        ({"error": "bad", "attempts": 2}, "generate_sql"),
        ({"error": "bad", "attempts": 3}, "answer"),
    ],
)
def test_route_after_validate(state: dict, expected: str) -> None:
    assert route_after_validate(state, max_retries=2) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"error": None, "attempts": 3}, "answer"),
        ({"error": "db", "attempts": 2}, "generate_sql"),
        ({"error": "db", "attempts": 3}, "answer"),
    ],
)
def test_route_after_execute(state: dict, expected: str) -> None:
    assert route_after_execute(state, max_retries=2) == expected


def _nodes_run(graph, question: str) -> tuple[list[str], dict]:
    order, state = [], {}
    for chunk in graph.stream({"question": question}, stream_mode="updates"):
        for node, update in chunk.items():
            order.append(node)
            state.update(update or {})
    return order, state


def test_happy_path(settings, retriever, ok_runner) -> None:
    llm = fake_llm("```sql\nSELECT emp_name, salary FROM employees\n```", "Employee_1.")
    graph = build_graph(llm=llm, retriever=retriever, query_runner=ok_runner, settings=settings)
    order, state = _nodes_run(graph, "Who earns the most?")

    assert order == ["retrieve_context", "generate_sql", "validate_sql", "execute_sql", "answer"]
    assert state["sql"].rstrip().endswith("LIMIT 50")
    assert state["result"].rows == [("Employee_1", 100.0)]
    assert state["answer"] == "Employee_1."
    assert state["attempts"] == 1


def test_retries_after_invalid_then_db_error(settings, retriever, ok_runner) -> None:
    calls = []

    def runner(sql: str) -> QueryResult:
        calls.append(sql)
        if len(calls) == 1:
            raise ProgrammingError(sql, {}, Exception('column "nope" does not exist'))
        return ok_runner(sql)

    llm = fake_llm(
        "DELETE FROM employees",
        "```sql\nSELECT nope FROM employees\n```",
        "```sql\nSELECT emp_name FROM employees\n```",
        "Done.",
    )
    graph = build_graph(llm=llm, retriever=retriever, query_runner=runner, settings=settings)
    order, state = _nodes_run(graph, "q")

    assert order.count("generate_sql") == 3
    assert order[-1] == "answer"
    assert state["attempts"] == 3
    assert state["error"] is None
    assert state["answer"] == "Done."


def test_gives_up_after_max_retries(settings, retriever, ok_runner) -> None:
    llm = fake_llm(*["DROP TABLE employees"] * 3)
    graph = build_graph(llm=llm, retriever=retriever, query_runner=ok_runner, settings=settings)
    order, state = _nodes_run(graph, "q")

    assert order.count("generate_sql") == 3  # 1 attempt + max_sql_retries (2)
    assert "execute_sql" not in order
    assert state["answer"].startswith("I couldn't produce a working SQL query after 3")

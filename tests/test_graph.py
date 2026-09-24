from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from sqlalchemy.exc import ProgrammingError

from rag_sql.agent.graph import build_graph, route_after_execute, route_after_validate
from rag_sql.db.query import QueryResult
from tests.conftest import EXAMPLE_DOC, TABLE_DOC, fake_llm


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


class PromptRecorder(BaseCallbackHandler):
    """Collects the text of every prompt sent to a chat model."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.prompts.append("\n".join(m.text for m in messages[0]))


def _nodes_run(
    graph, question: str, thread_id: str | None = None, callbacks: list | None = None
) -> tuple[list[str], dict]:
    order, state = [], {}
    for chunk in graph.stream(
        {"question": question, "thread_id": thread_id},
        {"callbacks": callbacks or []},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            order.append(node)
            state.update(update or {})
    return order, state


def test_happy_path(settings, retriever, ok_runner, chat_store) -> None:
    llm = fake_llm("```sql\nSELECT emp_name, salary FROM employees\n```", "Employee_1.")
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        query_runner=ok_runner,
        chat_store=chat_store,
        settings=settings,
    )
    order, state = _nodes_run(graph, "Who earns the most?")

    assert order == [
        "load_history",
        "condense_question",
        "retrieve_context",
        "generate_sql",
        "validate_sql",
        "execute_sql",
        "answer",
        "save_turn",
    ]
    assert state["standalone_question"] == "Who earns the most?"
    assert state["sql"].rstrip().endswith("LIMIT 50")
    assert state["result"].rows == [("Employee_1", 100.0)]
    assert state["answer"] == "Employee_1."
    assert state["attempts"] == 1
    # No thread_id: the turn is returned in state but not saved.
    assert [t["answer"] for t in state["history"]] == ["Employee_1."]
    assert chat_store.threads() == []


def test_retries_after_invalid_then_db_error(settings, retriever, ok_runner, chat_store) -> None:
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
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        query_runner=runner,
        chat_store=chat_store,
        settings=settings,
    )
    order, state = _nodes_run(graph, "q")

    assert order.count("generate_sql") == 3
    assert order[-2:] == ["answer", "save_turn"]
    assert state["attempts"] == 3
    assert state["error"] is None
    assert state["answer"] == "Done."


def test_gives_up_after_max_retries(settings, retriever, ok_runner, chat_store) -> None:
    llm = fake_llm(*["DROP TABLE employees"] * 3)
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        query_runner=ok_runner,
        chat_store=chat_store,
        settings=settings,
    )
    order, state = _nodes_run(graph, "q", thread_id="t1")

    assert order.count("generate_sql") == 3  # 1 attempt + max_sql_retries (2)
    assert "execute_sql" not in order
    assert state["answer"].startswith("I couldn't produce a working SQL query after 3")
    # The failed turn is saved too, so the next question knows it went unanswered.
    [saved] = chat_store.load("t1")
    assert saved["sql"] is None
    assert "Only SELECT" in saved["error"]


def test_follow_up_uses_history(settings, ok_runner, chat_store) -> None:
    retrieved_for: list[str] = []

    def retrieve(question: str) -> list:
        retrieved_for.append(question)
        return [TABLE_DOC, EXAMPLE_DOC]

    llm = fake_llm(
        # turn 1: no condense call
        "```sql\nSELECT department, avg(salary) FROM employees GROUP BY 1 ORDER BY 2 DESC\n```",
        "Sales has the highest average salary.",
        # turn 2
        "Which department has the lowest average salary?",
        "```sql\nSELECT department, avg(salary) FROM employees GROUP BY 1 ORDER BY 2\n```",
        "HR has the lowest.",
    )
    graph = build_graph(
        llm=llm,
        retriever=RunnableLambda(retrieve),
        query_runner=ok_runner,
        chat_store=chat_store,
        settings=settings,
    )
    recorder = PromptRecorder()

    _nodes_run(graph, "Which department has the highest average salary?", "t1", [recorder])
    order, state = _nodes_run(graph, "And the lowest?", "t1", [recorder])

    assert order[:2] == ["load_history", "condense_question"]
    assert state["standalone_question"] == "Which department has the lowest average salary?"
    assert state["attempts"] == 1
    assert retrieved_for[-1] == "Which department has the lowest average salary?"

    condense_prompt, sql_prompt, answer_prompt = recorder.prompts[2:]
    assert "Sales has the highest average salary." in condense_prompt
    assert "Follow-up question: And the lowest?" in condense_prompt
    assert "ORDER BY\n  2 DESC" in sql_prompt  # the previous turn's SQL, as normalized
    assert "Question: Which department has the lowest average salary?" in sql_prompt
    assert "Sales has the highest average salary." in answer_prompt

    saved = chat_store.load("t1")
    assert [t["question"] for t in saved] == [
        "Which department has the highest average salary?",
        "And the lowest?",
    ]
    assert saved[1]["standalone"] == "Which department has the lowest average salary?"
    assert saved[1]["answer"] == "HR has the lowest."


def test_threads_are_isolated(settings, retriever, ok_runner, chat_store) -> None:
    llm = fake_llm("SELECT 1", "One.", "SELECT 2", "Two.")
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        query_runner=ok_runner,
        chat_store=chat_store,
        settings=settings,
    )
    _nodes_run(graph, "first", "a")
    _, state = _nodes_run(graph, "second", "b")

    # Thread "b" has no history, so there is no condense call and no carried-over turn.
    assert state["standalone_question"] == "second"
    assert [t["question"] for t in state["history"]] == ["second"]
    assert [t["question"] for t in chat_store.load("a")] == ["first"]
    assert [t["question"] for t in chat_store.load("b")] == ["second"]

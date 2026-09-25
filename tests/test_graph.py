from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from sqlalchemy.exc import OperationalError, ProgrammingError

from rag_sql.agent.graph import (
    build_graph,
    route_after_execute,
    route_after_retrieve,
    route_after_validate,
)
from rag_sql.db.query import QueryResult
from tests.conftest import EXAMPLE_DOC, ROUTE_DATA, TABLE_DOC, fake_llm, route_reply


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
        ({"error": "db down", "attempts": 1, "db_unavailable": True}, "answer"),
    ],
)
def test_route_after_execute(state: dict, expected: str) -> None:
    assert route_after_execute(state, max_retries=2) == expected


def test_route_after_retrieve() -> None:
    assert route_after_retrieve({"intent": "data", "no_sql": False}) == "find_similar_queries"
    assert route_after_retrieve({"intent": "chat", "no_sql": True}) == "answer"
    assert route_after_retrieve({"intent": "clarify", "no_sql": True}) == "answer"


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


def test_happy_path(settings, retriever, ok_runner, chat_store, stores) -> None:
    llm = fake_llm(ROUTE_DATA, "```sql\nSELECT emp_name, salary FROM employees\n```", "Employee_1.")
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        query_runner=ok_runner,
        **stores,
        settings=settings,
    )
    order, state = _nodes_run(graph, "Who earns the most?")

    assert order == [
        "load_history",
        "route_question",
        "retrieve_context",
        "find_similar_queries",
        "generate_sql",
        "validate_sql",
        "execute_sql",
        "answer",
        "save_turn",
    ]
    assert (state["intent"], state["no_sql"]) == ("data", False)
    assert state["standalone_question"] == "Who earns the most?"
    assert state["sql"].rstrip().endswith("LIMIT 51")  # row_limit + 1, to detect truncation
    assert state["result"].rows == [("Employee_1", 100.0)]
    assert state["answer"] == "Employee_1."
    assert state["attempts"] == 1
    # No thread_id: the turn is returned in state but not saved.
    assert [t["answer"] for t in state["history"]] == ["Employee_1."]
    assert chat_store.threads() == []


def test_every_node_run_is_timed(settings, retriever, chat_store, stores) -> None:
    usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    llm = fake_llm(
        AIMessage(content=ROUTE_DATA, usage_metadata=usage),
        AIMessage(content="DROP TABLE employees", usage_metadata=usage),  # rejected: a retry
        AIMessage(content="```sql\nSELECT 1\n```", usage_metadata=usage),
        AIMessage(content="One.", usage_metadata=usage),
    )
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        query_runner=lambda _sql: QueryResult(columns=["a"], rows=[(1,)]),
        **stores,
        settings=settings,
    )
    state = graph.invoke({"question": "q", "thread_id": "t1"})

    # The reducer keeps one metric per node run, so both SQL attempts show.
    assert [(m["node"], m["attempt"]) for m in state["metrics"]] == [
        ("load_history", 0),
        ("route_question", 0),
        ("retrieve_context", 0),
        ("find_similar_queries", 0),
        ("generate_sql", 1),
        ("validate_sql", 1),
        ("generate_sql", 2),
        ("validate_sql", 2),
        ("execute_sql", 2),
        ("answer", 2),
        ("save_turn", 2),
    ]
    assert all(m["ms"] >= 0 for m in state["metrics"])
    assert [m["node"] for m in state["metrics"] if m["input_tokens"]] == [
        "route_question",
        "generate_sql",
        "generate_sql",
        "answer",
    ]
    assert "llm_usage" not in state  # moved into the metrics, never state

    [saved] = chat_store.load("t1")
    assert saved["metrics"] == state["turn_metrics"]
    # The turn's time ends with the answer: saving it is left out.
    assert saved["metrics"]["nodes"][-1]["node"] == "answer"
    assert (saved["metrics"]["attempts"], saved["metrics"]["input_tokens"]) == (2, 400)


def test_retries_after_invalid_then_db_error(
    settings, retriever, ok_runner, chat_store, stores
) -> None:
    calls = []

    def runner(sql: str) -> QueryResult:
        calls.append(sql)
        if len(calls) == 1:
            raise ProgrammingError(sql, {}, Exception('column "nope" does not exist'))
        return ok_runner(sql)

    llm = fake_llm(
        ROUTE_DATA,
        "DELETE FROM employees",
        "```sql\nSELECT nope FROM employees\n```",
        "```sql\nSELECT emp_name FROM employees\n```",
        "Done.",
    )
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        query_runner=runner,
        **stores,
        settings=settings,
    )
    order, state = _nodes_run(graph, "q")

    assert order.count("generate_sql") == 3
    assert order[-2:] == ["answer", "save_turn"]
    assert state["attempts"] == 3
    assert state["error"] is None
    assert state["answer"] == "Done."


def test_unavailable_database_is_not_retried(settings, retriever, chat_store, stores) -> None:
    def runner(sql: str) -> QueryResult:
        raise OperationalError(sql, {}, Exception("connection refused"))

    llm = fake_llm(ROUTE_DATA, "```sql\nSELECT emp_name FROM employees\n```")  # no retry replies
    graph = build_graph(
        llm=llm, retriever=retriever, query_runner=runner, **stores, settings=settings
    )
    order, state = _nodes_run(graph, "q", thread_id="t1")

    assert order.count("generate_sql") == 1
    assert order[-3:] == ["execute_sql", "answer", "save_turn"]
    assert state["db_unavailable"] is True
    assert state["answer"].startswith("I couldn't run the query because the database")
    [saved] = chat_store.load("t1")
    assert saved["error"] == "connection refused"


def test_gives_up_after_max_retries(settings, retriever, ok_runner, chat_store, stores) -> None:
    llm = fake_llm(ROUTE_DATA, *["DROP TABLE employees"] * 3)
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        query_runner=ok_runner,
        **stores,
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


def test_follow_up_uses_history(settings, ok_runner, chat_store, stores) -> None:
    retrieved_for: list[str] = []

    def retrieve(question: str) -> list:
        retrieved_for.append(question)
        return [TABLE_DOC, EXAMPLE_DOC]

    llm = fake_llm(
        # turn 1
        ROUTE_DATA,
        "```sql\nSELECT department, avg(salary) FROM employees GROUP BY 1 ORDER BY 2 DESC\n```",
        "Sales has the highest average salary.",
        # turn 2
        route_reply(standalone="Which department has the lowest average salary?"),
        "```sql\nSELECT department, avg(salary) FROM employees GROUP BY 1 ORDER BY 2\n```",
        "HR has the lowest.",
    )
    graph = build_graph(
        llm=llm,
        retriever=RunnableLambda(retrieve),
        query_runner=ok_runner,
        **stores,
        settings=settings,
    )
    recorder = PromptRecorder()

    _nodes_run(graph, "Which department has the highest average salary?", "t1", [recorder])
    order, state = _nodes_run(graph, "And the lowest?", "t1", [recorder])

    assert order[:2] == ["load_history", "route_question"]
    assert state["standalone_question"] == "Which department has the lowest average salary?"
    assert state["attempts"] == 1
    assert retrieved_for[-1] == "Which department has the lowest average salary?"

    route_prompt, sql_prompt, answer_prompt = recorder.prompts[3:]
    assert "Sales has the highest average salary." in route_prompt
    assert "Latest message: And the lowest?" in route_prompt
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


def test_threads_are_isolated(settings, retriever, ok_runner, chat_store, stores) -> None:
    llm = fake_llm(ROUTE_DATA, "SELECT 1", "One.", ROUTE_DATA, "SELECT 2", "Two.")
    graph = build_graph(
        llm=llm,
        retriever=retriever,
        query_runner=ok_runner,
        **stores,
        settings=settings,
    )
    _nodes_run(graph, "first", "a")
    _, state = _nodes_run(graph, "second", "b")

    # Thread "b" has no history: no carried-over turn.
    assert state["standalone_question"] == "second"
    assert [t["question"] for t in state["history"]] == ["second"]
    assert [t["question"] for t in chat_store.load("a")] == ["first"]
    assert [t["question"] for t in chat_store.load("b")] == ["second"]


def test_past_query_is_retrieved_in_another_thread(
    settings, retriever, ok_runner, stores, query_history
) -> None:
    llm = fake_llm(
        ROUTE_DATA,
        "```sql\nSELECT department, avg(salary) AS avg_salary FROM employees GROUP BY 1\n```",
        "Per department.",
        ROUTE_DATA,
        "```sql\nSELECT 1\n```",
        "Done.",
    )
    graph = build_graph(
        llm=llm, retriever=retriever, query_runner=ok_runner, settings=settings, **stores
    )
    recorder = PromptRecorder()

    _, first = _nodes_run(graph, "What is the average salary per department?", "a")
    # A thumbs up in the web UI saves it.
    query_history.add(first["standalone_question"], first["sql"], first["result"].row_count)
    order, second = _nodes_run(graph, "Average salary by department?", "b", [recorder])

    assert order.index("find_similar_queries") == order.index("retrieve_context") + 1
    [similar] = second["similar_queries"]
    assert similar.question == "What is the average salary per department?"
    assert similar.similarity == pytest.approx(1.0, abs=1e-3)
    sql_prompt = recorder.prompts[1]  # after the router's prompt
    assert "Similar questions answered earlier" in sql_prompt
    assert "AS avg_salary" in sql_prompt


def test_runs_never_save_query_examples(
    settings, retriever, ok_runner, stores, query_history
) -> None:
    graph = build_graph(
        llm=fake_llm(ROUTE_DATA, "SELECT 1", "One row."),
        retriever=retriever,
        query_runner=ok_runner,
        settings=settings,
        **stores,
    )
    _, state = _nodes_run(graph, "Average salary of remote employees?", "a")

    # Even a successful run: examples are only added by a user's thumbs up.
    assert state["result"].row_count == 1
    assert query_history.examples() == []


def _must_not_run(_sql: str) -> QueryResult:
    raise AssertionError("no query should run")


def test_chat_message_runs_nothing(settings, retriever, chat_store, stores) -> None:
    llm = fake_llm(route_reply("chat"), "Hi! I answer questions about the employees table.")
    graph = build_graph(
        llm=llm, retriever=retriever, query_runner=_must_not_run, **stores, settings=settings
    )
    recorder = PromptRecorder()
    order, state = _nodes_run(graph, "hello, who are you?", "t1", [recorder])

    assert order == ["load_history", "route_question", "retrieve_context", "answer", "save_turn"]
    assert state["answer"] == "Hi! I answer questions about the employees table."
    assert (state["intent"], state["no_sql"], state.get("sql")) == ("chat", True, None)
    assert "attempts" not in state  # no SQL was generated
    # The reply is written from the table names, not from a query result.
    assert "Tables: employees" in recorder.prompts[1]
    [turn] = chat_store.load("t1")
    assert (turn["sql"], turn["error"], turn["row_count"]) == (None, None, None)


def test_vague_question_gets_a_clarifying_question(settings, retriever, chat_store, stores) -> None:
    llm = fake_llm(
        route_reply("clarify", unclear="'Best' has no measure."),
        "Best by salary or by tenure?",
    )
    graph = build_graph(
        llm=llm, retriever=retriever, query_runner=_must_not_run, **stores, settings=settings
    )
    recorder = PromptRecorder()
    order, state = _nodes_run(graph, "Who are the best employees?", "t1", [recorder])

    assert order == ["load_history", "route_question", "retrieve_context", "answer", "save_turn"]
    assert (state["intent"], state["unclear"]) == ("clarify", "'Best' has no measure.")
    assert state["answer"] == "Best by salary or by tenure?"
    # The question is asked from the retrieved schema, with what the router found unclear.
    assert "Table employees" in recorder.prompts[1]
    assert "What is unclear: 'Best' has no measure." in recorder.prompts[1]
    [turn] = chat_store.load("t1")
    assert (turn["sql"], turn["error"], turn["answer"]) == (None, None, state["answer"])


def test_reply_to_a_clarifying_question_becomes_a_data_question(
    settings, retriever, ok_runner, chat_store, stores
) -> None:
    llm = fake_llm(
        route_reply("clarify", unclear="No measure."),
        "Best by salary or by tenure?",
        route_reply(standalone="Who are the highest paid employees?"),
        "```sql\nSELECT emp_name FROM employees ORDER BY salary DESC\n```",
        "Employee_1.",
    )
    graph = build_graph(
        llm=llm, retriever=retriever, query_runner=ok_runner, **stores, settings=settings
    )
    recorder = PromptRecorder()
    _nodes_run(graph, "Who are the best employees?", "t1", [recorder])
    order, state = _nodes_run(graph, "By salary", "t1", [recorder])

    assert "generate_sql" in order
    assert state["standalone_question"] == "Who are the highest paid employees?"
    # The router sees its own clarifying question, so it can combine it with the reply.
    route_prompt = recorder.prompts[2]
    assert "Answer: Best by salary or by tenure?" in route_prompt
    assert "Latest message: By salary" in route_prompt
    # The SQL model sees the earlier vague turn as one that got no SQL.
    assert "(no SQL was written for this message)" in recorder.prompts[3]


def test_unreadable_router_reply_falls_back_to_a_data_question(
    settings, retriever, ok_runner, stores
) -> None:
    llm = fake_llm("I think this is a data question.", "```sql\nSELECT 1\n```", "One.")
    graph = build_graph(
        llm=llm, retriever=retriever, query_runner=ok_runner, **stores, settings=settings
    )
    _, state = _nodes_run(graph, "How many?")

    assert (state["intent"], state["standalone_question"], state["answer"]) == (
        "data",
        "How many?",
        "One.",
    )


def test_router_has_its_own_model(settings, retriever, stores) -> None:
    graph = build_graph(
        llm=fake_llm("Hello!"),
        router_llm=fake_llm(route_reply("chat")),
        retriever=retriever,
        query_runner=_must_not_run,
        **stores,
        settings=settings,
    )
    _, state = _nodes_run(graph, "hi")

    assert (state["intent"], state["answer"]) == ("chat", "Hello!")


def test_router_replies_in_json_without_thinking(settings, retriever, ok_runner, stores) -> None:
    seen: list[dict] = []

    class Spy(BaseCallbackHandler):
        def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
            seen.append(kwargs["invocation_params"])

    graph = build_graph(
        llm=fake_llm(ROUTE_DATA, "SELECT 1", "One."),
        retriever=retriever,
        query_runner=ok_runner,
        **stores,
        settings=settings,
    )
    _nodes_run(graph, "q", callbacks=[Spy()])

    router_params, sql_params, _ = seen
    assert router_params["reasoning"] is False
    assert router_params["format"]["required"] == ["intent", "standalone_question", "unclear"]
    assert "format" not in sql_params and "reasoning" not in sql_params


class OllamaLikeModel(BaseChatModel):
    """Streams its replies token by token with usage only on the final chunk, like ChatOllama."""

    replies: list[str]

    @property
    def _llm_type(self) -> str:
        return "ollama-like"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        raise AssertionError("only called through streaming in this test")

    def _stream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        text = self.replies.pop(0)
        for token in text:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
        usage = {"input_tokens": 50, "output_tokens": len(text), "total_tokens": 50 + len(text)}
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata=usage,
                response_metadata={"total_duration": 2_000_000_000, "load_duration": 1_500_000_000},
            )
        )


def test_usage_survives_streamed_model_calls(settings, retriever, ok_runner, stores) -> None:
    # The web UI streams with "messages", which makes llm.invoke() stream: the usage on the
    # final chunk must still reach the node's metric.
    llm = OllamaLikeModel(replies=[ROUTE_DATA, "```sql\nSELECT 1\n```", "One."])
    graph = build_graph(
        llm=llm, retriever=retriever, query_runner=ok_runner, **stores, settings=settings
    )
    usage = {}
    for mode, payload in graph.stream({"question": "q"}, stream_mode=["updates", "messages"]):
        if mode == "updates":
            for node, update in payload.items():
                for m in (update or {}).get("metrics", []):
                    if m["input_tokens"]:
                        usage[node] = (m["llm_ms"], m["load_ms"], m["input_tokens"])
    assert usage == {
        "route_question": (2000, 1500, 50),
        "generate_sql": (2000, 1500, 50),
        "answer": (2000, 1500, 50),
    }

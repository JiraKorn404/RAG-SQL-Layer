import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.exc import OperationalError, ProgrammingError

from rag_sql.agent import nodes
from rag_sql.db.query import QueryResult
from rag_sql.history.chat import InMemoryChatStore
from rag_sql.history.queries import InMemoryQueryHistory
from rag_sql.metrics import node_metric
from tests.conftest import (
    EXAMPLE_DOC,
    TABLE_DOC,
    KeywordEmbeddings,
    fake_llm,
    make_turn,
    route_reply,
)


def test_generate_sql_increments_attempts_and_clears_error() -> None:
    llm = fake_llm(
        AIMessage(
            content="```sql\nSELECT emp_name FROM employees\n```",
            additional_kwargs={"reasoning_content": "Use employees."},
        )
    )
    update = nodes.generate_sql(
        {"question": "q", "context": [TABLE_DOC], "attempts": 1, "error": "boom", "sql": "x"},
        llm=llm,
        row_limit=10,
    )
    assert update == {
        "reasoning": "Use employees.",
        "sql": "SELECT emp_name FROM employees",
        "error": None,
        "result": None,
        "attempts": 2,
        "llm_usage": None,  # the fake model reports no usage
    }


def test_answer_to_chat_replies_without_a_result() -> None:
    llm = fake_llm("Hi! I answer questions about the employees table.")
    state = {
        "question": "hello, who are you?",
        "intent": "chat",
        "no_sql": True,
        "context": [TABLE_DOC],
    }
    assert nodes.answer(state, llm=llm) == {
        "answer": "Hi! I answer questions about the employees table.",
        "answer_reasoning": None,
        "llm_usage": None,
    }


def test_answer_to_a_vague_question_asks_back() -> None:
    llm = fake_llm(
        AIMessage(content="Best by salary?", additional_kwargs={"reasoning_content": "r"})
    )
    state = {
        "question": "best employees",
        "intent": "clarify",
        "unclear": "No measure.",
        "no_sql": True,
        "context": [TABLE_DOC],
    }
    assert nodes.answer(state, llm=llm) == {
        "answer": "Best by salary?",
        "answer_reasoning": "r",
        "llm_usage": None,
    }


def test_validate_sql_node() -> None:
    ok = nodes.validate_sql({"sql": "SELECT 1"}, row_limit=5)
    assert ok["error"] is None and "LIMIT 6" in ok["sql"]
    bad = nodes.validate_sql({"sql": "DELETE FROM employees"}, row_limit=5)
    assert "Only SELECT" in bad["error"]


def test_execute_sql_node_reports_db_error() -> None:
    def failing(_sql: str) -> QueryResult:
        raise ProgrammingError("SELECT nope", {}, Exception('column "nope" does not exist\nLINE 1'))

    update = nodes.execute_sql({"sql": "SELECT nope"}, run_query=failing)
    assert update == {
        "result": None,
        "error": 'column "nope" does not exist',
        "db_unavailable": False,
    }


def test_execute_sql_node_flags_unavailable_database() -> None:
    def failing(_sql: str) -> QueryResult:
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    update = nodes.execute_sql({"sql": "SELECT 1"}, run_query=failing)
    assert update == {"result": None, "error": "connection refused", "db_unavailable": True}


def test_answer_without_result_skips_llm() -> None:
    update = nodes.answer({"question": "q", "attempts": 3, "error": "bad"}, llm=fake_llm())
    assert "3 attempt(s)" in update["answer"] and "bad" in update["answer"]


def test_answer_reports_unavailable_database() -> None:
    state = {"question": "q", "attempts": 1, "error": "refused", "db_unavailable": True}
    update = nodes.answer(state, llm=fake_llm())
    assert update["answer"] == (
        "I couldn't run the query because the database is unavailable. Error: refused"
    )


def test_answer_uses_llm() -> None:
    state = {
        "question": "Top earner?",
        "sql": "SELECT 1",
        "result": QueryResult(columns=["emp_name"], rows=[("Employee_1",)]),
    }
    assert nodes.answer(state, llm=fake_llm("Employee_1 earns the most.")) == {
        "answer": "Employee_1 earns the most.",
        "answer_reasoning": None,
        "llm_usage": None,
    }


def test_answer_keeps_reasoning() -> None:
    state = {
        "question": "Top earner?",
        "sql": "SELECT 1",
        "result": QueryResult(columns=["emp_name"], rows=[("Employee_1",)]),
    }
    reply = AIMessage(content="Employee_1.", additional_kwargs={"reasoning_content": "One row."})
    assert nodes.answer(state, llm=fake_llm(reply)) == {
        "answer": "Employee_1.",
        "answer_reasoning": "One row.",
        "llm_usage": None,
    }


def test_answer_reports_model_usage() -> None:
    state = {
        "question": "Top earner?",
        "sql": "SELECT 1",
        "result": QueryResult(columns=["emp_name"], rows=[("Employee_1",)]),
    }
    reply = AIMessage(
        content="Employee_1.",
        usage_metadata={"input_tokens": 120, "output_tokens": 8, "total_tokens": 128},
        response_metadata={"total_duration": 2_500_000_000, "load_duration": 1_200_000_000},
    )
    assert nodes.answer(state, llm=fake_llm(reply))["llm_usage"] == {
        "llm_ms": 2500,
        "load_ms": 1200,
        "input_tokens": 120,
        "output_tokens": 8,
    }


# --- chat history ----------------------------------------------------------------------------


def test_load_history_from_store(chat_store) -> None:
    for i in range(4):
        chat_store.append("t1", make_turn(f"q{i}"))
    update = nodes.load_history({"thread_id": "t1"}, store=chat_store, max_turns=2)
    assert [t["question"] for t in update["history"]] == ["q2", "q3"]


def test_load_history_without_thread_keeps_passed_history(chat_store) -> None:
    passed = [make_turn("a"), make_turn("b")]
    update = nodes.load_history({"history": passed}, store=chat_store, max_turns=1)
    assert [t["question"] for t in update["history"]] == ["b"]


def test_load_history_disabled(chat_store) -> None:
    chat_store.append("t1", make_turn("q"))
    assert nodes.load_history({"thread_id": "t1"}, store=chat_store, max_turns=0) == {"history": []}


def test_route_question_data_without_history() -> None:
    update = nodes.route_question({"question": "q", "history": []}, llm=fake_llm(route_reply()))
    assert update == {
        "intent": "data",
        "standalone_question": "q",
        "unclear": None,
        "no_sql": False,
        "llm_usage": None,
    }


def test_route_question_rewrites_follow_up() -> None:
    reply = "<think>resolve it</think>\n" + route_reply(standalone="Lowest paid in Sales?")
    state = {"question": "And the lowest?", "history": [make_turn("Highest paid in Sales?")]}
    update = nodes.route_question(state, llm=fake_llm(AIMessage(content=reply)))
    assert (update["intent"], update["standalone_question"]) == ("data", "Lowest paid in Sales?")


def test_route_question_chat_keeps_the_message_and_runs_no_sql() -> None:
    update = nodes.route_question(
        {"question": "hello!"}, llm=fake_llm(route_reply("chat", standalone="Greeting"))
    )
    assert (update["intent"], update["standalone_question"], update["no_sql"]) == (
        "chat",
        "hello!",
        True,
    )


def test_route_question_clarify_keeps_what_is_unclear() -> None:
    reply = route_reply("clarify", unclear=" No measure. ")
    update = nodes.route_question({"question": "best products"}, llm=fake_llm(reply))
    assert (update["intent"], update["unclear"], update["no_sql"]) == (
        "clarify",
        "No measure.",
        True,
    )


def test_route_question_reads_json_in_a_code_block() -> None:
    reply = "```json\n" + route_reply("chat") + "\n```"
    assert nodes.route_question({"question": "hi"}, llm=fake_llm(reply))["intent"] == "chat"


@pytest.mark.parametrize("reply", ["", "not json", '{"intent": "poem"}', "[1, 2]"])
def test_route_question_falls_back_to_a_data_question(reply: str) -> None:
    update = nodes.route_question({"question": "How many?"}, llm=fake_llm(reply))
    assert (update["intent"], update["standalone_question"], update["no_sql"]) == (
        "data",
        "How many?",
        False,
    )


def test_save_turn_appends_and_saves(chat_store) -> None:
    state = {
        "thread_id": "t1",
        "history": [make_turn("q0")],
        "question": "And?",
        "standalone_question": "Full question?",
        "sql": "SELECT 1",
        "result": QueryResult(columns=["a"], rows=[(1,), (2,)]),
        "answer": "Two rows.",
        "reasoning": "Count them.",
        "answer_reasoning": "Two.",
    }
    update = nodes.save_turn(state, store=chat_store, model="qwen3.5:9b")
    turn = {
        "question": "And?",
        "standalone": "Full question?",
        "sql": "SELECT 1",
        "row_count": 2,
        "answer": "Two rows.",
        "error": None,
        "sql_reasoning": "Count them.",
        "answer_reasoning": "Two.",
        "model": "qwen3.5:9b",
        "metrics": None,  # no node metrics in the state
    }
    assert update["history"] == [make_turn("q0"), turn]
    assert chat_store.load("t1") == [{**turn, "id": update["turn_id"]}]
    assert update["turn_metrics"] is None


def test_save_turn_summarizes_node_metrics(chat_store) -> None:
    generate = node_metric(
        "generate_sql",
        attempt=1,
        ms=900,
        usage={"llm_ms": 800, "load_ms": 0, "input_tokens": 300, "output_tokens": 40},
    )
    execute = node_metric("execute_sql", attempt=1, ms=100, usage=None)
    state = {"thread_id": "t1", "question": "q", "answer": "A.", "metrics": [generate, execute]}

    update = nodes.save_turn(state, store=chat_store)
    expected = {
        "total_ms": 1000,
        "attempts": 1,
        "llm_ms": 800,
        "load_ms": 0,
        "input_tokens": 300,
        "output_tokens": 40,
        "nodes": [generate, execute],
    }
    assert update["turn_metrics"] == expected
    assert chat_store.load("t1")[0]["metrics"] == expected


def test_save_turn_failure_is_logged_not_raised(caplog) -> None:
    class BrokenStore(InMemoryChatStore):
        def append(self, thread_id, turn) -> None:
            raise RuntimeError("db down")

    state = {"thread_id": "t1", "question": "q", "sql": "bad", "error": "boom", "answer": "x"}
    update = nodes.save_turn(state, store=BrokenStore())
    assert update["history"][0]["sql"] is None
    assert update["history"][0]["error"] == "boom"
    assert "Could not save the turn" in caplog.text


# --- query history ---------------------------------------------------------------------------


def test_find_similar_queries_drops_curated_duplicates(query_history) -> None:
    query_history.add("Average salary per department?", "SELECT 1", 3)
    query_history.add(EXAMPLE_DOC.page_content, "SELECT 2", 5)  # same as the curated example
    state = {"question": "average salary by department", "context": [TABLE_DOC, EXAMPLE_DOC]}
    update = nodes.find_similar_queries(state, query_history=query_history, k=5, min_similarity=0.0)
    assert [q.question for q in update["similar_queries"]] == ["Average salary per department?"]


def test_find_similar_queries_uses_standalone_question(query_history) -> None:
    query_history.add("Highest salary per city?", "SELECT 1", 3)
    state = {"question": "And per city?", "standalone_question": "Highest salary per city?"}
    update = nodes.find_similar_queries(state, query_history=query_history, k=5, min_similarity=0.9)
    assert [q.sql for q in update["similar_queries"]] == ["SELECT 1"]


def test_find_similar_queries_failure_is_skipped(caplog) -> None:
    class Broken(InMemoryQueryHistory):
        def search(self, question, k, min_similarity):
            raise RuntimeError("db down")

    update = nodes.find_similar_queries(
        {"question": "q"}, query_history=Broken(KeywordEmbeddings()), k=5, min_similarity=0.5
    )
    assert update == {"similar_queries": []}
    assert "Query history search failed" in caplog.text


def _finished_state(**overrides) -> dict:
    state = {
        "thread_id": "t1",
        "turn_id": 7,
        "question": "And per city?",
        "standalone_question": "Average salary per city?",
        "sql": "SELECT city, avg(salary) FROM employees GROUP BY 1",
        "result": QueryResult(columns=["city", "avg"], rows=[("Paris", 1)]),
        "error": None,
        "answer": "Paris: 1.",
    }
    return state | overrides


def test_save_turn_returns_turn_id(chat_store) -> None:
    assert nodes.save_turn(_finished_state(), store=chat_store)["turn_id"] == 1
    assert nodes.save_turn(_finished_state(thread_id=None), store=chat_store)["turn_id"] is None

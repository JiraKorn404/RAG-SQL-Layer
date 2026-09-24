import pytest
from langchain_core.messages import AIMessage
from sqlalchemy.exc import ProgrammingError

from rag_sql.agent import nodes
from rag_sql.db.query import QueryResult
from rag_sql.memory import InMemoryChatStore
from rag_sql.query_history import InMemoryQueryHistory, PastQuery
from tests.conftest import EXAMPLE_DOC, TABLE_DOC, KeywordEmbeddings, fake_llm, make_turn


def test_extract_sql_prefers_sql_block() -> None:
    content = "Here you go:\n```sql\nSELECT 1\n```\nand\n```sql\nSELECT 2\n```"
    assert nodes.extract_sql(content) == "SELECT 2"


def test_extract_sql_falls_back_to_any_block_then_raw() -> None:
    assert nodes.extract_sql("```\nSELECT 3\n```") == "SELECT 3"
    assert nodes.extract_sql("  SELECT 4  ") == "SELECT 4"


def test_split_reasoning_from_additional_kwargs() -> None:
    msg = AIMessage(
        content="```sql\nSELECT 1\n```", additional_kwargs={"reasoning_content": " hmm "}
    )
    assert nodes.split_reasoning(msg) == ("hmm", "```sql\nSELECT 1\n```")


def test_split_reasoning_from_think_tags() -> None:
    msg = AIMessage(content="<think>plan it</think>\nSELECT 1")
    assert nodes.split_reasoning(msg) == ("plan it", "SELECT 1")


def test_split_reasoning_none() -> None:
    assert nodes.split_reasoning(AIMessage(content="SELECT 1")) == (None, "SELECT 1")


def test_format_schema_and_examples_split_by_kind() -> None:
    context = [TABLE_DOC, EXAMPLE_DOC]
    assert nodes.format_schema(context) == TABLE_DOC.page_content
    assert "ORDER BY salary DESC" in nodes.format_examples(context)
    assert nodes.format_examples([TABLE_DOC]) == "(none)"


def test_format_rows_markdown_table() -> None:
    result = QueryResult(columns=["a", "b"], rows=[(1, None), (2, "x|y")])
    assert nodes.format_rows(result) == "| a | b |\n|---|---|\n| 1 | NULL |\n| 2 | x\\|y |"
    assert nodes.format_rows(QueryResult(columns=["a"])) == "(no rows)"


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
    }


def test_validate_sql_node() -> None:
    ok = nodes.validate_sql({"sql": "SELECT 1"}, row_limit=5)
    assert ok["error"] is None and "LIMIT 5" in ok["sql"]
    bad = nodes.validate_sql({"sql": "DELETE FROM employees"}, row_limit=5)
    assert "Only SELECT" in bad["error"]


def test_execute_sql_node_reports_db_error() -> None:
    def failing(_sql: str) -> QueryResult:
        raise ProgrammingError("SELECT nope", {}, Exception('column "nope" does not exist\nLINE 1'))

    update = nodes.execute_sql({"sql": "SELECT nope"}, run_query=failing)
    assert update == {"result": None, "error": 'column "nope" does not exist'}


def test_answer_without_result_skips_llm() -> None:
    update = nodes.answer({"question": "q", "attempts": 3, "error": "bad"}, llm=fake_llm())
    assert "3 attempt(s)" in update["answer"] and "bad" in update["answer"]


def test_answer_uses_llm() -> None:
    state = {
        "question": "Top earner?",
        "sql": "SELECT 1",
        "result": QueryResult(columns=["emp_name"], rows=[("Employee_1",)]),
    }
    assert nodes.answer(state, llm=fake_llm("Employee_1 earns the most.")) == {
        "answer": "Employee_1 earns the most.",
        "answer_reasoning": None,
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
    }


# --- chat history ----------------------------------------------------------------------------


def test_format_history_sql_and_answers() -> None:
    history = [make_turn("q1", sql="SELECT 1", answer="a1"), make_turn("q2", sql=None)]
    with_sql = nodes.format_history(history, sql=True, answers=False)
    assert with_sql == (
        "Question: q1\n```sql\nSELECT 1\n```\n\nQuestion: q2\n(no working SQL; error: boom)"
    )
    assert nodes.format_history(history[:1]) == "Question: q1\nAnswer: a1"


def test_format_history_truncates_long_answers() -> None:
    text = nodes.format_history([make_turn("q", answer="x" * 1000)])
    assert text.endswith(" …")
    assert len(text) < nodes.HISTORY_ANSWER_CHARS + 50


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


def test_condense_question_skips_llm_without_history() -> None:
    update = nodes.condense_question({"question": "q", "history": []}, llm=fake_llm())
    assert update == {"standalone_question": "q"}


def test_condense_question_rewrites_follow_up() -> None:
    llm = fake_llm(AIMessage(content="<think>resolve it</think>\nLowest paid in Sales?"))
    state = {"question": "And the lowest?", "history": [make_turn("Highest paid in Sales?")]}
    assert nodes.condense_question(state, llm=llm) == {
        "standalone_question": "Lowest paid in Sales?"
    }


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
    }
    assert update["history"] == [make_turn("q0"), turn]
    assert chat_store.load("t1") == [turn]


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


def test_format_similar_queries() -> None:
    queries = [PastQuery("q1", "SELECT 1", 0.9), PastQuery("q2", "SELECT 2", 0.8)]
    assert nodes.format_similar_queries(queries) == (
        "Question: q1\n```sql\nSELECT 1\n```\n\nQuestion: q2\n```sql\nSELECT 2\n```"
    )


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


def test_save_query_example_stores_standalone_question(query_history) -> None:
    assert nodes.save_query_example(_finished_state(), query_history=query_history) == {
        "example_saved": True
    }
    [found] = query_history.search("Average salary per city?", 5, 0.9)
    assert found.sql == "SELECT city, avg(salary) FROM employees GROUP BY 1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"thread_id": None},  # not a saved conversation
        {"result": None, "error": "boom"},  # every attempt failed
        {"result": QueryResult(columns=["city"])},  # no rows
    ],
)
def test_save_query_example_skips_unsuccessful_turns(query_history, overrides) -> None:
    state = _finished_state(**overrides)
    assert nodes.save_query_example(state, query_history=query_history) == {"example_saved": False}
    assert query_history.search("Average salary per city?", 5, 0.0) == []


def test_save_turn_returns_turn_id(chat_store) -> None:
    assert nodes.save_turn(_finished_state(), store=chat_store)["turn_id"] == 1
    assert nodes.save_turn(_finished_state(thread_id=None), store=chat_store)["turn_id"] is None

"""Tests against the real Postgres (and Ollama). Run with: uv run pytest -m integration"""

from collections.abc import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from rag_sql.config import get_settings
from rag_sql.db.connection import get_engine
from rag_sql.db.introspect import introspect_tables
from rag_sql.db.query import is_database_unavailable, run_query, validate_sql
from rag_sql.memory import get_chat_store, new_thread_id
from rag_sql.query_history import PostgresQueryHistory, list_examples, set_enabled
from tests.conftest import KeywordEmbeddings, make_turn

pytestmark = pytest.mark.integration


def test_readonly_role_cannot_write() -> None:
    with pytest.raises(DBAPIError, match="read-only"), get_engine().connect() as conn:
        conn.exec_driver_sql("CREATE TABLE should_fail (a int)")


def test_run_query_only_runs_queries() -> None:
    # Rows come from a server-side cursor (DECLARE ... CURSOR FOR), which only accepts queries.
    with pytest.raises(DBAPIError, match="syntax error"):
        run_query(get_engine(), "CREATE TABLE should_fail (a int)", row_limit=1, timeout_ms=2000)


def test_added_limit_still_reports_truncation() -> None:
    sql = validate_sql("SELECT emp_name FROM employees", 10)
    result = run_query(get_engine(), sql, row_limit=10, timeout_ms=5000)
    assert (result.row_count, result.truncated) == (10, True)


def test_unreachable_database_is_unavailable() -> None:
    engine = get_engine(settings=get_settings().model_copy(update={"postgres_port": 1}))
    with pytest.raises(DBAPIError) as info:
        run_query(engine, "SELECT 1", row_limit=1, timeout_ms=2000)
    assert is_database_unavailable(info.value)


def test_query_error_is_not_unavailable() -> None:
    with pytest.raises(DBAPIError) as info:
        run_query(get_engine(), "SELECT nope FROM employees", row_limit=1, timeout_ms=2000)
    assert not is_database_unavailable(info.value)


def test_statement_timeout() -> None:
    with pytest.raises(DBAPIError, match="statement timeout"):
        run_query(get_engine(), "SELECT pg_sleep(2)", row_limit=1, timeout_ms=200)


def test_query_employees_and_truncation() -> None:
    result = run_query(
        get_engine(),
        validate_sql("SELECT emp_name FROM employees", 500),
        row_limit=10,
        timeout_ms=5000,
    )
    assert result.columns == ["emp_name"]
    assert result.row_count == 10
    assert result.truncated


def test_percent_and_colon_literals_are_not_placeholders() -> None:
    sql = "SELECT count(*) AS n, ':x' AS lit FROM employees WHERE emp_name LIKE 'Employee_1%'"
    result = run_query(get_engine(), sql, row_limit=5, timeout_ms=5000)
    assert result.rows[0][0] > 0
    assert result.rows[0][1] == ":x"


def test_introspection_skips_vector_tables() -> None:
    names = {t.name for t in introspect_tables(get_engine("admin"))}
    assert "employees" in names
    assert not any(n.startswith("langchain_pg_") for n in names)


def test_end_to_end() -> None:
    from rag_sql.agent import build_graph

    state = build_graph().invoke({"question": "How many employees are there?"})
    assert state.get("result") is not None, state.get("error")
    assert state["answer"]


# --- chat history ----------------------------------------------------------------------------


@pytest.fixture
def test_thread() -> Iterator[str]:
    """A throwaway thread id; its rows are removed as admin (the chat role cannot delete)."""
    thread_id = f"pytest-{new_thread_id()}"
    yield thread_id
    with get_engine("admin").begin() as conn:
        conn.execute(
            text("DELETE FROM chat_memory.chat_turns WHERE thread_id = :t"), {"t": thread_id}
        )


def test_chat_store_round_trip(test_thread: str) -> None:
    store = get_chat_store()
    first = make_turn(
        "first",
        sql_reasoning="Plan the SQL.",
        answer_reasoning="Read the rows.",
        model="qwen3.5:9b",
    )
    store.append(test_thread, first)
    store.append(test_thread, make_turn("second", sql=None, answer="failed"))

    assert store.load(test_thread) == [first, make_turn("second", sql=None, answer="failed")]
    assert [t["question"] for t in store.load(test_thread, 1)] == ["second"]
    [summary] = [t for t in store.threads(limit=1000) if t.thread_id == test_thread]
    assert (summary.turns, summary.first_question) == (2, "first")


def test_reader_role_cannot_read_chat_history() -> None:
    with pytest.raises(DBAPIError, match="permission denied"):
        run_query(
            get_engine(), "SELECT * FROM chat_memory.chat_turns", row_limit=1, timeout_ms=2000
        )


def test_chat_role_cannot_read_agent_tables() -> None:
    with (
        pytest.raises(DBAPIError, match="permission denied"),
        get_engine("memory").connect() as conn,
    ):
        conn.execute(text("SELECT 1 FROM employees LIMIT 1"))


def test_chat_role_cannot_delete_history(test_thread: str) -> None:
    get_chat_store().append(test_thread, make_turn("keep me"))
    with pytest.raises(DBAPIError, match="permission denied"), get_engine("memory").begin() as conn:
        conn.execute(
            text("DELETE FROM chat_memory.chat_turns WHERE thread_id = :t"), {"t": test_thread}
        )


def test_introspection_skips_chat_history() -> None:
    names = {t.qualified_name for t in introspect_tables(get_engine("admin"))}
    assert not any("chat_turns" in n for n in names)


# --- query history ---------------------------------------------------------------------------


@pytest.fixture
def pg_history() -> Iterator[PostgresQueryHistory]:
    """Query history under a throwaway embedding model name; its rows are removed as admin."""
    model = f"pytest-{new_thread_id()}"
    yield PostgresQueryHistory(get_engine("memory"), KeywordEmbeddings(), embed_model=model)
    with get_engine("admin").begin() as conn:
        conn.execute(
            text("DELETE FROM chat_memory.query_examples WHERE embed_model = :m"), {"m": model}
        )


def test_query_history_round_trip(pg_history: PostgresQueryHistory, test_thread: str) -> None:
    turn_id = get_chat_store().append(test_thread, make_turn("Average salary per department?"))
    assert pg_history.add("Average salary per department?", "SELECT 1", 4, turn_id=turn_id)
    assert pg_history.add("Highest salary per city?", "SELECT 2", 9)
    assert not pg_history.add("Average salary per department?", "SELECT 1", 4)  # duplicate

    found = pg_history.search("average salary by department", k=5, min_similarity=0.3)
    assert [q.sql for q in found][:2] == ["SELECT 1", "SELECT 2"]
    assert found[0].similarity == pytest.approx(1.0, abs=1e-3)
    assert [q.sql for q in pg_history.search("average salary department", 5, 0.9)] == ["SELECT 1"]


def test_query_history_ignores_other_embedding_models(pg_history: PostgresQueryHistory) -> None:
    pg_history.add("Average salary per department?", "SELECT 1", 4)
    other = PostgresQueryHistory(get_engine("memory"), KeywordEmbeddings(), embed_model="other")
    assert other.search("Average salary per department?", 5, 0.0) == []


def test_disabled_example_is_hidden_and_not_re_added(pg_history: PostgresQueryHistory) -> None:
    pg_history.add("Average salary per department?", "SELECT 1", 4)
    [stored] = [
        e
        for e in list_examples(get_engine("memory"), limit=1000)
        if e.question == "Average salary per department?"
        and e.embed_model == pg_history._embed_model
    ]
    assert set_enabled(get_engine("admin"), [stored.id], enabled=False) == 1
    try:
        assert pg_history.search("Average salary per department?", 5, 0.0) == []
        assert not pg_history.add("Average salary per department?", "SELECT 1", 4)
    finally:
        set_enabled(get_engine("admin"), [stored.id], enabled=True)
    assert pg_history.search("Average salary per department?", 5, 0.0)[0].sql == "SELECT 1"


def test_backfill_adds_successful_turns_only(
    pg_history: PostgresQueryHistory, test_thread: str
) -> None:
    store = get_chat_store()
    store.append(test_thread, make_turn("Remote employees per city?", sql="SELECT 7"))
    store.append(test_thread, make_turn("Broken question about salary?", sql=None))

    assert pg_history.backfill() >= 1
    assert pg_history.backfill() == 0  # nothing left for this model
    sqls = [q.sql for q in pg_history.search("Remote employees per city?", 5, 0.99)]
    assert "SELECT 7" in sqls
    assert not pg_history.search("Broken question about salary?", 5, 0.99)


def test_reader_role_cannot_read_query_history() -> None:
    with pytest.raises(DBAPIError, match="permission denied"):
        run_query(
            get_engine(), "SELECT * FROM chat_memory.query_examples", row_limit=1, timeout_ms=2000
        )


def test_chat_role_cannot_enable_or_disable_examples() -> None:
    with pytest.raises(DBAPIError, match="permission denied"):
        set_enabled(get_engine("memory"), [1], enabled=False)

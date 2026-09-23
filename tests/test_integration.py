"""Tests against the real Postgres (and Ollama). Run with: uv run pytest -m integration"""

import pytest
from sqlalchemy.exc import DBAPIError

from rag_sql.db.connection import get_engine
from rag_sql.db.introspect import introspect_tables
from rag_sql.db.query import run_query, validate_sql

pytestmark = pytest.mark.integration


def test_readonly_role_cannot_write() -> None:
    with pytest.raises(DBAPIError, match="read-only"):
        run_query(get_engine(), "CREATE TABLE should_fail (a int)", row_limit=1, timeout_ms=2000)


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
    names = {t.name for t in introspect_tables(get_engine(readonly=False))}
    assert "employees" in names
    assert not any(n.startswith("langchain_pg_") for n in names)


def test_end_to_end() -> None:
    from rag_sql.agent import build_graph

    state = build_graph().invoke({"question": "How many employees are there?"})
    assert state.get("result") is not None, state.get("error")
    assert state["answer"]

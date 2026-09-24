import pytest
from sqlalchemy.exc import (
    DataError,
    DBAPIError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)

from rag_sql.db.query import SQLValidationError, is_database_unavailable, validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM employees",
        "select city, avg(salary) from employees group by city;",
        "WITH a AS (SELECT 1 AS x) SELECT x FROM a",
        "SELECT 1 UNION ALL SELECT 2",
        "SELECT emp_name FROM employees WHERE emp_name ILIKE '%1%' AND join_date::date > '2020-01-01'",  # noqa: E501
    ],
)
def test_accepts_read_only_queries(sql: str) -> None:
    assert validate_sql(sql, 100)


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("", "Empty"),
        ("   ", "Empty"),
        ("DELETE FROM employees", "Only SELECT"),
        ("UPDATE employees SET salary = 0", "Only SELECT"),
        ("INSERT INTO employees (emp_name) VALUES ('x')", "Only SELECT"),
        ("DROP TABLE employees", "Only SELECT"),
        ("TRUNCATE employees", "Only SELECT"),
        ("CREATE TABLE x (a int)", "Only SELECT"),
        ("COPY employees TO '/tmp/x'", "Only SELECT"),
        ("SELECT 1; DROP TABLE employees", "exactly one statement"),
        ("SELECT 1; SELECT 2", "exactly one statement"),
        ("WITH d AS (DELETE FROM employees RETURNING *) SELECT * FROM d", "DELETE"),
        ("SELECT * INTO backup FROM employees", "INTO"),
        ("SELECT * FROM employees FOR UPDATE", "LOCK"),
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
        ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
        ("SELECT * FROM employees WHERE (", "could not be parsed"),
        ("SELEC * FRM employees", "Only SELECT"),
    ],
)
def test_rejects_unsafe_or_invalid_sql(sql: str, message: str) -> None:
    with pytest.raises(SQLValidationError, match=message):
        validate_sql(sql, 100)


def test_adds_limit_when_missing() -> None:
    # One row over the limit, so run_query can tell the result was truncated.
    assert validate_sql("SELECT * FROM employees", 25).rstrip().endswith("LIMIT 26")
    assert validate_sql("SELECT * FROM employees LIMIT ALL", 25).rstrip().endswith("LIMIT 26")


def test_keeps_existing_limit() -> None:
    sql = validate_sql("SELECT * FROM employees ORDER BY salary DESC LIMIT 5 OFFSET 10", 25)
    assert "LIMIT 5" in sql
    assert "OFFSET 10" in sql
    assert "LIMIT 26" not in sql
    assert "FETCH FIRST 5 ROWS ONLY" in validate_sql(
        "SELECT * FROM employees FETCH FIRST 5 ROWS ONLY", 25
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM employees LIMIT 1000000",
        "SELECT * FROM employees FETCH FIRST 1000 ROWS ONLY",
        "SELECT * FROM employees LIMIT 10 + 5",  # not a plain number
        "SELECT 1 UNION SELECT 2 LIMIT 500",
    ],
)
def test_caps_large_or_computed_limit(sql: str) -> None:
    capped = validate_sql(sql, 25)
    assert capped.rstrip().endswith("LIMIT 26")
    assert "1000" not in capped and "500" not in capped


def test_subquery_limit_is_left_alone() -> None:
    sql = validate_sql("SELECT * FROM (SELECT * FROM employees LIMIT 1000) AS e", 25)
    assert "LIMIT 1000" in sql
    assert sql.rstrip().endswith("LIMIT 26")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT query_to_xml('SELECT pg_sleep(60)', true, true, '')",
        "SELECT pg_catalog.query_to_xml_and_xmlschema('SELECT 1', true, true, '')",
        "SELECT cursor_to_xml('c', 10, true, true, '')",
    ],
)
def test_rejects_functions_that_run_sql_strings(sql: str) -> None:
    with pytest.raises(SQLValidationError, match="not allowed"):
        validate_sql(sql, 100)


class _PgError(Exception):
    """Stands in for a psycopg error, which carries the server's SQLSTATE."""

    def __init__(self, sqlstate: str | None) -> None:
        super().__init__("error")
        self.sqlstate = sqlstate


@pytest.mark.parametrize(
    ("error", "unavailable"),
    [
        (OperationalError("SELECT 1", {}, _PgError(None)), True),  # refused, DNS, lost socket
        (InterfaceError("SELECT 1", {}, _PgError(None)), True),
        (OperationalError("SELECT 1", {}, _PgError("08006")), True),  # connection failure
        (OperationalError("SELECT 1", {}, _PgError("28P01")), True),  # bad password
        (OperationalError("SELECT 1", {}, _PgError("57P01")), True),  # admin shutdown
        (OperationalError("SELECT 1", {}, _PgError("53300")), True),  # too many connections
        (OperationalError("SELECT 1", {}, _PgError("57014")), False),  # statement timeout
        (ProgrammingError("SELECT 1", {}, _PgError("42703")), False),  # undefined column
        (ProgrammingError("SELECT 1", {}, _PgError("42501")), False),  # permission denied
        (DataError("SELECT 1", {}, _PgError(None)), False),  # client-side conversion error
        (DBAPIError("SELECT 1", {}, _PgError("42703"), connection_invalidated=True), True),
    ],
)
def test_is_database_unavailable(error: DBAPIError, unavailable: bool) -> None:
    assert is_database_unavailable(error) is unavailable


def test_parse_error_has_no_ansi_codes() -> None:
    with pytest.raises(SQLValidationError) as info:
        validate_sql("SELECT * FROM employees WHERE (", 10)
    assert "\x1b" not in str(info.value)

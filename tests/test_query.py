import pytest

from rag_sql.db.query import SQLValidationError, validate_sql


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
    assert validate_sql("SELECT * FROM employees", 25).rstrip().endswith("LIMIT 25")


def test_keeps_existing_limit() -> None:
    sql = validate_sql("SELECT * FROM employees ORDER BY salary DESC LIMIT 5", 25)
    assert "LIMIT 5" in sql
    assert "LIMIT 25" not in sql


def test_parse_error_has_no_ansi_codes() -> None:
    with pytest.raises(SQLValidationError) as info:
        validate_sql("SELECT * FROM employees WHERE (", 10)
    assert "\x1b" not in str(info.value)

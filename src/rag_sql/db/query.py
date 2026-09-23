"""Validate and run agent SQL. Read-only three ways: SQL parse check, read-only role and txn."""

from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlalchemy import Engine, text
from sqlglot import exp
from sqlglot.errors import ParseError

DIALECT = "postgres"

# Nodes that write, lock or create objects, wherever they appear in the tree
# (e.g. a data-modifying CTE: WITH d AS (DELETE ... RETURNING *) SELECT ...).
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Command,
    exp.Into,  # SELECT ... INTO new_table
    exp.Lock,  # SELECT ... FOR UPDATE / FOR SHARE
)

# Functions with side effects or server access. The read-only role blocks most of these already.
_FORBIDDEN_FUNCTION_PREFIXES = (
    "pg_sleep",
    "pg_read_",
    "pg_ls_",
    "pg_stat_file",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_advisory",
    "pg_reload_conf",
    "set_config",
    "lo_",
    "dblink",
)


class SQLValidationError(ValueError):
    """The statement is not a single read-only query."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    truncated: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _function_name(node: exp.Func) -> str:
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def validate_sql(sql: str, row_limit: int) -> str:
    """Return a normalized, safe version of `sql`, or raise SQLValidationError.

    Accepts exactly one SELECT / WITH ... SELECT / set operation (UNION, ...). Adds
    `LIMIT row_limit` when the outer query has none. The returned SQL is regenerated from the
    parsed tree, so what runs is exactly what was checked.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("Empty SQL statement.")
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except ParseError as e:
        # str(e) embeds ANSI highlighting; build a plain message for the model instead.
        details = "; ".join(
            f"{err.get('description')} (line {err.get('line')}, col {err.get('col')})"
            for err in e.errors
        )
        raise SQLValidationError(f"SQL could not be parsed: {details or e}") from e

    if len(statements) != 1:
        raise SQLValidationError(f"Expected exactly one statement, got {len(statements)}.")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise SQLValidationError(
            f"Only SELECT queries are allowed, got {tree.key.upper()} statement."
        )

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLValidationError(
                f"Only read-only queries are allowed; found {node.key.upper()}."
            )
        if isinstance(node, exp.Func) and _function_name(node).startswith(
            _FORBIDDEN_FUNCTION_PREFIXES
        ):
            raise SQLValidationError(f"Function {_function_name(node)}() is not allowed.")

    if tree.args.get("limit") is None:
        tree = tree.limit(row_limit)
    return tree.sql(dialect=DIALECT, pretty=True)


def run_query(engine: Engine, sql: str, *, row_limit: int, timeout_ms: int) -> QueryResult:
    """Run an already validated query in a read-only transaction with a statement timeout.

    Fetches at most `row_limit` rows; `truncated` is set when more were available.
    """
    with engine.connect() as conn:
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        # set_config(..., true) is SET LOCAL, but accepts a bound parameter.
        conn.execute(
            text("SELECT set_config('statement_timeout', :timeout, true)"),
            {"timeout": str(timeout_ms)},
        )
        # no_parameters: the driver gets the statement as-is, so '%' and ':name' in it are never
        # treated as placeholders. Errors still surface as sqlalchemy.exc.DBAPIError.
        result = conn.execution_options(no_parameters=True).exec_driver_sql(sql)
        if result.returns_rows:
            columns = list(result.keys())
            rows = [tuple(r) for r in result.fetchmany(row_limit + 1)]
        else:
            columns, rows = [], []
        result.close()
        conn.rollback()

    truncated = len(rows) > row_limit
    return QueryResult(columns=columns, rows=rows[:row_limit], truncated=truncated)

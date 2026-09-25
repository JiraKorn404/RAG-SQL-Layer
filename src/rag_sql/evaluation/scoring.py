"""Compare the agent's query result with the reference result. Pure: no database, no model.

Results are compared by value, not by SQL text or column names:
- whole numbers must be equal (counts); other numbers may differ by rounding: up to ABS_TOL
  (so `avg(x)` matches `round(avg(x), 2)`) or REL_TOL of the value; dates become ISO strings;
- each reference column must match some column of the agent's result, in any order; extra
  columns in the agent's result are allowed but scored separately ("correct_extra_columns");
- rows are compared as multisets, or as sequences when the reference SQL has a top-level
  ORDER BY.
Values must match as values: TRUE and 'Remote' differ, so write cases whose answer doesn't
depend on how labels are spelled.
"""

import datetime as dt
import itertools
import math
from collections.abc import Sequence
from decimal import Decimal
from typing import Literal

import sqlglot
from sqlglot import exp

from rag_sql.db.query import DIALECT, QueryResult

Outcome = Literal[
    "correct",  # same values in the same shape (or NO_SQL for a question not about the data)
    "correct_extra_columns",  # every reference column is there, plus more
    "wrong_result",  # the SQL ran, but the result differs
    "declined",  # the agent replied NO_SQL to a question about the data
    "unneeded_sql",  # the agent wrote SQL for a question that isn't about the data
    "sql_failed",  # no SQL ran: retries ran out
    "db_unavailable",  # the database couldn't be used
    "agent_error",  # the run raised an exception
]
CORRECT: tuple[Outcome, ...] = ("correct", "correct_extra_columns")

# How far two numbers that aren't both whole may differ: rounding to 2 decimals, or 0.01 %.
ABS_TOL = 0.005
REL_TOL = 1e-4

# Column mappings tried before giving up (columns with identical values multiply them).
_MAX_MAPPINGS = 10_000

Value = None | bool | int | float | str


def normalize(value: object) -> Value:
    """Whole numbers become int, other numbers float, dates ISO strings, the rest str."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float | Decimal):
        number = float(value)
        return int(number) if number.is_integer() else number
    if isinstance(value, dt.date | dt.time):  # datetime is a date
        return value.isoformat()
    return str(value)


def _is_number(value: Value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def same_value(a: Value, b: Value) -> bool:
    if not (_is_number(a) and _is_number(b)):
        return type(a) is type(b) and a == b  # TRUE is not 1, '5' is not 5
    if isinstance(a, int) and isinstance(b, int):
        return a == b  # whole numbers (counts) must be exact
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _sort_key(value: Value) -> tuple[int, float | str]:
    """Orders values so that close numbers end up side by side."""
    match value:
        case None:
            return (0, 0)
        case bool():
            return (1, int(value))
        case int() | float():
            return (2, value)
        case _:
            return (3, value)


def _same_rows(expected: Sequence[tuple[Value, ...]], actual: Sequence[tuple[Value, ...]]) -> bool:
    return len(expected) == len(actual) and all(
        all(same_value(a, b) for a, b in zip(e, x, strict=True))
        for e, x in zip(expected, actual, strict=True)
    )


def _sorted(rows: Sequence[tuple[Value, ...]]) -> list[tuple[Value, ...]]:
    return sorted(rows, key=lambda row: tuple(_sort_key(v) for v in row))


def is_ordered(sql: str) -> bool:
    """Whether the query's top level has an ORDER BY, so the row order is part of the answer."""
    tree = sqlglot.parse_one(sql, read=DIALECT)
    return isinstance(tree, exp.Query) and tree.args.get("order") is not None


def compare(expected: QueryResult, actual: QueryResult, *, ordered: bool) -> Outcome:
    """Score `actual` against `expected`: "correct", "correct_extra_columns" or "wrong_result"."""
    if len(actual.rows) != len(expected.rows) or len(actual.columns) < len(expected.columns):
        return "wrong_result"
    extra = len(actual.columns) > len(expected.columns)
    if not expected.rows:
        return "correct_extra_columns" if extra else "correct"

    reference = [tuple(normalize(v) for v in row) for row in expected.rows]
    rows = [tuple(normalize(v) for v in row) for row in actual.rows]

    def column(table: list[tuple[Value, ...]], i: int) -> list[tuple[Value, ...]]:
        return _sorted([(row[i],) for row in table])

    # A reference column can only map to an actual column with the same values.
    candidates = [
        [j for j in range(len(actual.columns)) if _same_rows(column(reference, i), column(rows, j))]
        for i in range(len(expected.columns))
    ]
    if not ordered:
        reference = _sorted(reference)
    for mapping in itertools.islice(itertools.product(*candidates), _MAX_MAPPINGS):
        if len(set(mapping)) < len(mapping):
            continue  # two reference columns on one actual column
        projected = [tuple(row[j] for j in mapping) for row in rows]
        if _same_rows(reference, projected if ordered else _sorted(projected)):
            return "correct_extra_columns" if extra else "correct"
    return "wrong_result"

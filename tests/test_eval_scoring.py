import datetime as dt
from decimal import Decimal

import pytest

from rag_sql.db.query import QueryResult
from rag_sql.evaluation.scoring import compare, is_ordered, normalize, same_value


def _result(columns: str, *rows: tuple) -> QueryResult:
    return QueryResult(columns=columns.split(), rows=list(rows))


def test_normalize() -> None:
    assert normalize(Decimal("58698.1666")) == 58698.1666
    assert normalize(Decimal("5.00")) == 5 and isinstance(normalize(Decimal("5.00")), int)
    assert normalize(True) is True  # not 1: bool is an int, but a different answer
    assert normalize(dt.date(2024, 1, 2)) == "2024-01-02"
    assert normalize(None) is None


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        (58698.1666, 58698.17, True),  # rounded to 2 decimals
        (1366.3773, 1366.37, True),  # difference of rounded averages
        (0.1234, 0.12, True),
        (0.1234, 0.13, False),
        (12.34, 12.3, False),  # rounded to 1 decimal: too far
        (1_000_000.5, 1_000_050.5, True),  # within 0.01 %
        (115521, 115529, False),  # counts must be exact
        (5, 5.001, True),
        (True, 1, False),
        ("5", 5, False),
        (None, None, True),
        (None, 0, False),
    ],
)
def test_same_value(a, b, same: bool) -> None:
    assert same_value(normalize(a), normalize(b)) is same
    assert same_value(normalize(b), normalize(a)) is same


@pytest.mark.parametrize(
    ("sql", "ordered"),
    [
        ("SELECT city FROM employees ORDER BY count(*) DESC LIMIT 1", True),
        ("SELECT a FROM t", False),
        ("SELECT a FROM (SELECT a FROM t ORDER BY a) AS s", False),  # only the outer query
        ("SELECT 1 UNION SELECT 2 ORDER BY 1", True),
    ],
)
def test_is_ordered(sql: str, ordered: bool) -> None:
    assert is_ordered(sql) is ordered


def test_same_values_any_column_names_and_row_order() -> None:
    expected = _result("remote_work avg_salary", (False, Decimal("59069.17")), (True, 60435.54))
    actual = _result("remote avg", (True, 60435.5412), (False, 59069.1666))
    assert compare(expected, actual, ordered=False) == "correct"


def test_columns_can_be_in_another_order() -> None:
    expected = _result("department products", ("bulk", 38), ("other", 548))
    actual = _result("n dept", (548, "other"), (38, "bulk"))
    assert compare(expected, actual, ordered=False) == "correct"


def test_extra_columns_are_scored_separately() -> None:
    expected = _result("city", ("Delhi",))
    actual = _result("city employees", ("Delhi", 37))
    assert compare(expected, actual, ordered=True) == "correct_extra_columns"


@pytest.mark.parametrize(
    "actual",
    [
        _result("city", ("Pune",)),  # different value
        _result("city", ("Delhi",), ("Pune",)),  # extra row
        _result("n", (37,)),  # the reference column is missing
        _result("city"),  # no rows
    ],
)
def test_wrong_results(actual: QueryResult) -> None:
    assert compare(_result("city", ("Delhi",)), actual, ordered=True) == "wrong_result"


def test_order_matters_only_when_ordered() -> None:
    expected = _result("a", (1,), (2,))
    actual = _result("a", (2,), (1,))
    assert compare(expected, actual, ordered=False) == "correct"
    assert compare(expected, actual, ordered=True) == "wrong_result"


def test_rows_must_match_not_just_columns() -> None:
    # Each column has the same values, but paired differently.
    expected = _result("name n", ("a", 1), ("b", 2))
    actual = _result("name n", ("a", 2), ("b", 1))
    assert compare(expected, actual, ordered=False) == "wrong_result"


def test_columns_with_identical_values_are_matched_either_way() -> None:
    expected = _result("x y label", (1, 1, "p"), (2, 2, "q"))
    actual = _result("y x label", (1, 1, "p"), (2, 2, "q"))
    assert compare(expected, actual, ordered=False) == "correct"


def test_duplicate_rows_count() -> None:
    expected = _result("a", (1,), (1,), (2,))
    assert compare(expected, _result("a", (1,), (2,), (2,)), ordered=False) == "wrong_result"
    assert compare(expected, _result("a", (2,), (1,), (1,)), ordered=False) == "correct"


def test_empty_results() -> None:
    assert compare(_result("a"), _result("b"), ordered=False) == "correct"
    assert compare(_result("a"), _result("a b"), ordered=False) == "correct_extra_columns"


def test_near_equal_numbers_pair_up_across_rows() -> None:
    expected = _result("dept avg", ("a", Decimal("10.004")), ("b", Decimal("10.001")))
    actual = _result("dept avg", ("b", 10.0), ("a", 10.0))
    assert compare(expected, actual, ordered=False) == "correct"

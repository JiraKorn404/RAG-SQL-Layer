import pytest

from rag_sql.query_history import InMemoryQueryHistory, PastQuery, normalize_question, pick_best
from tests.conftest import KeywordEmbeddings


def test_normalize_question() -> None:
    assert normalize_question("  Average   Salary?\n") == "average salary?"


def test_pick_best_orders_cuts_off_and_dedupes() -> None:
    candidates = [
        PastQuery("b", "SELECT 2", 0.80),
        PastQuery("a", "SELECT 1", 0.95),
        PastQuery("A ", "SELECT 3", 0.90),  # same question as "a", different SQL
        PastQuery("c", "SELECT 4", 0.60),  # below the cutoff
    ]
    assert [q.sql for q in pick_best(candidates, k=5, min_similarity=0.7)] == [
        "SELECT 1",
        "SELECT 2",
    ]
    assert [q.sql for q in pick_best(candidates, k=1, min_similarity=0.0)] == ["SELECT 1"]


@pytest.fixture
def history() -> InMemoryQueryHistory:
    history = InMemoryQueryHistory(KeywordEmbeddings())
    history.add("Average salary per department?", "SELECT 1", 4)
    history.add("Highest salary per city?", "SELECT 2", 9)
    history.add("How many remote employees?", "SELECT 3", 1)
    return history


def test_search_best_first_with_cutoff(history) -> None:
    found = history.search("average salary by department", k=5, min_similarity=0.3)
    assert [q.sql for q in found][:2] == ["SELECT 1", "SELECT 2"]
    assert found[0].similarity == pytest.approx(1.0, abs=1e-3)
    assert [q.sql for q in history.search("average salary by department", 5, 0.9)] == ["SELECT 1"]
    assert history.search("average salary by department", k=0, min_similarity=0.0) == []


def test_add_skips_duplicates(history) -> None:
    assert history.add("Average salary per department?", "SELECT 1", 4) is False
    assert history.add("Average salary per department?", "SELECT 99", 4) is True


def test_disabled_examples_are_hidden_and_never_re_added(history) -> None:
    history.set_enabled([1], enabled=False)
    assert "SELECT 1" not in [q.sql for q in history.search("average salary department", 5, 0.0)]
    assert history.add("Average salary per department?", "SELECT 1", 4) is False

    history.set_enabled([1], enabled=True)
    assert history.search("average salary department", 1, 0.0)[0].sql == "SELECT 1"

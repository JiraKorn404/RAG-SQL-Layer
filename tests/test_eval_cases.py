from pathlib import Path

import pytest

from rag_sql.evaluation.cases import EvalCase, load_cases, select_cases
from rag_sql.history.queries import normalize_question
from rag_sql.retrieval import load_examples


def test_the_committed_cases_are_valid() -> None:
    cases = load_cases()
    assert len(cases) >= 5
    assert len({c.id for c in cases}) == len(cases)
    few_shot = {normalize_question(ex["question"]) for ex in load_examples()}
    assert not few_shot & {normalize_question(c.question) for c in cases}
    assert all(c.tags for c in cases)
    # Questions that aren't about the data have no reference SQL.
    assert {c.id for c in cases if c.sql is None} == {"chat-greeting", "chat-general-knowledge"}


def test_follow_up_history_becomes_turns() -> None:
    [case] = [c for c in load_cases() if c.history]
    [turn] = [t.to_turn() for t in case.history]
    assert turn["question"] == turn["standalone"] == "Which department has the most products?"
    assert turn["sql"].startswith("SELECT") and turn["answer"].startswith("Personal care")
    assert turn["error"] is None and turn["metrics"] is None


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- id: a\n  sql: SELECT 1\n", "needs non-empty"),
        ("- id: a\n  question: Q?\n", "either 'sql' or 'no_sql: true'"),
        ("- {id: a, question: 'Q?', sql: SELECT 1, no_sql: true}\n", "either 'sql'"),
        ("- {id: a, question: 'Q?', no_sql: yes-please}\n", "either 'sql'"),
        (
            "- {id: a, question: 'Q1?', sql: SELECT 1}\n"
            "- {id: a, question: 'Q2?', sql: SELECT 2}\n",
            "duplicate id",
        ),
        (
            "- {id: a, question: 'How many employees joined each year?', sql: SELECT 1}\n",
            "few-shot",
        ),
        ("- {id: a, question: 'Q?', sql: SELECT 1, history: [{question: 'P?'}]}\n", "history"),
        ("id: a\n", "expected a list"),
    ],
)
def test_invalid_cases_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_cases(_write(tmp_path, text))


def test_select_cases() -> None:
    cases = [
        EvalCase("a", "A?", "SELECT 1", tags=("employees",)),
        EvalCase("b", "B?", "SELECT 1", tags=("imba", "join")),
        EvalCase("c", "C?", "SELECT 1", tags=("imba",)),
    ]
    assert [c.id for c in select_cases(cases)] == ["a", "b", "c"]
    assert [c.id for c in select_cases(cases, tags=["join", "employees"])] == ["a", "b"]
    assert [c.id for c in select_cases(cases, tags=["imba"], limit=1)] == ["b"]

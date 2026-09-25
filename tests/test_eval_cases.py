from collections import Counter
from pathlib import Path

import pytest

from rag_sql.agent.prompts import ROUTER_PROMPT
from rag_sql.evaluation.cases import EvalCase, load_cases, select_cases
from rag_sql.history.queries import normalize_question
from rag_sql.retrieval import load_examples


def test_the_committed_cases_are_valid() -> None:
    cases = load_cases()
    assert len({c.id for c in cases}) == len(cases)
    few_shot = {normalize_question(ex["question"]) for ex in load_examples()}
    assert not few_shot & {normalize_question(c.question) for c in cases}
    # Every case has a group tag and a difficulty, so reports can break results down.
    difficulties = {"easy", "medium", "hard"}
    assert all(len(difficulties & set(c.tags)) == 1 for c in cases)
    assert all(len(set(c.tags) - difficulties) >= 1 for c in cases)
    # Only retail exists in the database.
    assert all("employees" not in (c.sql or "").replace("retail.employees", "") for c in cases)
    assert not any("imba" in (c.sql or "") for c in cases)


def test_the_committed_cases_cover_every_kind() -> None:
    kinds = Counter(c.kind for c in load_cases())
    assert kinds["sql"] >= 30
    assert kinds["chat"] >= 5 and kinds["clarify"] >= 6
    # Cases that guard against asking back on clear questions, and follow-ups.
    tags = Counter(tag for c in load_cases() for tag in c.tags)
    assert tags["not-vague"] >= 3 and tags["followup"] >= 5


def test_cases_are_not_shown_to_the_router() -> None:
    """The router prompt has its own examples: a case that repeats one would test memory."""
    prompt = " ".join(m.prompt.template for m in ROUTER_PROMPT.messages if hasattr(m, "prompt"))
    assert [c.id for c in load_cases() if c.question in prompt] == []


def test_follow_up_history_becomes_turns() -> None:
    case = next(c for c in load_cases() if c.id == "followup-fewest-late-city")
    [turn] = [t.to_turn() for t in case.history]
    assert turn["question"] == turn["standalone"] == "Which store city has the most late shipments?"
    assert turn["sql"].startswith("SELECT") and turn["answer"].startswith("Mumbai")
    assert turn["error"] is None and turn["metrics"] is None


def test_clarified_follow_ups_have_a_turn_without_sql() -> None:
    case = next(c for c in load_cases() if c.id == "followup-clarified-city")
    [turn] = [t.to_turn() for t in case.history]
    assert turn["sql"] is None and turn["answer"].endswith("number of orders?")
    assert case.kind == "sql"  # the reply is a data question once combined with it


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- id: a\n  sql: SELECT 1\n", "needs non-empty"),
        ("- id: a\n  question: Q?\n", "exactly one of 'sql'"),
        ("- {id: a, question: 'Q?', sql: SELECT 1, no_sql: true}\n", "exactly one of 'sql'"),
        ("- {id: a, question: 'Q?', no_sql: true, clarify: true}\n", "exactly one of 'sql'"),
        ("- {id: a, question: 'Q?', no_sql: yes-please}\n", "exactly one of 'sql'"),
        (
            "- {id: a, question: 'Q1?', sql: SELECT 1}\n"
            "- {id: a, question: 'Q2?', sql: SELECT 2}\n",
            "duplicate id",
        ),
        (
            "- {id: a, question: 'How many orders were placed in each year?', sql: SELECT 1}\n",
            "few-shot",
        ),
        (
            "- {id: a, question: 'Q?', sql: SELECT 1, history: [{question: 'P?'}]}\n",
            "'sql' or 'answer'",
        ),
        (
            "- {id: a, question: 'Q?', sql: SELECT 1, history: [{answer: 'A'}]}\n",
            "need a 'question'",
        ),
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


def test_case_kinds(tmp_path: Path) -> None:
    cases = load_cases(
        _write(
            tmp_path,
            "- {id: a, question: 'Q1?', sql: SELECT 1}\n"
            "- {id: b, question: 'Q2?', no_sql: true}\n"
            "- {id: c, question: 'Q3?', clarify: true}\n",
        )
    )
    assert [(c.id, c.kind, c.sql is None) for c in cases] == [
        ("a", "sql", False),
        ("b", "chat", True),
        ("c", "clarify", True),
    ]


def test_history_turn_answered_with_a_question_has_no_sql(tmp_path: Path) -> None:
    text = (
        "- id: a\n  question: By revenue\n  sql: SELECT 1\n"
        "  history:\n    - {question: 'Best customers?', answer: 'By revenue or orders?'}\n"
    )
    [case] = load_cases(_write(tmp_path, text))
    [turn] = [t.to_turn() for t in case.history]
    assert (turn["sql"], turn["answer"], turn["error"]) == (None, "By revenue or orders?", None)

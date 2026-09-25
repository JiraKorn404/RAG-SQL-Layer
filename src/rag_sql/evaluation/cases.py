"""Evaluation cases: questions with reference SQL, loaded from evaluation/cases.yaml."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from rag_sql.config import PROJECT_ROOT
from rag_sql.history.chat import Turn
from rag_sql.history.queries import normalize_question
from rag_sql.retrieval import DEFAULT_EXAMPLES_PATH, load_examples

DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation" / "cases.yaml"

# What the agent should do with a case's question: answer it with SQL, reply without SQL because
# it isn't about the data (chat), or ask back because it is too vague (clarify).
Kind = Literal["sql", "chat", "clarify"]


@dataclass(frozen=True)
class EarlierTurn:
    """A turn before the case's question, for follow-ups. A turn the agent answered with a question
    back has no SQL."""

    question: str
    sql: str | None = None
    answer: str = ""

    def to_turn(self) -> Turn:
        return {
            "question": self.question,
            "standalone": self.question,
            "sql": self.sql.strip() if self.sql else None,
            "row_count": None,
            "answer": self.answer,
            "error": None,
            "sql_reasoning": None,
            "answer_reasoning": None,
            "model": None,
            "metrics": None,
        }


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    # Reference: its result is the right answer. None: the agent must not write SQL, because the
    # message isn't about the data (chat) or is too vague (`clarify`).
    sql: str | None
    tags: tuple[str, ...] = ()
    history: tuple[EarlierTurn, ...] = ()
    clarify: bool = False

    @property
    def kind(self) -> Kind:
        if self.clarify:
            return "clarify"
        return "chat" if self.sql is None else "sql"


def load_cases(
    path: Path = DEFAULT_CASES_PATH, *, few_shot_path: Path = DEFAULT_EXAMPLES_PATH
) -> list[EvalCase]:
    """Read and check the cases: required fields (reference `sql`, `no_sql: true` for a message that
    isn't about the data, or `clarify: true` for a question too vague to answer), unique ids, no
    question from the few-shot file (the model would be shown its answer)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of cases")
    few_shot = {normalize_question(ex["question"]) for ex in load_examples(few_shot_path)}

    cases: list[EvalCase] = []
    for i, item in enumerate(data):
        where = f"{path}: case #{i}"
        if not isinstance(item, dict) or not all(item.get(k) for k in ("id", "question")):
            raise ValueError(f"{where} needs non-empty 'id' and 'question'")
        expected = [bool(item.get("sql")), item.get("no_sql") is True, item.get("clarify") is True]
        if sum(expected) != 1:
            raise ValueError(
                f"{where} ({item['id']}) needs exactly one of 'sql', 'no_sql: true' or "
                "'clarify: true'"
            )
        history = []
        for turn in item.get("history") or []:
            if not isinstance(turn, dict) or not turn.get("question"):
                raise ValueError(f"{where} ({item['id']}): history turns need a 'question'")
            if not (turn.get("sql") or turn.get("answer")):
                raise ValueError(f"{where} ({item['id']}): history turns need 'sql' or 'answer'")
            history.append(EarlierTurn(turn["question"], turn.get("sql"), turn.get("answer", "")))
        case = EvalCase(
            id=str(item["id"]),
            question=item["question"].strip(),
            sql=item["sql"].strip() if item.get("sql") else None,
            tags=tuple(item.get("tags") or ()),
            history=tuple(history),
            clarify=item.get("clarify") is True,
        )
        if any(c.id == case.id for c in cases):
            raise ValueError(f"{where}: duplicate id {case.id!r}")
        if normalize_question(case.question) in few_shot:
            raise ValueError(f"{where} ({case.id}): the question is a few-shot example")
        cases.append(case)
    return cases


def select_cases(
    cases: list[EvalCase], *, tags: list[str] | None = None, limit: int | None = None
) -> list[EvalCase]:
    """Cases with any of `tags` (all when None), the first `limit` of them."""
    selected = [c for c in cases if not tags or set(tags) & set(c.tags)]
    return selected[:limit] if limit else selected

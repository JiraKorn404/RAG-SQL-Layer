import pytest

from rag_sql.db.query import QueryResult
from rag_sql.metrics import node_metric, summarize
from rag_sql.ui.steps import (
    NO_SQL_NOTE,
    NO_SQL_SAVED_NOTE,
    ExampleCandidate,
    escape_md,
    example_candidate,
    steps_from_turn,
    steps_from_update,
)
from tests.conftest import TABLE_DOC, make_turn


def _steps(node: str, update: dict, question: str = "Q?", attempts: int = 1) -> list:
    return steps_from_update(node, update, question=question, attempts=attempts, max_retries=2)


def test_rewritten_question_shown_only_when_it_differs() -> None:
    same = {"intent": "data", "standalone_question": "Q?", "no_sql": False}
    assert _steps("route_question", same) == []
    rewritten = {"intent": "data", "standalone_question": "Q about sales?", "no_sql": False}
    [step] = _steps("route_question", rewritten)
    assert (step.kind, step.text) == ("interpreted", "Q about sales?")


def test_chat_message_shows_a_note() -> None:
    update = {"intent": "chat", "standalone_question": "Q?", "no_sql": True}
    assert [(s.kind, s.text) for s in _steps("route_question", update)] == [
        ("caption", NO_SQL_NOTE)
    ]


def test_vague_question_shows_what_is_unclear() -> None:
    update = {
        "intent": "clarify",
        "standalone_question": "Best products?",
        "unclear": "No measure.",
        "no_sql": True,
    }
    steps = _steps("route_question", update, question="best?")
    assert [(s.kind, s.text) for s in steps] == [
        ("interpreted", "Best products?"),
        ("caption", "No SQL yet: the question is too vague. No measure."),
    ]


def test_history_caption_only_with_earlier_turns() -> None:
    assert _steps("load_history", {"history": []}) == []
    [step] = _steps("load_history", {"history": [make_turn("A?"), make_turn("B?")]})
    assert step.text.startswith("2 earlier")


def test_context_caption_lists_tables() -> None:
    [step] = _steps("retrieve_context", {"context": [TABLE_DOC]})
    assert "employees" in step.text


def test_generation_without_reasoning_skips_thinking() -> None:
    steps = _steps("generate_sql", {"sql": "SELECT 1", "reasoning": None, "attempts": 1})
    assert [s.kind for s in steps] == ["sql"]
    assert steps[0].label == "SQL"


def test_retry_generation_is_labelled_with_attempt() -> None:
    steps = _steps("generate_sql", {"sql": "SELECT 1", "reasoning": "hmm", "attempts": 2})
    assert [(s.kind, s.label) for s in steps] == [
        ("thinking", "Thinking (attempt 2)"),
        ("sql", "SQL (attempt 2)"),
    ]


def test_saved_no_sql_turn_shows_the_note() -> None:
    turn = make_turn("hello?", sql=None, answer="Hi!")
    turn["error"] = None  # answered without SQL and without an error: chat, or a question back
    steps = steps_from_turn(turn)
    assert [(s.kind, s.text) for s in steps] == [("caption", NO_SQL_SAVED_NOTE), ("answer", "Hi!")]


@pytest.mark.parametrize(
    ("node", "attempts", "update", "label"),
    [
        ("validate_sql", 1, {"error": "boom"}, "Validation error · retry 1 of 2"),
        ("execute_sql", 3, {"error": "boom"}, "Execution error · no retries left"),
        (
            "execute_sql",
            1,
            {"error": "boom", "db_unavailable": True},
            "Execution error · database unavailable, not retried",
        ),
    ],
)
def test_error_step(node: str, attempts: int, update: dict, label: str) -> None:
    [step] = _steps(node, update, attempts=attempts)
    assert (step.kind, step.label, step.text) == ("error", label, "boom")


def test_result_step_has_dataframe_and_truncation_note() -> None:
    result = QueryResult(columns=["a"], rows=[(1,), (2,)], truncated=True)
    [step] = _steps("execute_sql", {"result": result, "error": None})
    assert step.data["a"].tolist() == [1, 2]
    assert "truncated" in step.text


def test_example_candidate_is_the_standalone_question_and_sql() -> None:
    turn = make_turn("And per city?", "SELECT city FROM employees") | {
        "standalone": "Average salary per city?",
        "row_count": 4,
        "id": 7,
    }
    assert example_candidate(turn) == ExampleCandidate(
        "Average salary per city?", "SELECT city FROM employees", 4, 7
    )
    # Turns not loaded from a store have no id yet.
    assert example_candidate(make_turn("Q?")).turn_id is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"sql": None, "row_count": None, "error": "boom"},  # every attempt failed
        {"row_count": 0},  # no rows
        {"error": "Interrupted"},  # the run didn't finish
        {"sql": None, "row_count": None},  # no SQL: not a question about the data
    ],
)
def test_no_example_candidate_without_rows(overrides) -> None:
    assert example_candidate(make_turn("Q?") | overrides) is None


def test_answer_step_includes_its_thinking() -> None:
    steps = _steps("answer", {"answer": "A.", "answer_reasoning": "One row."})
    assert [(s.kind, s.label) for s in steps] == [("thinking", "Thinking (answer)"), ("answer", "")]
    assert [s.kind for s in _steps("answer", {"answer": "A.", "answer_reasoning": None})] == [
        "answer"
    ]


def test_steps_from_saved_turn() -> None:
    kinds = [s.kind for s in steps_from_turn(make_turn("Q?"))]
    assert kinds == ["sql", "caption", "answer"]
    failed = [s.kind for s in steps_from_turn(make_turn("Q?", sql=None))]
    assert failed == ["error", "answer"]


def test_saved_turn_without_answer_has_no_answer_step() -> None:
    # An interrupted run can be saved before it answered.
    turn = make_turn("Q?", sql=None, answer="")
    assert [s.kind for s in steps_from_turn(turn)] == ["error"]


def test_saved_rewritten_question_is_shown() -> None:
    turn = {**make_turn("And?"), "standalone": "Full question?"}
    assert steps_from_turn(turn)[0].kind == "interpreted"


def test_steps_from_saved_turn_keep_thinking() -> None:
    turn = make_turn("Q?", sql_reasoning="Plan.", answer_reasoning="Read.")
    assert [(s.kind, s.label, s.text) for s in steps_from_turn(turn) if s.kind == "thinking"] == [
        ("thinking", "Thinking", "Plan."),
        ("thinking", "Thinking (answer)", "Read."),
    ]


def test_saved_turn_shows_its_model_first() -> None:
    [first, *_] = steps_from_turn(make_turn("Q?", model="qwen3.5:9b"))
    assert (first.kind, first.text) == ("model", "qwen3.5:9b")


def _metrics() -> dict:
    return summarize(
        [
            node_metric("retrieve_context", attempt=0, ms=300, usage=None),
            node_metric(
                "generate_sql",
                attempt=1,
                ms=2200,
                usage={"llm_ms": 2000, "load_ms": None, "input_tokens": 900, "output_tokens": 50},
            ),
        ]
    )


def test_metrics_step_after_save_turn() -> None:
    assert _steps("save_turn", {"turn_id": None, "turn_metrics": None}) == []
    [step] = _steps("save_turn", {"turn_id": 1, "turn_metrics": _metrics()})
    assert (step.kind, step.text) == ("metrics", "2.5 s · 1 attempt · 950 tokens")
    assert step.data["node"].tolist() == ["retrieve_context", "generate_sql"]
    # Nodes without a model call have empty cells, not NaN.
    assert step.data["tokens in"].isna().tolist() == [True, False]
    assert str(step.data["tokens in"].dtype) == "Int64"


def test_saved_turn_shows_its_metrics_last() -> None:
    steps = steps_from_turn(make_turn("Q?", metrics=_metrics()))
    assert (steps[-1].kind, steps[-1].text) == ("metrics", "2.5 s · 1 attempt · 950 tokens")
    assert all(s.kind != "metrics" for s in steps_from_turn(make_turn("Q?")))


def test_escape_md_keeps_dollars_literal() -> None:
    assert escape_md("$5 and $6") == r"\$5 and \$6"

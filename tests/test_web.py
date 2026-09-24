import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from rag_sql.agent.graph import build_graph
from rag_sql.db.query import QueryResult
from rag_sql.llm import ChatModelInfo
from rag_sql.web import escape_md, model_label, steps_from_turn, steps_from_update
from tests.conftest import TABLE_DOC, fake_llm, make_turn

SQL_REPLY = AIMessage(
    content="```sql\nSELECT emp_name, salary FROM employees\n```",
    additional_kwargs={"reasoning_content": "Order employees by salary."},
)
ANSWER_REPLY = AIMessage(
    content="Employee_1 earns $100.", additional_kwargs={"reasoning_content": "One row."}
)


def _steps(node: str, update: dict, question: str = "Q?", attempts: int = 1) -> list:
    return steps_from_update(node, update, question=question, attempts=attempts, max_retries=2)


def test_condensed_question_shown_only_when_rewritten() -> None:
    assert _steps("condense_question", {"standalone_question": "Q?"}) == []
    [step] = _steps("condense_question", {"standalone_question": "Q about sales?"})
    assert step.kind == "caption"
    assert "Q about sales?" in step.text


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


@pytest.mark.parametrize(
    ("node", "attempts", "label"),
    [
        ("validate_sql", 1, "Validation error · retry 1 of 2"),
        ("execute_sql", 3, "Execution error · no retries left"),
    ],
)
def test_error_step(node: str, attempts: int, label: str) -> None:
    [step] = _steps(node, {"error": "boom"}, attempts=attempts)
    assert (step.kind, step.label, step.text) == ("error", label, "boom")


def test_result_step_has_dataframe_and_truncation_note() -> None:
    result = QueryResult(columns=["a"], rows=[(1,), (2,)], truncated=True)
    [step] = _steps("execute_sql", {"result": result, "error": None})
    assert step.data["a"].tolist() == [1, 2]
    assert "truncated" in step.text


def test_query_example_caption_only_when_saved() -> None:
    assert _steps("save_query_example", {"example_saved": False}) == []
    assert len(_steps("save_query_example", {"example_saved": True})) == 1


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


def test_steps_from_saved_turn_keep_thinking() -> None:
    turn = make_turn("Q?", sql_reasoning="Plan.", answer_reasoning="Read.")
    assert [(s.kind, s.label, s.text) for s in steps_from_turn(turn) if s.kind == "thinking"] == [
        ("thinking", "Thinking", "Plan."),
        ("thinking", "Thinking (answer)", "Read."),
    ]


def test_saved_turn_shows_its_model_first() -> None:
    [first, *_] = steps_from_turn(make_turn("Q?", model="qwen3.5:9b"))
    assert (first.kind, first.text) == ("caption", "Model `qwen3.5:9b`")


def test_model_label() -> None:
    assert model_label(ChatModelInfo("qwen3.5:9b", "9.7B", thinking=True)) == "qwen3.5:9b · 9.7B"
    assert model_label(ChatModelInfo("tiny", None, thinking=False)) == "tiny · no thinking"


def test_escape_md_keeps_dollars_literal() -> None:
    assert escape_md("$5 and $6") == r"\$5 and \$6"


def _fake_graph(settings, retriever, ok_runner, stores, *replies):
    return build_graph(
        llm=fake_llm(*replies),
        retriever=retriever,
        query_runner=ok_runner,
        **stores,
        settings=settings,
    )


def test_messages_mode_streams_thinking_per_node(settings, retriever, ok_runner, stores) -> None:
    # The web UI relies on this: nodes call llm.invoke(), and stream_mode="messages" still
    # delivers the tokens, with the reasoning in additional_kwargs.
    graph = _fake_graph(settings, retriever, ok_runner, stores, SQL_REPLY, ANSWER_REPLY)
    reasoning: dict[str, str] = {}
    for mode, payload in graph.stream(
        {"question": "Who earns the most?"}, stream_mode=["updates", "messages"]
    ):
        if mode == "messages":
            message, metadata = payload
            node = metadata["langgraph_node"]
            reasoning[node] = reasoning.get(node, "") + (
                message.additional_kwargs.get("reasoning_content") or ""
            )
    assert reasoning == {"generate_sql": "Order employees by salary.", "answer": "One row."}


APP = """
import streamlit as st
from rag_sql.web import main

main(**st.session_state["deps"])
"""

MODELS = [
    ChatModelInfo("gemma4:e4b", "8.0B", thinking=True),
    ChatModelInfo("qwen3.5:9b", "9.7B", thinking=True),
]


class NamedFakeChatModel(GenericFakeChatModel):
    """A fake chat model with a name, like ChatOllama's `model`, so turns record it."""

    model: str


def _app(settings, chat_store, graph_for, list_models=lambda: MODELS) -> AppTest:
    at = AppTest.from_string(APP, default_timeout=30)
    at.session_state["deps"] = {
        "graph_for": graph_for,
        "store": chat_store,
        "settings": settings.model_copy(update={"ollama_chat_model": "gemma4:e4b"}),
        "list_models": list_models,
    }
    return at


def _captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


def test_app_runs_a_turn_and_lists_the_thread(
    settings, retriever, ok_runner, stores, chat_store
) -> None:
    graph = build_graph(
        llm=NamedFakeChatModel(messages=iter([SQL_REPLY, ANSWER_REPLY]), model="gemma4:e4b"),
        retriever=retriever,
        query_runner=ok_runner,
        **stores,
        settings=settings,
    )
    at = _app(settings, chat_store, lambda _model: graph)
    at.run()
    assert not at.exception
    assert "No saved conversations yet." in [c.value for c in at.sidebar.caption]

    at.chat_input[0].set_value("Who earns the most?").run()
    assert not at.exception

    assert "Model `gemma4:e4b`" in _captions(at)
    assert [c.value for c in at.code] == ["SELECT emp_name, salary FROM employees"]
    assert at.dataframe[0].value["emp_name"].tolist() == ["Employee_1"]
    # AppTest lists an expander that has an icon under `status`.
    assert {e.label for e in [*at.expander, *at.status]} >= {"Thinking", "Thinking (answer)"}
    assert any(m.value == r"Employee_1 earns \$100." for m in at.markdown)

    [thread] = chat_store.threads()
    assert at.session_state["thread_id"] == thread.thread_id
    assert [b.label for b in at.sidebar.button] == ["New chat", "Who earns the most?"]
    assert chat_store.load(thread.thread_id)[0]["model"] == "gemma4:e4b"

    # Leave the conversation and come back: the saved model and thinking are shown again.
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert not at.code
    at.sidebar.button[1].click().run()
    assert not at.exception
    assert "Model `gemma4:e4b`" in _captions(at)
    # The saved SQL is the validated one: formatted, with the LIMIT added.
    [code] = at.code
    assert code.value.endswith("LIMIT 50")
    assert {e.label for e in [*at.expander, *at.status]} >= {"Thinking", "Thinking (answer)"}


def test_app_uses_the_chosen_model(settings, retriever, ok_runner, stores, chat_store) -> None:
    chosen: list[str] = []

    def graph_for(model: ChatModelInfo):
        chosen.append(model.name)
        return _fake_graph(settings, retriever, ok_runner, stores, SQL_REPLY, ANSWER_REPLY)

    at = _app(settings, chat_store, graph_for)
    at.run()
    [picker] = at.sidebar.selectbox
    assert picker.options == ["gemma4:e4b · 8.0B", "qwen3.5:9b · 9.7B"]
    assert picker.value == "gemma4:e4b"  # OLLAMA_CHAT_MODEL

    picker.select("qwen3.5:9b").run()
    at.chat_input[0].set_value("Who earns the most?").run()
    assert not at.exception
    assert chosen == ["qwen3.5:9b"]
    assert "Model `qwen3.5:9b`" in _captions(at)
    assert at.sidebar.selectbox[0].value == "qwen3.5:9b"  # kept for the next question


def test_app_falls_back_when_models_cannot_be_listed(
    settings, retriever, ok_runner, stores, chat_store
) -> None:
    chosen: list[ChatModelInfo] = []

    def graph_for(model: ChatModelInfo):
        chosen.append(model)
        return _fake_graph(settings, retriever, ok_runner, stores, SQL_REPLY, ANSWER_REPLY)

    at = _app(settings, chat_store, graph_for, list_models=lambda: None)
    at.run()
    assert not at.sidebar.selectbox
    assert at.sidebar.warning[0].value == "Couldn't list the Ollama models."

    at.chat_input[0].set_value("Who earns the most?").run()
    assert not at.exception
    assert chosen == [ChatModelInfo("gemma4:e4b", None, thinking=True)]

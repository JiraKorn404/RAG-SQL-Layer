import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from streamlit.testing.v1 import AppTest

from rag_sql.agent.graph import build_graph
from rag_sql.history.queries import InMemoryQueryHistory
from rag_sql.llm import ChatModelInfo
from rag_sql.ui.steps import Step
from rag_sql.ui.web import model_label
from rag_sql.ui.web_chat import INTERRUPTED, run_turn
from rag_sql.ui.web_examples import filter_examples
from tests.conftest import ROUTE_DATA, KeywordEmbeddings, fake_llm, make_turn

SQL_REPLY = AIMessage(
    content="```sql\nSELECT emp_name, salary FROM employees\n```",
    additional_kwargs={"reasoning_content": "Order employees by salary."},
)
ANSWER_REPLY = AIMessage(
    content="Employee_1 earns $100.", additional_kwargs={"reasoning_content": "One row."}
)


def test_model_label() -> None:
    assert model_label(ChatModelInfo("qwen3.5:9b", "9.7B", thinking=True)) == "qwen3.5:9b · 9.7B"
    assert model_label(ChatModelInfo("tiny", None, thinking=False)) == "tiny · no thinking"


def _fake_graph(settings, retriever, ok_runner, stores, *replies):
    return build_graph(
        llm=fake_llm(ROUTE_DATA, *replies),
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
    assert reasoning == {
        "route_question": "",  # the router doesn't think
        "generate_sql": "Order employees by salary.",
        "answer": "One row.",
    }


class ScriptedGraph:
    """Stands in for a compiled graph: streams the given items, then raises `error` if set."""

    def __init__(self, items: list, error: BaseException | None = None) -> None:
        self.items = items
        self.error = error

    def stream(self, _input: dict, config: dict | None = None, stream_mode: list[str] = ()):
        yield from self.items
        if self.error is not None:
            raise self.error


class Stop(BaseException):
    """Like Streamlit's RerunException for a new question: a BaseException, not an Exception."""


def _thinking_token(node: str, text: str) -> tuple:
    chunk = AIMessageChunk(content="", additional_kwargs={"reasoning_content": text})
    return ("messages", (chunk, {"langgraph_node": node}))


def test_interrupted_run_is_saved_with_what_it_streamed() -> None:
    items = [
        ("updates", {"route_question": {"standalone_question": "Who earns the most money?"}}),
        _thinking_token("generate_sql", "Sort by salary."),
    ]
    saved, steps = [], []
    with pytest.raises(Stop):
        run_turn(
            ScriptedGraph(items, Stop()),
            "Who earns most?",
            "t1",
            steps,
            2,
            save_unfinished=saved.append,
        )

    [turn] = saved
    assert turn["error"] == INTERRUPTED
    assert turn["standalone"] == "Who earns the most money?"
    assert turn["sql_reasoning"] == "Sort by salary."
    assert (turn["sql"], turn["answer"]) == (None, "")
    # Kept in the session, so the next run redraws them.
    assert [s.kind for s in steps] == ["interpreted", "thinking", "error"]
    assert steps[-1] == Step("error", INTERRUPTED, label="Interrupted")


def test_failed_run_is_saved_with_its_error() -> None:
    items = [("updates", {"generate_sql": {"sql": "SELECT 1", "attempts": 1, "reasoning": None}})]
    saved, steps = [], []
    run_turn(
        ScriptedGraph(items, RuntimeError("Ollama is down")),
        "Q?",
        "t1",
        steps,
        2,
        save_unfinished=saved.append,
    )

    [turn] = saved
    assert turn["error"] == "Ollama is down"
    assert steps[-1] == Step("error", "Ollama is down", label="The agent failed")


def test_finished_run_is_not_saved_again() -> None:
    items = [("updates", {"answer": {"answer": "A."}}), ("updates", {"save_turn": {"turn_id": 1}})]
    saved: list = []
    run_turn(ScriptedGraph(items), "Q?", "t1", [], 2, save_unfinished=saved.append)
    assert saved == []


def test_run_turn_returns_the_saved_turn_with_its_id() -> None:
    turn = make_turn("Q?")
    items = [("updates", {"save_turn": {"history": [turn], "turn_id": 5}})]
    assert run_turn(ScriptedGraph(items), "Q?", "t1", [], 2) == {**turn, "id": 5}


def test_run_turn_returns_a_failed_turn_with_its_error() -> None:
    items = [("updates", {"generate_sql": {"sql": "SELECT 1", "attempts": 1, "reasoning": None}})]
    turn = run_turn(ScriptedGraph(items, RuntimeError("Ollama is down")), "Q?", "t1", [], 2)
    assert (turn["error"], "id" in turn) == ("Ollama is down", False)


def test_failed_save_of_unfinished_run_is_only_logged(caplog) -> None:
    def broken(_turn) -> None:
        raise RuntimeError("db down")

    run_turn(ScriptedGraph([], RuntimeError("boom")), "Q?", "t1", [], 2, save_unfinished=broken)
    assert "Could not save the unfinished turn" in caplog.text


APP = """
import streamlit as st
from rag_sql.ui.web import main

main(**st.session_state["deps"])
"""

MODELS = [
    ChatModelInfo("gemma4:e4b", "8.0B", thinking=True),
    ChatModelInfo("qwen3.5:9b", "9.7B", thinking=True),
]


class NamedFakeChatModel(GenericFakeChatModel):
    """A fake chat model with a name, like ChatOllama's `model`, so turns record it."""

    model: str


def _app(
    settings, chat_store, graph_for, list_models=lambda: MODELS, query_history=None
) -> AppTest:
    at = AppTest.from_string(APP, default_timeout=30)
    at.session_state["deps"] = {
        "graph_for": graph_for,
        "store": chat_store,
        "settings": settings.model_copy(update={"ollama_chat_model": "gemma4:e4b"}),
        "list_models": list_models,
        "query_history": query_history or InMemoryQueryHistory(KeywordEmbeddings()),
    }
    return at


def _captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


def test_app_runs_a_turn_and_lists_the_thread(
    settings, retriever, ok_runner, stores, chat_store
) -> None:
    graph = build_graph(
        llm=NamedFakeChatModel(
            messages=iter([AIMessage(content=ROUTE_DATA), SQL_REPLY, ANSWER_REPLY]),
            model="gemma4:e4b",
        ),
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
    assert any(e.label.endswith(" s · 1 attempt") for e in at.status)  # timing, live

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
    # The saved SQL is the validated one: formatted, with the LIMIT (row_limit + 1) added.
    [code] = at.code
    assert code.value.endswith("LIMIT 51")
    assert {e.label for e in [*at.expander, *at.status]} >= {"Thinking", "Thinking (answer)"}
    assert any(e.label.endswith(" s · 1 attempt") for e in at.status)  # saved timing


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


# --- thumbs up and the Query examples page ------------------------------------------------------


def _thumbs(at: AppTest) -> list:
    return [b for b in at.button if b.label == "Good answer"]


def test_thumbs_up_saves_the_example(
    settings, retriever, ok_runner, stores, chat_store, query_history
) -> None:
    graph = _fake_graph(settings, retriever, ok_runner, stores, SQL_REPLY, ANSWER_REPLY)
    at = _app(settings, chat_store, lambda _model: graph, query_history=query_history)
    at.run()
    at.chat_input[0].set_value("Who earns the most?").run()
    assert not at.exception
    assert query_history.examples() == []  # nothing is saved without a thumbs up

    [thumb] = _thumbs(at)
    thumb.click().run()
    assert not at.exception
    [example] = query_history.examples()
    [turn] = chat_store.load(at.session_state["thread_id"])
    assert (example.question, example.sql) == ("Who earns the most?", turn["sql"])
    assert not _thumbs(at)
    assert any("Saved as an example" in c for c in _captions(at))

    # Reopened from the sidebar, the reply still shows it is saved.
    at.sidebar.button[0].click().run()
    at.sidebar.button[1].click().run()
    assert not at.exception
    assert not _thumbs(at)
    assert any("Saved as an example" in c for c in _captions(at))


def test_hidden_example_cannot_be_saved_again(
    settings, retriever, ok_runner, stores, chat_store, query_history
) -> None:
    graph = _fake_graph(settings, retriever, ok_runner, stores, SQL_REPLY, ANSWER_REPLY)
    at = _app(settings, chat_store, lambda _model: graph, query_history=query_history)
    at.run()
    at.chat_input[0].set_value("Who earns the most?").run()
    [turn] = chat_store.load(at.session_state["thread_id"])
    query_history.add(turn["standalone"], turn["sql"], 1)
    query_history.hide(turn["standalone"], turn["sql"])

    at.run()
    assert not at.exception
    assert not _thumbs(at)
    assert any("Hidden from the query examples" in c for c in _captions(at))


def test_no_thumbs_up_when_the_sql_failed(
    settings, retriever, ok_runner, stores, chat_store
) -> None:
    # Rejected by validation every time, so no query runs.
    graph = _fake_graph(settings, retriever, ok_runner, stores, *["DROP TABLE employees"] * 3)
    at = _app(settings, chat_store, lambda _model: graph)
    at.run()
    at.chat_input[0].set_value("Who earns the most?").run()
    assert not at.exception
    assert at.error  # the rejected SQL is shown
    assert not _thumbs(at)


EXAMPLES_APP = """
import streamlit as st
from rag_sql.ui.web_examples import examples_page

examples_page(st.session_state["query_history"])
"""


def _examples_app(query_history) -> AppTest:
    at = AppTest.from_string(EXAMPLES_APP, default_timeout=30)
    at.session_state["query_history"] = query_history
    return at


def test_filter_examples_matches_every_word_in_question_or_sql(query_history) -> None:
    query_history.add("Average salary per city?", "SELECT city, avg(salary) FROM employees", 4)
    query_history.add("How many remote employees?", "SELECT count(*) FROM employees", 1)
    examples = query_history.examples()
    assert [e.id for e in filter_examples(examples, "  ")] == [2, 1]
    assert [e.id for e in filter_examples(examples, "CITY avg")] == [1]
    assert [e.id for e in filter_examples(examples, "count remote")] == [2]
    assert filter_examples(examples, "city remote") == []


def test_examples_page_lists_filters_and_hides(query_history) -> None:
    query_history.add("Average salary per city?", "SELECT city, avg(salary) FROM employees", 4)
    query_history.add("How many remote employees?", "SELECT count(*) FROM employees", 1)
    at = _examples_app(query_history)
    at.run()
    assert not at.exception
    assert [c.value for c in at.code] == [
        "SELECT count(*) FROM employees",
        "SELECT city, avg(salary) FROM employees",
    ]

    at.text_input[0].set_value("city").run()
    assert [c.value for c in at.code] == ["SELECT city, avg(salary) FROM employees"]
    assert "1 of 2 example(s)" in _captions(at)
    at.text_input[0].set_value("").run()

    # Hide asks to confirm; Cancel keeps the example.
    at.button(key="hide-1").click().run()
    assert at.button(key="confirm-hide-1")
    at.button(key="cancel-hide-1").click().run()
    assert len(query_history.examples()) == 2

    at.button(key="hide-1").click().run()
    at.button(key="confirm-hide-1").click().run()
    assert not at.exception
    assert [e.id for e in query_history.examples()] == [2]
    assert [c.value for c in at.code] == ["SELECT count(*) FROM employees"]


def test_examples_page_without_examples(query_history) -> None:
    at = _examples_app(query_history)
    at.run()
    assert not at.exception
    assert at.info[0].value.startswith("No examples yet.")

"""Streamlit chat app: page, sidebar (conversations, model picker) and session state. Each reply
is drawn by web_chat.py, which streams the agent's steps and the model's thinking live.

Run with `uv run streamlit run src/rag_sql/ui/web.py`, or in Docker (the compose service `web`).
"""

import logging
from collections.abc import Callable
from typing import Any

import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from rag_sql.agent.graph import build_graph, default_query_runner
from rag_sql.config import Settings, get_settings
from rag_sql.history.chat import ChatStore, ThreadSummary, get_chat_store, new_thread_id
from rag_sql.history.queries import get_query_history
from rag_sql.llm import ChatModelInfo, get_chat_model, list_chat_models
from rag_sql.retrieval import get_retriever
from rag_sql.tracing import run_config
from rag_sql.ui.steps import Step, escape_md, model_step, steps_from_turn
from rag_sql.ui.web_chat import render_step, run_turn

logger = logging.getLogger(__name__)

# Conversations listed in the sidebar, most recent first.
SIDEBAR_THREADS = 30

# Seconds the Ollama model list is cached, so newly pulled models show up without a restart.
MODEL_LIST_TTL_S = 60

# Left-aligns the sidebar's conversation buttons: Streamlit centers button labels (in wrappers
# inside the button) and has no option for it. Relies on Streamlit's markup (checked with 1.64):
# look at the sidebar again after a Streamlit upgrade.
SIDEBAR_CSS = """
.st-key-chat-history button,
.st-key-chat-history button > div,
.st-key-chat-history button > div > span { justify-content: flex-start; text-align: left; }
"""


def model_label(model: ChatModelInfo) -> str:
    """How a model is listed in the sidebar, e.g. "qwen3.5:9b · 9.7B"."""
    parts = [model.name]
    if model.parameter_size:
        parts.append(model.parameter_size)
    if not model.thinking:
        parts.append("no thinking")
    return " · ".join(parts)


def thread_title(thread: ThreadSummary) -> str:
    """The first question on one line. The sidebar cuts it to the button width with "…"."""
    return escape_md(" ".join(thread.first_question.split()))


# --- session and sidebar -------------------------------------------------------------------------


@st.cache_resource
def _default_store() -> ChatStore:
    return get_chat_store()


@st.cache_resource
def _shared_deps() -> dict[str, Any]:
    """Dependencies every model's graph shares, so a model doesn't open its own DB pools."""
    s = get_settings()
    return {
        "retriever": get_retriever(s),
        "query_runner": default_query_runner(s),
        "chat_store": _default_store(),
        "query_history": get_query_history(s),
    }


@st.cache_resource
def _default_graph(model: str, thinking: bool) -> CompiledStateGraph:
    s = get_settings()
    llm = get_chat_model(s, model=model, reasoning=s.ollama_reasoning and thinking)
    return build_graph(llm=llm, settings=s, **_shared_deps())


@st.cache_data(ttl=MODEL_LIST_TTL_S, show_spinner=False)
def _default_models() -> list[ChatModelInfo] | None:
    """Chat models on the Ollama server; None when it can't be reached (retried after the TTL)."""
    try:
        return list_chat_models()
    except Exception:
        logger.warning("Could not list the Ollama models", exc_info=True)
        return None


def _new_thread() -> None:
    st.session_state.thread_id = new_thread_id()
    st.session_state.messages = []
    st.query_params["thread"] = st.session_state.thread_id


def _open_thread(store: ChatStore, thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    st.query_params["thread"] = thread_id
    try:
        turns = store.load(thread_id)
    except Exception:
        logger.exception("Could not load thread %s", thread_id)
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": [Step("error", "Couldn't load this conversation.", label="Error")],
            }
        ]
        return
    st.session_state.messages = [
        message
        for turn in turns
        for message in (
            {"role": "user", "content": turn["question"]},
            {"role": "assistant", "content": steps_from_turn(turn)},
        )
    ]


def _model_picker(models: list[ChatModelInfo] | None, settings: Settings) -> ChatModelInfo:
    """The model chosen in the sidebar, kept per browser session. Defaults to OLLAMA_CHAT_MODEL."""
    configured = settings.ollama_chat_model
    if not models:
        st.warning(
            "Couldn't list the Ollama models."
            if models is None
            else "No chat models on the Ollama server.",
            icon=":material/warning:",
        )
        st.caption(f"Model `{configured}` (OLLAMA_CHAT_MODEL)")
        # Capabilities unknown: use the configured model and reasoning setting as they are.
        return ChatModelInfo(configured, None, thinking=True)

    by_name = {m.name: m for m in models}
    if st.session_state.get("model") not in by_name:
        st.session_state.model = configured if configured in by_name else models[0].name
    name = st.selectbox(
        "Model", list(by_name), key="model", format_func=lambda n: model_label(by_name[n])
    )
    if configured not in by_name:
        st.caption(f"`{configured}` (OLLAMA_CHAT_MODEL) isn't on the Ollama server.")
    return by_name[name]


def _sidebar(
    store: ChatStore, settings: Settings, models: list[ChatModelInfo] | None
) -> ChatModelInfo:
    with st.sidebar:
        st.button("New chat", icon=":material/add:", on_click=_new_thread, width="stretch")
        model = _model_picker(models, settings)
        st.subheader("Chat history")
        st.html(f"<style>{SIDEBAR_CSS}</style>")
        try:
            threads = store.threads(SIDEBAR_THREADS)
        except Exception:
            logger.exception("Could not list threads")
            st.warning("Couldn't load chat history.")
            threads = []
        if not threads:
            st.caption("No saved conversations yet.")
        with st.container(key="chat-history", gap=None):
            for thread in threads:
                current = thread.thread_id == st.session_state.thread_id
                st.button(
                    thread_title(thread),
                    key=f"thread-{thread.thread_id}",
                    help=(
                        f"{escape_md(thread.first_question)}  \n{thread.turns} question(s) · "
                        f"last {thread.last_at:%Y-%m-%d %H:%M}"
                    ),
                    type="primary" if current else "tertiary",
                    on_click=_open_thread,
                    args=(store, thread.thread_id),
                    width="stretch",
                    wrap=False,
                )
    return model


def main(
    graph_for: Callable[[ChatModelInfo], CompiledStateGraph] | None = None,
    store: ChatStore | None = None,
    settings: Settings | None = None,
    list_models: Callable[[], list[ChatModelInfo] | None] | None = None,
) -> None:
    """The app. Pass fakes to run it without Postgres or Ollama (tests).

    `graph_for` builds the agent graph for a chosen model; `list_models` returns the chat models
    to choose from, or None when the server can't be reached.
    """
    st.set_page_config(page_title="RAG-SQL chat", page_icon=":material/database:", layout="wide")
    store = store or _default_store()
    settings = settings or get_settings()
    graph_for = graph_for or (lambda m: _default_graph(m.name, m.thinking))
    list_models = list_models or _default_models

    if "thread_id" not in st.session_state:
        if thread_id := st.query_params.get("thread"):
            _open_thread(store, thread_id)
        else:
            _new_thread()

    model = _sidebar(store, settings, list_models())

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(escape_md(message["content"]))
            else:
                for step in message["content"]:
                    render_step(step)

    if question := st.chat_input("Ask a question about the database"):
        steps = [model_step(model.name)]
        st.session_state.messages += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": steps},
        ]
        with st.chat_message("user"):
            st.markdown(escape_md(question))
        thread_id = st.session_state.thread_id
        with st.chat_message("assistant"):
            render_step(steps[0])
            run_turn(
                graph_for(model),
                question,
                thread_id,
                steps,
                settings.max_sql_retries,
                config=run_config("web", session_id=thread_id, model=model.name, settings=settings),
                save_unfinished=lambda turn: store.append(thread_id, {**turn, "model": model.name}),
            )
        st.rerun()  # redraw the sidebar, which now lists this conversation first


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()

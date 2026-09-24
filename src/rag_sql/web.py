"""Streamlit chat UI: stream the agent graph and show each step, with the model's thinking live.

Run with `uv run streamlit run src/rag_sql/web.py`, or in Docker (the compose service `web`).
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import streamlit as st
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph

from rag_sql.agent.graph import build_graph, default_query_runner
from rag_sql.config import Settings, get_settings
from rag_sql.llm import ChatModelInfo, get_chat_model, list_chat_models
from rag_sql.memory import ChatStore, ThreadSummary, Turn, get_chat_store, new_thread_id
from rag_sql.query_history import PastQuery, get_query_history
from rag_sql.retrieval import get_retriever

logger = logging.getLogger(__name__)

# Nodes whose model calls are streamed token by token. condense_question is short and not shown.
STREAMED_NODES = ("generate_sql", "answer")

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

# Seconds between redraws of streamed text; redrawing on every token slows the page down.
REDRAW_INTERVAL = 0.1

StepKind = Literal["caption", "similar", "thinking", "sql", "error", "result", "answer"]


@dataclass
class Step:
    """One element of an assistant reply. Kept in session state so reruns can redraw the chat."""

    kind: StepKind
    text: str = ""
    label: str = ""
    data: Any = None  # "similar": list[PastQuery]; "result": pandas.DataFrame


# --- pure mapping from graph output to steps (no Streamlit calls) ------------------------------


def thinking_label(node: str, attempt: int) -> str:
    if node == "answer":
        return "Thinking (answer)"
    return "Thinking" if attempt <= 1 else f"Thinking (attempt {attempt})"


def steps_from_update(
    node: str, update: dict[str, Any], *, question: str, attempts: int, max_retries: int
) -> list[Step]:
    """The steps to show for one node's update from `graph.stream(stream_mode="updates")`."""
    if node == "load_history":
        earlier = len(update.get("history") or [])
        return [Step("caption", f"{earlier} earlier question(s) in context")] if earlier else []
    if node == "condense_question":
        standalone = update.get("standalone_question") or ""
        if standalone.strip() != question.strip():
            return [Step("caption", f"Interpreted as: *{escape_md(standalone)}*")]
        return []
    if node == "retrieve_context":
        docs = update.get("context") or []
        tables = [d.metadata.get("table") for d in docs if d.metadata.get("kind") == "table"]
        examples = sum(1 for d in docs if d.metadata.get("kind") == "example")
        return [
            Step(
                "caption", f"Context: tables {', '.join(tables) or 'none'} · {examples} example(s)"
            )
        ]
    if node == "find_similar_queries":
        queries: list[PastQuery] = update.get("similar_queries") or []
        if not queries:
            return []
        return [Step("similar", label=f"Similar past queries ({len(queries)})", data=queries)]
    if node == "generate_sql":
        attempt = update.get("attempts", 1)
        steps = []
        if reasoning := update.get("reasoning"):
            steps.append(Step("thinking", reasoning, label=thinking_label(node, attempt)))
        label = "SQL" if attempt <= 1 else f"SQL (attempt {attempt})"
        return [*steps, Step("sql", update.get("sql") or "", label=label)]
    if node in ("validate_sql", "execute_sql") and update.get("error"):
        step = "Validation" if node == "validate_sql" else "Execution"
        retry = (
            f"retry {attempts} of {max_retries}" if attempts <= max_retries else "no retries left"
        )
        return [Step("error", update["error"], label=f"{step} error · {retry}")]
    if node == "execute_sql" and (result := update.get("result")) is not None:
        note = f"{result.row_count} row(s)"
        if result.truncated:
            note += " (truncated at the row limit)"
        frame = pd.DataFrame.from_records(result.rows, columns=result.columns)
        return [Step("result", note, data=frame)]
    if node == "answer":
        steps = []
        if reasoning := update.get("answer_reasoning"):
            steps.append(Step("thinking", reasoning, label=thinking_label(node, attempts)))
        return [*steps, Step("answer", update.get("answer", ""))]
    if node == "save_query_example" and update.get("example_saved"):
        return [Step("caption", "Saved to query history as an example for similar questions.")]
    return []


def model_step(model: str) -> Step:
    return Step("caption", f"Model `{model}`")


def model_label(model: ChatModelInfo) -> str:
    """How a model is listed in the sidebar, e.g. "qwen3.5:9b · 9.7B"."""
    parts = [model.name]
    if model.parameter_size:
        parts.append(model.parameter_size)
    if not model.thinking:
        parts.append("no thinking")
    return " · ".join(parts)


def steps_from_turn(turn: Turn) -> list[Step]:
    """Steps for a saved turn. Result rows aren't saved, so only the row count is shown."""
    steps = [model_step(turn["model"])] if turn["model"] else []
    if turn["standalone"].strip() != turn["question"].strip():
        steps.append(Step("caption", f"Interpreted as: *{escape_md(turn['standalone'])}*"))
    if turn["sql_reasoning"]:
        steps.append(Step("thinking", turn["sql_reasoning"], label="Thinking"))
    if turn["sql"]:
        steps.append(Step("sql", turn["sql"], label="SQL"))
    if turn["row_count"] is not None:
        steps.append(Step("caption", f"{turn['row_count']} row(s) · result rows aren't saved"))
    if turn["error"]:
        steps.append(Step("error", turn["error"], label="Error"))
    if turn["answer_reasoning"]:
        steps.append(Step("thinking", turn["answer_reasoning"], label="Thinking (answer)"))
    steps.append(Step("answer", turn["answer"]))
    return steps


def escape_md(text: str) -> str:
    """Keep `$` literal: Streamlit Markdown renders `$...$` as LaTeX (e.g. two dollar amounts)."""
    return text.replace("$", r"\$")


def thread_title(thread: ThreadSummary) -> str:
    """The first question on one line. The sidebar cuts it to the button width with "…"."""
    return escape_md(" ".join(thread.first_question.split()))


# --- rendering ----------------------------------------------------------------------------------


def render_step(step: Step) -> None:
    match step.kind:
        case "caption":
            st.caption(step.text)
        case "similar":
            with st.expander(step.label, icon=":material/history:"):
                for q in step.data:
                    st.markdown(f"{escape_md(q.question)} · similarity {q.similarity:.2f}")
                    st.code(q.sql, language="sql")
        case "thinking":
            with st.expander(step.label, icon=":material/psychology:"):
                st.markdown(escape_md(step.text))
        case "sql":
            st.markdown(f"**{step.label}**")
            st.code(step.text, language="sql")
        case "error":
            st.error(f"**{step.label}**  \n{escape_md(step.text)}", icon=":material/error:")
        case "result":
            st.markdown("**SQL output**")
            st.caption(step.text)
            st.dataframe(step.data, hide_index=True)
        case "answer":
            st.markdown("**Answer**")
            st.markdown(escape_md(step.text))


class LiveCall:
    """Live widgets for one streamed model call: its thinking and, for the answer, its text."""

    def __init__(self, node: str, attempt: int) -> None:
        self.node = node
        self.label = thinking_label(node, attempt)
        self.reasoning = ""
        self.content = ""
        self._status: Any = None  # created on the first thinking token
        self._thinking: Any = None
        self._answer: Any = None  # created on the first answer token
        self._drawn_at = 0.0

    def add(self, message: BaseMessage) -> None:
        self.reasoning += message.additional_kwargs.get("reasoning_content") or ""
        if self.node == "answer":
            self.content += message.text
        if time.monotonic() - self._drawn_at >= REDRAW_INTERVAL:
            self._draw()

    def _draw(self) -> None:
        if self.reasoning:
            if self._status is None:
                self._status = st.status(f"{self.label}…", expanded=True)
                self._thinking = self._status.empty()
            self._thinking.markdown(escape_md(self.reasoning))
        if self.content:
            if self._answer is None:
                st.markdown("**Answer**")
                self._answer = st.empty()
            self._answer.markdown(escape_md(self.content))
        self._drawn_at = time.monotonic()

    def finish(self, steps: list[Step], *, failed: bool = False) -> tuple[list[Step], list[Step]]:
        """Close the live widgets once the node's update arrived.

        Returns (steps to keep, steps still to render): the thinking and answer were already
        shown live. When the steps have no thinking (the run failed), the streamed one is kept.
        """
        self._draw()
        if self._status is not None:
            self._status.update(
                label=self.label, state="error" if failed else "complete", expanded=False
            )
        keep, pending = [], []
        if self.reasoning and not any(s.kind == "thinking" for s in steps):
            keep.append(Step("thinking", self.reasoning.strip(), label=self.label))
        for step in steps:
            keep.append(step)
            if step.kind == "thinking" and self._status is not None:
                continue
            if step.kind == "answer" and self._answer is not None:
                self._answer.markdown(escape_md(step.text))
                continue
            pending.append(step)
        return keep, pending


def run_turn(
    graph: CompiledStateGraph, question: str, thread_id: str, steps: list[Step], max_retries: int
) -> None:
    """Run the agent on `question`, rendering as it goes and appending to `steps`.

    `steps` is filled in place, so a run cut short by a new question keeps what it showed.
    """
    attempts = 0
    live: LiveCall | None = None
    try:
        for mode, payload in graph.stream(
            {"question": question, "thread_id": thread_id}, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                message, metadata = payload
                node = metadata.get("langgraph_node")
                if node in STREAMED_NODES:
                    live = live or LiveCall(node, attempts + 1)
                    live.add(message)
                continue
            for node, update in payload.items():
                if not update:
                    continue
                attempts = update.get("attempts", attempts)
                new = steps_from_update(
                    node, update, question=question, attempts=attempts, max_retries=max_retries
                )
                pending = new
                if live is not None and live.node == node:
                    new, pending = live.finish(new)
                    live = None
                for step in pending:
                    render_step(step)
                steps.extend(new)
    except Exception as e:
        logger.exception("Agent run failed")
        if live is not None:
            kept, _ = live.finish([], failed=True)
            steps.extend(kept)
        error = Step("error", str(e) or type(e).__name__, label="The agent failed")
        render_step(error)
        steps.append(error)


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
        with st.chat_message("assistant"):
            render_step(steps[0])
            run_turn(
                graph_for(model),
                question,
                st.session_state.thread_id,
                steps,
                settings.max_sql_retries,
            )
        st.rerun()  # redraw the sidebar, which now lists this conversation first


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()

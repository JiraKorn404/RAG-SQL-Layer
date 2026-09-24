"""Notebook rendering: stream the agent graph and show each step as it completes."""

import html
from functools import lru_cache
from typing import Any

import pandas as pd
from IPython.display import HTML, Markdown, display
from langgraph.graph.state import CompiledStateGraph

from rag_sql.agent.graph import build_graph
from rag_sql.config import get_settings
from rag_sql.db.query import QueryResult
from rag_sql.memory import ChatStore, get_chat_store, new_thread_id
from rag_sql.query_history import PastQuery

# Rows rendered in the notebook table; the full result stays in the returned state.
DISPLAY_MAX_ROWS = 50

_MUTED = "color:#6b7280;"


@lru_cache(maxsize=1)
def _default_store() -> ChatStore:
    return get_chat_store()


@lru_cache(maxsize=1)
def _default_graph() -> CompiledStateGraph:
    return build_graph(chat_store=_default_store())


def _muted(text: str) -> None:
    display(HTML(f'<div style="{_MUTED}font-size:0.9em">{text}</div>'))


def _show_thread(thread_id: str, update: dict[str, Any]) -> None:
    earlier = len(update.get("history") or [])
    status = f"{earlier} earlier question(s) in context" if earlier else "new conversation"
    _muted(f"Thread <code>{html.escape(thread_id)}</code> · {status}")


def _show_condensed(question: str, update: dict[str, Any]) -> None:
    standalone = update.get("standalone_question") or ""
    if standalone.strip() != question.strip():
        _muted(f"Interpreted as: <i>{html.escape(standalone)}</i>")


def _show_context(update: dict[str, Any]) -> None:
    docs = update.get("context") or []
    tables = [d.metadata.get("table") for d in docs if d.metadata.get("kind") == "table"]
    examples = sum(1 for d in docs if d.metadata.get("kind") == "example")
    _muted(f"Context: tables {html.escape(', '.join(tables) or 'none')} · {examples} example(s)")


def _show_similar(update: dict[str, Any]) -> None:
    queries: list[PastQuery] = update.get("similar_queries") or []
    if not queries:
        return
    items = "".join(
        f'<li style="margin-bottom:6px">{html.escape(q.question)} '
        f"<span>(similarity {q.similarity:.2f})</span>"
        f'<pre style="white-space:pre-wrap;margin:2px 0">{html.escape(q.sql)}</pre></li>'
        for q in queries
    )
    display(
        HTML(
            f'<details style="{_MUTED}font-size:0.9em"><summary>Similar past queries '
            f"({len(queries)})</summary><ol>{items}</ol></details>"
        )
    )


def _show_generation(update: dict[str, Any]) -> None:
    attempt = update.get("attempts", 1)
    if reasoning := update.get("reasoning"):
        display(
            HTML(
                f'<details style="{_MUTED}"><summary><b>Thinking</b>'
                f"{f' (attempt {attempt})' if attempt > 1 else ''}</summary>"
                f'<pre style="white-space:pre-wrap;{_MUTED}">{html.escape(reasoning)}</pre>'
                "</details>"
            )
        )
    title = "SQL" if attempt == 1 else f"SQL (attempt {attempt})"
    display(Markdown(f"### {title}\n```sql\n{update.get('sql') or ''}\n```"))


def _show_error(node: str, error: str, attempts: int, max_retries: int) -> None:
    step = "Validation" if node == "validate_sql" else "Execution"
    retry = f"retry {attempts} of {max_retries}" if attempts <= max_retries else "no retries left"
    display(
        HTML(
            '<div style="color:#b91c1c;border-left:3px solid #b91c1c;padding:4px 8px;'
            f'margin:4px 0"><b>{step} error</b> · {retry}<br>'
            f"<code>{html.escape(error)}</code></div>"
        )
    )


def _show_result(result: QueryResult) -> None:
    df = pd.DataFrame.from_records(result.rows, columns=result.columns)
    note = f"{result.row_count} row(s)"
    if result.truncated:
        note += " (truncated at the row limit)"
    display(Markdown(f"### SQL output\n{note}"))
    with pd.option_context("display.max_rows", DISPLAY_MAX_ROWS, "display.min_rows", 20):
        display(df)


def run_and_display(
    question: str, graph: CompiledStateGraph | None = None, *, thread_id: str | None = None
) -> None:
    """Run the agent on `question` and render each step as it completes.

    The turn is saved to chat history under `thread_id`; without one, it starts a new thread.
    To ask follow-up questions, use `Chat`. Returns nothing so a notebook cell doesn't echo the
    state; for the state itself, use `build_graph().invoke({"question": ...})`.
    """
    graph = graph or _default_graph()
    thread_id = thread_id or new_thread_id()
    max_retries = get_settings().max_sql_retries

    display(Markdown(f"**Question:** {question}"))
    attempts = 1
    for chunk in graph.stream(
        {"question": question, "thread_id": thread_id}, stream_mode="updates"
    ):
        for node, update in chunk.items():
            if not update:
                continue
            attempts = update.get("attempts", attempts)
            if node == "load_history":
                _show_thread(thread_id, update)
            elif node == "condense_question":
                _show_condensed(question, update)
            elif node == "retrieve_context":
                _show_context(update)
            elif node == "find_similar_queries":
                _show_similar(update)
            elif node == "generate_sql":
                _show_generation(update)
            elif node in ("validate_sql", "execute_sql") and update.get("error"):
                _show_error(node, update["error"], attempts, max_retries)
            elif node == "execute_sql" and update.get("result") is not None:
                _show_result(update["result"])
            elif node == "answer":
                display(Markdown(f"### Answer\n{update.get('answer', '')}"))
            elif node == "save_query_example" and update.get("example_saved"):
                _muted("Saved to query history as an example for similar questions.")


class Chat:
    """A conversation: each question sees the earlier ones, so follow-ups work.

    chat = Chat()                # new conversation
    chat = Chat("<thread id>")   # continue a saved one (see show_threads())
    chat.ask("Which department has the highest average salary?")
    chat.ask("And the lowest?")
    """

    def __init__(
        self,
        thread_id: str | None = None,
        *,
        graph: CompiledStateGraph | None = None,
        store: ChatStore | None = None,
    ) -> None:
        self.thread_id = thread_id or new_thread_id()
        self._graph = graph
        self._store = store

    def __repr__(self) -> str:
        return f"Chat(thread_id={self.thread_id!r})"

    def ask(self, question: str) -> None:
        run_and_display(question, self._graph, thread_id=self.thread_id)

    def reset(self) -> None:
        """Start a new conversation. The previous one stays saved."""
        self.thread_id = new_thread_id()

    def show_history(self) -> None:
        turns = (self._store or _default_store()).load(self.thread_id)
        if not turns:
            _muted("No questions in this conversation yet.")
            return
        # The thinking columns are long; the web UI shows them.
        display(pd.DataFrame(turns).drop(columns=["sql_reasoning", "answer_reasoning"]))


def show_threads(limit: int = 20, store: ChatStore | None = None) -> None:
    """Saved conversations, most recent first. Pass a thread_id to `Chat` to continue one."""
    threads = (store or _default_store()).threads(limit)
    if not threads:
        _muted("No saved conversations yet.")
        return
    display(pd.DataFrame(threads))

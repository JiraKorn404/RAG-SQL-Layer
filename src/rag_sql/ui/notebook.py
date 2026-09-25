"""Notebook rendering: stream the agent graph and show each step as it completes.

What each node update shows is decided in steps.py (shared with the web UI); this module only
draws the steps.
"""

import html
from functools import lru_cache

import pandas as pd
from IPython.display import HTML, Markdown, display
from langgraph.graph.state import CompiledStateGraph

from rag_sql.agent.graph import build_graph
from rag_sql.config import get_settings
from rag_sql.history.chat import ChatStore, get_chat_store, new_thread_id
from rag_sql.tracing import run_config
from rag_sql.ui.steps import Step, escape_md, steps_from_update

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


def render_step(step: Step) -> None:
    match step.kind:
        case "caption":
            _muted(html.escape(step.text))
        case "interpreted":
            _muted(f"Interpreted as: <i>{html.escape(step.text)}</i>")
        case "model":
            _muted(f"Model <code>{html.escape(step.text)}</code>")
        case "similar":
            items = "".join(
                f'<li style="margin-bottom:6px">{html.escape(q.question)} '
                f"<span>(similarity {q.similarity:.2f})</span>"
                f'<pre style="white-space:pre-wrap;margin:2px 0">{html.escape(q.sql)}</pre></li>'
                for q in step.data
            )
            display(
                HTML(
                    f'<details style="{_MUTED}font-size:0.9em"><summary>{html.escape(step.label)}'
                    f"</summary><ol>{items}</ol></details>"
                )
            )
        case "thinking":
            display(
                HTML(
                    f'<details style="{_MUTED}"><summary><b>{html.escape(step.label)}</b></summary>'
                    f'<pre style="white-space:pre-wrap;{_MUTED}">{html.escape(step.text)}</pre>'
                    "</details>"
                )
            )
        case "sql":
            display(Markdown(f"### {step.label}\n```sql\n{step.text}\n```"))
        case "error":
            display(
                HTML(
                    '<div style="color:#b91c1c;border-left:3px solid #b91c1c;padding:4px 8px;'
                    f'margin:4px 0"><b>{html.escape(step.label)}</b><br>'
                    f"<code>{html.escape(step.text)}</code></div>"
                )
            )
        case "result":
            display(Markdown(f"### SQL output\n{step.text}"))
            with pd.option_context("display.max_rows", DISPLAY_MAX_ROWS, "display.min_rows", 20):
                display(step.data)
        case "answer":
            display(Markdown(f"### Answer\n{escape_md(step.text)}"))
        case "metrics":
            table = step.data.to_html(index=False, na_rep="", border=0)
            display(
                HTML(
                    f'<details style="{_MUTED}font-size:0.9em"><summary>'
                    f"{html.escape(step.text)}</summary>{table}</details>"
                )
            )


def run_and_display(
    question: str, graph: CompiledStateGraph | None = None, *, thread_id: str | None = None
) -> None:
    """Run the agent on `question` and render each step as it completes.

    The turn is saved to chat history under `thread_id`; without one, it starts a new thread.
    To ask follow-up questions, use `Chat`. Returns nothing so a notebook cell doesn't echo the
    state; for the state itself, use `build_graph().invoke({"question": ...})`.
    """
    settings = get_settings()
    # The default graph uses the configured model; another graph's model is on its trace anyway.
    model = settings.ollama_chat_model if graph is None else None
    graph = graph or _default_graph()
    thread_id = thread_id or new_thread_id()
    max_retries = settings.max_sql_retries

    display(Markdown(f"**Question:** {escape_md(question)}"))
    _muted(f"Thread <code>{html.escape(thread_id)}</code>")
    attempts = 0
    for chunk in graph.stream(
        {"question": question, "thread_id": thread_id},
        config=run_config("notebook", session_id=thread_id, model=model, settings=settings),
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if not update:
                continue
            attempts = update.get("attempts", attempts)
            for step in steps_from_update(
                node, update, question=question, attempts=attempts, max_retries=max_retries
            ):
                render_step(step)


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
        # The thinking columns are long, and the metrics nested; the web UI shows them.
        display(pd.DataFrame(turns).drop(columns=["sql_reasoning", "answer_reasoning", "metrics"]))


def show_threads(limit: int = 20, store: ChatStore | None = None) -> None:
    """Saved conversations, most recent first. Pass a thread_id to `Chat` to continue one."""
    threads = (store or _default_store()).threads(limit)
    if not threads:
        _muted("No saved conversations yet.")
        return
    display(pd.DataFrame(threads))

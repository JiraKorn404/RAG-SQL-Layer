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

# Rows rendered in the notebook table; the full result stays in the returned state.
DISPLAY_MAX_ROWS = 50

_MUTED = "color:#6b7280;"


@lru_cache(maxsize=1)
def _default_graph() -> CompiledStateGraph:
    return build_graph()


def _show_context(update: dict[str, Any]) -> None:
    docs = update.get("context") or []
    tables = [d.metadata.get("table") for d in docs if d.metadata.get("kind") == "table"]
    examples = sum(1 for d in docs if d.metadata.get("kind") == "example")
    display(
        HTML(
            f'<div style="{_MUTED}font-size:0.9em">Context: tables '
            f"{html.escape(', '.join(tables) or 'none')} · {examples} example(s)</div>"
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


def run_and_display(question: str, graph: CompiledStateGraph | None = None) -> None:
    """Run the agent on `question` and render each step as it completes.

    Returns nothing so a notebook cell doesn't echo the state; for the state itself, use
    `build_graph().invoke({"question": ...})`.
    """
    graph = graph or _default_graph()
    max_retries = get_settings().max_sql_retries

    display(Markdown(f"**Question:** {question}"))
    attempts = 1
    for chunk in graph.stream({"question": question}, stream_mode="updates"):
        for node, update in chunk.items():
            if not update:
                continue
            attempts = update.get("attempts", attempts)
            if node == "retrieve_context":
                _show_context(update)
            elif node == "generate_sql":
                _show_generation(update)
            elif node in ("validate_sql", "execute_sql") and update.get("error"):
                _show_error(node, update["error"], attempts, max_retries)
            elif node == "execute_sql" and update.get("result") is not None:
                _show_result(update["result"])
            elif node == "answer":
                display(Markdown(f"### Answer\n{update.get('answer', '')}"))

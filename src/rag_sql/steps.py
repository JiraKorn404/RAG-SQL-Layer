"""What to show for each step of an agent run, whatever the UI.

Graph updates (`graph.stream(stream_mode="updates")`) and saved turns become `Step`s here. The
front ends, display.py (Jupyter) and web.py (Streamlit), only decide how each kind is drawn.
"""

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from rag_sql.memory import Turn
from rag_sql.metrics import NodeMetric, TurnMetrics, format_summary
from rag_sql.query_history import PastQuery

StepKind = Literal[
    "caption",  # text: a muted status line
    "interpreted",  # text: the standalone rewrite of a follow-up question
    "model",  # text: name of the chat model that answered
    "similar",  # label; data: list[PastQuery]
    "thinking",  # label; text: the model's reasoning (Markdown)
    "sql",  # label; text: the SQL
    "error",  # label; text: the error message
    "result",  # text: row count note; data: pandas.DataFrame
    "answer",  # text: the answer (Markdown)
    "metrics",  # text: one-line summary (time, attempts, tokens); data: pandas.DataFrame per node
]


@dataclass
class Step:
    """One element of an assistant reply. `text` is plain text unless its kind says Markdown."""

    kind: StepKind
    text: str = ""
    label: str = ""
    data: Any = None


def escape_md(text: str) -> str:
    """Keep `$` literal: Streamlit and Jupyter Markdown render `$...$` as LaTeX."""
    return text.replace("$", r"\$")


def thinking_label(node: str, attempt: int) -> str:
    if node == "answer":
        return "Thinking (answer)"
    return "Thinking" if attempt <= 1 else f"Thinking (attempt {attempt})"


def model_step(model: str) -> Step:
    return Step("model", model)


def node_table(nodes: list[NodeMetric]) -> pd.DataFrame:
    """One row per node run: its time and model usage."""
    frame = pd.DataFrame.from_records(
        [
            {
                "node": n["node"],
                "attempt": n["attempt"],
                "ms": n["ms"],
                "model ms": n["llm_ms"],
                "load ms": n["load_ms"],
                "tokens in": n["input_tokens"],
                "tokens out": n["output_tokens"],
            }
            for n in nodes
        ],
        columns=["node", "attempt", "ms", "model ms", "load ms", "tokens in", "tokens out"],
    )
    # Nullable integers: empty cells instead of NaN floats for nodes without a model call.
    return frame.astype({c: "Int64" for c in frame.columns if c != "node"})


def metrics_step(metrics: TurnMetrics) -> Step:
    return Step("metrics", format_summary(metrics), data=node_table(metrics["nodes"]))


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
            return [Step("interpreted", standalone)]
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
        if update.get("db_unavailable"):
            retry = "database unavailable, not retried"
        elif attempts <= max_retries:
            retry = f"retry {attempts} of {max_retries}"
        else:
            retry = "no retries left"
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
    if node == "save_turn" and (metrics := update.get("turn_metrics")):
        return [metrics_step(metrics)]
    if node == "save_query_example" and update.get("example_saved"):
        return [Step("caption", "Saved to query history as an example for similar questions.")]
    return []


def steps_from_turn(turn: Turn) -> list[Step]:
    """Steps for a saved turn. Result rows aren't saved, so only the row count is shown."""
    steps = [model_step(turn["model"])] if turn["model"] else []
    if turn["standalone"].strip() != turn["question"].strip():
        steps.append(Step("interpreted", turn["standalone"]))
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
    if turn["answer"]:
        steps.append(Step("answer", turn["answer"]))
    if metrics := turn.get("metrics"):
        steps.append(metrics_step(metrics))
    return steps

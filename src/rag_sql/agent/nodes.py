"""Graph nodes. Each takes the state (plus injected dependencies) and returns a partial update."""

import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from sqlalchemy.exc import DBAPIError

from rag_sql.agent.prompts import ANSWER_PROMPT, SQL_GENERATION_PROMPT, SQL_RETRY_FEEDBACK
from rag_sql.agent.state import AgentState
from rag_sql.db.query import QueryResult, SQLValidationError
from rag_sql.db.query import validate_sql as check_sql

logger = logging.getLogger(__name__)

# Rows of the SQL result shown to the model when writing the answer.
ANSWER_MAX_ROWS = 50

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_BLOCK_RE = re.compile(r"```\w*\s*(.*?)```", re.DOTALL)


# --- helpers ---------------------------------------------------------------------------------


def split_reasoning(message: BaseMessage) -> tuple[str | None, str]:
    """Return (reasoning, content) from a model reply.

    Reasoning comes from `additional_kwargs["reasoning_content"]` (ChatOllama with
    reasoning=True), or from inline <think>...</think> tags some models emit.
    """
    content = message.text
    reasoning = message.additional_kwargs.get("reasoning_content") or None
    inline = _THINK_RE.findall(content)
    if inline:
        content = _THINK_RE.sub("", content)
        reasoning = reasoning or "\n".join(t.strip() for t in inline)
    return (reasoning.strip() if reasoning else None), content.strip()


def extract_sql(content: str) -> str:
    """Pull the SQL out of a model reply: last ```sql block, else last code block, else raw."""
    for pattern in (_SQL_BLOCK_RE, _ANY_BLOCK_RE):
        blocks = pattern.findall(content)
        if blocks:
            return blocks[-1].strip()
    return content.strip()


def format_schema(context: list[Document]) -> str:
    tables = [d.page_content for d in context if d.metadata.get("kind") == "table"]
    return "\n\n".join(tables) or "(no schema found)"


def format_examples(context: list[Document]) -> str:
    examples = [
        f"Question: {d.page_content}\n```sql\n{d.metadata['sql']}\n```"
        for d in context
        if d.metadata.get("kind") == "example"
    ]
    return "\n\n".join(examples) or "(none)"


def format_rows(result: QueryResult, max_rows: int = ANSWER_MAX_ROWS) -> str:
    """Markdown table of the first `max_rows` rows."""
    if not result.columns:
        return "(no columns)"
    if not result.rows:
        return "(no rows)"

    def cell(value: Any) -> str:
        return "NULL" if value is None else str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(result.columns) + " |",
        "|" + "---|" * len(result.columns),
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in result.rows[:max_rows]]
    return "\n".join(lines)


# --- nodes -----------------------------------------------------------------------------------


def retrieve_context(state: AgentState, *, retriever: Runnable[str, list[Document]]) -> dict:
    docs = retriever.invoke(state["question"])
    logger.debug("Retrieved %d docs", len(docs))
    return {"context": docs}


def generate_sql(state: AgentState, *, llm: BaseChatModel, row_limit: int) -> dict:
    feedback = ""
    if state.get("error") and state.get("sql") is not None:
        feedback = SQL_RETRY_FEEDBACK.format(sql=state["sql"], error=state["error"])

    context = state.get("context", [])
    message = (SQL_GENERATION_PROMPT | llm).invoke(
        {
            "schema": format_schema(context),
            "examples": format_examples(context),
            "feedback": feedback,
            "question": state["question"],
            "row_limit": row_limit,
        }
    )
    reasoning, content = split_reasoning(message)
    return {
        "reasoning": reasoning,
        "sql": extract_sql(content),
        "error": None,
        "result": None,
        "attempts": state.get("attempts", 0) + 1,
    }


def validate_sql(state: AgentState, *, row_limit: int) -> dict:
    try:
        return {"sql": check_sql(state.get("sql") or "", row_limit), "error": None}
    except SQLValidationError as e:
        return {"error": str(e)}


def execute_sql(state: AgentState, *, run_query: Callable[[str], QueryResult]) -> dict:
    try:
        return {"result": run_query(state["sql"]), "error": None}
    except DBAPIError as e:
        # The driver's message (first line) is what the model needs to fix the query.
        message = str(e.orig).strip().splitlines()[0] if e.orig else str(e)
        return {"result": None, "error": message}


def answer(state: AgentState, *, llm: BaseChatModel) -> dict:
    result = state.get("result")
    if result is None:
        # Retries exhausted: report the failure without another model call.
        return {
            "answer": (
                f"I couldn't produce a working SQL query after {state.get('attempts', 0)} "
                f"attempt(s). Last error: {state.get('error') or 'unknown'}"
            )
        }

    shown = min(result.row_count, ANSWER_MAX_ROWS)
    row_summary = f"{result.row_count} row(s)"
    if shown < result.row_count:
        row_summary += f", first {shown} shown"
    if result.truncated:
        row_summary += ", truncated at the row limit"

    message = (ANSWER_PROMPT | llm).invoke(
        {
            "question": state["question"],
            "sql": state["sql"],
            "row_summary": row_summary,
            "rows": format_rows(result),
        }
    )
    _, content = split_reasoning(message)
    return {"answer": content}

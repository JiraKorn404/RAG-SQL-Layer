"""Prompt inputs from the agent state, and parsing of the model's replies (thinking, SQL,
NO_SQL)."""

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from rag_sql.agent.prompts import NO_SQL
from rag_sql.db.query import QueryResult
from rag_sql.history.chat import Turn
from rag_sql.history.queries import PastQuery

# Rows of the SQL result shown to the model when writing the answer.
ANSWER_MAX_ROWS = 50

# Characters of each earlier answer shown to the model.
HISTORY_ANSWER_CHARS = 500

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_BLOCK_RE = re.compile(r"```\w*\s*(.*?)```", re.DOTALL)


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


def is_no_sql(content: str) -> bool:
    """Whether the SQL model replied NO_SQL (not a question about the data), allowing for a code
    block, quotes or a trailing period around it."""
    return extract_sql(content).strip(" \t\r\n`'\".").upper() == NO_SQL


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


def format_table_names(context: list[Document]) -> str:
    tables = [d.metadata.get("table") for d in context if d.metadata.get("kind") == "table"]
    return ", ".join(t for t in tables if t) or "(none found)"


def format_similar_queries(queries: list[PastQuery]) -> str:
    return "\n\n".join(f"Question: {q.question}\n```sql\n{q.sql}\n```" for q in queries)


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


def format_history(history: list[Turn], *, sql: bool = False, answers: bool = True) -> str:
    """Earlier turns, oldest first: each question, plus its SQL and/or answer."""
    blocks = []
    for turn in history:
        lines = [f"Question: {turn['standalone'] or turn['question']}"]
        if sql:
            if turn["sql"]:
                lines.append(f"```sql\n{turn['sql']}\n```")
            elif turn["error"]:
                lines.append(f"(no working SQL; error: {turn['error']})")
            else:
                lines.append("(no SQL: not a question about the data)")
        if answers:
            text = turn["answer"]
            if len(text) > HISTORY_ANSWER_CHARS:
                text = text[:HISTORY_ANSWER_CHARS].rstrip() + " …"
            lines.append(f"Answer: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

"""Graph nodes. Each takes the state (plus injected dependencies) and returns a partial update.

Nodes that call the model also return `llm_usage` (metrics.usage_of). It never reaches the state:
the timing wrapper in graph.py moves it into the node's metric. Prompt inputs are built, and model
replies parsed, in formatting.py.
"""

import logging
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from sqlalchemy.exc import DBAPIError

from rag_sql.agent.formatting import (
    ANSWER_MAX_ROWS,
    extract_sql,
    format_examples,
    format_history,
    format_rows,
    format_schema,
    format_similar_queries,
    format_table_names,
    is_no_sql,
    split_reasoning,
)
from rag_sql.agent.prompts import (
    ANSWER_HISTORY,
    ANSWER_PROMPT,
    CHAT_REPLY_PROMPT,
    CONDENSE_PROMPT,
    SIMILAR_QUERIES,
    SQL_GENERATION_PROMPT,
    SQL_HISTORY,
    SQL_RETRY_FEEDBACK,
)
from rag_sql.agent.state import AgentState
from rag_sql.db.query import QueryResult, SQLValidationError, is_database_unavailable
from rag_sql.db.query import validate_sql as check_sql
from rag_sql.history.chat import ChatStore, Turn
from rag_sql.history.queries import QueryHistory, normalize_question
from rag_sql.metrics import summarize, usage_of

logger = logging.getLogger(__name__)


def current_question(state: AgentState) -> str:
    """The question the SQL must answer: the standalone rewrite when there is one."""
    return state.get("standalone_question") or state["question"]


# --- nodes -----------------------------------------------------------------------------------


def load_history(state: AgentState, *, store: ChatStore, max_turns: int) -> dict:
    """Load the thread's last `max_turns` turns. Without a thread_id, keep any history passed in."""
    if max_turns <= 0:
        return {"history": []}
    thread_id = state.get("thread_id")
    if thread_id:
        return {"history": store.load(thread_id, max_turns)}
    return {"history": state.get("history", [])[-max_turns:]}


def condense_question(state: AgentState, *, llm: BaseChatModel) -> dict:
    """Rewrite a follow-up into a standalone question. No model call on the first turn."""
    question = state["question"]
    history = state.get("history") or []
    if not history:
        return {"standalone_question": question}

    message = (CONDENSE_PROMPT | llm).invoke(
        {"history": format_history(history), "question": question}
    )
    _, content = split_reasoning(message)
    return {"standalone_question": content or question, "llm_usage": usage_of(message)}


def retrieve_context(state: AgentState, *, retriever: Runnable[str, list[Document]]) -> dict:
    docs = retriever.invoke(current_question(state))
    logger.debug("Retrieved %d docs", len(docs))
    return {"context": docs}


def find_similar_queries(
    state: AgentState, *, query_history: QueryHistory, k: int, min_similarity: float
) -> dict:
    """Similar past successful queries, minus ones already given as curated examples.

    A failed search is logged and skipped: the agent still works without these examples.
    """
    try:
        found = query_history.search(current_question(state), k, min_similarity)
    except Exception:
        logger.warning("Query history search failed; continuing without it", exc_info=True)
        return {"similar_queries": []}

    curated = {
        normalize_question(d.page_content)
        for d in state.get("context", [])
        if d.metadata.get("kind") == "example"
    }
    return {"similar_queries": [q for q in found if normalize_question(q.question) not in curated]}


def generate_sql(state: AgentState, *, llm: BaseChatModel, row_limit: int) -> dict:
    """Write SQL for the question, or set `no_sql` (and no SQL) when the model replied NO_SQL."""
    feedback = ""
    if state.get("error") and state.get("sql") is not None:
        feedback = SQL_RETRY_FEEDBACK.format(sql=state["sql"], error=state["error"])
    history = state.get("history") or []
    similar = state.get("similar_queries") or []

    context = state.get("context", [])
    message = (SQL_GENERATION_PROMPT | llm).invoke(
        {
            "schema": format_schema(context),
            "examples": format_examples(context),
            "similar": (
                SIMILAR_QUERIES.format(queries=format_similar_queries(similar)) if similar else ""
            ),
            "history": (
                SQL_HISTORY.format(turns=format_history(history, sql=True, answers=False))
                if history
                else ""
            ),
            "feedback": feedback,
            "question": current_question(state),
            "row_limit": row_limit,
        }
    )
    reasoning, content = split_reasoning(message)
    no_sql = is_no_sql(content)
    return {
        "reasoning": reasoning,
        "sql": None if no_sql else extract_sql(content),
        "no_sql": no_sql,
        "error": None,
        "result": None,
        "attempts": state.get("attempts", 0) + 1,
        "llm_usage": usage_of(message),
    }


def validate_sql(state: AgentState, *, row_limit: int) -> dict:
    try:
        return {"sql": check_sql(state.get("sql") or "", row_limit), "error": None}
    except SQLValidationError as e:
        return {"error": str(e)}


def execute_sql(state: AgentState, *, run_query: Callable[[str], QueryResult]) -> dict:
    """Run the SQL. An error is kept for a retry, unless the database itself is unavailable."""
    try:
        return {"result": run_query(state["sql"]), "error": None, "db_unavailable": False}
    except DBAPIError as e:
        # The driver's message (first line) is what the model needs to fix the query.
        message = str(e.orig).strip().splitlines()[0] if e.orig else str(e)
        return {"result": None, "error": message, "db_unavailable": is_database_unavailable(e)}


def answer(state: AgentState, *, llm: BaseChatModel) -> dict:
    result = state.get("result")
    if result is None and state.get("db_unavailable"):
        return {
            "answer": (
                "I couldn't run the query because the database is unavailable. "
                f"Error: {state.get('error') or 'unknown'}"
            ),
            "answer_reasoning": None,
        }
    if state.get("no_sql"):
        return chat_reply(state, llm=llm)
    if result is None:
        # Retries exhausted: report the failure without another model call.
        return {
            "answer": (
                f"I couldn't produce a working SQL query after {state.get('attempts', 0)} "
                f"attempt(s). Last error: {state.get('error') or 'unknown'}"
            ),
            "answer_reasoning": None,
        }

    shown = min(result.row_count, ANSWER_MAX_ROWS)
    row_summary = f"{result.row_count} row(s)"
    if shown < result.row_count:
        row_summary += f", first {shown} shown"
    if result.truncated:
        row_summary += ", truncated at the row limit"

    history = state.get("history") or []
    message = (ANSWER_PROMPT | llm).invoke(
        {
            "history": ANSWER_HISTORY.format(turns=format_history(history)) if history else "",
            "question": current_question(state),
            "sql": state["sql"],
            "row_summary": row_summary,
            "rows": format_rows(result),
        }
    )
    reasoning, content = split_reasoning(message)
    return {"answer": content, "answer_reasoning": reasoning, "llm_usage": usage_of(message)}


def chat_reply(state: AgentState, *, llm: BaseChatModel) -> dict:
    """The answer to a message that isn't about the data: no SQL ran, so no result to report.

    Part of the answer node (its update and streamed tokens are the answer's), not a node itself.
    """
    history = state.get("history") or []
    message = (CHAT_REPLY_PROMPT | llm).invoke(
        {
            "tables": format_table_names(state.get("context", [])),
            "history": ANSWER_HISTORY.format(turns=format_history(history)) if history else "",
            "question": state["question"],
        }
    )
    reasoning, content = split_reasoning(message)
    return {"answer": content, "answer_reasoning": reasoning, "llm_usage": usage_of(message)}


def turn_from_state(state: AgentState, model: str | None = None) -> Turn:
    """The chat turn for a run's state. The SQL and row count only when the SQL ran.

    `model` is the name of the chat model that answered. Also used for runs that didn't finish.
    The metrics cover the nodes that ran so far (save_turn itself and later nodes are left out).
    """
    result = state.get("result")
    return {
        "question": state["question"],
        "standalone": current_question(state),
        "sql": state.get("sql") if result is not None else None,
        "row_count": result.row_count if result is not None else None,
        "answer": state.get("answer", ""),
        "error": None if result is not None else state.get("error"),
        "sql_reasoning": state.get("reasoning"),
        "answer_reasoning": state.get("answer_reasoning"),
        "model": model,
        "metrics": summarize(state.get("metrics") or []),
    }


def save_turn(state: AgentState, *, store: ChatStore, model: str | None = None) -> dict:
    """Append this turn to the history, and save it when the run has a thread_id.

    `model` is the name of the chat model that answered, recorded with the turn.

    A failed save is logged, not raised: the answer has already been produced.
    """
    turn = turn_from_state(state, model)
    turn_id = None
    if thread_id := state.get("thread_id"):
        try:
            turn_id = store.append(thread_id, turn)
        except Exception:
            logger.exception("Could not save the turn to chat history (thread %s)", thread_id)
    return {
        "history": [*state.get("history", []), turn],
        "turn_id": turn_id,
        "turn_metrics": turn["metrics"],
    }

"""Graph state shared by nodes, graph routing and the display layer."""

from typing import TypedDict

from langchain_core.documents import Document

from rag_sql.db.query import QueryResult
from rag_sql.memory import Turn
from rag_sql.query_history import PastQuery


class AgentState(TypedDict, total=False):
    thread_id: str | None  # conversation id; nothing is saved without one
    history: list[Turn]  # earlier turns of the conversation, oldest first (+ this one at the end)
    question: str
    standalone_question: str  # question rewritten to stand on its own; == question on turn 1
    context: list[Document]  # retrieved table docs + few-shot example docs
    similar_queries: list[PastQuery]  # similar past successful queries, best first
    reasoning: str | None  # model thinking behind the latest SQL, if the model produced any
    sql: str | None  # latest SQL (normalized by validate_sql once it passes)
    error: str | None  # validation or execution error of the latest SQL
    attempts: int  # number of SQL generations so far
    result: QueryResult | None
    answer: str
    turn_id: int | None  # id of the saved chat turn
    example_saved: bool  # whether this turn was added to the query history

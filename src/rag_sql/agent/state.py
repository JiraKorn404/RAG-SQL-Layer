"""Graph state shared by nodes, graph routing and the display layer."""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document

from rag_sql.db.query import QueryResult
from rag_sql.history.chat import Turn
from rag_sql.history.queries import PastQuery
from rag_sql.metrics import NodeMetric, TurnMetrics


class AgentState(TypedDict, total=False):
    thread_id: str | None  # conversation id; nothing is saved without one
    history: list[Turn]  # earlier turns of the conversation, oldest first (+ this one at the end)
    question: str
    standalone_question: str  # question rewritten to stand on its own; == question on turn 1
    context: list[Document]  # retrieved table docs + few-shot example docs
    similar_queries: list[PastQuery]  # similar past successful queries, best first
    reasoning: str | None  # model thinking behind the latest SQL, if the model produced any
    sql: str | None  # latest SQL (normalized by validate_sql once it passes)
    no_sql: bool  # the model replied NO_SQL: not a question about the data, so nothing runs
    error: str | None  # validation or execution error of the latest SQL
    db_unavailable: bool  # the database couldn't be used (not a SQL problem), so no retry
    attempts: int  # number of SQL generations so far
    result: QueryResult | None
    answer: str
    answer_reasoning: str | None  # model thinking behind the answer, if the model produced any
    turn_id: int | None  # id of the saved chat turn
    example_saved: bool  # whether this turn was added to the query history
    # One per node run, added by the timing wrapper in graph.py; the reducer appends, so each
    # SQL attempt keeps its own entries.
    metrics: Annotated[list[NodeMetric], operator.add]
    turn_metrics: TurnMetrics | None  # summary of `metrics`, saved with the turn by save_turn

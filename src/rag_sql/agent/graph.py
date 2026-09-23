"""Graph wiring and routing."""

from collections.abc import Callable
from functools import partial
from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_sql.agent import nodes
from rag_sql.agent.state import AgentState
from rag_sql.config import Settings, get_settings
from rag_sql.db.connection import get_engine
from rag_sql.db.query import QueryResult, run_query
from rag_sql.llm import get_chat_model
from rag_sql.retrieval import get_retriever


def _should_retry(state: AgentState, max_retries: int) -> bool:
    # attempts counts generations; the first one is not a retry.
    return state.get("attempts", 0) <= max_retries


def route_after_validate(
    state: AgentState, max_retries: int
) -> Literal["execute_sql", "generate_sql", "answer"]:
    if not state.get("error"):
        return "execute_sql"
    return "generate_sql" if _should_retry(state, max_retries) else "answer"


def route_after_execute(state: AgentState, max_retries: int) -> Literal["answer", "generate_sql"]:
    if not state.get("error"):
        return "answer"
    return "generate_sql" if _should_retry(state, max_retries) else "answer"


def build_graph(
    *,
    llm: BaseChatModel | None = None,
    retriever: Runnable[str, list[Document]] | None = None,
    query_runner: Callable[[str], QueryResult] | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Compile the agent graph. Pass fakes for any dependency to run without network or DB."""
    s = settings or get_settings()

    llm = llm or get_chat_model(s)
    retriever = retriever or get_retriever(s)
    if query_runner is None:
        query_runner = partial(
            run_query,
            get_engine(readonly=True, settings=s),
            row_limit=s.sql_row_limit,
            timeout_ms=s.sql_timeout_ms,
        )

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_context", partial(nodes.retrieve_context, retriever=retriever))
    graph.add_node("generate_sql", partial(nodes.generate_sql, llm=llm, row_limit=s.sql_row_limit))
    graph.add_node("validate_sql", partial(nodes.validate_sql, row_limit=s.sql_row_limit))
    graph.add_node("execute_sql", partial(nodes.execute_sql, run_query=query_runner))
    graph.add_node("answer", partial(nodes.answer, llm=llm))

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql",
        partial(route_after_validate, max_retries=s.max_sql_retries),
        ["execute_sql", "generate_sql", "answer"],
    )
    graph.add_conditional_edges(
        "execute_sql",
        partial(route_after_execute, max_retries=s.max_sql_retries),
        ["answer", "generate_sql"],
    )
    graph.add_edge("answer", END)
    return graph.compile()

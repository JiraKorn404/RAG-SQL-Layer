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
from rag_sql.llm import chat_model_name, get_chat_model
from rag_sql.memory import ChatStore, get_chat_store
from rag_sql.query_history import QueryHistory, get_query_history
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


def default_query_runner(settings: Settings | None = None) -> Callable[[str], QueryResult]:
    """Runs agent SQL as the read-only role, with the configured row limit and timeout."""
    s = settings or get_settings()
    return partial(
        run_query,
        get_engine("reader", settings=s),
        row_limit=s.sql_row_limit,
        timeout_ms=s.sql_timeout_ms,
    )


def build_graph(
    *,
    llm: BaseChatModel | None = None,
    retriever: Runnable[str, list[Document]] | None = None,
    query_runner: Callable[[str], QueryResult] | None = None,
    chat_store: ChatStore | None = None,
    query_history: QueryHistory | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Compile the agent graph. Pass fakes for any dependency to run without network or DB.

    Chat history and query history are saved only for runs whose input has a `thread_id`.
    """
    s = settings or get_settings()

    llm = llm or get_chat_model(s)
    retriever = retriever or get_retriever(s)
    if chat_store is None:
        chat_store = get_chat_store(s)
    if query_history is None:
        query_history = get_query_history(s)
    if query_runner is None:
        query_runner = default_query_runner(s)

    graph = StateGraph(AgentState)
    graph.add_node(
        "load_history",
        partial(nodes.load_history, store=chat_store, max_turns=s.chat_history_turns),
    )
    graph.add_node("condense_question", partial(nodes.condense_question, llm=llm))
    graph.add_node("retrieve_context", partial(nodes.retrieve_context, retriever=retriever))
    graph.add_node(
        "find_similar_queries",
        partial(
            nodes.find_similar_queries,
            query_history=query_history,
            k=s.query_history_k,
            min_similarity=s.query_history_min_similarity,
        ),
    )
    graph.add_node("generate_sql", partial(nodes.generate_sql, llm=llm, row_limit=s.sql_row_limit))
    graph.add_node("validate_sql", partial(nodes.validate_sql, row_limit=s.sql_row_limit))
    graph.add_node("execute_sql", partial(nodes.execute_sql, run_query=query_runner))
    graph.add_node("answer", partial(nodes.answer, llm=llm))
    graph.add_node(
        "save_turn", partial(nodes.save_turn, store=chat_store, model=chat_model_name(llm))
    )
    graph.add_node(
        "save_query_example", partial(nodes.save_query_example, query_history=query_history)
    )

    graph.add_edge(START, "load_history")
    graph.add_edge("load_history", "condense_question")
    graph.add_edge("condense_question", "retrieve_context")
    graph.add_edge("retrieve_context", "find_similar_queries")
    graph.add_edge("find_similar_queries", "generate_sql")
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
    graph.add_edge("answer", "save_turn")
    graph.add_edge("save_turn", "save_query_example")
    graph.add_edge("save_query_example", END)
    return graph.compile()

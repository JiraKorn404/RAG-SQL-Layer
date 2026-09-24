from collections.abc import Callable

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_sql.config import Settings
from rag_sql.db.query import QueryResult
from rag_sql.memory import InMemoryChatStore, Turn
from rag_sql.query_history import InMemoryQueryHistory

TABLE_DOC = Document(
    page_content="Table employees\n  emp_name TEXT\n  salary NUMERIC(12, 2)",
    metadata={"kind": "table", "table": "employees"},
)
EXAMPLE_DOC = Document(
    page_content="Who are the 5 highest paid employees?",
    metadata={
        "kind": "example",
        "sql": "SELECT emp_name FROM employees ORDER BY salary DESC LIMIT 5;",
    },
)


def fake_llm(*replies: str | AIMessage) -> GenericFakeChatModel:
    messages = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return GenericFakeChatModel(messages=iter(messages))


@pytest.fixture
def settings() -> Settings:
    # _env_file=None: tests never read the developer's .env.
    return Settings(_env_file=None, max_sql_retries=2, sql_row_limit=50)


@pytest.fixture
def retriever() -> RunnableLambda:
    return RunnableLambda(lambda _question: [TABLE_DOC, EXAMPLE_DOC])


class KeywordEmbeddings(Embeddings):
    """Word counts over a small vocabulary: questions sharing words get similar vectors."""

    VOCAB = ("salary", "department", "average", "highest", "lowest", "employees", "city", "remote")

    def embed_query(self, text: str) -> list[float]:
        words = [w.strip("?.,") for w in text.lower().split()]
        # The constant keeps the vector non-zero for questions with no vocabulary words.
        return [float(words.count(v)) for v in self.VOCAB] + [0.01]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


@pytest.fixture
def chat_store() -> InMemoryChatStore:
    return InMemoryChatStore()


@pytest.fixture
def query_history() -> InMemoryQueryHistory:
    return InMemoryQueryHistory(KeywordEmbeddings())


@pytest.fixture
def stores(chat_store, query_history) -> dict:
    """The storage dependencies of build_graph, all in memory."""
    return {"chat_store": chat_store, "query_history": query_history}


def make_turn(
    question: str,
    sql: str | None = "SELECT 1",
    answer: str = "A.",
    *,
    sql_reasoning: str | None = None,
    answer_reasoning: str | None = None,
    model: str | None = None,
) -> Turn:
    return {
        "question": question,
        "standalone": question,
        "sql": sql,
        "row_count": 1 if sql else None,
        "answer": answer,
        "error": None if sql else "boom",
        "sql_reasoning": sql_reasoning,
        "answer_reasoning": answer_reasoning,
        "model": model,
    }


@pytest.fixture
def ok_runner() -> Callable[[str], QueryResult]:
    return lambda _sql: QueryResult(columns=["emp_name", "salary"], rows=[("Employee_1", 100.0)])

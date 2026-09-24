"""Schema + few-shot retrieval over pgvector: build the index (admin) and query it (read-only)."""

import logging
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_postgres import PGVector
from sqlalchemy import Engine
from sqlalchemy.exc import DBAPIError

from rag_sql.config import PROJECT_ROOT, Settings, get_settings
from rag_sql.db.connection import get_engine
from rag_sql.db.introspect import TableDoc, introspect_tables
from rag_sql.llm import get_embeddings

logger = logging.getLogger(__name__)

DEFAULT_EXAMPLES_PATH = PROJECT_ROOT / "examples" / "few_shot.yaml"

# Document kinds, stored in metadata and used to filter searches.
TABLE = "table"
EXAMPLE = "example"


def table_to_document(table: TableDoc) -> Document:
    return Document(
        id=f"{TABLE}:{table.qualified_name}",
        page_content=table.to_text(),
        metadata={"kind": TABLE, "table": table.qualified_name},
    )


def example_to_document(index: int, example: dict[str, str]) -> Document:
    # Only the question is embedded, so examples match on question similarity.
    return Document(
        id=f"{EXAMPLE}:{index}",
        page_content=example["question"].strip(),
        metadata={"kind": EXAMPLE, "sql": example["sql"].strip()},
    )


def load_examples(path: Path = DEFAULT_EXAMPLES_PATH) -> list[dict[str, str]]:
    if not path.exists():
        logger.warning("No few-shot examples file at %s", path)
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict) or not ex.get("question") or not ex.get("sql"):
            raise ValueError(f"{path}: example #{i} needs non-empty 'question' and 'sql'")
    return data


def build_documents(tables: list[TableDoc], examples: list[dict[str, str]]) -> list[Document]:
    return [table_to_document(t) for t in tables] + [
        example_to_document(i, ex) for i, ex in enumerate(examples)
    ]


def _vector_store(
    engine: Engine, embeddings: Embeddings, settings: Settings, **kwargs: Any
) -> PGVector:
    return PGVector(
        embeddings=embeddings,
        connection=engine,
        collection_name=settings.vector_collection,
        use_jsonb=True,
        **kwargs,
    )


def build_index(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    embeddings: Embeddings | None = None,
    examples_path: Path = DEFAULT_EXAMPLES_PATH,
) -> int:
    """Rebuild the vector collection from the live schema and the few-shot file.

    Uses the admin engine: it creates the pgvector tables and replaces the collection.
    Returns the number of documents indexed.
    """
    s = settings or get_settings()
    engine = engine or get_engine("admin", settings=s)
    embeddings = embeddings or get_embeddings(s)

    tables = introspect_tables(engine)
    examples = load_examples(examples_path)
    docs = build_documents(tables, examples)
    logger.info("Indexing %d tables and %d examples", len(tables), len(examples))

    store = _vector_store(engine, embeddings, s, pre_delete_collection=True)
    if docs:
        store.add_documents(docs, ids=[d.id for d in docs])
    return len(docs)


def get_retriever(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    embeddings: Embeddings | None = None,
    k: int | None = None,
) -> Runnable[str, list[Document]]:
    """Retriever: question -> up to `k` table docs followed by up to `k // 2` example docs.

    Tables and examples are searched separately, so examples can never crowd the schema out.
    Uses the read-only engine; the index must already exist (`uv run rag-sql-index`).
    """
    s = settings or get_settings()
    engine = engine or get_engine("reader", settings=s)
    embeddings = embeddings or get_embeddings(s)
    k = k or s.retrieval_k

    try:
        store = _vector_store(engine, embeddings, s, create_extension=False)
    except DBAPIError as e:
        raise RuntimeError(
            f"Vector collection {s.vector_collection!r} is missing or unreadable. "
            "Build it with `uv run rag-sql-index`."
        ) from e

    def retrieve(question: str) -> list[Document]:
        vector = embeddings.embed_query(question)
        tables = store.similarity_search_by_vector(vector, k=k, filter={"kind": TABLE})
        examples = store.similarity_search_by_vector(
            vector, k=max(1, k // 2), filter={"kind": EXAMPLE}
        )
        return tables + examples

    return RunnableLambda(retrieve, name="schema_retriever")


def main() -> None:
    """CLI entry point: `uv run rag-sql-index`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    count = build_index()
    logger.info("Indexed %d documents into collection %r", count, get_settings().vector_collection)


if __name__ == "__main__":
    main()

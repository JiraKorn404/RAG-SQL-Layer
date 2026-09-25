"""Query history: successful past questions + SQL, retrieved by vector similarity as examples.

When a saved turn's SQL runs and returns rows, its standalone question is embedded and stored
with the SQL in `chat_memory.query_examples` (created by db/init/04-chat-memory.sh), through the
chat role (CHAT_DB_USER), which can only read and insert. Before writing SQL, the agent retrieves
the most similar ones. The agent's read-only role has no access to the table. Stored examples are
managed with `uv run rag-sql-history` (cli.py).
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from langchain_core.embeddings import Embeddings
from sqlalchemy import Engine, bindparam, text

from rag_sql.config import Settings, get_settings
from rag_sql.db.connection import get_engine
from rag_sql.llm import get_embeddings

# Candidates fetched per result wanted, so duplicates and weak matches can be dropped.
_OVERFETCH = 3


@dataclass(frozen=True)
class PastQuery:
    question: str
    sql: str
    similarity: float  # cosine similarity of the questions, 1.0 = same direction


class QueryHistory(Protocol):
    def add(self, question: str, sql: str, row_count: int, turn_id: int | None = None) -> bool:
        """Store a successful query. Returns False when skipped: already stored, or disabled."""
        ...

    def search(self, question: str, k: int, min_similarity: float) -> list[PastQuery]:
        """Up to `k` past queries at least `min_similarity` similar, best first."""
        ...


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def pick_best(candidates: list[PastQuery], k: int, min_similarity: float) -> list[PastQuery]:
    """Best first, above the cutoff, one per question."""
    picked: list[PastQuery] = []
    seen: set[str] = set()
    for c in sorted(candidates, key=lambda c: c.similarity, reverse=True):
        key = normalize_question(c.question)
        if c.similarity < min_similarity or key in seen:
            continue
        seen.add(key)
        picked.append(c)
        if len(picked) == k:
            break
    return picked


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


@dataclass
class _Row:
    id: int
    question: str
    sql: str
    vector: list[float]
    enabled: bool = True


class InMemoryQueryHistory:
    """Process-local history, for tests and runs without the database."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._rows: list[_Row] = []

    def add(self, question: str, sql: str, row_count: int, turn_id: int | None = None) -> bool:
        if any(r.question == question and r.sql == sql for r in self._rows):
            return False
        vector = self._embeddings.embed_query(question)
        self._rows.append(_Row(len(self._rows) + 1, question, sql, vector))
        return True

    def search(self, question: str, k: int, min_similarity: float) -> list[PastQuery]:
        if k <= 0 or not self._rows:
            return []
        vector = self._embeddings.embed_query(question)
        candidates = [
            PastQuery(r.question, r.sql, _cosine(vector, r.vector)) for r in self._rows if r.enabled
        ]
        return pick_best(candidates, k, min_similarity)

    def set_enabled(self, ids: list[int], enabled: bool) -> None:
        for row in self._rows:
            if row.id in ids:
                row.enabled = enabled


class PostgresQueryHistory:
    """History in `chat_memory.query_examples`. Use an engine for the "memory" role."""

    def __init__(self, engine: Engine, embeddings: Embeddings, embed_model: str) -> None:
        self._engine = engine
        self._embeddings = embeddings
        self._embed_model = embed_model

    def add(self, question: str, sql: str, row_count: int, turn_id: int | None = None) -> bool:
        vector = self._embeddings.embed_query(question)
        with self._engine.begin() as conn:
            inserted = conn.execute(
                text(
                    "INSERT INTO chat_memory.query_examples "
                    "(turn_id, question, sql, row_count, embedding, embed_model) "
                    "SELECT CAST(:turn_id AS bigint), :question, :sql, "
                    "CAST(:row_count AS integer), CAST(:embedding AS vector), :embed_model "
                    # A disabled pair stays out, also under a new embedding model.
                    "WHERE NOT EXISTS (SELECT 1 FROM chat_memory.query_examples "
                    "WHERE NOT enabled AND question = :question AND sql = :sql) "
                    "ON CONFLICT (embed_model, md5(question), md5(sql)) DO NOTHING"
                ),
                {
                    "turn_id": turn_id,
                    "question": question,
                    "sql": sql,
                    "row_count": row_count,
                    "embedding": _vector_literal(vector),
                    "embed_model": self._embed_model,
                },
            ).rowcount
        return inserted == 1

    def search(self, question: str, k: int, min_similarity: float) -> list[PastQuery]:
        if k <= 0:
            return []
        vector = _vector_literal(self._embeddings.embed_query(question))
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT question, sql, 1 - (embedding <=> CAST(:embedding AS vector)) "
                    "FROM chat_memory.query_examples "
                    "WHERE enabled AND embed_model = :embed_model "
                    "ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT :limit"
                ),
                {"embedding": vector, "embed_model": self._embed_model, "limit": k * _OVERFETCH},
            ).all()
        return pick_best([PastQuery(q, s, float(sim)) for q, s, sim in rows], k, min_similarity)

    def backfill(self) -> int:
        """Add successful chat turns that have no example for the current embedding model."""
        with self._engine.connect() as conn:
            turns = conn.execute(
                text(
                    "SELECT t.id, t.standalone, t.sql, t.row_count FROM chat_memory.chat_turns t "
                    "WHERE t.sql IS NOT NULL AND t.error IS NULL AND t.row_count > 0 "
                    "AND NOT EXISTS (SELECT 1 FROM chat_memory.query_examples e "
                    "WHERE e.question = t.standalone AND e.sql = t.sql "
                    "AND (e.embed_model = :embed_model OR NOT e.enabled)) "
                    "ORDER BY t.id"
                ),
                {"embed_model": self._embed_model},
            ).all()
        return sum(self.add(question, sql, rows, turn_id) for turn_id, question, sql, rows in turns)


def get_query_history(settings: Settings | None = None) -> PostgresQueryHistory:
    s = settings or get_settings()
    return PostgresQueryHistory(
        get_engine("memory", settings=s), get_embeddings(s), embed_model=s.ollama_embed_model
    )


# --- admin -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredExample:
    id: int
    enabled: bool
    question: str
    sql: str
    row_count: int
    embed_model: str
    created_at: datetime


def list_examples(
    engine: Engine, *, include_disabled: bool = False, limit: int = 50
) -> list[StoredExample]:
    """Stored examples, newest first."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, enabled, question, sql, row_count, embed_model, created_at "
                "FROM chat_memory.query_examples WHERE enabled OR :include_disabled "
                "ORDER BY id DESC LIMIT :limit"
            ),
            {"include_disabled": include_disabled, "limit": limit},
        ).all()
    return [StoredExample(*row) for row in rows]


def set_enabled(engine: Engine, ids: list[int], enabled: bool) -> int:
    """Enable or disable examples by id. Needs the admin engine. Returns the rows changed."""
    with engine.begin() as conn:
        return conn.execute(
            text(
                "UPDATE chat_memory.query_examples SET enabled = :enabled WHERE id IN :ids"
            ).bindparams(bindparam("ids", expanding=True)),
            {"enabled": enabled, "ids": ids},
        ).rowcount

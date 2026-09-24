"""Chat history: one row per question, grouped into conversations by thread_id, kept forever.

Stored in `chat_memory.chat_turns` (created by db/init/04-chat-memory.sh) through the chat role
(CHAT_DB_USER), which can only read and insert there. The agent's read-only role has no access
to that schema, so generated SQL can never read past conversations.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypedDict

from sqlalchemy import Engine, text

from rag_sql.config import Settings, get_settings
from rag_sql.db.connection import get_engine


class Turn(TypedDict):
    question: str  # as the user asked it
    standalone: str  # rewritten to stand on its own (same as question when there was no history)
    sql: str | None  # SQL that produced the answer; None when every attempt failed
    row_count: int | None
    answer: str
    error: str | None  # last error when the turn failed


@dataclass(frozen=True)
class ThreadSummary:
    thread_id: str
    turns: int
    first_question: str
    last_at: datetime


class ChatStore(Protocol):
    def load(self, thread_id: str, limit: int | None = None) -> list[Turn]:
        """The last `limit` turns of a thread (all of them when None), oldest first."""
        ...

    def append(self, thread_id: str, turn: Turn) -> int:
        """Save a turn and return its id."""
        ...

    def threads(self, limit: int = 20) -> list[ThreadSummary]:
        """Conversations, most recently active first."""
        ...


def new_thread_id() -> str:
    return uuid.uuid4().hex


def _last(turns: list[Turn], limit: int | None) -> list[Turn]:
    if limit is None:
        return turns
    return turns[-limit:] if limit > 0 else []


class InMemoryChatStore:
    """Process-local store, for tests and runs without the database."""

    def __init__(self) -> None:
        self._rows: dict[str, list[tuple[datetime, Turn]]] = {}
        self._count = 0

    def load(self, thread_id: str, limit: int | None = None) -> list[Turn]:
        return _last([turn.copy() for _, turn in self._rows.get(thread_id, [])], limit)

    def append(self, thread_id: str, turn: Turn) -> int:
        self._rows.setdefault(thread_id, []).append((datetime.now(UTC), turn.copy()))
        self._count += 1
        return self._count

    def threads(self, limit: int = 20) -> list[ThreadSummary]:
        summaries = [
            ThreadSummary(thread_id, len(rows), rows[0][1]["question"], rows[-1][0])
            for thread_id, rows in self._rows.items()
        ]
        return sorted(summaries, key=lambda s: s.last_at, reverse=True)[:limit]


class PostgresChatStore:
    """Store backed by `chat_memory.chat_turns`. Use an engine for the "memory" role."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def load(self, thread_id: str, limit: int | None = None) -> list[Turn]:
        if limit is not None and limit <= 0:
            return []
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT question, standalone, sql, row_count, answer, error "
                        "FROM chat_memory.chat_turns WHERE thread_id = :thread_id "
                        "ORDER BY id DESC LIMIT :limit"
                    ),
                    {"thread_id": thread_id, "limit": limit},
                )
                .mappings()
                .all()
            )
        return [Turn(**row) for row in reversed(rows)]

    def append(self, thread_id: str, turn: Turn) -> int:
        with self._engine.begin() as conn:
            return conn.execute(
                text(
                    "INSERT INTO chat_memory.chat_turns "
                    "(thread_id, question, standalone, sql, row_count, answer, error) VALUES "
                    "(:thread_id, :question, :standalone, :sql, :row_count, :answer, :error) "
                    "RETURNING id"
                ),
                {"thread_id": thread_id, **turn},
            ).scalar_one()

    def threads(self, limit: int = 20) -> list[ThreadSummary]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT thread_id, count(*) AS turns, "
                    "(array_agg(question ORDER BY id))[1] AS first_question, "
                    "max(created_at) AS last_at "
                    "FROM chat_memory.chat_turns GROUP BY thread_id "
                    "ORDER BY last_at DESC LIMIT :limit"
                ),
                {"limit": limit},
            ).all()
        return [ThreadSummary(*row) for row in rows]


def get_chat_store(settings: Settings | None = None) -> ChatStore:
    return PostgresChatStore(get_engine("memory", settings=settings or get_settings()))

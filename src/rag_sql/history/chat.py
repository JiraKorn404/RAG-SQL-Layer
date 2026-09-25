"""Chat history: one row per question, grouped into conversations by thread_id, kept forever.

Stored in `chat_memory.chat_turns`, with each turn's node metrics in `chat_memory.turn_metrics`
(both created by db/init/04-chat-memory.sh), through the chat role (CHAT_DB_USER), which can only
read and insert there. The agent's read-only role has no access to that schema, so generated SQL
can never read past conversations.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypedDict

from sqlalchemy import Connection, Engine, bindparam, text

from rag_sql.config import Settings, get_settings
from rag_sql.db.connection import get_engine
from rag_sql.metrics import COLD_LOAD_MS, NodeMetric, TurnMetrics, summarize


class Turn(TypedDict):
    question: str  # as the user asked it
    standalone: str  # rewritten to stand on its own (same as question when there was no history)
    sql: str | None  # SQL that produced the answer; None when every attempt failed
    row_count: int | None
    answer: str
    error: str | None  # last error when the turn failed
    sql_reasoning: str | None  # model thinking behind the last SQL attempt, if any
    answer_reasoning: str | None  # model thinking behind the answer, if any
    model: str | None  # chat model that answered
    metrics: TurnMetrics | None  # time and tokens per node; None for turns saved without them


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
                        "SELECT id, question, standalone, sql, row_count, answer, error, "
                        "sql_reasoning, answer_reasoning, model "
                        "FROM chat_memory.chat_turns WHERE thread_id = :thread_id "
                        "ORDER BY id DESC LIMIT :limit"
                    ),
                    {"thread_id": thread_id, "limit": limit},
                )
                .mappings()
                .all()
            )
            nodes = _node_metrics(conn, [row["id"] for row in rows])
        return [
            Turn(**{k: v for k, v in row.items() if k != "id"}, metrics=summarize(nodes[row["id"]]))
            for row in reversed(rows)
        ]

    def append(self, thread_id: str, turn: Turn) -> int:
        """Save the turn and its node metrics, in one transaction."""
        fields = {k: v for k, v in turn.items() if k != "metrics"}
        with self._engine.begin() as conn:
            turn_id = conn.execute(
                text(
                    "INSERT INTO chat_memory.chat_turns "
                    "(thread_id, question, standalone, sql, row_count, answer, error, "
                    "sql_reasoning, answer_reasoning, model) VALUES "
                    "(:thread_id, :question, :standalone, :sql, :row_count, :answer, :error, "
                    ":sql_reasoning, :answer_reasoning, :model) "
                    "RETURNING id"
                ),
                {"thread_id": thread_id, **fields},
            ).scalar_one()
            if metrics := turn.get("metrics"):
                conn.execute(
                    text(
                        "INSERT INTO chat_memory.turn_metrics "
                        "(turn_id, seq, node, attempt, ms, llm_ms, load_ms, input_tokens, "
                        "output_tokens) VALUES "
                        "(:turn_id, :seq, :node, :attempt, :ms, :llm_ms, :load_ms, "
                        ":input_tokens, :output_tokens)"
                    ),
                    [
                        {"turn_id": turn_id, "seq": seq, **node}
                        for seq, node in enumerate(metrics["nodes"])
                    ],
                )
            return turn_id

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


def _node_metrics(conn: Connection, turn_ids: list[int]) -> defaultdict[int, list[NodeMetric]]:
    """The stored node metrics of these turns, in run order, by turn id."""
    by_turn: defaultdict[int, list[NodeMetric]] = defaultdict(list)
    if not turn_ids:
        return by_turn
    rows = conn.execute(
        text(
            "SELECT turn_id, node, attempt, ms, llm_ms, load_ms, input_tokens, output_tokens "
            "FROM chat_memory.turn_metrics WHERE turn_id IN :ids ORDER BY turn_id, seq"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": turn_ids},
    ).mappings()
    for row in rows:
        by_turn[row["turn_id"]].append(
            NodeMetric(**{k: v for k, v in row.items() if k != "turn_id"})
        )
    return by_turn


def get_chat_store(settings: Settings | None = None) -> ChatStore:
    return PostgresChatStore(get_engine("memory", settings=settings or get_settings()))


# --- stats -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelStats:
    model: str
    turns: int
    # Share of turns answered without error: their SQL ran, or no SQL was needed (NO_SQL).
    success_rate: float
    avg_attempts: float
    p50_ms: float
    p95_ms: float
    avg_tokens: float
    cold_load_rate: float  # share of turns that waited for the model to load


def turn_stats(engine: Engine, *, days: int = 30, model: str | None = None) -> list[ModelStats]:
    """Per chat model, over the turns of the last `days` days that have metrics. Most used first."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "WITH per_turn AS ("
                " SELECT turn_id, sum(ms) AS total_ms, max(attempt) AS attempts,"
                " sum(coalesce(input_tokens, 0) + coalesce(output_tokens, 0)) AS tokens,"
                " bool_or(coalesce(load_ms, 0) >= :cold_ms) AS cold_load"
                " FROM chat_memory.turn_metrics GROUP BY turn_id) "
                "SELECT coalesce(t.model, 'unknown') AS model, count(*) AS turns,"
                " avg((t.error IS NULL)::int) AS success_rate,"
                " avg(p.attempts) AS avg_attempts,"
                " percentile_cont(0.5) WITHIN GROUP (ORDER BY p.total_ms) AS p50_ms,"
                " percentile_cont(0.95) WITHIN GROUP (ORDER BY p.total_ms) AS p95_ms,"
                " avg(p.tokens) AS avg_tokens,"
                " avg(p.cold_load::int) AS cold_load_rate "
                "FROM chat_memory.chat_turns t JOIN per_turn p ON p.turn_id = t.id "
                "WHERE t.created_at >= now() - make_interval(days => :days)"
                " AND (CAST(:model AS text) IS NULL OR t.model = :model) "
                "GROUP BY 1 ORDER BY turns DESC, model"
            ),
            {"days": days, "model": model, "cold_ms": COLD_LOAD_MS},
        ).all()
    return [ModelStats(row[0], row[1], *(float(v) for v in row[2:])) for row in rows]

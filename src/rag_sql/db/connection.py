"""SQLAlchemy engine factory."""

from functools import lru_cache
from typing import Literal

from sqlalchemy import URL, Engine, create_engine

from rag_sql.config import Settings, get_settings

Role = Literal["reader", "admin", "memory"]

# Seconds to wait for each address of the database host. Without it, connecting to a host that
# is down can hang for minutes before the agent can report that the database is unavailable.
CONNECT_TIMEOUT_S = 5


def get_engine(role: Role = "reader", settings: Settings | None = None) -> Engine:
    """Engine for one of the database roles:

    - "reader" (default): the read-only agent role. The only engine that runs LLM-generated SQL.
    - "admin": the Postgres owner, for indexing only.
    - "memory": the chat history role; reads and inserts `chat_memory.chat_turns`, nothing else.

    Never run LLM-generated SQL on an engine other than "reader".

    Engines are shared: the same role and connection settings always get the same engine, and so
    one connection pool (e.g. the chat store and the query history share the "memory" one).
    """
    s = settings or get_settings()
    user, password = {
        "reader": (s.app_db_user, s.app_db_password),
        "admin": (s.postgres_user, s.postgres_password),
        "memory": (s.chat_db_user, s.chat_db_password),
    }[role]

    url = URL.create(
        "postgresql+psycopg",
        username=user,
        password=password.get_secret_value(),
        host=s.postgres_host,
        port=s.postgres_port,
        database=s.postgres_db,
    )
    return _create_engine(url, read_only=role == "reader")


@lru_cache
def _create_engine(url: URL, *, read_only: bool) -> Engine:
    connect_args: dict[str, object] = {"connect_timeout": CONNECT_TIMEOUT_S}
    if read_only:
        connect_args["options"] = "-c default_transaction_read_only=on"
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)

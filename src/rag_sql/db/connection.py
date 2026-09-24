"""SQLAlchemy engine factory."""

from typing import Literal

from sqlalchemy import URL, Engine, create_engine

from rag_sql.config import Settings, get_settings

Role = Literal["reader", "admin", "memory"]


def get_engine(role: Role = "reader", settings: Settings | None = None) -> Engine:
    """Engine for one of the database roles:

    - "reader" (default): the read-only agent role. The only engine that runs LLM-generated SQL.
    - "admin": the Postgres owner, for indexing only.
    - "memory": the chat history role; reads and inserts `chat_memory.chat_turns`, nothing else.

    Never run LLM-generated SQL on an engine other than "reader".
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
    connect_args = {"options": "-c default_transaction_read_only=on"} if role == "reader" else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)

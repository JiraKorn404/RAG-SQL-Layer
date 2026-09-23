"""SQLAlchemy engine factory."""

from sqlalchemy import URL, Engine, create_engine

from rag_sql.config import Settings, get_settings


def get_engine(readonly: bool = True, settings: Settings | None = None) -> Engine:
    """Engine for the read-only agent role (default) or the admin user (indexing only).

    Never run LLM-generated SQL on an engine created with `readonly=False`.
    """
    s = settings or get_settings()
    if readonly:
        user, password = s.app_db_user, s.app_db_password
    else:
        user, password = s.postgres_user, s.postgres_password

    url = URL.create(
        "postgresql+psycopg",
        username=user,
        password=password.get_secret_value(),
        host=s.postgres_host,
        port=s.postgres_port,
        database=s.postgres_db,
    )
    connect_args = {"options": "-c default_transaction_read_only=on"} if readonly else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)

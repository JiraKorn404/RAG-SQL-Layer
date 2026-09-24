from typing import Any

from rag_sql.config import Settings
from rag_sql.db import connection
from rag_sql.db.connection import get_engine


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_same_role_and_settings_share_one_engine() -> None:
    # Separate but equal Settings objects, like the chat store's and the query history's.
    assert get_engine("memory", _settings()) is get_engine("memory", _settings())


def test_roles_and_connection_settings_get_their_own_engine() -> None:
    s = _settings()
    assert get_engine("reader", s) is not get_engine("memory", s)
    assert get_engine("reader", s) is not get_engine("reader", _settings(postgres_port=5433))


def test_only_the_reader_engine_is_read_only(monkeypatch) -> None:
    created: dict[str, dict[str, Any]] = {}

    def fake_create_engine(url, **kwargs):
        created[url.username] = kwargs["connect_args"]
        return object()

    monkeypatch.setattr(connection, "create_engine", fake_create_engine)
    # A port no other test uses, so the engines aren't cached yet.
    s = _settings(postgres_port=15432)
    for role in ("reader", "admin", "memory"):
        get_engine(role, s)

    timeout = {"connect_timeout": connection.CONNECT_TIMEOUT_S}
    assert created == {
        s.app_db_user: {**timeout, "options": "-c default_transaction_read_only=on"},
        s.postgres_user: timeout,
        s.chat_db_user: timeout,
    }

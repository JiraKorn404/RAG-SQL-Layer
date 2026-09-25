import pytest
from pydantic import ValidationError

from rag_sql.config import Settings


@pytest.mark.parametrize(
    "overrides",
    [
        {"sql_row_limit": 0},
        {"sql_timeout_ms": 0},  # Postgres would read it as "no timeout"
        {"max_sql_retries": -1},
        {"retrieval_k": 0},
        {"chat_history_turns": -1},
        {"query_history_k": -1},
        {"query_history_min_similarity": 1.5},
        {"postgres_port": 70000},
        {"db_schemas": "public,chat_memory"},
        {"langfuse_enabled": True},  # without keys
        {"langfuse_enabled": True, "langfuse_public_key": "pk-lf-x"},
    ],
)
def test_rejects_out_of_range_values(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_router_model_defaults_to_the_chat_model() -> None:
    assert Settings(_env_file=None).ollama_router_model == ""


def test_zero_turns_and_examples_turn_features_off() -> None:
    s = Settings(_env_file=None, chat_history_turns=0, query_history_k=0, max_sql_retries=0)
    assert (s.chat_history_turns, s.query_history_k, s.max_sql_retries) == (0, 0, 0)


def test_schemas_are_split() -> None:
    assert Settings(_env_file=None, db_schemas=" public , imba ").db_schemas == ("public", "imba")

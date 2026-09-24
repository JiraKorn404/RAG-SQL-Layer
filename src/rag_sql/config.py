"""Application settings. The only module that reads environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repo root, so `.env` is found whatever the working directory is (the notebook runs in notebooks/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag"
    postgres_user: str = "postgres"
    postgres_password: SecretStr = SecretStr("")

    # Read-only role used by the agent
    app_db_user: str = "rag_reader"
    app_db_password: SecretStr = SecretStr("")

    # Chat history role: reads and inserts chat_memory.chat_turns, nothing else
    chat_db_user: str = "rag_memory"
    chat_db_password: SecretStr = SecretStr("")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:14b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_reasoning: bool = True

    # Agent / SQL execution
    sql_row_limit: int = 200
    sql_timeout_ms: int = 15000
    max_sql_retries: int = 3

    # Retrieval
    vector_collection: str = "schema_docs"
    retrieval_k: int = 6
    # Schemas whose tables are indexed and shown to the model (comma-separated in .env)
    db_schemas: Annotated[tuple[str, ...], NoDecode] = ("public", "imba")

    # Chat history: earlier turns shown to the model (0 = none; turns are still saved)
    chat_history_turns: int = 5

    # Query history: similar past successful queries given to the model as examples
    query_history_k: int = 5  # most retrieved (0 = off; queries are still saved)
    query_history_min_similarity: float = 0.75  # cosine similarity cutoff

    @field_validator("db_schemas", mode="before")
    @classmethod
    def _split_schemas(cls, value: object) -> object:
        if isinstance(value, str):
            value = [s.strip() for s in value.split(",") if s.strip()]
        if not value:
            raise ValueError("DB_SCHEMAS needs at least one schema")
        if "chat_memory" in value:
            raise ValueError("chat_memory holds chat history and must never be shown to the model")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Application settings. The only module that reads environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, SecretStr, field_validator, model_validator
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
    postgres_port: int = Field(5432, gt=0, le=65535)
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
    sql_row_limit: int = Field(200, gt=0)
    sql_timeout_ms: int = Field(15000, gt=0)  # 0 would mean no timeout in Postgres
    max_sql_retries: int = Field(3, ge=0)

    # Retrieval
    vector_collection: str = "schema_docs"
    retrieval_k: int = Field(6, gt=0)
    # Schemas whose tables are indexed and shown to the model (comma-separated in .env)
    db_schemas: Annotated[tuple[str, ...], NoDecode] = ("public", "imba")

    # Chat history: earlier turns shown to the model (0 = none; turns are still saved)
    chat_history_turns: int = Field(5, ge=0)

    # Query history: similar past successful queries given to the model as examples
    query_history_k: int = Field(5, ge=0)  # most retrieved (0 = off; queries are still saved)
    query_history_min_similarity: float = Field(0.75, ge=0, le=1)  # cosine similarity cutoff

    # Langfuse tracing (tracing.py); the server is docker-compose.langfuse.yml
    langfuse_enabled: bool = False
    langfuse_base_url: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")

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

    @model_validator(mode="after")
    def _langfuse_keys(self) -> Self:
        if self.langfuse_enabled and not (
            self.langfuse_public_key and self.langfuse_secret_key.get_secret_value()
        ):
            raise ValueError("LANGFUSE_ENABLED needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

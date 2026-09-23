"""LLM and embedding factories. The only module that knows the provider (Ollama)."""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_sql.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None, **overrides: Any) -> BaseChatModel:
    """Chat model used for SQL generation and answering.

    With `reasoning=True`, the model's thinking is returned in
    `message.additional_kwargs["reasoning_content"]`, separate from the content.
    """
    s = settings or get_settings()
    params: dict[str, Any] = {
        "model": s.ollama_chat_model,
        "base_url": s.ollama_base_url,
        "reasoning": s.ollama_reasoning,
        "temperature": 0,
    }
    params.update(overrides)
    return ChatOllama(**params)


def get_embeddings(settings: Settings | None = None, **overrides: Any) -> Embeddings:
    s = settings or get_settings()
    params: dict[str, Any] = {"model": s.ollama_embed_model, "base_url": s.ollama_base_url}
    params.update(overrides)
    return OllamaEmbeddings(**params)

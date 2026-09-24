"""LLM and embedding factories. The only module that knows the provider (Ollama)."""

from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ollama import Client

from rag_sql.config import Settings, get_settings

# Seconds to wait for the Ollama server when listing models.
LIST_TIMEOUT_S = 10


@dataclass(frozen=True)
class ChatModelInfo:
    name: str  # e.g. "qwen3.5:9b"
    parameter_size: str | None  # as the server reports it, e.g. "9.7B"
    thinking: bool  # returns its reasoning separately; the others reject reasoning=True


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


def chat_model_name(llm: BaseChatModel) -> str | None:
    """The model name of a chat model (ChatOllama and most providers expose `model`)."""
    return getattr(llm, "model", None)


def list_chat_models(
    settings: Settings | None = None, *, client: Client | None = None
) -> list[ChatModelInfo]:
    """Chat models on the Ollama server, by name. Embedding models are left out.

    Uses the capabilities the server reports (`/api/show`): a chat model can do "completion" and
    is not an "embedding" model. Names and families don't tell them apart (qwen3-embedding).
    """
    s = settings or get_settings()
    client = client or Client(host=s.ollama_base_url, timeout=LIST_TIMEOUT_S)
    models = []
    for listed in client.list().models:
        if not listed.model:
            continue
        capabilities = set(client.show(listed.model).capabilities or [])
        if "completion" not in capabilities or "embedding" in capabilities:
            continue
        size = listed.details.parameter_size if listed.details else None
        models.append(ChatModelInfo(listed.model, size or None, "thinking" in capabilities))
    return sorted(models, key=lambda m: m.name)

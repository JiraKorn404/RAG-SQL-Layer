"""LLM and embedding factories. The only module that knows the provider (Ollama)."""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ollama import Client

from rag_sql.config import Settings, get_settings

# Seconds to wait for the Ollama server when listing models.
LIST_TIMEOUT_S = 10

# Query embeddings remembered (see CachedEmbeddings).
EMBED_CACHE_SIZE = 256


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


def get_router_model(chat_llm: BaseChatModel, settings: Settings | None = None) -> Runnable:
    """Model of the router agent: a short classification, so it never thinks.

    `OLLAMA_ROUTER_MODEL` when set, else the chat model itself with thinking off (same model, so
    Ollama doesn't have to load a second one).
    """
    s = settings or get_settings()
    if s.ollama_router_model:
        return get_chat_model(s, model=s.ollama_router_model, reasoning=False)
    return chat_llm.bind(reasoning=False)


def with_json_schema(llm: Runnable, schema: dict[str, Any]) -> Runnable:
    """The model constrained to reply with JSON that fits `schema` (Ollama's `format`)."""
    return llm.bind(format=schema)


class CachedEmbeddings(Embeddings):
    """Remembers the most recent query embeddings.

    In one turn, the retriever, the query history search and the query history save all embed
    the same standalone question: with a shared instance, that is one call to the server.
    """

    def __init__(self, inner: Embeddings, maxsize: int = EMBED_CACHE_SIZE) -> None:
        self.inner = inner
        self._maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()  # the web UI embeds from several sessions at once

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            if (vector := self._cache.get(text)) is not None:
                self._cache.move_to_end(text)
                return list(vector)
        vector = self.inner.embed_query(text)
        with self._lock:
            self._cache[text] = list(vector)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return list(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)


def get_embeddings(settings: Settings | None = None, **overrides: Any) -> Embeddings:
    """Embedding model. Without overrides, callers with the same model share one instance, and so
    one query cache (see CachedEmbeddings)."""
    s = settings or get_settings()
    if overrides:
        params: dict[str, Any] = {"model": s.ollama_embed_model, "base_url": s.ollama_base_url}
        params.update(overrides)
        return CachedEmbeddings(OllamaEmbeddings(**params))
    return _shared_embeddings(s.ollama_embed_model, s.ollama_base_url)


@lru_cache
def _shared_embeddings(model: str, base_url: str) -> CachedEmbeddings:
    return CachedEmbeddings(OllamaEmbeddings(model=model, base_url=base_url))


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

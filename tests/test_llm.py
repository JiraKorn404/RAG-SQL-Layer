from types import SimpleNamespace

from langchain_core.embeddings import Embeddings

from rag_sql.config import Settings
from rag_sql.llm import (
    CachedEmbeddings,
    ChatModelInfo,
    chat_model_name,
    get_chat_model,
    get_embeddings,
    list_chat_models,
)

# What the Ollama server reported (0.34.3) for the models on the dev machine, plus two odd cases.
SERVER_MODELS = {
    "qwen3.5:9b": ("9.7B", ["completion", "vision", "tools", "thinking"]),
    "gemma4:e4b": ("8.0B", ["completion", "vision", "audio", "tools", "thinking"]),
    "qwen3-embedding:0.6b": ("595.78M", ["tools", "thinking", "embedding"]),
    "old-model:7b": ("7B", None),  # a server that reports no capabilities
    "plain-chat:1b": ("", ["completion"]),
}


class FakeOllamaClient:
    def list(self) -> SimpleNamespace:
        return SimpleNamespace(
            models=[
                SimpleNamespace(model=name, details=SimpleNamespace(parameter_size=size))
                for name, (size, _) in SERVER_MODELS.items()
            ]
        )

    def show(self, model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=SERVER_MODELS[model][1])


def test_list_chat_models_keeps_chat_models_only() -> None:
    assert list_chat_models(Settings(_env_file=None), client=FakeOllamaClient()) == [
        ChatModelInfo("gemma4:e4b", "8.0B", thinking=True),
        ChatModelInfo("plain-chat:1b", None, thinking=False),
        ChatModelInfo("qwen3.5:9b", "9.7B", thinking=True),
    ]


def test_chat_model_name() -> None:
    llm = get_chat_model(Settings(_env_file=None), model="qwen3.5:9b", reasoning=False)
    assert chat_model_name(llm) == "qwen3.5:9b"
    assert llm.reasoning is False


class CountingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def test_cached_embeddings_embed_each_query_once() -> None:
    inner = CountingEmbeddings()
    cached = CachedEmbeddings(inner, maxsize=2)
    assert cached.embed_query("a") == cached.embed_query("a") == [1.0]
    cached.embed_query("bb")
    cached.embed_query("ccc")  # evicts "a", the least recently used
    cached.embed_query("a")
    assert inner.calls == ["a", "bb", "ccc", "a"]


def test_cached_vectors_cannot_be_changed_by_callers() -> None:
    cached = CachedEmbeddings(CountingEmbeddings())
    cached.embed_query("a").append(99.0)
    assert cached.embed_query("a") == [1.0]


def test_embeddings_are_shared_per_model() -> None:
    s = Settings(_env_file=None)
    assert get_embeddings(s) is get_embeddings(Settings(_env_file=None))
    assert get_embeddings(s) is not get_embeddings(s, model="other-embed")

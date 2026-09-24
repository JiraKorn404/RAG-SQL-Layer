from types import SimpleNamespace

from rag_sql.config import Settings
from rag_sql.llm import ChatModelInfo, chat_model_name, get_chat_model, list_chat_models

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

import logging
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import SecretStr

from rag_sql import tracing
from rag_sql.agent.graph import build_graph
from rag_sql.config import Settings
from tests.conftest import fake_llm


@pytest.fixture
def traced(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "langfuse_enabled": True,
            "langfuse_public_key": "pk-lf-test",
            "langfuse_secret_key": SecretStr("sk-lf-test"),
        }
    )


class FakeHandler:
    def __init__(self, *, public_key: str) -> None:
        self.public_key = public_key


@pytest.fixture
def clients(monkeypatch) -> list[tuple]:
    """The Langfuse clients built (none of them real), and handlers that aren't real either."""
    built: list[tuple] = []
    monkeypatch.setattr(tracing, "_cached_client", lambda *args: built.append(args))
    monkeypatch.setattr(tracing, "CallbackHandler", FakeHandler)
    return built


def test_disabled_tracing_gives_an_empty_config(settings, clients) -> None:
    assert tracing.run_config("web", session_id="t1", settings=settings) == {}
    assert clients == []


def test_enabled_tracing_adds_a_handler_and_trace_attributes(traced, clients) -> None:
    config = tracing.run_config(
        "eval", session_id="eval-1", model="qwen3:14b", tags=["case_a"], settings=traced
    )
    [handler] = config["callbacks"]
    assert handler.public_key == "pk-lf-test"
    assert clients == [("http://localhost:3000", "pk-lf-test", "sk-lf-test")]
    assert config["run_name"] == tracing.TRACE_NAME
    assert config["metadata"] == {
        "langfuse_trace_name": tracing.TRACE_NAME,
        "langfuse_tags": ["eval", "qwen3:14b", "case_a"],
        "langfuse_session_id": "eval-1",
    }


def test_each_run_gets_its_own_handler(traced, clients) -> None:
    first = tracing.run_config("web", settings=traced)["callbacks"][0]
    assert tracing.run_config("web", settings=traced)["callbacks"][0] is not first
    assert "langfuse_session_id" not in tracing.run_config("web", settings=traced)["metadata"]


def test_unavailable_client_runs_untraced(traced, monkeypatch, caplog) -> None:
    def broken(*_args):
        raise RuntimeError("bad config")

    monkeypatch.setattr(tracing, "_cached_client", broken)
    with caplog.at_level(logging.WARNING):
        assert tracing.run_config("notebook", settings=traced) == {}
    assert "running untraced" in caplog.text


class Recorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.chains: list[str] = []
        self.model_calls = 0

    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        self.chains.append(kwargs.get("name") or "")

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.model_calls += 1


def test_run_callbacks_reach_the_nodes_and_their_model_calls(
    settings, retriever, ok_runner, stores
) -> None:
    # Tracing relies on this: the nodes need no changes to be traced.
    llm = fake_llm("```sql\nSELECT emp_name FROM employees\n```", "Employee_1.")
    graph = build_graph(
        llm=llm, retriever=retriever, query_runner=ok_runner, **stores, settings=settings
    )
    recorder = Recorder()
    graph.invoke({"question": "Who earns the most?"}, config={"callbacks": [recorder]})
    assert {"generate_sql", "execute_sql", "answer"} <= set(recorder.chains)
    assert recorder.model_calls == 2  # generate_sql and answer (turn 1: no condense call)

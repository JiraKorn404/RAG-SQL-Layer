from langchain_core.messages import AIMessage

from rag_sql.metrics import format_summary, node_metric, summarize, usage_of


def test_usage_from_usage_and_response_metadata() -> None:
    message = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        response_metadata={"total_duration": 1_234_567_890, "load_duration": 0},
    )
    assert usage_of(message) == {
        "llm_ms": 1235,
        "load_ms": None,  # 0: the model was already loaded
        "input_tokens": 10,
        "output_tokens": 3,
    }


def test_usage_falls_back_to_ollama_counts() -> None:
    message = AIMessage(content="x", response_metadata={"prompt_eval_count": 7, "eval_count": 2})
    assert usage_of(message) == {
        "llm_ms": None,
        "load_ms": None,
        "input_tokens": 7,
        "output_tokens": 2,
    }


def test_no_usage_reported() -> None:
    assert usage_of(AIMessage(content="x")) is None


def _usage(llm_ms=None, load_ms=None, input_tokens=None, output_tokens=None) -> dict:
    return {
        "llm_ms": llm_ms,
        "load_ms": load_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def test_summarize_totals_and_attempts() -> None:
    nodes = [
        node_metric("retrieve_context", attempt=0, ms=50, usage=None),
        node_metric("generate_sql", attempt=1, ms=4000, usage=_usage(3900, 2500, 900, 60)),
        node_metric("generate_sql", attempt=2, ms=3000, usage=_usage(2900, None, 950, 50)),
        node_metric("answer", attempt=2, ms=1000, usage=_usage(950, None, 400, 30)),
    ]
    assert summarize(nodes) == {
        "total_ms": 8050,
        "attempts": 2,
        "llm_ms": 7750,
        "load_ms": 2500,
        "input_tokens": 2250,
        "output_tokens": 140,
        "nodes": nodes,
    }
    assert summarize([]) is None


def test_format_summary() -> None:
    metrics = summarize(
        [
            node_metric("generate_sql", attempt=2, ms=11_000, usage=_usage(9000, 4100, 1500, 300)),
            node_metric("answer", attempt=2, ms=1300, usage=None),
        ]
    )
    assert format_summary(metrics) == "12.3 s · 2 attempts · 1.8k tokens · model load 4.1 s"


def test_format_summary_leaves_out_what_is_not_there() -> None:
    # No SQL generated (e.g. an interrupted run), no model usage reported, no cold start.
    metrics = summarize([node_metric("retrieve_context", attempt=0, ms=420, usage=None)])
    assert format_summary(metrics) == "0.4 s"
    metrics = summarize([node_metric("generate_sql", attempt=1, ms=900, usage=_usage(800, 300))])
    assert format_summary(metrics) == "0.9 s · 1 attempt"

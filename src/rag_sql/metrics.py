"""Turn metrics: the time, tokens and SQL attempt of each node of a run.

`build_graph` times every node and adds a `NodeMetric` to the state; the nodes that call the
model also report its usage (`usage_of`). `save_turn` summarizes them into `TurnMetrics`, stored
per node in `chat_memory.turn_metrics` (memory.py) and shown by steps.py. This module is pure.

CLI (`uv run rag-sql-metrics [--days N] [--model M]`): latency, tokens and success per model.
"""

import argparse
import logging
from typing import TypedDict

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# A model load at least this long counts as a cold start (the model wasn't in memory).
COLD_LOAD_MS = 1000


class LLMUsage(TypedDict):
    llm_ms: int | None  # model time reported by Ollama (total_duration), without the network
    load_ms: int | None  # part of it spent loading the model into memory
    input_tokens: int | None
    output_tokens: int | None


class NodeMetric(TypedDict):
    node: str
    attempt: int  # SQL attempt the node ran in; 0 before the first SQL generation
    ms: int  # wall time of the node, including model and database calls
    llm_ms: int | None  # None for nodes without a model call, or models that report no usage
    load_ms: int | None
    input_tokens: int | None
    output_tokens: int | None


class TurnMetrics(TypedDict):
    total_ms: int
    attempts: int  # SQL generations
    llm_ms: int
    load_ms: int
    input_tokens: int
    output_tokens: int
    nodes: list[NodeMetric]  # in run order


def _ns_to_ms(ns: object) -> int | None:
    return round(ns / 1_000_000) if isinstance(ns, int | float) and ns > 0 else None


def usage_of(message: BaseMessage) -> LLMUsage | None:
    """The usage of one model reply, or None when the model reported none (e.g. a fake)."""
    meta = message.response_metadata or {}
    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", meta.get("prompt_eval_count"))
    output_tokens = usage.get("output_tokens", meta.get("eval_count"))
    llm_ms = _ns_to_ms(meta.get("total_duration"))
    if llm_ms is None and input_tokens is None and output_tokens is None:
        return None
    return LLMUsage(
        llm_ms=llm_ms,
        load_ms=_ns_to_ms(meta.get("load_duration")),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def node_metric(node: str, *, attempt: int, ms: int, usage: LLMUsage | None) -> NodeMetric:
    usage = usage or LLMUsage(llm_ms=None, load_ms=None, input_tokens=None, output_tokens=None)
    return NodeMetric(node=node, attempt=attempt, ms=ms, **usage)


def summarize(nodes: list[NodeMetric]) -> TurnMetrics | None:
    """Totals over a run's node metrics; None when there are none."""
    if not nodes:
        return None

    def total(key: str) -> int:
        return sum(n[key] or 0 for n in nodes)

    return TurnMetrics(
        total_ms=total("ms"),
        attempts=max(n["attempt"] for n in nodes),
        llm_ms=total("llm_ms"),
        load_ms=total("load_ms"),
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        nodes=list(nodes),
    )


def _tokens(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def format_summary(metrics: TurnMetrics) -> str:
    """One line, e.g. "12.3 s · 2 attempts · 1.8k tokens · model load 4.1 s"."""
    parts = [f"{metrics['total_ms'] / 1000:.1f} s"]
    if metrics["attempts"]:
        parts.append(f"{metrics['attempts']} attempt{'s' if metrics['attempts'] > 1 else ''}")
    tokens = metrics["input_tokens"] + metrics["output_tokens"]
    if tokens:
        parts.append(f"{_tokens(tokens)} tokens")
    if metrics["load_ms"] >= COLD_LOAD_MS:
        parts.append(f"model load {metrics['load_ms'] / 1000:.1f} s")
    return " · ".join(parts)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `uv run rag-sql-metrics`."""
    # memory.py imports this module, so it is imported here.
    from rag_sql.db.connection import get_engine
    from rag_sql.memory import turn_stats

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="rag-sql-metrics", description="Latency, tokens and success of saved turns, per model."
    )
    parser.add_argument("--days", type=int, default=30, help="only turns of the last N days")
    parser.add_argument("--model", help="only this chat model")
    args = parser.parse_args(argv)

    stats = turn_stats(get_engine("memory"), days=args.days, model=args.model)
    if not stats:
        print(f"No turns with metrics in the last {args.days} day(s).")
        return
    print(
        f"{'model':<24} {'turns':>6} {'success':>8} {'attempts':>9} {'p50 s':>7} {'p95 s':>7} "
        f"{'tokens':>8} {'cold load':>10}"
    )
    for s in stats:
        print(
            f"{s.model:<24} {s.turns:>6} {s.success_rate:>8.0%} {s.avg_attempts:>9.2f} "
            f"{s.p50_ms / 1000:>7.1f} {s.p95_ms / 1000:>7.1f} {_tokens(round(s.avg_tokens)):>8} "
            f"{s.cold_load_rate:>10.0%}"
        )


if __name__ == "__main__":
    main()

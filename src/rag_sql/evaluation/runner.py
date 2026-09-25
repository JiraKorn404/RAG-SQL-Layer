"""Run the evaluation cases against chat models; report accuracy, retries, latency and tokens.

CLI (`uv run rag-sql-eval`):
  run [--models A,B] [--tags T,U] [--limit N] [--repeat K] [--with-history] [--out DIR]
      one JSON file per model in DIR (default eval_results/, not committed)
  compare OLD.json NEW.json
      accuracy change and the cases whose outcome changed

Runs have no thread_id, so nothing is saved to chat or query history. Past queries are not used
unless --with-history (then they are searched, still never saved).
"""

import argparse
import itertools
import json
import logging
import math
import re
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from rag_sql.agent.graph import build_graph, default_query_runner
from rag_sql.config import PROJECT_ROOT, Settings, get_settings
from rag_sql.db.query import QueryResult, validate_sql
from rag_sql.evaluation.cases import DEFAULT_CASES_PATH, EvalCase, load_cases, select_cases
from rag_sql.evaluation.scoring import CORRECT, Outcome, compare, is_ordered
from rag_sql.llm import ChatModelInfo, get_chat_model, get_embeddings, list_chat_models
from rag_sql.memory import InMemoryChatStore
from rag_sql.metrics import TurnMetrics
from rag_sql.query_history import InMemoryQueryHistory, get_query_history
from rag_sql.retrieval import get_retriever

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = PROJECT_ROOT / "eval_results"


class EvalError(RuntimeError):
    """The evaluation can't run: a bad reference, or a model that isn't available."""


@dataclass
class CaseResult:
    case_id: str
    tags: list[str]
    run: int  # repeat number, from 1
    outcome: Outcome
    sql: str | None  # the SQL that ran, if any
    error: str | None
    answer: str
    attempts: int
    seconds: float  # whole run, including saving
    metrics: TurnMetrics | None  # up to the answer, as the chat history records it

    @property
    def correct(self) -> bool:
        return self.outcome in CORRECT


# --- running ------------------------------------------------------------------------------------


def reference_results(
    cases: Iterable[EvalCase], run_query: Callable[[str], QueryResult], row_limit: int
) -> dict[str, QueryResult]:
    """Run each reference SQL (read-only, like the agent's). A reference must fit the row limit,
    or the comparison would only see part of it."""
    results = {}
    for case in cases:
        try:
            result = run_query(validate_sql(case.sql, row_limit))
        except Exception as e:
            raise EvalError(f"Reference SQL of {case.id!r} failed: {e}") from e
        if result.truncated:
            raise EvalError(f"Reference of {case.id!r} returns more than {row_limit} rows")
        results[case.id] = result
    return results


def run_case(
    graph: CompiledStateGraph, case: EvalCase, expected: QueryResult, *, run: int = 1
) -> CaseResult:
    start = time.perf_counter()
    history = [turn.to_turn() for turn in case.history]
    try:
        state = graph.invoke({"question": case.question, "history": history})
    except Exception as e:
        logger.warning("Case %s failed", case.id, exc_info=True)
        return CaseResult(
            case_id=case.id,
            tags=list(case.tags),
            run=run,
            outcome="agent_error",
            sql=None,
            error=str(e) or type(e).__name__,
            answer="",
            attempts=0,
            seconds=time.perf_counter() - start,
            metrics=None,
        )

    result: QueryResult | None = state.get("result")
    if result is not None:
        outcome = compare(expected, result, ordered=is_ordered(case.sql))
    else:
        outcome = "db_unavailable" if state.get("db_unavailable") else "sql_failed"
    return CaseResult(
        case_id=case.id,
        tags=list(case.tags),
        run=run,
        outcome=outcome,
        sql=state.get("sql") if result is not None else None,
        error=state.get("error"),
        answer=state.get("answer", ""),
        attempts=state.get("attempts", 0),
        seconds=time.perf_counter() - start,
        metrics=state.get("turn_metrics"),
    )


def run_cases(
    graph: CompiledStateGraph,
    cases: list[EvalCase],
    expected: dict[str, QueryResult],
    *,
    repeat: int = 1,
) -> list[CaseResult]:
    results = []
    total = len(cases) * repeat
    for run in range(1, repeat + 1):
        for case in cases:
            result = run_case(graph, case, expected[case.id], run=run)
            results.append(result)
            progress = f"[{len(results)}/{total}] {case.id}"
            logger.info(
                "%s: %s (%.1f s, %d attempt(s))",
                progress,
                result.outcome,
                result.seconds,
                result.attempts,
            )
    return results


# --- report -------------------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; 0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    seconds = [(r.metrics["total_ms"] / 1000) if r.metrics else r.seconds for r in results]
    tokens = [r.metrics["input_tokens"] + r.metrics["output_tokens"] for r in results if r.metrics]
    by_tag: dict[str, list[int]] = {}
    for r in results:
        for tag in r.tags:
            counts = by_tag.setdefault(tag, [0, 0])
            counts[0] += r.correct
            counts[1] += 1
    n = len(results)
    return {
        "runs": n,
        "correct": sum(r.correct for r in results),
        "exact": sum(r.outcome == "correct" for r in results),
        "accuracy": sum(r.correct for r in results) / n if n else 0.0,
        "avg_attempts": sum(r.attempts for r in results) / n if n else 0.0,
        "p50_s": _percentile(seconds, 0.5),
        "p95_s": _percentile(seconds, 0.95),
        "avg_tokens": sum(tokens) / len(tokens) if tokens else None,
        "outcomes": dict(Counter(r.outcome for r in results)),
        "by_tag": {tag: {"correct": c, "runs": t} for tag, (c, t) in sorted(by_tag.items())},
    }


def format_report(model: str, results: list[CaseResult]) -> str:
    s = summarize_results(results)
    tokens = f"{s['avg_tokens'] / 1000:.1f}k" if s["avg_tokens"] is not None else "n/a"
    lines = [
        f"Model {model}: {s['correct']}/{s['runs']} correct ({s['accuracy']:.0%}), "
        f"{s['exact']} exact, {s['avg_attempts']:.2f} attempts, "
        f"p50 {s['p50_s']:.1f} s, p95 {s['p95_s']:.1f} s, {tokens} tokens/turn",
        "  by tag: "
        + ", ".join(f"{tag} {v['correct']}/{v['runs']}" for tag, v in s["by_tag"].items()),
    ]
    for r in results:
        mark = "ok  " if r.correct else "FAIL"
        run = f" #{r.run}" if s["runs"] > len({x.case_id for x in results}) else ""
        detail = f" ({r.error})" if r.outcome in ("sql_failed", "agent_error") and r.error else ""
        lines.append(f"  {mark} {r.case_id}{run}: {r.outcome}{detail}")
    return "\n".join(lines)


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        )
    except OSError, subprocess.SubprocessError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def write_results(
    out_dir: Path,
    model: str,
    results: list[CaseResult],
    *,
    settings: Settings,
    cases_path: Path,
    with_history: bool,
    repeat: int,
) -> Path:
    """One JSON file per model and run, named by time and model."""
    created = datetime.now()
    status = _git("status", "--porcelain")
    payload = {
        "model": model,
        "created_at": created.isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "cases_file": str(cases_path),
        "repeat": repeat,
        "with_history": with_history,
        "settings": {
            "ollama_embed_model": settings.ollama_embed_model,
            "ollama_reasoning": settings.ollama_reasoning,
            "retrieval_k": settings.retrieval_k,
            "sql_row_limit": settings.sql_row_limit,
            "sql_timeout_ms": settings.sql_timeout_ms,
            "max_sql_retries": settings.max_sql_retries,
            "chat_history_turns": settings.chat_history_turns,
            "query_history_k": settings.query_history_k if with_history else 0,
        },
        "summary": summarize_results(results),
        "results": [asdict(r) for r in results],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{created:%Y%m%d-%H%M%S}-{re.sub(r'[^A-Za-z0-9._-]+', '-', model)}"
    path = out_dir / f"{stem}.json"
    for n in itertools.count(2):  # the same model twice in one second: keep both files
        if not path.exists():
            break
        path = out_dir / f"{stem}-{n}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def compare_runs(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Accuracy change between two result files, and the cases whose outcome changed."""

    def by_case(payload: dict[str, Any]) -> dict[str, tuple[int, int, str]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in payload["results"]:
            grouped.setdefault(r["case_id"], []).append(r)
        return {
            case_id: (
                sum(r["outcome"] in CORRECT for r in runs),
                len(runs),
                Counter(r["outcome"] for r in runs).most_common(1)[0][0],
            )
            for case_id, runs in grouped.items()
        }

    def label(payload: dict[str, Any]) -> str:
        s = payload["summary"]
        return (
            f"{payload['model']} @ {payload.get('git_commit') or '?'}: "
            f"{s['correct']}/{s['runs']} ({s['accuracy']:.0%})"
        )

    before, after = by_case(old), by_case(new)
    lines = [f"old  {label(old)}", f"new  {label(new)}"]
    changed = []
    for case_id in sorted(before.keys() | after.keys()):
        if case_id not in before or case_id not in after:
            where = "new" if case_id not in before else "old"
            changed.append(f"  ? {case_id}: only in the {where} run")
            continue
        (c0, n0, o0), (c1, n1, o1) = before[case_id], after[case_id]
        if c0 / n0 != c1 / n1 or o0 != o1:
            mark = "+" if c1 / n1 > c0 / n0 else "-" if c1 / n1 < c0 / n0 else "~"
            changed.append(f"  {mark} {case_id}: {o0} ({c0}/{n0}) -> {o1} ({c1}/{n1})")
    lines += changed or ["  no case changed"]
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------------------------


def resolve_models(names: list[str], settings: Settings) -> list[ChatModelInfo]:
    try:
        available = {m.name: m for m in list_chat_models(settings)}
    except Exception as e:
        raise EvalError(f"Could not list the Ollama models at {settings.ollama_base_url}") from e
    missing = [n for n in names if n not in available]
    if missing:
        raise EvalError(
            f"Not on the Ollama server: {', '.join(missing)}. Available: {', '.join(available)}"
        )
    return [available[n] for n in names]


def eval_graph(model: ChatModelInfo, settings: Settings, *, with_history: bool, shared: dict):
    """The agent graph for one model: in-memory chat store, and no past queries unless asked."""
    s = settings if with_history else settings.model_copy(update={"query_history_k": 0})
    llm = get_chat_model(s, model=model.name, reasoning=s.ollama_reasoning and model.thinking)
    query_history = (
        shared["query_history"] if with_history else InMemoryQueryHistory(get_embeddings(s))
    )
    return build_graph(
        llm=llm,
        retriever=shared["retriever"],
        query_runner=shared["query_runner"],
        chat_store=InMemoryChatStore(),
        query_history=query_history,
        settings=s,
    )


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _run(args: argparse.Namespace) -> None:
    s = get_settings()
    cases = select_cases(load_cases(args.cases), tags=_split(args.tags), limit=args.limit)
    if not cases:
        raise EvalError("No cases selected")
    models = resolve_models(_split(args.models) or [s.ollama_chat_model], s)
    query_runner = default_query_runner(s)
    expected = reference_results(cases, query_runner, s.sql_row_limit)
    shared = {
        "retriever": get_retriever(s),
        "query_runner": query_runner,
        "query_history": get_query_history(s) if args.with_history else None,
    }
    logger.info("%d case(s), %d run(s) each, %d model(s)", len(cases), args.repeat, len(models))

    reports = []
    for model in models:
        graph = eval_graph(model, s, with_history=args.with_history, shared=shared)
        results = run_cases(graph, cases, expected, repeat=args.repeat)
        path = write_results(
            args.out,
            model.name,
            results,
            settings=s,
            cases_path=args.cases,
            with_history=args.with_history,
            repeat=args.repeat,
        )
        reports.append(f"{format_report(model.name, results)}\n  -> {path}")
    print("\n\n".join(reports))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `uv run rag-sql-eval`."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="rag-sql-eval", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the cases against one or more chat models")
    run.add_argument("--models", help="comma-separated; default OLLAMA_CHAT_MODEL")
    run.add_argument("--tags", help="comma-separated: only cases with any of these tags")
    run.add_argument("--limit", type=int, help="only the first N cases")
    run.add_argument("--repeat", type=int, default=1, help="run each case N times")
    run.add_argument("--with-history", action="store_true", help="use past queries as examples")
    run.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    run.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    diff = commands.add_parser("compare", help="compare two result files")
    diff.add_argument("old", type=Path)
    diff.add_argument("new", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            _run(args)
        else:
            old, new = (json.loads(p.read_text(encoding="utf-8")) for p in (args.old, args.new))
            print(compare_runs(old, new))
    except EvalError as e:
        parser.exit(1, f"rag-sql-eval: {e}\n")


if __name__ == "__main__":
    main()

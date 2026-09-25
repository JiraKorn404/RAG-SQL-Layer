"""Run the evaluation cases against chat models (`uv run rag-sql-eval run`, see cli.py).

Runs have no thread_id, so nothing is saved to chat or query history. Past queries are not used
unless `with_history` (then they are searched, still never saved).
"""

import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from rag_sql.agent.graph import build_graph, default_query_runner
from rag_sql.config import Settings, get_settings
from rag_sql.db.query import QueryResult, validate_sql
from rag_sql.evaluation.cases import DEFAULT_CASES_PATH, EvalCase, load_cases, select_cases
from rag_sql.evaluation.results import DEFAULT_OUT_DIR, CaseResult, format_report, write_results
from rag_sql.evaluation.scoring import Outcome, compare, is_ordered
from rag_sql.history.chat import InMemoryChatStore
from rag_sql.history.queries import InMemoryQueryHistory, get_query_history
from rag_sql.llm import ChatModelInfo, get_chat_model, get_embeddings, list_chat_models
from rag_sql.retrieval import get_retriever

logger = logging.getLogger(__name__)


class EvalError(RuntimeError):
    """The evaluation can't run: a bad reference, or a model that isn't available."""


def reference_results(
    cases: Iterable[EvalCase], run_query: Callable[[str], QueryResult], row_limit: int
) -> dict[str, QueryResult]:
    """Run each reference SQL (read-only, like the agent's). A reference must fit the row limit,
    or the comparison would only see part of it. Cases without reference SQL (no_sql) are left
    out."""
    results = {}
    for case in cases:
        if case.sql is None:
            continue
        try:
            result = run_query(validate_sql(case.sql, row_limit))
        except Exception as e:
            raise EvalError(f"Reference SQL of {case.id!r} failed: {e}") from e
        if result.truncated:
            raise EvalError(f"Reference of {case.id!r} returns more than {row_limit} rows")
        results[case.id] = result
    return results


def run_case(
    graph: CompiledStateGraph, case: EvalCase, expected: QueryResult | None, *, run: int = 1
) -> CaseResult:
    """Run one case. `expected` is the reference result; None for a no_sql case, which is correct
    only when the agent replied NO_SQL."""
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
    outcome: Outcome
    if expected is None or case.sql is None:
        outcome = "correct" if state.get("no_sql") else "unneeded_sql"
    elif state.get("no_sql"):
        outcome = "declined"
    elif result is not None:
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
            result = run_case(graph, case, expected.get(case.id), run=run)
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


def evaluate(
    models: list[str] | None = None,
    *,
    cases_path: Path = DEFAULT_CASES_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    tags: list[str] | None = None,
    limit: int | None = None,
    repeat: int = 1,
    with_history: bool = False,
    settings: Settings | None = None,
) -> str:
    """Run the selected cases against each model (default OLLAMA_CHAT_MODEL), write one result
    file per model to `out_dir`, and return the reports."""
    s = settings or get_settings()
    cases = select_cases(load_cases(cases_path), tags=tags, limit=limit)
    if not cases:
        raise EvalError("No cases selected")
    chosen = resolve_models(models or [s.ollama_chat_model], s)
    query_runner = default_query_runner(s)
    expected = reference_results(cases, query_runner, s.sql_row_limit)
    shared = {
        "retriever": get_retriever(s),
        "query_runner": query_runner,
        "query_history": get_query_history(s) if with_history else None,
    }
    logger.info("%d case(s), %d run(s) each, %d model(s)", len(cases), repeat, len(chosen))

    reports = []
    for model in chosen:
        graph = eval_graph(model, s, with_history=with_history, shared=shared)
        results = run_cases(graph, cases, expected, repeat=repeat)
        path = write_results(
            out_dir,
            model.name,
            results,
            settings=s,
            cases_path=cases_path,
            with_history=with_history,
            repeat=repeat,
        )
        reports.append(f"{format_report(model.name, results)}\n  -> {path}")
    return "\n\n".join(reports)

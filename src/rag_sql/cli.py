"""Command-line entry points ([project.scripts] in pyproject.toml). Output is ASCII only.

rag-sql-index                      rebuild the schema + few-shot index (retrieval.py)
rag-sql-history backfill           embed successful chat turns that have no example yet
                                   (also after changing OLLAMA_EMBED_MODEL)
rag-sql-history list [--all] [--sql]
                                   show stored examples (enabled only, unless --all)
rag-sql-history disable|enable ID...
                                   exclude examples from retrieval, or include them again
                                   (admin role); disabled ones are never re-added
rag-sql-metrics [--days N] [--model M]
                                   latency, tokens and success of saved turns, per model
rag-sql-eval run [--models A,B] [--tags T,U] [--limit N] [--repeat K] [--with-history]
                                   run the evaluation cases; one JSON file per model in
                                   evaluation/results/ (not committed)
rag-sql-eval compare OLD.json NEW.json
                                   accuracy change and the cases whose outcome changed
"""

import argparse
import json
import logging
from pathlib import Path

from rag_sql.config import get_settings
from rag_sql.db.connection import get_engine
from rag_sql.evaluation.cases import DEFAULT_CASES_PATH
from rag_sql.evaluation.results import DEFAULT_OUT_DIR, compare_runs
from rag_sql.evaluation.runner import EvalError, evaluate
from rag_sql.history.chat import turn_stats
from rag_sql.history.queries import get_query_history, list_examples, set_enabled
from rag_sql.metrics import format_tokens
from rag_sql.retrieval import build_index

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def index_main() -> None:
    """`uv run rag-sql-index`: rebuild the vector index from the live schema and few-shot file."""
    _setup_logging()
    count = build_index()
    logger.info("Indexed %d documents into collection %r", count, get_settings().vector_collection)


def history_main(argv: list[str] | None = None) -> None:
    """`uv run rag-sql-history <command>`: manage the query history used as SQL examples."""
    _setup_logging()
    parser = argparse.ArgumentParser(
        prog="rag-sql-history", description="Manage the query history used as SQL examples."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("backfill", help="embed successful chat turns that have no example yet")
    list_cmd = commands.add_parser("list", help="show stored examples, newest first")
    list_cmd.add_argument("--all", action="store_true", help="include disabled examples")
    list_cmd.add_argument("--sql", action="store_true", help="print each example's SQL")
    list_cmd.add_argument("--limit", type=int, default=50)
    for name in ("disable", "enable"):
        cmd = commands.add_parser(name, help=f"{name} examples by id")
        cmd.add_argument("ids", type=int, nargs="+")
    args = parser.parse_args(argv)

    s = get_settings()
    if args.command == "backfill":
        added = get_query_history(s).backfill()
        logger.info("Added %d example(s) for embedding model %r", added, s.ollama_embed_model)
    elif args.command == "list":
        examples = list_examples(
            get_engine("memory", settings=s), include_disabled=args.all, limit=args.limit
        )
        for e in examples:
            state = "on " if e.enabled else "off"
            print(
                f"{e.id:>6}  {state}  {e.row_count:>6} rows  {e.created_at:%Y-%m-%d}  {e.question}"
            )
            if args.sql:
                print("        " + e.sql.replace("\n", "\n        "))
        if not examples:
            print("No examples stored.")
    else:
        changed = set_enabled(get_engine("admin", settings=s), args.ids, args.command == "enable")
        logger.info("%sd %d of %d example(s)", args.command.capitalize(), changed, len(args.ids))


def metrics_main(argv: list[str] | None = None) -> None:
    """`uv run rag-sql-metrics`: latency, tokens and success of saved turns, per model."""
    _setup_logging()
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
            f"{s.p50_ms / 1000:>7.1f} {s.p95_ms / 1000:>7.1f} "
            f"{format_tokens(round(s.avg_tokens)):>8} {s.cold_load_rate:>10.0%}"
        )


def eval_main(argv: list[str] | None = None) -> None:
    """`uv run rag-sql-eval`: run the evaluation cases, or compare two result files."""
    _setup_logging()
    parser = argparse.ArgumentParser(
        prog="rag-sql-eval",
        description="Run the evaluation cases against chat models; report accuracy, retries, "
        "latency and tokens.",
    )
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
            print(
                evaluate(
                    _split(args.models),
                    cases_path=args.cases,
                    out_dir=args.out,
                    tags=_split(args.tags),
                    limit=args.limit,
                    repeat=args.repeat,
                    with_history=args.with_history,
                )
            )
        else:
            old, new = (json.loads(p.read_text(encoding="utf-8")) for p in (args.old, args.new))
            print(compare_runs(old, new))
    except EvalError as e:
        parser.exit(1, f"rag-sql-eval: {e}\n")

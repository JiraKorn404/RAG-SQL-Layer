"""Evaluation results: one CaseResult per case run, summaries, the text report, the JSON result
files (evaluation/results/, not committed) and the comparison of two of them."""

import itertools
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_sql.agent.prompts import prompt_version
from rag_sql.config import PROJECT_ROOT, Settings
from rag_sql.evaluation.scoring import CORRECT, EXPECTED_INTENT, Outcome
from rag_sql.metrics import TurnMetrics

DEFAULT_OUT_DIR = PROJECT_ROOT / "evaluation" / "results"


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
    expected: str = "sql"  # the kind of case: sql, chat or clarify
    intent: str | None = None  # what the router chose (None for runs before the router)
    standalone: str | None = None  # the question as the router rewrote it

    @property
    def correct(self) -> bool:
        return self.outcome in CORRECT

    @property
    def routed_right(self) -> bool | None:
        """Whether the router chose the intent the case expects; None when it recorded none."""
        if self.intent is None:
            return None
        return self.intent == EXPECTED_INTENT[self.expected]


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
    by_kind: dict[str, list[int]] = {}
    for r in results:
        for counts in [by_tag.setdefault(tag, [0, 0]) for tag in r.tags] + [
            by_kind.setdefault(r.expected, [0, 0])
        ]:
            counts[0] += r.correct
            counts[1] += 1
    routed = [r.routed_right for r in results if r.routed_right is not None]
    runs_by_case: dict[str, set[bool]] = {}
    for r in results:
        runs_by_case.setdefault(r.case_id, set()).add(r.correct)
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
        "by_kind": {kind: {"correct": c, "runs": t} for kind, (c, t) in sorted(by_kind.items())},
        "route": {"right": sum(routed), "runs": len(routed)},
        # Cases that passed in some runs and failed in others (with --repeat).
        "flaky": sum(len(outcomes) > 1 for outcomes in runs_by_case.values()),
    }


def format_report(model: str, results: list[CaseResult], *, router_only: bool = False) -> str:
    s = summarize_results(results)
    tokens = f"{s['avg_tokens'] / 1000:.1f}k" if s["avg_tokens"] is not None else "n/a"
    if router_only:
        # Only the router ran: "correct" means it chose the right intent.
        lines = [
            f"Router of {model}: {s['correct']}/{s['runs']} right intent ({s['accuracy']:.0%}), "
            f"p50 {s['p50_s']:.1f} s, p95 {s['p95_s']:.1f} s"
        ]
    else:
        lines = [
            f"Model {model}: {s['correct']}/{s['runs']} correct ({s['accuracy']:.0%}), "
            f"{s['exact']} exact, {s['avg_attempts']:.2f} attempts, "
            f"p50 {s['p50_s']:.1f} s, p95 {s['p95_s']:.1f} s, {tokens} tokens/turn"
        ]
        if s["route"]["runs"]:
            lines.append(f"  router: {s['route']['right']}/{s['route']['runs']} right intent")
    lines += [
        "  by kind: "
        + ", ".join(f"{kind} {v['correct']}/{v['runs']}" for kind, v in s["by_kind"].items()),
        "  by tag: "
        + ", ".join(f"{tag} {v['correct']}/{v['runs']}" for tag, v in s["by_tag"].items()),
    ]
    if s["flaky"]:
        lines.append(f"  flaky: {s['flaky']} case(s) passed in some runs and failed in others")
    for r in results:
        mark = "ok  " if r.correct else "FAIL"
        run = f" #{r.run}" if s["runs"] > len({x.case_id for x in results}) else ""
        detail = f" ({r.error})" if r.outcome in ("sql_failed", "agent_error") and r.error else ""
        lines.append(f"  {mark} {r.case_id}{run}: {r.outcome}{detail}")
        if not r.correct and router_only and r.intent:
            lines.append(f"       chose {r.intent}, expected {EXPECTED_INTENT[r.expected]}")
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
    router_only: bool = False,
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
        "router_only": router_only,
        "prompt_version": prompt_version(),
        "settings": {
            "ollama_router_model": settings.ollama_router_model or None,
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
        route = s.get("route") or {}
        routed = f", router {route['right']}/{route['runs']}" if route.get("runs") else ""
        return (
            f"{payload['model']} @ {payload.get('git_commit') or '?'}"
            f" prompts {payload.get('prompt_version') or '?'}: "
            f"{s['correct']}/{s['runs']} ({s['accuracy']:.0%}){routed}"
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

import json
from pathlib import Path

import pytest

from rag_sql import cli


def _result_file(path: Path, outcome: str) -> Path:
    payload = {
        "model": "qwen3:14b",
        "git_commit": "abc1234",
        "summary": {"correct": int(outcome == "correct"), "runs": 1, "accuracy": 1.0},
        "results": [{"case_id": "a", "outcome": outcome}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_eval_compare(tmp_path: Path, capsys) -> None:
    old = _result_file(tmp_path / "old.json", "correct")
    new = _result_file(tmp_path / "new.json", "wrong_result")
    cli.eval_main(["compare", str(old), str(new)])
    assert "  - a: correct (1/1) -> wrong_result (0/1)" in capsys.readouterr().out


@pytest.mark.parametrize("main", [cli.history_main, cli.metrics_main, cli.eval_main])
def test_help_needs_no_database(main, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_eval_run_passes_its_options(monkeypatch, capsys) -> None:
    seen: dict = {}

    def fake_evaluate(models, **kwargs) -> str:
        seen.update(models=models, **kwargs)
        return "report"

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    cli.eval_main(["run", "--models", "gemma4:e4b", "--router-only", "--repeat", "3"])

    assert (seen["models"], seen["router_only"], seen["repeat"]) == (["gemma4:e4b"], True, 3)
    assert capsys.readouterr().out.strip() == "report"

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

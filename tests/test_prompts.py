from rag_sql.agent import prompts


def test_prompt_version_is_stable_and_short() -> None:
    assert prompts.prompt_version() == prompts.prompt_version()
    assert len(prompts.prompt_version()) == 8


def test_prompt_version_changes_with_a_prompt(monkeypatch) -> None:
    before = prompts.prompt_version()
    monkeypatch.setattr(prompts, "SQL_HISTORY", prompts.SQL_HISTORY + "More.")
    assert prompts.prompt_version() != before

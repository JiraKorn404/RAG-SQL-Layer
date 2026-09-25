import re

import pytest

from rag_sql.agent.graph import build_graph
from rag_sql.ui import notebook
from tests.conftest import fake_llm


@pytest.fixture
def shown(monkeypatch) -> list:
    """What the notebook would display, instead of displaying it."""
    outputs: list = []
    monkeypatch.setattr(notebook, "display", outputs.append)
    return outputs


def _text(output) -> str:
    return output.data if hasattr(output, "data") else output.to_string()


def test_run_and_display_renders_each_step(
    shown, settings, retriever, ok_runner, stores, chat_store
) -> None:
    graph = build_graph(
        llm=fake_llm("```sql\nSELECT emp_name, salary FROM employees\n```", "Costs $5 and $6."),
        retriever=retriever,
        query_runner=ok_runner,
        **stores,
        settings=settings,
    )
    notebook.run_and_display("Who earns the most?", graph, thread_id="t1")

    texts = [_text(o) for o in shown]
    assert texts[0] == "**Question:** Who earns the most?"
    assert "Thread <code>t1</code>" in texts[1]
    assert any("Context: tables employees" in t for t in texts)
    assert any(t.startswith("### SQL\n```sql\nSELECT") for t in texts)
    assert any(t.startswith("### SQL output\n1 row(s)") for t in texts)
    assert any("Employee_1" in t for t in texts)  # the result table
    assert texts[-3] == r"### Answer" + "\n" + r"Costs \$5 and \$6."
    # Timing: summary line, with the per-node table inside.
    assert re.search(r"<summary>\d+\.\d s · 1 attempt</summary>", texts[-2])
    assert "<td>generate_sql</td>" in texts[-2]
    assert "Saved to query history" in texts[-1]
    assert [t["question"] for t in chat_store.load("t1")] == ["Who earns the most?"]


def test_error_step_is_escaped(shown) -> None:
    notebook.render_step(notebook.Step("error", "<b>boom</b>", label="Execution error"))
    [output] = shown
    assert "&lt;b&gt;boom&lt;/b&gt;" in output.data

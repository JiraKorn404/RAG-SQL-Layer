from langchain_core.messages import AIMessage
from sqlalchemy.exc import ProgrammingError

from rag_sql.agent import nodes
from rag_sql.db.query import QueryResult
from tests.conftest import EXAMPLE_DOC, TABLE_DOC, fake_llm


def test_extract_sql_prefers_sql_block() -> None:
    content = "Here you go:\n```sql\nSELECT 1\n```\nand\n```sql\nSELECT 2\n```"
    assert nodes.extract_sql(content) == "SELECT 2"


def test_extract_sql_falls_back_to_any_block_then_raw() -> None:
    assert nodes.extract_sql("```\nSELECT 3\n```") == "SELECT 3"
    assert nodes.extract_sql("  SELECT 4  ") == "SELECT 4"


def test_split_reasoning_from_additional_kwargs() -> None:
    msg = AIMessage(
        content="```sql\nSELECT 1\n```", additional_kwargs={"reasoning_content": " hmm "}
    )
    assert nodes.split_reasoning(msg) == ("hmm", "```sql\nSELECT 1\n```")


def test_split_reasoning_from_think_tags() -> None:
    msg = AIMessage(content="<think>plan it</think>\nSELECT 1")
    assert nodes.split_reasoning(msg) == ("plan it", "SELECT 1")


def test_split_reasoning_none() -> None:
    assert nodes.split_reasoning(AIMessage(content="SELECT 1")) == (None, "SELECT 1")


def test_format_schema_and_examples_split_by_kind() -> None:
    context = [TABLE_DOC, EXAMPLE_DOC]
    assert nodes.format_schema(context) == TABLE_DOC.page_content
    assert "ORDER BY salary DESC" in nodes.format_examples(context)
    assert nodes.format_examples([TABLE_DOC]) == "(none)"


def test_format_rows_markdown_table() -> None:
    result = QueryResult(columns=["a", "b"], rows=[(1, None), (2, "x|y")])
    assert nodes.format_rows(result) == "| a | b |\n|---|---|\n| 1 | NULL |\n| 2 | x\\|y |"
    assert nodes.format_rows(QueryResult(columns=["a"])) == "(no rows)"


def test_generate_sql_increments_attempts_and_clears_error() -> None:
    llm = fake_llm(
        AIMessage(
            content="```sql\nSELECT emp_name FROM employees\n```",
            additional_kwargs={"reasoning_content": "Use employees."},
        )
    )
    update = nodes.generate_sql(
        {"question": "q", "context": [TABLE_DOC], "attempts": 1, "error": "boom", "sql": "x"},
        llm=llm,
        row_limit=10,
    )
    assert update == {
        "reasoning": "Use employees.",
        "sql": "SELECT emp_name FROM employees",
        "error": None,
        "result": None,
        "attempts": 2,
    }


def test_validate_sql_node() -> None:
    ok = nodes.validate_sql({"sql": "SELECT 1"}, row_limit=5)
    assert ok["error"] is None and "LIMIT 5" in ok["sql"]
    bad = nodes.validate_sql({"sql": "DELETE FROM employees"}, row_limit=5)
    assert "Only SELECT" in bad["error"]


def test_execute_sql_node_reports_db_error() -> None:
    def failing(_sql: str) -> QueryResult:
        raise ProgrammingError("SELECT nope", {}, Exception('column "nope" does not exist\nLINE 1'))

    update = nodes.execute_sql({"sql": "SELECT nope"}, run_query=failing)
    assert update == {"result": None, "error": 'column "nope" does not exist'}


def test_answer_without_result_skips_llm() -> None:
    update = nodes.answer({"question": "q", "attempts": 3, "error": "bad"}, llm=fake_llm())
    assert "3 attempt(s)" in update["answer"] and "bad" in update["answer"]


def test_answer_uses_llm() -> None:
    state = {
        "question": "Top earner?",
        "sql": "SELECT 1",
        "result": QueryResult(columns=["emp_name"], rows=[("Employee_1",)]),
    }
    assert nodes.answer(state, llm=fake_llm("Employee_1 earns the most.")) == {
        "answer": "Employee_1 earns the most."
    }

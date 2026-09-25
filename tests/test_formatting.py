from langchain_core.messages import AIMessage

from rag_sql.agent import formatting
from rag_sql.db.query import QueryResult
from rag_sql.history.queries import PastQuery
from tests.conftest import EXAMPLE_DOC, TABLE_DOC, make_turn


def test_extract_sql_prefers_sql_block() -> None:
    content = "Here you go:\n```sql\nSELECT 1\n```\nand\n```sql\nSELECT 2\n```"
    assert formatting.extract_sql(content) == "SELECT 2"


def test_extract_sql_falls_back_to_any_block_then_raw() -> None:
    assert formatting.extract_sql("```\nSELECT 3\n```") == "SELECT 3"
    assert formatting.extract_sql("  SELECT 4  ") == "SELECT 4"


def test_parse_route_reads_the_decision() -> None:
    reply = '{"intent": "clarify", "standalone_question": "Best?", "unclear": "No measure."}'
    decision = formatting.parse_route(reply, "best")
    assert (decision.intent, decision.standalone_question, decision.unclear) == (
        "clarify",
        "Best?",
        "No measure.",
    )


def test_parse_route_finds_json_around_other_text() -> None:
    reply = 'Sure:\n```json\n{"intent": "chat", "standalone_question": "hi"}\n```'
    decision = formatting.parse_route(reply, "hi")
    assert (decision.intent, decision.unclear) == ("chat", "")


def test_parse_route_falls_back_to_a_data_question() -> None:
    for reply in ("", "SELECT 1", '{"intent": "unknown"}', '{"standalone_question": "x"}'):
        decision = formatting.parse_route(reply, "How many?")
        assert (decision.intent, decision.standalone_question) == ("data", "How many?"), reply


def test_format_table_names() -> None:
    assert formatting.format_table_names([TABLE_DOC, EXAMPLE_DOC]) == "employees"
    assert formatting.format_table_names([]) == "(none found)"


def test_split_reasoning_from_additional_kwargs() -> None:
    msg = AIMessage(
        content="```sql\nSELECT 1\n```", additional_kwargs={"reasoning_content": " hmm "}
    )
    assert formatting.split_reasoning(msg) == ("hmm", "```sql\nSELECT 1\n```")


def test_split_reasoning_from_think_tags() -> None:
    msg = AIMessage(content="<think>plan it</think>\nSELECT 1")
    assert formatting.split_reasoning(msg) == ("plan it", "SELECT 1")


def test_split_reasoning_none() -> None:
    assert formatting.split_reasoning(AIMessage(content="SELECT 1")) == (None, "SELECT 1")


def test_format_schema_and_examples_split_by_kind() -> None:
    context = [TABLE_DOC, EXAMPLE_DOC]
    assert formatting.format_schema(context) == TABLE_DOC.page_content
    assert "ORDER BY salary DESC" in formatting.format_examples(context)
    assert formatting.format_examples([TABLE_DOC]) == "(none)"


def test_format_rows_markdown_table() -> None:
    result = QueryResult(columns=["a", "b"], rows=[(1, None), (2, "x|y")])
    assert formatting.format_rows(result) == "| a | b |\n|---|---|\n| 1 | NULL |\n| 2 | x\\|y |"
    assert formatting.format_rows(QueryResult(columns=["a"])) == "(no rows)"


def test_format_history_sql_and_answers() -> None:
    history = [make_turn("q1", sql="SELECT 1", answer="a1"), make_turn("q2", sql=None)]
    with_sql = formatting.format_history(history, sql=True, answers=False)
    assert with_sql == (
        "Question: q1\n```sql\nSELECT 1\n```\n\nQuestion: q2\n(no working SQL; error: boom)"
    )
    assert formatting.format_history(history[:1]) == "Question: q1\nAnswer: a1"


def test_format_history_no_sql_turn() -> None:
    turn = make_turn("hello?", sql=None, answer="Hi!")
    turn["error"] = None  # a chat message, or a question that was asked back
    text = formatting.format_history([turn], sql=True, answers=False)
    assert text == "Question: hello?\n(no SQL was written for this message)"


def test_format_history_truncates_long_answers() -> None:
    text = formatting.format_history([make_turn("q", answer="x" * 1000)])
    assert text.endswith(" …")
    assert len(text) < formatting.HISTORY_ANSWER_CHARS + 50


def test_format_similar_queries() -> None:
    queries = [PastQuery("q1", "SELECT 1", 0.9), PastQuery("q2", "SELECT 2", 0.8)]
    assert formatting.format_similar_queries(queries) == (
        "Question: q1\n```sql\nSELECT 1\n```\n\nQuestion: q2\n```sql\nSELECT 2\n```"
    )

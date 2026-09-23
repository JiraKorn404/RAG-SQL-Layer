"""All prompt templates. Edit prompts here only."""

from langchain_core.prompts import ChatPromptTemplate

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert PostgreSQL analyst. Write one SQL query that answers the user's "
            "question using only the tables and columns described below.\n\n"
            "Rules:\n"
            "- Write a single read-only SELECT statement (WITH ... SELECT is fine). Never modify "
            "data or schema.\n"
            "- Use only tables and columns that appear in the schema. Do not invent names.\n"
            "- Match text values exactly as described in the column comments.\n"
            "- When grouping or ranking by a column that can be NULL, exclude NULL values unless "
            "the question is about missing or unknown data.\n"
            "- Give computed columns clear aliases and order results in a meaningful way.\n"
            "- Return at most {row_limit} rows.\n"
            "- Reply with the query in a single ```sql code block and nothing else.",
        ),
        (
            "human",
            "Database schema:\n{schema}\n\n"
            "Example questions with correct SQL:\n{examples}\n\n"
            "{feedback}"
            "Question: {question}",
        ),
    ]
)

# Filled into {feedback} when a previous attempt failed validation or execution.
SQL_RETRY_FEEDBACK = (
    "Your previous query failed.\n"
    "Previous query:\n```sql\n{sql}\n```\n"
    "Error: {error}\n"
    "Fix the problem and write a corrected query.\n\n"
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a database. Use only the SQL result provided. "
            "Be concise and state the key numbers. If the result is empty, say that no matching "
            "data was found. If the result was truncated, mention that only part of it is shown. "
            "Do not repeat the SQL.",
        ),
        (
            "human",
            "Question: {question}\n\nSQL:\n```sql\n{sql}\n```\n\nResult ({row_summary}):\n{rows}",
        ),
    ]
)

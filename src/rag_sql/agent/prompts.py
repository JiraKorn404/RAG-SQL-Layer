"""All prompt templates. Edit prompts here only."""

from langchain_core.prompts import ChatPromptTemplate

# The whole reply of the SQL model when the question can't be answered from the database.
NO_SQL = "NO_SQL"

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite follow-up questions about a database so they can be understood without "
            "the conversation.\n\n"
            "Rules:\n"
            '- Use the conversation to resolve references such as "it", "they", "that '
            'department", "the same" or "and the lowest?".\n'
            "- Keep every filter, grouping and number from the follow-up. Carry over conditions "
            "from earlier questions only when the follow-up clearly continues them.\n"
            "- If the follow-up already stands on its own, return it unchanged.\n"
            "- Reply with the rewritten question only: no explanation, no SQL, no quotes.",
        ),
        (
            "human",
            "Conversation so far (oldest first):\n{history}\n\nFollow-up question: {question}",
        ),
    ]
)

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
            "- Write table names exactly as the schema shows them, including any schema prefix "
            "(e.g. imba.orders).\n"
            "- Match text values exactly as described in the column comments.\n"
            "- When grouping or ranking by a column that can be NULL, exclude NULL values unless "
            "the question is about missing or unknown data.\n"
            "- Give computed columns clear aliases and order results in a meaningful way.\n"
            "- If earlier questions from the conversation are shown and the new question builds "
            "on one, start from its SQL.\n"
            "- Return at most {row_limit} rows.\n"
            "- Reply with the query in a single ```sql code block and nothing else.\n"
            "- Exception: if the question is not about the data at all (a greeting, small talk, "
            "a question about you, or general knowledge the database doesn't hold), don't write "
            f"a query: reply with exactly {NO_SQL} and nothing else. Never write a placeholder "
            "query that answers a different question.",
        ),
        (
            "human",
            "Database schema:\n{schema}\n\n"
            "Example questions with correct SQL:\n{examples}\n\n"
            "{similar}"
            "{history}"
            "{feedback}"
            "Question: {question}",
        ),
    ]
)

# Filled into {similar} of SQL_GENERATION_PROMPT when similar past queries were found.
SIMILAR_QUERIES = (
    "Similar questions answered earlier, with SQL that ran successfully "
    "(check it fits this question before reusing it):\n{queries}\n\n"
)

# Filled into {history} of SQL_GENERATION_PROMPT when the conversation has earlier turns.
SQL_HISTORY = "Earlier questions in this conversation and their SQL (oldest first):\n{turns}\n\n"

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
            "If the result doesn't answer the question, say that you couldn't answer it from the "
            "database instead of reporting the result. Do not repeat the SQL.",
        ),
        (
            "human",
            "{history}Question: {question}\n\nSQL:\n```sql\n{sql}\n```\n\n"
            "Result ({row_summary}):\n{rows}",
        ),
    ]
)

# Filled into {history} of ANSWER_PROMPT and CHAT_REPLY_PROMPT when the conversation has earlier
# turns.
ANSWER_HISTORY = "Earlier in this conversation (oldest first):\n{turns}\n\n"

# Used by the answer node instead of ANSWER_PROMPT when the SQL model replied NO_SQL.
CHAT_REPLY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the assistant of a question-answering app over a PostgreSQL database: you "
            "answer questions about its data by writing and running SQL. The user's message is "
            "not a question about the data.\n\n"
            "Reply in two or three sentences: respond to a greeting or a question about you, say "
            "that you answer questions about the data in the tables below, and suggest one or two "
            "example questions about them. Never state facts or numbers about the data.",
        ),
        ("human", "Tables: {tables}\n\n{history}Message: {question}"),
    ]
)

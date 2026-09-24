# CLAUDE.md

This file guides Claude Code when working in this repository.

## Project overview

**RAG-SQL-Layer** is a RAG-powered text-to-SQL agent. A user asks a question in natural language, the agent retrieves the relevant schema context, writes a SQL query, runs it against a self-hosted PostgreSQL database, and answers from the query result.

The main interface is a Jupyter notebook. Running a cell streams each step of the agent so the user sees:

1. **Thinking**: the model's reasoning.
2. **SQL statement**: the generated query, highlighted.
3. **SQL output**: the result rows, as a table.
4. **Answer**: the final natural-language answer.

The design priority is **modularity**. Each concern (config, LLM, database, retrieval, agent, display) lives in its own module behind a small interface, so a model, prompt, database, or display can be swapped without touching the rest.

## Tech stack

| Concern | Choice |
|---|---|
| Language / packaging | Python 3.14, managed with **uv** (`pyproject.toml`, `.python-version`) |
| Relational DB | Self-hosted PostgreSQL 17 + pgvector, via `docker-compose.yml` |
| LLM + embeddings | Self-hosted **Ollama** on a separate machine, reached over **Tailscale**, via `langchain_ollama` (`ChatOllama`, `OllamaEmbeddings`) |
| Orchestration | **LangGraph** (`StateGraph`) with LangChain core primitives |
| Vector store | pgvector in the same Postgres instance (`langchain_postgres.PGVector`) |
| DB access | SQLAlchemy 2.x + psycopg 3 |
| SQL validation | `sqlglot` (parse, and enforce read-only statements) |
| Config | `pydantic-settings`, loaded from `.env` |
| Output | Jupyter (`notebooks/`), rendered with `IPython.display` + pandas |

## Directory layout

```
RAG-SQL-Layer/
├── CLAUDE.md
├── README.md
├── pyproject.toml              # deps + [project.scripts] rag-sql-index = "rag_sql.retrieval:main"
├── docker-compose.yml          # Postgres + pgvector
├── .env.example                # all config keys, no secrets
├── db/
│   └── init/                   # SQL run in filename order on first container start:
│                               #   01-extensions.sql, 02-roles.sh, 03-employees.sql,
│                               #   04-chat-memory.sh (chat history schema + role)
├── src/rag_sql/
│   ├── __init__.py
│   ├── config.py               # Settings (pydantic-settings); the ONLY place env vars are read
│   ├── llm.py                  # get_chat_model(), get_embeddings() factories
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py       # get_engine("reader"|"admin"|"memory") -> SQLAlchemy engine
│   │   ├── introspect.py       # read tables/columns/FKs/comments -> TableDoc objects
│   │   └── query.py            # validate_sql() (sqlglot, pure) + run_query() -> QueryResult
│   ├── retrieval.py            # build Documents, (re)build pgvector index, get_retriever(), main() CLI
│   ├── memory.py               # chat history: Turn, ChatStore (in-memory + Postgres), get_chat_store()
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py            # AgentState TypedDict (shared by nodes, graph, display)
│   │   ├── prompts.py          # all prompt templates (no prompts elsewhere)
│   │   ├── nodes.py            # one function per graph node
│   │   └── graph.py            # build_graph() -> compiled LangGraph
│   └── display.py              # run_and_display(), Chat, show_threads(): notebook rendering
├── examples/
│   └── few_shot.yaml           # question -> SQL pairs used for retrieval
├── notebooks/
│   └── demo.ipynb              # main entry point
└── tests/
```

Code lives under `src/rag_sql/` (src layout). The notebook imports from the package and holds no business logic.

Structure rules:
- A subpackage exists only when it holds more than one real module. Otherwise use a single module (`llm.py`, `display.py`, `retrieval.py`).
- Merge files only when they share one concern and change together. Keep `agent/state.py` separate (shared contract, avoids circular imports) and `agent/prompts.py` separate (most-edited file).
- `db/introspect.py` (admin, used by indexing) and `db/query.py` (read-only, used by the agent) stay apart, so the agent's code path never touches admin credentials.
- `memory.py` is the only module that touches chat history, and only through the "memory" engine.
- Split `retrieval.py` back into `retrieval/indexer.py` + `retrieval/retriever.py` if it grows past about 250 lines.

## Agent graph (LangGraph)

```
START
  └─> load_history       # ChatStore: last CHAT_HISTORY_TURNS turns of the thread (if thread_id set)
  └─> condense_question  # LLM: follow-up + history -> standalone question (no call on turn 1)
  └─> retrieve_context   # retriever (on the standalone question): tables/columns + few-shot examples
  └─> generate_sql       # LLM: question + context + earlier Q/SQL (+ previous error) -> SQL, with reasoning
  └─> validate_sql       # db/query.py validate_sql(): parse, reject non-read-only, add LIMIT if missing
        ├─ invalid ──> generate_sql   (while attempts < MAX_SQL_RETRIES)
  └─> execute_sql        # db/query.py run_query()
        ├─ DB error ─> generate_sql   (while attempts < MAX_SQL_RETRIES)
  └─> answer             # LLM: question + SQL + rows (+ earlier Q/A) -> natural-language answer
  └─> save_turn          # ChatStore: append this turn (also failed ones); a failed save is only logged
END
```

`AgentState` (in `agent/state.py`) carries: `thread_id`, `history` (earlier `Turn`s), `question`, `standalone_question`, `context` (retrieved docs), `reasoning`, `sql`, `error`, `attempts`, `result` (columns + rows), `answer`.

Chat history: every run is one turn. History is loaded and saved only when the input has a `thread_id`; without one, `history` passed in the input is used and the new turn is returned in state, so a stateless caller can carry it. Stored turns hold question, standalone question, SQL, row count, answer and error, never result rows.

Rules:
- A node is a pure function `(state) -> partial state update`. Its dependencies (LLM, engine, retriever) are injected when the graph is built in `build_graph(...)`, never created at import time.
- Routing logic lives in small `route_*` functions in `graph.py`, not inside nodes.
- Every node writes its output to state, so the display layer can render it from `stream_mode="updates"`.

## Notebook output contract

`display.py` exposes `run_and_display(question, graph=None, *, thread_id=None)` (no `thread_id` = new thread). It iterates `graph.stream(..., stream_mode="updates")` and renders each node's update as it arrives:

| Node update | Rendered as |
|---|---|
| `load_history.history` | Muted line: thread id + number of earlier questions in context |
| `condense_question.standalone_question` | Muted "Interpreted as: ..." line, only when it differs from the question |
| `generate_sql.reasoning` | Collapsible / muted "Thinking" block (Markdown) |
| `generate_sql.sql` | Header "SQL" + syntax-highlighted ```sql block |
| `validate_sql` / `execute_sql` error | Red warning showing the error and retry count |
| `execute_sql.result` | Header "SQL output" + `pandas.DataFrame` (truncated to display limit) |
| `answer.answer` | Header "Answer" + Markdown |

Thinking comes from Ollama reasoning models via `ChatOllama(reasoning=True)` (read from `message.additional_kwargs["reasoning_content"]`). If the model doesn't produce reasoning, skip the block. Don't fail.

A notebook cell should be as simple as:

```python
from rag_sql.display import run_and_display

run_and_display("Top 5 customers by revenue last quarter?")
```

For follow-up questions, use a `Chat` (one thread): `chat = Chat()`, then `chat.ask(...)`. `Chat("<thread id>")` continues a saved conversation; `show_threads()` lists them and `chat.show_history()` shows one.

Keep rendering separate from agent logic. The graph must also be usable without Jupyter (scripts, tests, a future API).

## Configuration

All settings are read in `src/rag_sql/config.py` through a single `Settings` class (`get_settings()` cached). No module reads `os.environ` directly. Every key goes into `.env.example`.

| Key | Purpose | Example |
|---|---|---|
| `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_DB` | DB location | `localhost` / `5432` / `rag` |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | Admin user (docker, indexing) | |
| `APP_DB_USER` / `APP_DB_PASSWORD` | **Read-only** role the agent queries with | `rag_reader` |
| `CHAT_DB_USER` / `CHAT_DB_PASSWORD` | Chat history role (`SELECT`, `INSERT` on `chat_memory.chat_turns` only) | `rag_memory` |
| `OLLAMA_BASE_URL` | Ollama over Tailscale | `http://<tailscale-host>:11434` |
| `OLLAMA_CHAT_MODEL` | Chat / SQL model | e.g. `qwen3:14b` |
| `OLLAMA_EMBED_MODEL` | Embedding model | e.g. `nomic-embed-text` |
| `OLLAMA_REASONING` | Enable thinking output | `true` |
| `SQL_ROW_LIMIT` | Max rows returned / auto `LIMIT` | `200` |
| `SQL_TIMEOUT_MS` | `statement_timeout` for agent queries | `15000` |
| `MAX_SQL_RETRIES` | Regenerate attempts on invalid/failed SQL | `3` |
| `VECTOR_COLLECTION` | pgvector collection name | `schema_docs` |
| `RETRIEVAL_K` | Docs retrieved per question | `6` |
| `CHAT_HISTORY_TURNS` | Earlier turns shown to the model (0 = none; turns are still saved) | `5` |

## Safety rules for SQL execution (must hold)

- The agent connects **only** as the read-only role (`APP_DB_USER`), which has only `SELECT` on the target schemas and `default_transaction_read_only = on`. It is created by `db/init/02-roles.sh`, which is safe to re-run. On an existing volume, run `docker compose exec postgres bash /docker-entrypoint-initdb.d/02-roles.sh` (from Git Bash, prefix it with `MSYS_NO_PATHCONV=1`).
- `validate_sql()` in `db/query.py` rejects anything that isn't a single `SELECT` / `WITH ... SELECT` statement (use sqlglot, not regex), and adds a `LIMIT` when missing.
- Every query runs inside a read-only transaction with `SET LOCAL statement_timeout`.
- Never interpolate user text into SQL outside the LLM-generated statement. Never run LLM output as the admin user.
- Chat history lives in its own schema, `chat_memory`, written only by the chat role (`CHAT_DB_USER`) through fixed, parameterized statements in `memory.py`. That role has `SELECT` + `INSERT` on `chat_memory.chat_turns` and nothing else (history is append-only and kept forever). `APP_DB_USER` has no access to `chat_memory`, so generated SQL can't read past conversations. Never run LLM output as the chat role. Created by `db/init/04-chat-memory.sh`, which is safe to re-run; on an existing volume run it like `02-roles.sh` above.

## Ollama over Tailscale

- Ollama runs on another device. On that machine set `OLLAMA_HOST=0.0.0.0` so it listens on the Tailscale interface.
- `OLLAMA_BASE_URL` uses the device's Tailscale MagicDNS name or `100.x.y.z` IP.
- `llm.py` is the only module that builds `ChatOllama` / `OllamaEmbeddings`. To switch providers, change this module and nothing else.
- Quick connectivity check: `curl $OLLAMA_BASE_URL/api/tags`.

## Commands

```sh
# Environment
uv sync                                   # install deps
uv add <pkg>                              # add a dependency (don't hand-edit versions)

# Database
docker compose up -d                      # start Postgres + pgvector
docker compose ps                         # wait for "healthy"
docker compose down                       # stop (data kept); add -v to wipe data

# Build / refresh the RAG index (run after schema or few-shot changes)
uv run rag-sql-index

# Notebook
uv run jupyter lab notebooks/demo.ipynb

# Quality
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Coding conventions

- Type hints everywhere. Use `TypedDict` for graph state and dataclasses/pydantic for value objects (`TableDoc`, `QueryResult`).
- Factories (`get_chat_model`, `get_engine`, `get_retriever`, `get_chat_store`, `build_graph`) are the seams for swapping implementations. Accept overrides as arguments so tests can inject fakes.
- All prompts live in `agent/prompts.py` as `ChatPromptTemplate`s. Don't put inline prompt strings in nodes.
- No side effects at import time: no network or DB connections when a module loads.
- Use `logging`, not `print`, in library code. Only `display.py` writes to notebook output.
- Keep the notebook thin: imports plus `run_and_display(...)` / `Chat` calls.

## Testing

- Unit-test `validate_sql()`, routing functions, and nodes with a fake LLM (`langchain_core.language_models.fake_chat_models.FakeListChatModel`) and no network. Use `InMemoryChatStore` for chat history.
- Integration tests that need Postgres or Ollama are marked `@pytest.mark.integration` and skipped by default.

## When changing things

- **New model**: change `OLLAMA_CHAT_MODEL` in `.env`. No code change.
- **New node / step**: add the function in `nodes.py`, the state fields in `state.py`, wire it in `graph.py`, and add rendering in `display.py`.
- **Schema changed or new few-shot examples**: re-run `uv run rag-sql-index`.
- **New config key**: add it to `Settings` and `.env.example` together.
- **Chat history table changed**: edit `db/init/04-chat-memory.sh` (keep it idempotent) and the statements in `memory.py`, then re-run the script.

# CLAUDE.md

This file guides Claude Code when working in this repository.

## Project overview

**RAG-SQL-Layer** is a RAG-powered text-to-SQL agent. A user asks a question in natural language, the agent retrieves the relevant schema context, writes a SQL query, runs it against a self-hosted PostgreSQL database, and answers from the query result.

There are two interfaces: a Jupyter notebook and a Streamlit web chat (run in Docker). Both stream each step of the agent so the user sees:

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
| CI | GitHub Actions (`.github/workflows/ci.yml`): ruff, stripped notebooks, unit tests |
| Tracing | Self-hosted **Langfuse** v4 (`docker-compose.langfuse.yml`), through its LangChain callback handler (`tracing.py`); off unless `LANGFUSE_ENABLED` |
| Metrics / evaluation | Per-node time and tokens in `chat_memory.turn_metrics` (`metrics.py`); scored test questions (`rag_sql/evaluation/`, `evaluation/cases.yaml`) |
| Config | `pydantic-settings`, loaded from `.env` |
| Output | Jupyter (`notebooks/`), rendered with `IPython.display` + pandas (optional extra `notebook`) |
| Web UI | **Streamlit** (`ui/web.py`, optional extra `web`), served by the compose service `web` |

## Directory layout

```
RAG-SQL-Layer/
├── pyproject.toml              # deps (+ extras "web", "notebook"); [project.scripts] -> rag_sql.cli
├── docker-compose.yml          # Postgres + pgvector, and the Streamlit web UI (service "web")
├── docker-compose.langfuse.yml # Langfuse (tracing): its own compose project and databases
├── Dockerfile                  # image of the web UI (.dockerignore keeps .env and data out)
├── .env.example                # all config keys, no secrets
├── .github/workflows/ci.yml    # CI: ruff check + format, nbstripout --verify, pytest
├── db/init/                    # run in filename order on first container start: 01-extensions,
│                               #   02-roles (read-only role), 03-employees, 04-chat-memory
│                               #   (chat_memory tables + role), 05-imba (Instacart -> schema imba),
│                               #   06-retail (retail data warehouse -> schema retail)
├── data/structured/            # CSVs loaded by db/init (instacart/ is large: local only, ignored)
├── examples/few_shot.yaml      # curated question -> SQL pairs, indexed for retrieval
├── evaluation/
│   ├── cases.yaml              # eval cases: question + reference SQL (+ history for follow-ups)
│   └── results/                # rag-sql-eval output, one JSON per model and run (local only, ignored)
├── notebooks/demo.ipynb        # main entry point (demo2.ipynb, ...: local scratch, ignored)
├── src/rag_sql/
│   ├── config.py               # Settings (pydantic-settings); the ONLY place env vars are read
│   ├── llm.py                  # get_chat_model(), get_embeddings() (caches query vectors), list_chat_models()
│   ├── metrics.py              # NodeMetric/TurnMetrics, usage_of(), summarize(), format_* (pure)
│   ├── retrieval.py            # schema + few-shot Documents, build_index(), get_retriever()
│   ├── tracing.py              # run_config(): Langfuse callback + trace attributes for one run
│   ├── cli.py                  # every CLI: rag-sql-index, -history, -metrics, -eval (argparse, printing)
│   ├── db/
│   │   ├── connection.py       # get_engine("reader"|"admin"|"memory"), one per role
│   │   ├── introspect.py       # tables/columns/FKs/comments -> TableDoc (admin, indexing only)
│   │   └── query.py            # validate_sql() (sqlglot, pure), run_query() -> QueryResult, is_database_unavailable()
│   ├── history/                # everything that touches the chat_memory schema
│   │   ├── chat.py             # Turn, ChatStore (in-memory + Postgres), get_chat_store(), turn_stats()
│   │   └── queries.py          # query examples (thumbs up, pgvector): QueryHistory, get_query_history()
│   ├── agent/
│   │   ├── state.py            # AgentState TypedDict (shared by nodes, graph, UI)
│   │   ├── prompts.py          # all prompt templates (no prompts elsewhere)
│   │   ├── formatting.py       # state -> prompt text (schema, rows, history), model reply -> thinking + SQL
│   │   ├── nodes.py            # one function per graph node (+ turn_from_state())
│   │   └── graph.py            # build_graph() -> compiled LangGraph; timed() wraps every node
│   ├── ui/
│   │   ├── steps.py            # Step, steps_from_update()/steps_from_turn(), example_candidate(): no UI code
│   │   ├── notebook.py         # Jupyter: run_and_display(), Chat, show_threads()
│   │   ├── web.py              # Streamlit app: pages, sidebar (history, model picker), thumbs up, session state
│   │   ├── web_chat.py         # Streamlit chat: render_step(), run_turn() with live thinking
│   │   └── web_examples.py     # Streamlit "Query examples" page: list, search, hide
│   └── evaluation/
│       ├── cases.py            # EvalCase, load_cases() (checked: ids, no few-shot overlap)
│       ├── scoring.py          # compare() agent result vs reference result (pure)
│       ├── runner.py           # run cases per model: reference_results(), run_case(), evaluate()
│       └── results.py          # CaseResult, summary, report, JSON result files, compare_runs()
└── tests/                      # one test_<module>.py per module; conftest.py has the fakes
```

Code lives under `src/rag_sql/` (src layout). The notebook imports from the package and holds no business logic.

Structure rules:
- A subpackage exists only when it holds more than one real module. Otherwise use a single module (`llm.py`, `retrieval.py`, `cli.py`).
- `ui/steps.py` decides what each node update or saved turn shows, and which turns can be saved as examples (`example_candidate()`); `ui/notebook.py` and `ui/web_chat.py` only draw `Step`s. Keep UI calls out of `steps.py`, and node-specific logic out of the front ends.
- Merge files only when they share one concern and change together. Keep `agent/state.py` separate (shared contract, avoids circular imports) and `agent/prompts.py` separate (most-edited file).
- `db/introspect.py` (admin, used by indexing) and `db/query.py` (read-only, used by the agent) stay apart, so the agent's code path never touches admin credentials.
- `history/chat.py` (chat turns, their metrics, `turn_stats()`) and `history/queries.py` (past queries) are the only modules that touch `chat_memory`, and only through the "memory" engine (the admin engine only for `rag-sql-history disable|enable`).
- `metrics.py` is pure (types, summaries); storing metrics is `history/chat.py`'s job. `evaluation/scoring.py` is pure too; everything that runs models or queries is in `evaluation/runner.py`, and reporting is in `evaluation/results.py`.
- `tracing.py` is the only module that imports `langfuse`. Nodes and `graph.py` know nothing of tracing: the callers of `graph.stream()` / `invoke()` pass `config=run_config(...)`.
- CLIs live in `cli.py` only: argument parsing and printing there, the work in the library modules (no `main()` elsewhere).
- Split `retrieval.py` back into `retrieval/indexer.py` + `retrieval/retriever.py` if it grows past about 250 lines.

## Agent graph (LangGraph)

```
START
  └─> load_history       # ChatStore: last CHAT_HISTORY_TURNS turns of the thread (if thread_id set)
  └─> condense_question  # LLM: follow-up + history -> standalone question (no call on turn 1)
  └─> retrieve_context   # retriever (on the standalone question): tables/columns + few-shot examples
  └─> find_similar_queries  # QueryHistory: up to QUERY_HISTORY_K similar query examples (thumbs up)
  └─> generate_sql       # LLM: question + context + similar queries + earlier Q/SQL (+ previous error) -> SQL
        ├─ NO_SQL ───> answer         (not a question about the data: nothing is validated or run)
  └─> validate_sql       # db/query.py validate_sql(): parse, reject non-read-only, cap LIMIT at row limit + 1
        ├─ invalid ──> generate_sql   (while attempts < MAX_SQL_RETRIES)
  └─> execute_sql        # db/query.py run_query()
        ├─ DB error ─> generate_sql   (while attempts < MAX_SQL_RETRIES; never when the database is unavailable)
  └─> answer             # LLM: question + SQL + rows (+ earlier Q/A) -> natural-language answer
                         #   (after NO_SQL: CHAT_REPLY_PROMPT, question + table names -> short reply)
  └─> save_turn          # ChatStore: append this turn (also failed ones); a failed save is only logged
END
```

Every node is wrapped by `timed()` in `graph.py`: its update also carries one `NodeMetric` (node, SQL attempt, wall ms, and for model calls Ollama's model ms, load ms and tokens). Nodes that call the model return `llm_usage` (`metrics.usage_of(message)`); the wrapper moves it into the metric, so it never reaches state.

`AgentState` (in `agent/state.py`) carries: `thread_id`, `history` (earlier `Turn`s), `question`, `standalone_question`, `context` (retrieved docs), `similar_queries` (`PastQuery`s), `reasoning`, `sql`, `no_sql` (the model replied `NO_SQL`: the question isn't about the data, so `sql` is None and nothing runs), `error`, `db_unavailable` (the error is a connection/availability problem, so it isn't retried), `attempts`, `result` (columns + rows), `answer`, `answer_reasoning`, `turn_id`, `metrics` (`NodeMetric`s; the reducer appends, so every SQL attempt keeps its entries) and `turn_metrics` (their summary, set by `save_turn`).

Chat history: every run is one turn. History is loaded and saved only when the input has a `thread_id`; without one, `history` passed in the input is used and the new turn is returned in state, so a stateless caller can carry it. Stored turns hold question, standalone question, SQL, row count, answer, error, the model's thinking (`sql_reasoning` of the last SQL attempt, `answer_reasoning`) the chat `model` that answered (injected into `save_turn` by `build_graph` from the LLM's name) and the turn's `metrics` (one row per node run in `chat_memory.turn_metrics`, written in the same transaction; `save_turn` is left out, so the time ends with the answer), never result rows. Loaded turns carry their row `id`. Thinking is never put into prompts.

Query history: the graph never adds to it. When a user gives a reply a thumbs up in the web UI (only for turns whose SQL ran without error and returned at least one row, `steps.example_candidate()`), its standalone question is embedded (`OLLAMA_EMBED_MODEL`) and stored with the SQL in `chat_memory.query_examples`. The web UI's Query examples page can hide one (`enabled = false`, for every embedding model; rows are never deleted). Before generating SQL, the agent searches it by cosine similarity (only rows of the current embedding model, only `enabled` ones), keeps matches at or above `QUERY_HISTORY_MIN_SIMILARITY`, one per question, drops ones that duplicate a curated few-shot example, and gives up to `QUERY_HISTORY_K` to the model as a separate prompt section. Retrieved SQL is only an example: it is never executed directly. Identical (question, SQL) pairs are stored once; hidden (disabled) pairs are never re-added. A failed search or save never blocks an answer.

Rules:
- A node is a pure function `(state) -> partial state update`. Its dependencies (LLM, engine, retriever) are injected when the graph is built in `build_graph(...)`, never created at import time.
- Routing logic lives in small `route_*` functions in `graph.py`, not inside nodes.
- Every node writes its output to state, so the display layer can render it from `stream_mode="updates"`.

## Notebook output contract

`ui/notebook.py` exposes `run_and_display(question, graph=None, *, thread_id=None)` (no `thread_id` = new thread). It shows the question and the thread id, then iterates `graph.stream(..., stream_mode="updates")`, turns each node's update into `Step`s with `steps.steps_from_update()` (shared with the web UI) and draws them as they arrive:

| Node update | Rendered as |
|---|---|
| `load_history.history` | Muted line: number of earlier questions in context; skipped when none |
| `condense_question.standalone_question` | Muted "Interpreted as: ..." line, only when it differs from the question |
| `find_similar_queries.similar_queries` | Collapsible muted "Similar past queries (n)": question, similarity, SQL; skipped when none |
| `generate_sql.reasoning` | Collapsible / muted "Thinking" block (Markdown) |
| `generate_sql.sql` | Header "SQL" + syntax-highlighted ```sql block |
| `generate_sql.no_sql` | Muted "No SQL: not a question about the data." line instead of the SQL (also for saved turns with no SQL and no error) |
| `validate_sql` / `execute_sql` error | Red warning showing the error and retry count (or "database unavailable, not retried") |
| `execute_sql.result` | Header "SQL output" + `pandas.DataFrame` (truncated to display limit) |
| `answer.answer_reasoning` | Collapsible "Thinking (answer)" block; skipped when none |
| `answer.answer` | Header "Answer" + Markdown (`$` kept literal, not LaTeX) |
| `save_turn.turn_metrics` | Collapsible muted summary ("12.3 s · 2 attempts · 1.8k tokens · model load 4.1 s") with a per-node table |

The notebook has no thumbs up: query examples are saved from the web UI.

Thinking comes from Ollama reasoning models via `ChatOllama(reasoning=True)` (read from `message.additional_kwargs["reasoning_content"]`). If the model doesn't produce reasoning, skip the block. Don't fail.

A notebook cell should be as simple as:

```python
from rag_sql.ui.notebook import run_and_display

run_and_display("Top 5 customers by revenue last quarter?")
```

For follow-up questions, use a `Chat` (one thread): `chat = Chat()`, then `chat.ask(...)`. `Chat("<thread id>")` continues a saved conversation; `show_threads()` lists them and `chat.show_history()` shows one.

Keep rendering separate from agent logic. The graph must also be usable without Jupyter (scripts, tests, a future API).

## Web UI contract

`ui/web.py` is a Streamlit app (`streamlit run src/rag_sql/ui/web.py`; `main(graph_for=, store=, settings=, list_models=, query_history=)` accepts fakes) with two pages (`st.navigation`, in the sidebar): **Chat** (the default) and **Query examples** (`ui/web_examples.py`). On the Chat page each reply is drawn and streamed by `ui/web_chat.py`. It renders the same node updates as the notebook table above, in chat bubbles, with these differences:

- It streams with `stream_mode=["updates", "messages"]`. Tokens of the `generate_sql` and `answer` model calls arrive in "messages" mode (tagged with `metadata["langgraph_node"]`), so the thinking (`additional_kwargs["reasoning_content"]`) and the answer text appear live. Nodes don't change for this: they call `llm.invoke()`, and LangGraph streams it. When a run fails mid-call, the thinking streamed so far is kept.
- Node updates become `Step`s through `steps.steps_from_update()`, the same as in the notebook. Steps are kept in `st.session_state`, so reruns redraw the chat without calling the agent.
- The sidebar lists saved conversations (`ChatStore.threads()`) and has "New chat". Each is one left-aligned line (`wrap=False`, plus a small CSS rule scoped to the `chat-history` container; recheck it after a Streamlit upgrade). Opening one loads its turns (`steps.steps_from_turn()`: with the saved thinking, but no result rows, since they aren't stored). The open thread is in the URL (`?thread=<id>`).
- The sidebar has a **Model** picker. `llm.list_chat_models()` lists the Ollama models whose capabilities (`/api/show`) include `completion` and not `embedding`, cached 60 s. It defaults to `OLLAMA_CHAT_MODEL`, the choice is kept per browser session, and models without the `thinking` capability get `reasoning=False` (Ollama rejects it for them). If Ollama can't be listed, only the configured model is used, with a warning. The embedding model isn't selectable (changing it needs a re-index).
- Each reply starts with a "Model `<name>`" caption, live and for saved turns, and ends with an expander titled with the timing summary (per-node table inside), for saved turns too.
- One graph per model, built on demand (`st.cache_resource` keyed by model name). The retriever, query runner, chat store and query history are built once and shared by all of them, so a model doesn't open its own DB pools. Everything is shared by all browser sessions.
- A new question sent while a run is going stops that run. What it had shown so far is kept. A run that doesn't reach `save_turn` (stopped by a new question, or failed with an exception, e.g. Ollama unreachable) is still saved, through `run_turn(save_unfinished=...)`, as a turn whose error says why (`INTERRUPTED` or the exception), with what it produced so far, metrics included (`run_turn` adds up the per-node `metrics` of the updates, like the graph's reducer). So a reloaded conversation, and the next follow-up, match what was shown.
- A reply whose SQL ran and returned rows ends with a **Good answer** (thumbs up) button, live (from the turn `run_turn()` returns) and for saved turns. It saves the standalone question and SQL as a query example (`QueryHistory.add`, linked to the turn id). Each redraw reads the status of the shown replies in one query (`QueryHistory.status`): a saved pair shows "Saved as an example" instead, and a hidden one says it can't be saved again. So a pair hidden on the Query examples page shows as hidden in the chat too.
- The **Query examples** page lists the examples that aren't hidden (`QueryHistory.examples()`: question and SQL, newest first, one per pair), with a search box (every word must appear in the question or SQL) and a Hide button per example, confirmed inline. Anyone who can open the web UI can hide examples; the admin can bring one back with `rag-sql-history enable`.

## Tracing (Langfuse)

`tracing.run_config(source, *, session_id=, model=, tags=)` returns the `RunnableConfig` of one run: `{}` when `LANGFUSE_ENABLED` is off, else a new Langfuse `CallbackHandler` (one per run) and trace attributes in `metadata` (`langfuse_trace_name`, `langfuse_session_id`, `langfuse_tags`). LangGraph passes the callbacks down to every node and to the `llm.invoke()` / `retriever.invoke()` calls in them, so one trace holds a span per node (every SQL attempt), the retrieved documents and each model call (prompt, reply, `reasoning_content`, tokens). Nodes need no change for it, as for streaming.

| Caller | Source tag | Session | Other tags |
|---|---|---|---|
| `ui/notebook.py` `run_and_display` | `notebook` | thread id | `OLLAMA_CHAT_MODEL` (default graph only) |
| `ui/web.py` -> `web_chat.run_turn(config=)` | `web` | thread id | chosen model |
| `evaluation/runner.py` `run_case(config=)` | `eval` | `eval-<timestamp>`, one per `rag-sql-eval run` | model, case id |

- The Langfuse client is built on the first traced run, with the keys from `Settings` (cached). Traces are sent in the background and flushed at exit. A client that can't be built is logged and the run goes on untraced; an unreachable server only loses traces.
- A run stopped mid-way (a new question in the web UI) ends its trace with the root span at level `ERROR`.
- The `langfuse` LangChain integration needs the `langchain` package (it checks its version), hence that dependency.
- Langfuse v4 serves observations at `/api/public/v2/observations` (the v1 trace endpoints return 404 in its events-only mode).
- `chat_memory.turn_metrics` stays the source of the UI timing summary and `rag-sql-metrics`; Langfuse is for inspecting runs.

## Configuration

All settings are read in `src/rag_sql/config.py` through a single `Settings` class (`get_settings()` cached). No module reads `os.environ` directly. Every key goes into `.env.example`. Numeric keys are range-checked (`Field(gt=..., ge=..., le=...)`), so a bad value fails at startup.

| Key | Purpose | Example |
|---|---|---|
| `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_DB` | DB location | `localhost` / `5432` / `rag` |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | Admin user (docker, indexing) | |
| `APP_DB_USER` / `APP_DB_PASSWORD` | **Read-only** role the agent queries with | `rag_reader` |
| `CHAT_DB_USER` / `CHAT_DB_PASSWORD` | Chat history, query history and metrics role (`SELECT`, `INSERT` on `chat_memory` tables, `UPDATE` of `query_examples.enabled`) | `rag_memory` |
| `OLLAMA_BASE_URL` | Ollama over Tailscale | `http://<tailscale-host>:11434` |
| `OLLAMA_CHAT_MODEL` | Chat / SQL model (the web UI's default choice) | e.g. `qwen3:14b` |
| `OLLAMA_EMBED_MODEL` | Embedding model | e.g. `nomic-embed-text` |
| `OLLAMA_REASONING` | Enable thinking output | `true` |
| `SQL_ROW_LIMIT` | Max rows returned (queries get `LIMIT` of this + 1, to detect truncation) | `200` |
| `SQL_TIMEOUT_MS` | `statement_timeout` for agent queries | `15000` |
| `MAX_SQL_RETRIES` | Regenerate attempts on invalid/failed SQL | `3` |
| `VECTOR_COLLECTION` | pgvector collection name | `schema_docs` |
| `RETRIEVAL_K` | Docs retrieved per question | `6` |
| `DB_SCHEMAS` | Schemas indexed and shown to the model (comma-separated; never `chat_memory`) | `public,imba,retail` |
| `CHAT_HISTORY_TURNS` | Earlier turns shown to the model (0 = none; turns are still saved) | `5` |
| `QUERY_HISTORY_K` | Most similar past queries given to the model (0 = none; queries are still saved) | `5` |
| `QUERY_HISTORY_MIN_SIMILARITY` | Cosine similarity cutoff for past queries (tune per embedding model) | `0.75` |
| `WEB_PORT` | Host port of the web UI, bound to `127.0.0.1` (docker compose only; not in `Settings`) | `8501` |
| `LANGFUSE_ENABLED` | Trace runs to Langfuse (needs both keys) | `false` |
| `LANGFUSE_BASE_URL` | Langfuse server (the web container uses `host.docker.internal`) | `http://localhost:3000` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Project keys; the server creates the project with them | `pk-lf-...` / `sk-lf-...` |
| `LANGFUSE_PORT`, `LANGFUSE_INIT_USER_EMAIL` / `_PASSWORD`, `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY`, `LANGFUSE_DB_PASSWORD`, `LANGFUSE_CLICKHOUSE_PASSWORD`, `LANGFUSE_REDIS_PASSWORD`, `LANGFUSE_MINIO_PASSWORD` | Langfuse server only (`docker-compose.langfuse.yml`; not in `Settings`); hex secrets | `openssl rand -hex 32` |

## Safety rules for SQL execution (must hold)

- The agent connects **only** as the read-only role (`APP_DB_USER`), which has only `SELECT` on the target schemas and `default_transaction_read_only = on`. It is created by `db/init/02-roles.sh`, which is safe to re-run. On an existing volume, run `docker compose exec postgres bash /docker-entrypoint-initdb.d/02-roles.sh` (from Git Bash, prefix it with `MSYS_NO_PATHCONV=1`).
- `validate_sql()` in `db/query.py` rejects anything that isn't a single `SELECT` / `WITH ... SELECT` statement (use sqlglot, not regex), and functions with side effects or that run a SQL string (`query_to_xml`, `dblink`, ...). It gives the outer query `LIMIT SQL_ROW_LIMIT + 1` when it has none, or a larger or non-literal one; the extra row tells `run_query` that the result was truncated.
- Every query runs inside a read-only transaction with `SET LOCAL statement_timeout`, through a server-side cursor (`stream_results`): at most `SQL_ROW_LIMIT + 1` rows are ever fetched, and only queries can run (`DECLARE ... CURSOR FOR` rejects anything else).
- Every engine has a `connect_timeout` (`db/connection.py`), so an unreachable database fails in seconds. `execute_sql` sets `db_unavailable` for connection/availability errors (`is_database_unavailable()`), and the graph answers right away instead of asking the model for new SQL.
- Never interpolate user text into SQL outside the LLM-generated statement. Never run LLM output as the admin user.
- `rag-sql-eval` runs the reference SQL of `evaluation/cases.yaml` through `validate_sql()` and `run_query()` as the read-only role, like the agent's SQL. Eval runs have no `thread_id`, so they save nothing to chat or query history.
- Chat history lives in its own schema, `chat_memory`, written only by the chat role (`CHAT_DB_USER`) through fixed, parameterized statements in `history/chat.py` and `history/queries.py`. That role has `SELECT` + `INSERT` on `chat_memory.chat_turns`, `chat_memory.query_examples` and `chat_memory.turn_metrics`, plus `UPDATE` of `query_examples.enabled` only (hiding an example from the web UI), and nothing else: history is append-only and kept forever, and no row is ever deleted. The admin can also disable or enable examples. `APP_DB_USER` has no access to `chat_memory`, so generated SQL can't read past conversations. Never run LLM output as the chat role. Created by `db/init/04-chat-memory.sh`, which is safe to re-run; on an existing volume run it like `02-roles.sh` above.
- The web container gets `.env` through compose with `POSTGRES_USER` / `POSTGRES_PASSWORD` blanked, so it never holds admin credentials; `.dockerignore` keeps `.env` out of the image. Its port is published on `127.0.0.1` only (Streamlit has no login, and the sidebar shows every saved conversation).
- Langfuse traces hold questions, SQL, result rows and thinking in full. Langfuse runs as its own compose project with its own Postgres, never the agent's database or roles; only its UI is published, on `127.0.0.1`, with sign-up disabled and telemetry off. Tracing never blocks an answer, and tests never trace (`tests/conftest.py` forces `LANGFUSE_ENABLED=false`).

## Ollama over Tailscale

- Ollama runs on another device. On that machine set `OLLAMA_HOST=0.0.0.0` so it listens on the Tailscale interface.
- `OLLAMA_BASE_URL` uses the device's Tailscale MagicDNS name or `100.x.y.z` IP.
- `llm.py` is the only module that builds `ChatOllama` / `OllamaEmbeddings`. To switch providers, change this module and nothing else.
- Quick connectivity check: `curl $OLLAMA_BASE_URL/api/tags`.

## Commands

```sh
# Environment
uv sync --extra web                       # install deps (+ Streamlit, needed by the web tests)
uv run nbstripout --install               # once per clone: commit notebooks without outputs
uv add <pkg>                              # add a dependency (don't hand-edit versions)

# Database
docker compose up -d                      # start Postgres + pgvector
docker compose ps                         # wait for "healthy"
docker compose down                       # stop (data kept); add -v to wipe data

# Build / refresh the RAG index (run after schema or few-shot changes)
uv run rag-sql-index

# Query history (examples saved with a thumbs up in the web UI; listed and hidden on its Query examples page)
uv run rag-sql-history backfill          # embed the stored examples for the current embed model (after changing it)
uv run rag-sql-history list [--all] [--sql]
uv run rag-sql-history disable <id>...   # exclude bad examples (admin); `enable` to undo

# Metrics of saved turns: latency, tokens, success, cold model loads per model
uv run rag-sql-metrics [--days 30] [--model M]

# Evaluation (needs Postgres + Ollama; results in evaluation/results/, local only)
uv run rag-sql-eval run [--models A,B] [--tags T] [--limit N] [--repeat K] [--with-history]
uv run rag-sql-eval compare evaluation/results/OLD.json evaluation/results/NEW.json

# Notebook
uv run jupyter lab notebooks/demo.ipynb

# Web UI (http://localhost:8501)
docker compose up -d --build              # starts Postgres + web; --build after code changes
uv sync --extra web && uv run streamlit run src/rag_sql/ui/web.py   # local dev, without Docker

# Tracing (Langfuse UI: http://localhost:3000; then LANGFUSE_ENABLED=true)
docker compose -f docker-compose.langfuse.yml up -d

# Quality (CI runs the same: .github/workflows/ci.yml)
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Coding conventions

- Type hints everywhere. Use `TypedDict` for graph state and dataclasses/pydantic for value objects (`TableDoc`, `QueryResult`).
- Factories (`get_chat_model`, `get_engine`, `get_retriever`, `get_chat_store`, `get_query_history`, `build_graph`) are the seams for swapping implementations. Accept overrides as arguments so tests can inject fakes.
- All prompts live in `agent/prompts.py` as `ChatPromptTemplate`s. Don't put inline prompt strings in nodes.
- No side effects at import time: no network or DB connections when a module loads.
- Use `logging`, not `print`, in library code. Only `ui/notebook.py` (Jupyter) and `ui/web.py` / `ui/web_chat.py` / `ui/web_examples.py` (Streamlit) produce UI output; `cli.py` prints the CLI results. CLI output is ASCII only (the Windows console may use a code page such as cp874).
- Keep the notebook thin: imports plus `run_and_display(...)` / `Chat` calls.

## Testing

- Unit-test `validate_sql()`, routing functions, and nodes with a fake LLM (`langchain_core.language_models.fake_chat_models.FakeListChatModel`) and no network. Use `InMemoryChatStore` for chat history and `InMemoryQueryHistory` with `tests.conftest.KeywordEmbeddings` for query history.
- Integration tests that need Postgres or Ollama are marked `@pytest.mark.integration` and skipped by default (and in CI).
- Give fakes usage with `AIMessage(usage_metadata=..., response_metadata={"total_duration": ...})` to test metrics; `tests/test_graph.py::OllamaLikeModel` streams like ChatOllama (usage on the last chunk).
- `tests/test_eval_*.py` cover cases, scoring, the runner and the results without network, `tests/test_cli.py` the CLIs; the references run in `tests/test_integration.py`.
- `tests/test_tracing.py` checks `run_config()` with a fake client and handler, and that a run's callbacks reach the nodes' model calls. An autouse fixture in `conftest.py` keeps tracing off, since code that falls back to `get_settings()` reads the developer's `.env`.
- Test what a node shows in `tests/test_steps.py` (pure); `tests/test_notebook.py` and `tests/test_web.py` only check the drawing.

## When changing things

- **New model**: pull it on the Ollama machine; the web UI lists it within a minute. To make it the default (and the notebook's model), change `OLLAMA_CHAT_MODEL` in `.env`. No code change. Compare it first: `uv run rag-sql-eval run --models old,new`.
- **Prompt, retrieval or agent change**: run `uv run rag-sql-eval run` before and after, then `rag-sql-eval compare` the two result files.
- **New eval case**: add it to `evaluation/cases.yaml` (answer independent of label spelling, reference columns only what the answer needs, fewer rows than `SQL_ROW_LIMIT`; a question that isn't about the data gets `no_sql: true` instead of `sql`, and scores "unneeded_sql" if the agent writes SQL, while a data question answered with `NO_SQL` scores "declined"); `uv run pytest -m integration -k eval_references` checks the references.
- **New node / step**: add the function in `nodes.py`, the state fields in `state.py`, wire it in `graph.py`, and map its update to `Step`s in `steps.steps_from_update()`. Touch `render_step()` in `ui/notebook.py` and `ui/web_chat.py` only for a new `StepKind`.
- **Schema changed or new few-shot examples**: re-run `uv run rag-sql-index`. Query examples that no longer fit the schema can be hidden on the web UI's Query examples page (or `uv run rag-sql-history disable <id>`).
- **New embedding model** (`OLLAMA_EMBED_MODEL`): re-run `uv run rag-sql-index` and `uv run rag-sql-history backfill`, then re-tune `QUERY_HISTORY_MIN_SIMILARITY`.
- **New config key**: add it to `Settings` and `.env.example` together.
- **Chat history, query history or metrics table changed**: edit `db/init/04-chat-memory.sh` (keep it idempotent) and the statements in `history/chat.py` / `history/queries.py`, then re-run the script.

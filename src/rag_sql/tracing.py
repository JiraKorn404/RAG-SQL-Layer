"""Langfuse tracing of agent runs. The only module that knows Langfuse.

`run_config()` is the config for one `graph.stream()` / `graph.invoke()`. With tracing enabled
(LANGFUSE_ENABLED), it holds a Langfuse callback handler, so the run becomes one trace: a span per
node (every SQL attempt), the retriever's documents, and each model call with its prompt, reply,
thinking and tokens. Nodes need nothing for it: LangGraph passes the callbacks down to the
`llm.invoke()` calls in them, as it does for streaming. With tracing off, the config is empty.

Traces are sent in the background (and flushed when the process exits). Tracing never blocks a
run: when the client can't be built, the run goes on untraced.
"""

import logging
from collections.abc import Iterable
from functools import lru_cache
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from rag_sql.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Name of the trace, and of its root span (the graph run), in Langfuse.
TRACE_NAME = "rag-sql"

# Where a run comes from; a tag on its trace.
Source = Literal["web", "notebook", "eval"]


def run_config(
    source: Source,
    *,
    session_id: str | None = None,
    model: str | None = None,
    tags: Iterable[str] = (),
    settings: Settings | None = None,
) -> RunnableConfig:
    """The config of one agent run: traced to Langfuse when enabled, else empty.

    `session_id` groups traces in Langfuse (a chat thread, or an eval run). `source`, `model` and
    `tags` become tags of the trace.
    """
    s = settings or get_settings()
    if not s.langfuse_enabled:
        return {}
    try:
        _client(s)
        # One handler per run: it keeps the state of the run it traces. It finds the client by
        # its public key.
        handler = CallbackHandler(public_key=s.langfuse_public_key)
    except Exception:
        logger.warning("Langfuse tracing unavailable; running untraced", exc_info=True)
        return {}
    metadata: dict[str, object] = {
        "langfuse_trace_name": TRACE_NAME,
        "langfuse_tags": [source, *([model] if model else []), *tags],
    }
    if session_id:
        metadata["langfuse_session_id"] = session_id
    return RunnableConfig(callbacks=[handler], run_name=TRACE_NAME, metadata=metadata)


def _client(s: Settings) -> Langfuse:
    return _cached_client(
        s.langfuse_base_url, s.langfuse_public_key, s.langfuse_secret_key.get_secret_value()
    )


@lru_cache
def _cached_client(base_url: str, public_key: str, secret_key: str) -> Langfuse:
    # Built on the first traced run, not at import.
    return Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)

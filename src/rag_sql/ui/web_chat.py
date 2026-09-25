"""Streamlit chat drawing: saved steps (`render_step`) and a live agent run (`run_turn`), which
streams the model's thinking and answer as they arrive. The app itself (page, sidebar, sessions)
is web.py.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import streamlit as st
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph

from rag_sql.agent.nodes import turn_from_state
from rag_sql.history.chat import Turn
from rag_sql.ui.steps import Step, escape_md, steps_from_update, thinking_label

logger = logging.getLogger(__name__)

# Nodes whose model calls are streamed token by token. condense_question is short and not shown.
STREAMED_NODES = ("generate_sql", "answer")

# Seconds between redraws of streamed text; redrawing on every token slows the page down.
REDRAW_INTERVAL = 0.1

# Saved as the error of a turn that a new question stopped before it finished.
INTERRUPTED = "Interrupted: a new question was sent before this one finished."


def render_step(step: Step) -> None:
    match step.kind:
        case "caption":
            st.caption(escape_md(step.text))
        case "interpreted":
            st.caption(f"Interpreted as: *{escape_md(step.text)}*")
        case "model":
            st.caption(f"Model `{step.text}`")
        case "similar":
            with st.expander(step.label, icon=":material/history:"):
                for q in step.data:
                    st.markdown(f"{escape_md(q.question)} · similarity {q.similarity:.2f}")
                    st.code(q.sql, language="sql")
        case "thinking":
            with st.expander(step.label, icon=":material/psychology:"):
                st.markdown(escape_md(step.text))
        case "sql":
            st.markdown(f"**{step.label}**")
            st.code(step.text, language="sql")
        case "error":
            st.error(f"**{step.label}**  \n{escape_md(step.text)}", icon=":material/error:")
        case "result":
            st.markdown("**SQL output**")
            st.caption(step.text)
            st.dataframe(step.data, hide_index=True)
        case "answer":
            st.markdown("**Answer**")
            st.markdown(escape_md(step.text))
        case "metrics":
            with st.expander(step.text, icon=":material/timer:"):
                st.dataframe(step.data, hide_index=True)


class LiveCall:
    """Live widgets for one streamed model call: its thinking and, for the answer, its text."""

    def __init__(self, node: str, attempt: int) -> None:
        self.node = node
        self.label = thinking_label(node, attempt)
        self.reasoning = ""
        self.content = ""
        self._status: Any = None  # created on the first thinking token
        self._thinking: Any = None
        self._answer: Any = None  # created on the first answer token
        self._drawn_at = 0.0

    def add(self, message: BaseMessage) -> None:
        self.reasoning += message.additional_kwargs.get("reasoning_content") or ""
        if self.node == "answer":
            self.content += message.text
        if time.monotonic() - self._drawn_at >= REDRAW_INTERVAL:
            self._draw()

    def _draw(self) -> None:
        if self.reasoning:
            if self._status is None:
                self._status = st.status(f"{self.label}…", expanded=True)
                self._thinking = self._status.empty()
            self._thinking.markdown(escape_md(self.reasoning))
        if self.content:
            if self._answer is None:
                st.markdown("**Answer**")
                self._answer = st.empty()
            self._answer.markdown(escape_md(self.content))
        self._drawn_at = time.monotonic()

    def streamed_steps(self) -> list[Step]:
        """What was streamed so far, as steps. Makes no Streamlit calls."""
        steps = []
        if self.reasoning:
            steps.append(Step("thinking", self.reasoning.strip(), label=self.label))
        if self.content:
            steps.append(Step("answer", self.content.strip()))
        return steps

    def finish(self, steps: list[Step], *, failed: bool = False) -> tuple[list[Step], list[Step]]:
        """Close the live widgets once the node's update arrived.

        Returns (steps to keep, steps still to render): the thinking and answer were already
        shown live. When the steps have no thinking (the run failed), the streamed one is kept.
        """
        self._draw()
        if self._status is not None:
            self._status.update(
                label=self.label, state="error" if failed else "complete", expanded=False
            )
        keep, pending = [], []
        if self.reasoning and not any(s.kind == "thinking" for s in steps):
            keep.append(Step("thinking", self.reasoning.strip(), label=self.label))
        for step in steps:
            keep.append(step)
            if step.kind == "thinking" and self._status is not None:
                continue
            if step.kind == "answer" and self._answer is not None:
                self._answer.markdown(escape_md(step.text))
                continue
            pending.append(step)
        return keep, pending


def unfinished_turn(state: dict[str, Any], error: str, live: LiveCall | None) -> Turn:
    """The turn for a run that stopped before save_turn, with what it produced so far."""
    turn = turn_from_state(state)
    turn["error"] = error
    if live is not None and live.node == "generate_sql" and live.reasoning:
        turn["sql_reasoning"] = live.reasoning.strip()
    if live is not None and live.node == "answer":
        turn["answer"] = live.content.strip()
        turn["answer_reasoning"] = live.reasoning.strip() or None
    return turn


def run_turn(
    graph: CompiledStateGraph,
    question: str,
    thread_id: str,
    steps: list[Step],
    max_retries: int,
    *,
    save_unfinished: Callable[[Turn], None] | None = None,
) -> None:
    """Run the agent on `question`, rendering as it goes and appending to `steps`.

    `steps` is filled in place, so a run cut short by a new question keeps what it showed. A run
    that doesn't reach save_turn, because it failed or a new question stopped it, is passed to
    `save_unfinished` with its error, so the saved conversation matches what was shown.
    """
    attempts = 0
    live: LiveCall | None = None
    state: dict[str, Any] = {"question": question, "thread_id": thread_id}
    saved = False
    # Streamlit stops a run for a new question by raising a BaseException in it (RerunException),
    # which `except Exception` doesn't catch, so the error stays INTERRUPTED then.
    error = INTERRUPTED
    try:
        for mode, payload in graph.stream(
            {"question": question, "thread_id": thread_id}, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                message, metadata = payload
                node = metadata.get("langgraph_node")
                if node in STREAMED_NODES:
                    live = live or LiveCall(node, attempts + 1)
                    live.add(message)
                continue
            for node, update in payload.items():
                if not update:
                    continue
                # Updates are per node: add up the metrics like the graph's reducer does.
                state.update({k: v for k, v in update.items() if k != "metrics"})
                state["metrics"] = [*state.get("metrics", []), *update.get("metrics", [])]
                saved = saved or node == "save_turn"
                attempts = update.get("attempts", attempts)
                new = steps_from_update(
                    node, update, question=question, attempts=attempts, max_retries=max_retries
                )
                pending = new
                if live is not None and live.node == node:
                    new, pending = live.finish(new)
                    live = None
                for step in pending:
                    render_step(step)
                steps.extend(new)
    except Exception as e:
        logger.exception("Agent run failed")
        error = str(e) or type(e).__name__
        if live is not None:
            kept, _ = live.finish([], failed=True)
            steps.extend(kept)
        failed = Step("error", error, label="The agent failed")
        render_step(failed)
        steps.append(failed)
    finally:
        if not saved:
            if error == INTERRUPTED:
                # No Streamlit calls while the run is being stopped: the next run draws these.
                if live is not None:
                    steps.extend(live.streamed_steps())
                steps.append(Step("error", INTERRUPTED, label="Interrupted"))
            if save_unfinished is not None:
                try:
                    save_unfinished(unfinished_turn(state, error, live))
                except Exception:
                    logger.exception("Could not save the unfinished turn (thread %s)", thread_id)

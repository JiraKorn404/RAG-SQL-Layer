"""Streamlit page of the query examples: the questions + SQL users marked as good in the chat.

The agent is shown the most similar ones when it writes SQL. Hiding one here stops that (it is
kept in the database with `enabled = false`, and a thumbs up can't save it again); the admin can
bring it back with `uv run rag-sql-history enable`. The app itself is web.py.
"""

import logging

import streamlit as st

from rag_sql.history.queries import Example, QueryHistory
from rag_sql.ui.steps import escape_md

logger = logging.getLogger(__name__)

# Session key of the example whose Hide is waiting for confirmation.
PENDING_HIDE = "examples-pending-hide"


def filter_examples(examples: list[Example], search: str) -> list[Example]:
    """The examples whose question or SQL contains every word of `search`, ignoring case."""
    words = search.casefold().split()
    return [e for e in examples if all(w in f"{e.question}\n{e.sql}".casefold() for w in words)]


def _ask_to_hide(example_id: int | None) -> None:
    st.session_state[PENDING_HIDE] = example_id


def _hide(query_history: QueryHistory, example: Example) -> None:
    st.session_state[PENDING_HIDE] = None
    try:
        query_history.hide(example.question, example.sql)
    except Exception:
        logger.exception("Could not hide query example %s", example.id)
        st.toast("Couldn't hide the example.", icon=":material/error:")


def _example_card(query_history: QueryHistory, example: Example) -> None:
    with st.container(border=True, key=f"example-{example.id}"):
        st.markdown(f"**{escape_md(example.question)}**")
        st.code(example.sql, language="sql")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"Saved {example.created_at:%Y-%m-%d %H:%M}")
            if st.session_state.get(PENDING_HIDE) != example.id:
                st.button(
                    "Hide",
                    key=f"hide-{example.id}",
                    icon=":material/visibility_off:",
                    type="tertiary",
                    on_click=_ask_to_hide,
                    args=(example.id,),
                )
                return
            st.caption("The agent won't be shown it again, and it can't be saved again.")
            st.button(
                "Hide",
                key=f"confirm-hide-{example.id}",
                type="primary",
                on_click=_hide,
                args=(query_history, example),
            )
            st.button(
                "Cancel", key=f"cancel-hide-{example.id}", on_click=_ask_to_hide, args=(None,)
            )


def examples_page(query_history: QueryHistory) -> None:
    st.title("Query examples")
    st.caption(
        "Questions and SQL marked as good in the chat. When the agent writes SQL, it is shown "
        "the most similar ones."
    )
    try:
        examples = query_history.examples()
    except Exception:
        logger.exception("Could not load the query examples")
        st.error("Couldn't load the query examples.", icon=":material/error:")
        return
    if not examples:
        st.info("No examples yet. Give a good answer a thumbs up in the chat to add one.")
        return

    search = st.text_input(
        "Search", placeholder="Words in the question or SQL", icon=":material/search:"
    )
    shown = filter_examples(examples, search)
    st.caption(f"{len(shown)} of {len(examples)} example(s)")
    for example in shown:
        _example_card(query_history, example)

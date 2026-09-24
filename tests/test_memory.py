from rag_sql.memory import InMemoryChatStore, new_thread_id
from tests.conftest import make_turn


def test_in_memory_store_load_order_and_limit() -> None:
    store = InMemoryChatStore()
    for q in ("a", "b", "c"):
        store.append("t1", make_turn(q))
    assert [t["question"] for t in store.load("t1")] == ["a", "b", "c"]
    assert [t["question"] for t in store.load("t1", 2)] == ["b", "c"]
    assert store.load("t1", 0) == []
    assert store.load("missing") == []


def test_in_memory_store_returns_copies() -> None:
    store = InMemoryChatStore()
    turn = make_turn("a")
    store.append("t1", turn)
    turn["answer"] = "changed"
    store.load("t1")[0]["answer"] = "changed too"
    assert store.load("t1")[0]["answer"] == "A."


def test_in_memory_store_threads_most_recent_first() -> None:
    store = InMemoryChatStore()
    store.append("old", make_turn("first old"))
    store.append("new", make_turn("first new"))
    store.append("old", make_turn("second old"))
    threads = store.threads()
    assert [(t.thread_id, t.turns, t.first_question) for t in threads] == [
        ("old", 2, "first old"),
        ("new", 1, "first new"),
    ]
    assert len(store.threads(limit=1)) == 1


def test_new_thread_ids_are_unique() -> None:
    assert len({new_thread_id() for _ in range(100)}) == 100

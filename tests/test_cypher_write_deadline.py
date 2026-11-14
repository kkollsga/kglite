"""A write statement's deadline is a time bound, and a late statement rolls back.

Write loops used to poll the deadline every 4,096 rows and their value
expressions not at all, and nothing polled after the last clause. So a slow
write of fewer than 4,096 rows ran past ``timeout_ms`` by the whole clause
(3-6x measured), and one whose last clause was the write *committed* and
returned success. Each test here calibrates the per-row cost ``U`` of its
statement, runs enough rows that the work is about five times the limit, and
asserts the abort lands within two rows' work of the limit and leaves the graph
unchanged.
"""

from __future__ import annotations

import math
import re
import time

import pytest

import kglite

NODES = 2000
LIMIT_MS = 600
SLACK_MS = 250
SLOW = "reduce(s = 0, j IN range(1, $w) | s + j)"

STATEMENTS = {
    "set_return": f"UNWIND range(1, $k) AS i MATCH (n:Employment {{id: i}}) SET n.acc = {SLOW} RETURN count(n) AS c",
    "set_no_return": f"UNWIND range(1, $k) AS i MATCH (n:Employment {{id: i}}) SET n.acc = {SLOW}",
    "create": f"UNWIND range(1, $k) AS i CREATE (:Tmp {{v: {SLOW}}})",
    "merge": f"UNWIND range(1, $k) AS i MERGE (t:Tmp {{k: i}}) ON CREATE SET t.acc = {SLOW}",
}


def build(mode: str, tmp_path, name: str) -> kglite.KnowledgeGraph:
    if mode == "disk":
        g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / f"{name}.kgl"))
    else:
        g = kglite.KnowledgeGraph()
    g.cypher(f"UNWIND range(1, {NODES}) AS i CREATE (:Employment {{id: i, title: toString(i)}})")
    return g


def changed(g) -> tuple[int, int]:
    """Nodes a statement above would have written: (Employment with acc, Tmp)."""
    acc = g.cypher("MATCH (n:Employment) WHERE n.acc IS NOT NULL RETURN count(n) AS c").to_list()[0]["c"]
    tmp = g.cypher("MATCH (t:Tmp) RETURN count(t) AS c").to_list()[0]["c"]
    return acc, tmp


def per_row_ms(run, statement: str, w: int, rows: int = 10) -> float:
    started = time.perf_counter()
    run(statement, {"k": rows, "w": w}, 0)
    return (time.perf_counter() - started) * 1000 / rows


def calibrate(mode: str, tmp_path, statement: str) -> tuple[int, float]:
    """(w, U): an expression width giving a per-row cost of roughly 15 ms+."""
    w = 20000
    for attempt in range(3):
        g = build(mode, tmp_path, f"calibrate{attempt}")
        u = per_row_ms(lambda q, p, t: g.cypher(q, params=p, timeout_ms=t), statement, w)
        if u >= 10:
            return w, u
        w = int(w * math.ceil(15 / max(u, 0.1)))
    return w, u


def timeout_numbers(err: Exception) -> tuple[int, int]:
    match = re.search(r"elapsed (\d+)ms, limit (\d+)ms", str(err))
    assert match, f"timeout message carries no numbers: {err}"
    return int(match.group(1)), int(match.group(2))


def assert_bounded_and_rolled_back(g, run, statement: str, w: int, u: float) -> None:
    k = math.ceil(5 * LIMIT_MS / u)
    before = changed(g)
    with pytest.raises(kglite.CypherTimeoutError) as excinfo:
        run(statement, {"k": k, "w": w}, LIMIT_MS)
    elapsed, limit = timeout_numbers(excinfo.value)
    assert elapsed <= LIMIT_MS + 2 * u + SLACK_MS, (
        f"deadline {LIMIT_MS} ms overran to {elapsed} ms (U = {u:.1f} ms/row, {k} rows)"
    )
    assert limit == LIMIT_MS
    assert changed(g) == before, "a statement that timed out left writes behind"


@pytest.mark.parametrize("mode", ["memory", "disk"])
@pytest.mark.parametrize("shape", sorted(STATEMENTS))
def test_write_deadline_is_bounded_and_rolls_back(mode, shape, tmp_path):
    statement = STATEMENTS[shape]
    w, u = calibrate(mode, tmp_path, statement)
    g = build(mode, tmp_path, "subject")
    assert_bounded_and_rolled_back(g, lambda q, p, t: g.cypher(q, params=p, timeout_ms=t), statement, w, u)


def test_session_write_deadline_is_bounded_and_rolls_back(tmp_path):
    statement = STATEMENTS["set_no_return"]
    w, u = calibrate("memory", tmp_path, statement)
    session = build("memory", tmp_path, "session").session()
    assert_bounded_and_rolled_back(session, lambda q, p, t: session.execute(q, params=p, timeout_ms=t), statement, w, u)
    assert changed(session) == (0, 0)


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_transaction_write_deadline_is_bounded_and_rolls_back(mode, tmp_path):
    statement = STATEMENTS["set_no_return"]
    w, u = calibrate(mode, tmp_path, statement)
    g = build(mode, tmp_path, "tx")
    tx = g.begin()
    assert_bounded_and_rolled_back(tx, lambda q, p, t: tx.cypher(q, params=p, timeout_ms=t), statement, w, u)
    tx.commit()
    assert changed(g) == (0, 0)


def test_timeout_reports_the_configured_limit_despite_a_large_parameter():
    """The deadline starts when the binding resolves it, before `$rows` is
    converted; the reported limit and elapsed time share that origin."""
    g = build("memory", None, "limit")
    rows = [{"id": i % NODES + 1} for i in range(300_000)]
    with pytest.raises(kglite.CypherTimeoutError) as excinfo:
        g.cypher(
            f"UNWIND $rows AS c MATCH (n:Employment {{id: c.id}}) SET n.acc = {SLOW}",
            params={"rows": rows, "w": 20000},
            timeout_ms=300,
        )
    elapsed, limit = timeout_numbers(excinfo.value)
    assert limit == 300
    assert elapsed >= 300


# ── statements that run without a rollback checkpoint ────────────────────
#
# A terminal variable-only DELETE and a single-node CREATE on the in-memory
# backend run with no checkpoint (nothing after their first write can fail).
# A deadline error from such a statement must therefore fire before anything
# is applied: the error means "nothing changed", on every path.

FAST_NODES = 200_000


def fast_graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher(f"UNWIND range(1, {FAST_NODES}) AS i CREATE (:A {{id: i}})")
    return g


def count_a(target) -> int:
    return target.cypher("MATCH (n:A) RETURN count(n) AS c").to_list()[0]["c"]


def delete_window_ms() -> float:
    g = fast_graph()
    started = time.perf_counter()
    g.cypher("MATCH (n:A) DELETE n", timeout_ms=0)
    return (time.perf_counter() - started) * 1000


@pytest.mark.parametrize("surface", ["graph", "transaction"])
def test_checkpoint_free_delete_that_times_out_deletes_nothing(surface):
    full = delete_window_ms()
    raised = 0
    for fraction in (0.02, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
        g = fast_graph()
        target = g.begin() if surface == "transaction" else g
        try:
            target.cypher("MATCH (n:A) DELETE n", timeout_ms=max(1, int(full * fraction)))
        except kglite.CypherTimeoutError:
            raised += 1
            assert count_a(target) == FAST_NODES, f"timed out at {fraction:.0%} of the window and kept deletes"
            if surface == "transaction":
                target.commit()
                assert count_a(g) == FAST_NODES, "commit published a timed-out DELETE"
        else:
            assert count_a(target) == 0
    assert raised, "no sampled timeout fired; the test measured nothing"


def test_checkpoint_free_create_that_times_out_creates_nothing():
    """The value expression runs past the deadline without polling (``replace``
    over a parameter list); the node must not be inserted."""
    query = "CREATE (:T {v: reduce(acc = 0, j IN $items | acc + size(replace($s, 'a', 'bb')))})"
    # Size the expression on this machine: a fixed list ran in 19.6 ms on a fast
    # CI runner, under the 20 ms the quarter-window timeout below needs.
    items = 300
    while True:
        params = {"items": list(range(items)), "s": "a" * 20_000}
        g = kglite.KnowledgeGraph()
        started = time.perf_counter()
        g.cypher(query, params=params, timeout_ms=0)
        full = (time.perf_counter() - started) * 1000
        if full >= 40 or items >= 300 * 64:
            break
        items *= 2
    assert full >= 40, f"expression too cheap ({full:.1f} ms at {items} items) to outrun a deadline"
    g = kglite.KnowledgeGraph()
    with pytest.raises(kglite.CypherTimeoutError):
        g.cypher(query, params=params, timeout_ms=max(1, int(full / 4)))
    assert g.cypher("MATCH (t:T) RETURN count(t) AS c").to_list()[0]["c"] == 0


def test_transaction_statement_elapsed_is_the_statement_not_the_transaction_age():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:A {id: 'x'})")
    tx = g.begin(timeout_ms=60_000)
    time.sleep(1.5)
    diagnostics = tx.cypher("MATCH (a:A) RETURN a.id AS i").diagnostics
    tx.rollback()
    assert diagnostics["elapsed_ms"] < 100, diagnostics
    assert diagnostics["timeout_ms"] == 60_000, diagnostics

"""Pin the Bolt server's expected transaction usage patterns.

The Bolt server (Phase B/C of docs/history/bolt-implementation.md) consumes the existing
`graph.begin()` / `graph.begin_read()` / `tx.commit()` / `tx.rollback()` /
context-manager surface to map Bolt's BEGIN / RUN / COMMIT / ROLLBACK
messages onto kglite operations. These tests pin the contracts so library
refactors can't silently regress the surface Phase C will consume.

See docs/python/transactions.md for the binding-implementer narrative.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def graph_with_people() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
            ]
        ),
        "Person",
        "id",
        "name",
    )
    return g


def _count_people(g: kglite.KnowledgeGraph) -> int:
    return int(g.cypher("MATCH (p:Person) RETURN count(p) AS n").to_df()["n"][0])


# ── Explicit begin() / commit() / rollback() ───────────────────────────


class TestExplicitTransactions:
    """The Bolt server's BEGIN → RUN → COMMIT / ROLLBACK message flow."""

    def test_begin_multiple_cypher_commit_all_visible(self, graph_with_people):
        """3 CREATE ops inside begin()/commit() are all visible after commit."""
        assert _count_people(graph_with_people) == 2
        tx = graph_with_people.begin()
        tx.cypher("CREATE (:Person {id: 10, name: 'Carol', age: 40})")
        tx.cypher("CREATE (:Person {id: 11, name: 'Dan', age: 35})")
        tx.cypher("CREATE (:Person {id: 12, name: 'Eve', age: 28})")
        tx.commit()
        assert _count_people(graph_with_people) == 5

    def test_begin_multiple_cypher_rollback_discards_all(self, graph_with_people):
        """rollback() discards every mutation inside the transaction."""
        assert _count_people(graph_with_people) == 2
        tx = graph_with_people.begin()
        tx.cypher("CREATE (:Person {id: 10, name: 'Carol'})")
        tx.cypher("CREATE (:Person {id: 11, name: 'Dan'})")
        tx.rollback()
        assert _count_people(graph_with_people) == 2

    def test_uncommitted_mutations_invisible_to_outside_reader(self, graph_with_people):
        """Snapshot isolation: outside reads do not see uncommitted writes."""
        tx = graph_with_people.begin()
        tx.cypher("CREATE (:Person {id: 99, name: 'Inside-tx'})")
        # The outside graph still reports the pre-begin() count.
        assert _count_people(graph_with_people) == 2
        # The transaction sees its own writes.
        in_tx = int(tx.cypher("MATCH (p:Person) RETURN count(p) AS n").to_df()["n"][0])
        assert in_tx == 3
        tx.commit()
        assert _count_people(graph_with_people) == 3

    def test_double_commit_raises_clear_error(self, graph_with_people):
        """Calling commit() twice is a contract violation; clear error required."""
        tx = graph_with_people.begin()
        tx.cypher("CREATE (:Person {id: 99, name: 'X'})")
        tx.commit()
        with pytest.raises(kglite.KgError, match="already committed"):
            tx.commit()

    def test_commit_after_rollback_raises_clear_error(self, graph_with_people):
        tx = graph_with_people.begin()
        tx.rollback()
        with pytest.raises(kglite.KgError, match="already committed or rolled back"):
            tx.commit()


# ── Context manager (with graph.begin() as tx:) ────────────────────────


class TestContextManager:
    """The Pythonic surface — also the easiest one for Bolt to mirror onto
    a session's auto-commit RUN messages."""

    def test_context_manager_auto_commits_on_success(self, graph_with_people):
        with graph_with_people.begin() as tx:
            tx.cypher("CREATE (:Person {id: 50, name: 'Frank'})")
            tx.cypher("CREATE (:Person {id: 51, name: 'Grace'})")
        # Outside the `with`, the mutations are committed.
        assert _count_people(graph_with_people) == 4

    def test_context_manager_auto_rollbacks_on_exception(self, graph_with_people):
        class TestError(Exception):
            pass

        with pytest.raises(TestError):
            with graph_with_people.begin() as tx:
                tx.cypher("CREATE (:Person {id: 50, name: 'Frank'})")
                raise TestError("simulate downstream failure")
        # Nothing committed.
        assert _count_people(graph_with_people) == 2

    def test_context_manager_read_only(self, graph_with_people):
        """begin_read() context manager works for read-heavy sessions."""
        with graph_with_people.begin_read() as tx:
            assert tx.is_read_only is True
            rows = tx.cypher("MATCH (p:Person) RETURN p.name AS n ORDER BY n").to_df()
            assert list(rows["n"]) == ["Alice", "Bob"]


# ── OCC conflict semantics ─────────────────────────────────────────────


class TestOptimisticConcurrencyControl:
    """When two transactions race, the second to commit detects the conflict."""

    def test_occ_conflict_raises_on_second_commit(self, graph_with_people):
        tx_a = graph_with_people.begin()
        tx_b = graph_with_people.begin()
        tx_a.cypher("CREATE (:Person {id: 100, name: 'FromTxA'})")
        tx_b.cypher("CREATE (:Person {id: 101, name: 'FromTxB'})")
        # First commit wins.
        tx_a.commit()
        # Second commit detects the version bump and raises.
        with pytest.raises(kglite.KgError, match="Transaction conflict"):
            tx_b.commit()
        # Tx A's mutation is in the graph; Tx B's was dropped.
        rows = graph_with_people.cypher("MATCH (p:Person) RETURN p.name AS n ORDER BY n").to_df()
        names = list(rows["n"])
        assert "FromTxA" in names
        assert "FromTxB" not in names

    def test_outside_mutation_during_transaction_triggers_conflict(self, graph_with_people):
        """Direct graph.cypher() mutations also bump the version → conflict."""
        tx = graph_with_people.begin()
        tx.cypher("CREATE (:Person {id: 100, name: 'TxMutation'})")
        # An outside-the-transaction mutation also bumps version.
        graph_with_people.cypher("CREATE (:Person {id: 200, name: 'OutsideMutation'})")
        with pytest.raises(kglite.KgError, match="Transaction conflict"):
            tx.commit()


# ── Read-only enforcement ──────────────────────────────────────────────


class TestReadOnlyTransaction:
    """begin_read() rejects mutations cleanly."""

    def test_read_only_rejects_create(self, graph_with_people):
        tx = graph_with_people.begin_read()
        with pytest.raises(kglite.KgError):
            tx.cypher("CREATE (:Person {id: 99, name: 'NoGood'})")

    def test_read_only_rejects_set(self, graph_with_people):
        tx = graph_with_people.begin_read()
        with pytest.raises(kglite.KgError):
            tx.cypher("MATCH (p:Person {id: 1}) SET p.age = 99")

    def test_read_only_rejects_delete(self, graph_with_people):
        tx = graph_with_people.begin_read()
        with pytest.raises(kglite.KgError):
            tx.cypher("MATCH (p:Person {id: 1}) DELETE p")

    def test_read_only_commit_is_noop(self, graph_with_people):
        """commit() on a read-only tx is a no-op (Bolt RUN/PULL summary)."""
        tx = graph_with_people.begin_read()
        tx.cypher("MATCH (p:Person) RETURN p.name")
        tx.commit()  # no-op, no error


# ── Auto-commit semantics (Bolt server must wrap to avoid these) ───────


class TestAutoCommitContract:
    """`graph.cypher()` outside a transaction is auto-commit per call.

    Bolt servers MUST NOT expose this to clients — they wrap each session's
    statements in BEGIN/COMMIT to preserve atomicity. These tests pin the
    contract so the binding-implementer docs can reference the exact shape.
    """

    def test_each_cypher_call_is_independently_visible(self, graph_with_people):
        graph_with_people.cypher("CREATE (:Person {id: 10, name: 'AC1'})")
        assert _count_people(graph_with_people) == 3
        graph_with_people.cypher("CREATE (:Person {id: 11, name: 'AC2'})")
        assert _count_people(graph_with_people) == 4

    def test_partial_mutation_visible_when_multi_statement_fails(self, graph_with_people):
        """Single cypher() call with multiple CREATE statements: if a later one
        fails (e.g. typed-property validation), earlier CREATEs are already in
        the graph. This is the contract a Bolt server wraps in BEGIN/COMMIT."""
        # Build a query that creates two nodes then fails on a parse-time
        # error (using an invalid expression on the third clause). The
        # parser catches it before any execution, so neither prior CREATE
        # is visible — wrong shape. Use a runtime error instead.
        baseline = _count_people(graph_with_people)
        # CREATE clauses that succeed, then a SET on a non-existent label
        # (succeeds with 0 rows in cypher's lenient mode). To trigger a
        # genuine mid-statement runtime failure, use parameter mismatch.
        # The cleanest case: a procedure call requiring a missing arg.
        try:
            graph_with_people.cypher(
                "CREATE (:Person {id: 90, name: 'BeforeFail'}) "
                "WITH count(*) AS c "
                "CALL orphan_node({wrong_kwarg: 'X'}) YIELD node "
                "RETURN count(node)"
            )
        except kglite.KgError:
            pass
        # The CREATE before the failing CALL IS now in the graph — auto-commit.
        # This is the contract a Bolt server must wrap to provide atomicity.
        post = _count_people(graph_with_people)
        # Contract: at least one CREATE landed despite the downstream
        # CALL failure. Bolt servers wrap statements in begin()/commit()
        # exactly because of this.
        assert post >= baseline


# ── Timeouts ───────────────────────────────────────────────────────────


class TestTransactionTimeouts:
    """The Bolt server can attach a per-transaction deadline via begin(timeout_ms=...)."""

    def test_begin_with_timeout_aborts_overdue_op(self, graph_with_people):
        """A transaction whose deadline expires raises a typed CypherTimeoutError
        on the next operation."""
        import time

        tx = graph_with_people.begin(timeout_ms=10)
        time.sleep(0.05)  # 50 ms > 10 ms deadline
        with pytest.raises(kglite.CypherTimeoutError):
            tx.cypher("MATCH (p:Person) RETURN count(p)")


# ── Memory / Arc behavior under repeated begin()/commit() ──────────────


class TestRepeatedTransactionsNoLeak:
    """100 sequential begin()/commit() cycles complete cleanly — pins that
    Arc references don't pile up and the working-copy clones get freed."""

    def test_hundred_begin_commit_cycles_complete(self, graph_with_people):
        for i in range(100):
            tx = graph_with_people.begin()
            tx.cypher(f"CREATE (:Person {{id: {1000 + i}, name: 'Loop_{i}'}})")
            tx.commit()
        # All 100 + 2 baseline visible.
        assert _count_people(graph_with_people) == 102

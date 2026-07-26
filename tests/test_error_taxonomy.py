"""The typed-error contract an application actually writes `except` clauses against.

Two guarantees are under test, and both were broken before this suite existed:

1. **A constraint violation is catchable by type, from every write path.**
   `ConstraintViolationError` was previously unreachable — a violation raised
   through Cypher arrived as `CypherExecutionError`, and one raised through the
   bulk loader as `ArgumentError`, so the only handler an app could write was a
   substring match on the message in its signup path.

2. **A commit conflict is catchable by type and carries a stable code.**
   It was previously `ArgumentError` ("Invalid argument: Transaction
   conflict…"), with no `.code` anywhere on the surface.

The message quality is asserted alongside the type on purpose: these messages
name the constraint, the property, and the offending value, and adding a type
must not be an excuse to regress them.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

# ─── A. Constraint violations are typed, on every write path ────────────────


@pytest.fixture
def users() -> kglite.KnowledgeGraph:
    """A graph whose `User.email` is a NODE KEY (unique *and* present)."""
    kg = kglite.KnowledgeGraph()
    kg.define_schema({"nodes": {"User": {"primary_key": "email", "required": ["email"]}}})
    return kg


def test_duplicate_signup_via_cypher_is_catchable_by_type(users):
    """The signup case: `except kglite.ConstraintViolationError`, no substring match."""
    users.cypher("CREATE (u:User {email: 'a@b.com', name: 'A'})")

    with pytest.raises(kglite.ConstraintViolationError) as excinfo:
        users.cypher("CREATE (u:User {email: 'a@b.com', name: 'B'})")

    exc = excinfo.value
    # Catchable at every level of the hierarchy an app might use.
    assert isinstance(exc, kglite.ConstraintError)
    assert isinstance(exc, kglite.KgError)
    # ...and *not* as the generic execution error it used to be.
    assert not isinstance(exc, kglite.CypherExecutionError)
    assert exc.code == "ConstraintViolation"

    # Message quality must survive the retyping: it still names the constraint,
    # the property, the offending value, and the remedy.
    message = str(exc)
    assert "already exists" in message
    assert "NODE KEY constraint on User.email" in message
    assert "'a@b.com'" in message
    assert "MERGE" in message


def test_not_null_via_cypher_is_catchable_by_type(users):
    with pytest.raises(kglite.ConstraintViolationError) as excinfo:
        users.cypher("CREATE (u:User {name: 'no email'})")

    message = str(excinfo.value)
    assert "must have the property 'email'" in message
    assert "NODE KEY constraint on User.email" in message
    assert excinfo.value.code == "ConstraintViolation"


def test_set_to_null_via_cypher_is_catchable_by_type(users):
    users.cypher("CREATE (u:User {email: 'c@d.com'})")
    with pytest.raises(kglite.ConstraintViolationError):
        users.cypher("MATCH (u:User) SET u.email = null")


def test_remove_required_property_via_cypher_is_catchable_by_type(users):
    users.cypher("CREATE (u:User {email: 'e@f.com'})")
    with pytest.raises(kglite.ConstraintViolationError):
        users.cypher("MATCH (u:User) REMOVE u.email")


def test_bulk_loader_violation_is_catchable_by_type():
    """`add_nodes` is the funnel users trust most for volume; it raised
    `ArgumentError` before, despite the docs promising the typed class."""
    kg = kglite.KnowledgeGraph()
    kg.define_schema({"nodes": {"P": {"primary_key": "id", "required": ["email"]}}})

    with pytest.raises(kglite.ConstraintViolationError) as excinfo:
        kg.add_nodes(pd.DataFrame([{"id": 1}]), "P", "id")

    assert "NOT NULL constraint on P.email" in str(excinfo.value)
    assert excinfo.value.code == "ConstraintViolation"


def test_constraint_creation_failure_is_typed_and_distinct():
    """Declaring a constraint the data already violates is a *different* fix
    (deduplicate, then re-declare), so it stays a distinct sibling class."""
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:U {id: 1, email: 'd@d.com'})")
    kg.cypher("CREATE (:U {id: 2, email: 'd@d.com'})")

    with pytest.raises(kglite.ConstraintCreationError) as excinfo:
        kg.define_schema({"nodes": {"U": {"unique": [["email"]]}}})

    exc = excinfo.value
    assert isinstance(exc, kglite.ConstraintError)
    assert not isinstance(exc, kglite.ConstraintViolationError)
    assert exc.code == "ConstraintCreationFailed"


def test_cypher_create_constraint_ddl_violation_is_typed():
    """The `CREATE CONSTRAINT` DDL path goes through its own raise sites."""
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:W {id: 1, email: 'x@x.com'})")
    kg.cypher("CREATE (:W {id: 2, email: 'x@x.com'})")

    with pytest.raises(kglite.ConstraintError) as excinfo:
        kg.cypher("CREATE CONSTRAINT FOR (w:W) REQUIRE w.email IS UNIQUE")

    assert excinfo.value.code in {"ConstraintViolation", "ConstraintCreationFailed"}


def test_a_non_constraint_cypher_failure_is_still_a_cypher_error(users):
    """The side channel must not mistype unrelated failures."""
    users.cypher("CREATE (u:User {email: 'g@h.com'})")

    with pytest.raises(kglite.CypherExecutionError) as excinfo:
        users.cypher("MATCH (u:User) RETURN nonexistent_function(u)")

    assert not isinstance(excinfo.value, kglite.ConstraintError)
    assert excinfo.value.code == "CypherExecution"


def test_a_successful_write_after_a_violation_is_unaffected(users):
    """A parked violation must never leak into a later, successful execution."""
    with pytest.raises(kglite.ConstraintViolationError):
        users.cypher("CREATE (u:User {name: 'no email'})")

    users.cypher("CREATE (u:User {email: 'ok@ok.com'})")
    assert users.cypher("MATCH (u:User) RETURN count(u) AS n").to_list() == [{"n": 1}]

    # ...and the *next* failure is still classified correctly, not stale.
    with pytest.raises(kglite.ConstraintViolationError) as excinfo:
        users.cypher("CREATE (u:User {email: 'ok@ok.com'})")
    assert "already exists" in str(excinfo.value)


# ─── B. Transaction conflicts are typed and carry a code ────────────────────


def _two_node_graph() -> kglite.KnowledgeGraph:
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:N {id: 1, v: 0})")
    kg.cypher("CREATE (:N {id: 2, v: 0})")
    return kg


def test_commit_conflict_raises_transaction_conflict_error():
    kg = _two_node_graph()
    t1 = kg.begin()
    t2 = kg.begin()
    t1.cypher("MATCH (n:N {id: 1}) SET n.v = 1")
    t2.cypher("MATCH (n:N {id: 1}) SET n.v = 2")
    t1.commit()

    with pytest.raises(kglite.TransactionConflictError) as excinfo:
        t2.commit()

    exc = excinfo.value
    assert isinstance(exc, kglite.KgError)
    assert not isinstance(exc, kglite.ArgumentError)
    assert exc.code == "TransactionConflict"
    # The advice that was already good, plus the version gap.
    message = str(exc)
    assert "Retry the transaction" in message
    assert "were not applied" in message


def test_conflict_code_is_readable_without_an_instance():
    """`.code` is a class constant too, so a dispatch table can be built up
    front rather than inside an `except` block."""
    assert kglite.TransactionConflictError.code == "TransactionConflict"
    assert kglite.ConstraintViolationError.code == "ConstraintViolation"
    assert kglite.CypherSyntaxError.code == "CypherSyntax"
    # The abstract bases span several codes and so carry None.
    assert kglite.KgError.code is None
    assert kglite.ConstraintError.code is None


def test_every_raised_error_carries_a_code():
    """`.code` is a property of the surface, not of one lucky path."""
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:U {id: 1})")

    with pytest.raises(kglite.CypherSyntaxError) as syntax:
        kg.cypher("MATCH (((")
    assert syntax.value.code == "CypherSyntax"

    with pytest.raises(kglite.KgError) as missing:
        kg.cypher("MATCH (u:U) RETURN $undefined_param")
    assert isinstance(missing.value.code, str) and missing.value.code


def test_disjoint_transactions_still_conflict_by_design():
    """Characterization test for a documented limitation.

    OCC here is a whole-graph version check, not a read/write-set
    intersection: a commit publishes the transaction's working copy by pointer
    swap, so t2's copy does not contain t1's write and applying it would
    silently revert t1. Rejecting is therefore *correct* for this commit model,
    not a spurious failure — see docs/concepts/concurrency.md. If KGLite ever
    gains a merging commit, this test is the one that should change.
    """
    kg = _two_node_graph()
    t1 = kg.begin()
    t2 = kg.begin()
    t1.cypher("MATCH (n:N {id: 1}) SET n.v = 111")
    t2.cypher("MATCH (n:N {id: 2}) SET n.v = 222")  # a different node
    t1.commit()

    # The lost update this rejection prevents: t2 still sees the pre-t1 value.
    assert t2.cypher("MATCH (n:N {id: 1}) RETURN n.v AS v").to_list() == [{"v": 0}]

    with pytest.raises(kglite.TransactionConflictError):
        t2.commit()

    # t1's write survived precisely because t2 was refused.
    assert kg.cypher("MATCH (n:N {id: 1}) RETURN n.v AS v").to_list() == [{"v": 111}]


# ─── C. The retry loop, end to end ──────────────────────────────────────────


def test_retry_on_conflict_succeeds_after_a_conflicting_commit():
    """The loop every correct app needs: a writer that lost one race and won
    on the retry, without the caller writing any retry code."""
    kg = _two_node_graph()
    attempts = []

    def work(tx):
        attempts.append(len(attempts) + 1)
        # Interleave one competing commit, but only on the first attempt, so
        # the first commit conflicts and the second succeeds.
        if len(attempts) == 1:
            kg.cypher("MATCH (n:N {id: 2}) SET n.v = 99")
        tx.cypher("MATCH (n:N {id: 1}) SET n.v = 42")
        return "done"

    result = kglite.retry_on_conflict(kg, work, base_delay=0, jitter=False)

    assert result == "done"
    assert len(attempts) == 2, "expected exactly one retry"
    assert kg.cypher("MATCH (n:N {id: 1}) RETURN n.v AS v").to_list() == [{"v": 42}]


def test_retry_on_conflict_commits_without_contention():
    kg = _two_node_graph()
    calls = []

    def work(tx):
        calls.append(1)
        tx.cypher("MATCH (n:N {id: 1}) SET n.v = 7")
        return tx

    kglite.retry_on_conflict(kg, work, base_delay=0, jitter=False)

    assert len(calls) == 1, "no conflict means no retry"
    assert kg.cypher("MATCH (n:N {id: 1}) RETURN n.v AS v").to_list() == [{"v": 7}]


def test_retry_on_conflict_reraises_after_exhausting_attempts():
    kg = _two_node_graph()

    def work(tx):
        # Guarantee a conflict on every single attempt.
        kg.cypher("MATCH (n:N {id: 2}) SET n.v = 1")
        tx.cypher("MATCH (n:N {id: 1}) SET n.v = 2")

    with pytest.raises(kglite.TransactionConflictError) as excinfo:
        kglite.retry_on_conflict(kg, work, attempts=3, base_delay=0, jitter=False)

    # The real error survives the loop, codes and all.
    assert excinfo.value.code == "TransactionConflict"


def test_retry_on_conflict_does_not_retry_other_errors():
    """Only conflicts are retried — a constraint violation is the caller's bug
    and must surface immediately rather than being hammered `attempts` times."""
    kg = kglite.KnowledgeGraph()
    kg.define_schema({"nodes": {"User": {"primary_key": "email"}}})
    kg.cypher("CREATE (u:User {email: 'dup@x.com'})")
    calls = []

    def work(tx):
        calls.append(1)
        tx.cypher("CREATE (u:User {email: 'dup@x.com'})")

    with pytest.raises(kglite.ConstraintViolationError):
        kglite.retry_on_conflict(kg, work, attempts=4, base_delay=0, jitter=False)

    assert len(calls) == 1, "a non-conflict error must not be retried"


def test_retry_on_conflict_rejects_a_nonsense_attempt_count():
    kg = _two_node_graph()
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        kglite.retry_on_conflict(kg, lambda tx: None, attempts=0)

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


# ─── C. One cause, one class: client mistakes are client errors ─────────────
#
# The three sections below are a *table*, deliberately. Each row names a cause
# and the (class, code) pair every surface that can produce it must answer
# with, so a future divergence fails by the name of the diverging row rather
# than as a message-substring surprise somewhere downstream.


@pytest.fixture
def people() -> kglite.KnowledgeGraph:
    kg = kglite.KnowledgeGraph()
    kg.add_nodes(pd.DataFrame([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]), "P", "id")
    return kg


def _write_on_session(kg):
    kg.session().cypher("CREATE (n:P {id: 9})")


def _write_on_frozen(kg):
    kg.freeze().cypher("CREATE (n:P {id: 9})")


def _write_on_read_transaction(kg):
    tx = kg.begin_read()
    try:
        tx.cypher("CREATE (n:P {id: 9})")
    finally:
        tx.rollback()


def _write_on_read_only_graph(kg):
    kg.read_only(True)
    try:
        kg.cypher("CREATE (n:P {id: 9})")
    finally:
        kg.read_only(False)


@pytest.mark.parametrize(
    "handle",
    [
        pytest.param(_write_on_session, id="Session.cypher"),
        pytest.param(_write_on_frozen, id="FrozenGraph.cypher"),
        pytest.param(_write_on_read_transaction, id="read-only Transaction"),
        pytest.param(_write_on_read_only_graph, id="read-only KnowledgeGraph"),
    ],
)
def test_a_write_on_a_read_handle_is_one_class_on_every_handle(people, handle):
    """Four handles, one policy — "this handle does not take writes" — so one
    class and one code. Before this test they answered `ValueError` (twice,
    with no `.code` at all), `ArgumentError` and `CypherExecutionError`, and no
    caller could route on the refusal without matching four things."""
    with pytest.raises(kglite.ArgumentError) as excinfo:
        handle(people)

    exc = excinfo.value
    assert exc.code == "InvalidArgument"
    assert isinstance(exc, kglite.KgError)
    # A client mistake must not be published as an execution failure: that code
    # maps to Neo.DatabaseError.Statement.ExecutionFailed on the Bolt wire.
    assert not isinstance(exc, kglite.CypherExecutionError)
    # The refusal still names the remedy it always named.
    assert "CREATE" in str(exc)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda kg: kg.properties("Nope"), id="properties"),
        pytest.param(lambda kg: kg.neighbors_schema("Nope"), id="neighbors_schema"),
        pytest.param(lambda kg: kg.sample("Nope"), id="sample"),
        pytest.param(lambda kg: kg.describe(types=["Nope"]), id="describe"),
        pytest.param(lambda kg: kg.set_parent_type("Nope", "P"), id="set_parent_type"),
        pytest.param(lambda kg: kg.set_temporal("Nope", "a", "b"), id="set_temporal"),
    ],
)
def test_an_unknown_node_type_is_one_class_on_every_method(people, call):
    """`KeyError` is reserved by docs/python/error-handling.md for a missing
    result column or mapping key. A node type is neither, so an unknown one is
    an argument mistake — on every method that takes a type name."""
    with pytest.raises(kglite.ArgumentError) as excinfo:
        call(people)
    assert excinfo.value.code == "InvalidArgument"
    assert "Nope" in str(excinfo.value)


def test_an_unknown_node_type_is_not_a_key_error(people):
    """The two methods that used to raise it, pinned by their old class."""
    for call in (people.properties, people.neighbors_schema):
        with pytest.raises(kglite.ArgumentError):
            call("Nope")


# ─── D. No leaked interpreter errors from a validated argument ──────────────


def test_a_negative_sample_count_is_an_argument_error(people):
    """`n: usize` made PyO3 refuse before any kglite code ran, so the caller
    saw `OverflowError: can't convert negative int to unsigned` — an
    interpreter detail naming neither the parameter nor the value."""
    with pytest.raises(kglite.ArgumentError) as excinfo:
        people.sample("P", -1)
    message = str(excinfo.value)
    assert "n" in message and "-1" in message
    assert excinfo.value.code == "InvalidArgument"


def test_a_negative_positional_sample_count_is_an_argument_error(people):
    """`sample(-1)` takes the count-only call shape and must refuse the same way."""
    with pytest.raises(kglite.ArgumentError):
        people.select("P").sample(-1)


def test_from_records_names_the_value_it_cannot_carry():
    """The refusal is the contract (a JSON spec has no temporal type, and
    writing one as text would demote it to a string property — see
    `test_property_roundtrip_matrix.py`). What was broken is the *message*:
    `json.dumps` answered "Object of type datetime is not JSON serializable",
    naming neither the value, the field, nor a way forward."""
    import datetime

    spec = {
        "nodes": [
            {
                "type": "Event",
                "id_field": "id",
                "records": [{"id": 1, "at": datetime.datetime(2024, 3, 9, 14, 30, 5)}],
            }
        ]
    }
    with pytest.raises(TypeError) as excinfo:
        kglite.from_records(spec)
    message = str(excinfo.value)
    assert "from_records" in message
    assert "datetime" in message
    assert "2024" in message, "the refusal must name the offending value"
    assert "ISO-8601" in message and "add_nodes" in message, "and a way forward"


def test_from_records_treats_a_missing_timestamp_as_null():
    """`pd.NaT` is a datetime subclass, so it reached the same refusal — but a
    missing value carries no temporal to demote."""
    kg = kglite.from_records(
        {
            "nodes": [
                {
                    "type": "Event",
                    "id_field": "id",
                    "records": [{"id": 1, "at": pd.NaT}],
                }
            ]
        }
    )
    assert kg.cypher("MATCH (n:Event) RETURN n.at AS at").to_list() == [{"at": None}]


def test_a_pandas_nat_parameter_is_null():
    """`pd.NaT` is a `datetime` subclass, so it reached `datetime_to_utc_naive`
    and failed inside with `'float' object cannot be interpreted as an
    integer`. It is pandas' missing value and binds as NULL, like NaN."""
    kg = kglite.KnowledgeGraph()
    rows = kg.cypher("RETURN $v AS v", params={"v": pd.NaT}).to_list()
    assert rows == [{"v": None}]


def test_a_nested_conversion_failure_names_the_parameter_path():
    """Every typed arm builds the `$v.a[0]` path; the arm that wraps a raw
    Python error used to return it unwrapped, discarding the path the stub
    promises for *any* nested failure."""

    import datetime

    class ExplodingZone(datetime.tzinfo):
        """A tzinfo whose `utcoffset` raises — the shortest route to a raw
        `PyErr` from inside the datetime arm, which is where the unwrapping
        happened."""

        def utcoffset(self, dt):
            raise ValueError("this zone refuses to answer")

        def tzname(self, dt):
            return "BOOM"

        def dst(self, dt):
            return None

    kg = kglite.KnowledgeGraph()
    aware = datetime.datetime(2024, 3, 9, 14, 30, tzinfo=ExplodingZone())
    with pytest.raises(Exception) as excinfo:
        kg.cypher("RETURN $v AS v", params={"v": {"a": [aware]}})
    assert "$v.a[0]" in str(excinfo.value)
    assert "this zone refuses to answer" in str(excinfo.value)


def test_a_numpy_bool_parameter_is_accepted():
    """Every other numpy scalar converts; `np.bool_` was rejected with a
    message naming `'bool'` — a type that *is* supported."""
    np = pytest.importorskip("numpy")
    kg = kglite.KnowledgeGraph()
    rows = kg.cypher("RETURN $v AS v", params={"v": np.bool_(True)}).to_list()
    assert rows == [{"v": True}]


def test_an_unsupported_numpy_type_is_named_with_its_module():
    """`'complex128'` alone is not a Python type name anyone can look up."""
    np = pytest.importorskip("numpy")
    kg = kglite.KnowledgeGraph()
    with pytest.raises(TypeError) as excinfo:
        kg.cypher("RETURN $v AS v", params={"v": np.complex128(1 + 2j)})
    assert "numpy.complex128" in str(excinfo.value)

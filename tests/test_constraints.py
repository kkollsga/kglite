"""Constraint completeness: per-constraint × per-write-path enforcement matrix.

This is the acceptance spec for arbitrary UNIQUE, write-time NOT NULL, and a
generalized primary key. The central property under test is that enforcement is
**not Cypher-only**: node creation has three funnels (see
`dir_graph/node_write.rs`), and the DataFrame-shaped bulk funnel
(`add_nodes`, and therefore blueprints / `from_records` / OKF / WAL replay /
`extend_graph`) is the one users trust most for volume. A constraint the bulk
path walks past is worse than no constraint at all, so every constraint kind is
asserted against every write path here rather than only through `CREATE`.

Declaration surface (all through the existing `define_schema`, so no new
per-constraint API):

    {"nodes": {"Person": {
        "primary_key": "email",         # NODE KEY: unique + not null
        "unique": [["email"], ["first", "last"]],   # arbitrary + composite
        "required": ["email"],          # NOT NULL, enforced at write time
    }}}

Run: pytest tests/test_constraints.py
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

MODES = ("memory", "mapped", "disk")

# A violation must be recognisable without depending on the exact prose.
UNIQUE_ERROR = "UNIQUE constraint"
NOT_NULL_ERROR = "NOT NULL constraint"
NODE_KEY_ERROR = "NODE KEY constraint"


def _fresh(mode: str, tmp_path) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    return KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))


def _count(kg: KnowledgeGraph, label: str) -> int:
    return kg.cypher(f"MATCH (n:{label}) RETURN count(n) AS c").to_dicts()[0]["c"]


def _snapshot(kg: KnowledgeGraph, label: str) -> list[dict]:
    """Order-independent full contents of a label, for before/after comparison."""
    rows = kg.cypher(f"MATCH (n:{label}) RETURN n.id AS id, n.email AS email, n.name AS name").to_dicts()
    return sorted(rows, key=lambda r: repr(r.get("id")))


# ===========================================================================
# UNIQUE on an arbitrary property — one row per write path
# ===========================================================================


@pytest.mark.parametrize("mode", MODES)
def test_unique_rejects_duplicate_via_cypher_create(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.cypher("CREATE (:Person {id: 2, email: 'a@b.c'})")
    assert _count(kg, "Person") == 1


@pytest.mark.parametrize("mode", MODES)
def test_unique_rejects_duplicate_via_cypher_set(mode, tmp_path):
    """SET is a write path too: moving a value onto an occupied tuple violates."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    kg.cypher("CREATE (:Person {id: 2, email: 'd@e.f'})")
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.cypher("MATCH (p:Person {id: 2}) SET p.email = 'a@b.c'")
    # The rejected SET left the original value intact.
    assert kg.cypher("MATCH (p:Person {id: 2}) RETURN p.email AS e").to_dicts()[0]["e"] == "d@e.f"


@pytest.mark.parametrize("mode", MODES)
def test_unique_allows_set_to_own_current_value(mode, tmp_path):
    """A node must not conflict with itself — rewriting a value is a no-op."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    kg.cypher("MATCH (p:Person {id: 1}) SET p.email = 'a@b.c'")
    assert _count(kg, "Person") == 1


@pytest.mark.parametrize("mode", MODES)
def test_unique_rejects_duplicate_within_one_add_nodes_batch(mode, tmp_path):
    """THE bulk-path test: a repeat inside one input is rejected, and nothing
    is written — the batch is validated before it reaches storage."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    df = pd.DataFrame({"id": [1, 2], "email": ["a@b.c", "a@b.c"]})
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.add_nodes(df, "Person", "id")
    assert _count(kg, "Person") == 0


@pytest.mark.parametrize("mode", MODES)
def test_unique_rejects_add_nodes_row_colliding_with_stored_node(mode, tmp_path):
    """Bulk vs the *existing* graph: a new id carrying an already-taken email."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.add_nodes(pd.DataFrame({"id": [1], "email": ["a@b.c"]}), "Person", "id")
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.add_nodes(pd.DataFrame({"id": [2], "email": ["a@b.c"]}), "Person", "id")
    assert _count(kg, "Person") == 1


@pytest.mark.parametrize("mode", MODES)
def test_unique_permits_add_nodes_upsert_of_same_node(mode, tmp_path):
    """Re-loading the same id keeps its own email — an upsert is not a duplicate."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    df = pd.DataFrame({"id": [1], "email": ["a@b.c"]})
    kg.add_nodes(df, "Person", "id")
    kg.add_nodes(df, "Person", "id")
    assert _count(kg, "Person") == 1


@pytest.mark.parametrize("mode", MODES)
def test_unique_rejects_duplicate_via_merge_on_create_branch(mode, tmp_path):
    """MERGE matching on a *different* property still has to honour UNIQUE when
    its create branch fires."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.cypher("MERGE (p:Person {id: 2}) ON CREATE SET p.email = 'a@b.c'")


# ===========================================================================
# UNIQUE semantics: NULL exemption, composite tuples, freed values
# ===========================================================================


@pytest.mark.parametrize("mode", MODES)
def test_unique_does_not_apply_to_nodes_missing_the_property(mode, tmp_path):
    """Neo4j semantics: uniqueness does not constrain nodes without the property,
    so many nodes may share 'no email'."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1})")
    kg.cypher("CREATE (:Person {id: 2})")
    assert _count(kg, "Person") == 2


@pytest.mark.parametrize("mode", MODES)
def test_composite_unique_requires_the_whole_tuple_to_collide(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"P": {"unique": [["first", "last"]]}}})
    kg.cypher("CREATE (:P {id: 1, first: 'A', last: 'B'})")
    # Same first, different last — no collision.
    kg.cypher("CREATE (:P {id: 2, first: 'A', last: 'C'})")
    assert _count(kg, "P") == 2
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.cypher("CREATE (:P {id: 3, first: 'A', last: 'B'})")


@pytest.mark.parametrize("mode", MODES)
def test_composite_unique_exempts_partial_tuples(mode, tmp_path):
    """A composite constraint needs every component present to apply."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"P": {"unique": [["first", "last"]]}}})
    kg.cypher("CREATE (:P {id: 1, first: 'A'})")
    kg.cypher("CREATE (:P {id: 2, first: 'A'})")
    assert _count(kg, "P") == 2


@pytest.mark.parametrize("mode", MODES)
def test_deleting_a_node_frees_its_unique_value(mode, tmp_path):
    """A deleted node must give up its tuple, or the value is reserved forever."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    kg.cypher("MATCH (p:Person {id: 1}) DETACH DELETE p")
    kg.cypher("CREATE (:Person {id: 2, email: 'a@b.c'})")
    assert _count(kg, "Person") == 1


@pytest.mark.parametrize("mode", MODES)
def test_set_frees_the_vacated_unique_value(mode, tmp_path):
    """Moving a value off a node releases it for reuse."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    kg.cypher("MATCH (p:Person {id: 1}) SET p.email = 'moved@b.c'")
    kg.cypher("CREATE (:Person {id: 2, email: 'a@b.c'})")
    assert _count(kg, "Person") == 2


# ===========================================================================
# Declaration-time behaviour
# ===========================================================================


@pytest.mark.parametrize("mode", MODES)
def test_declaring_unique_on_already_duplicated_data_is_rejected(mode, tmp_path):
    """Installing a constraint the data violates would make it lie about the
    rows already present."""
    kg = _fresh(mode, tmp_path)
    kg.add_nodes(pd.DataFrame({"id": [1, 2], "email": ["a@b.c", "a@b.c"]}), "Person", "id")
    with pytest.raises(Exception, match="duplicate"):
        kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    # Rejected declaration left the data untouched.
    assert _count(kg, "Person") == 2


@pytest.mark.parametrize("mode", MODES)
def test_declaring_unique_on_clean_data_succeeds_and_then_enforces(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.add_nodes(pd.DataFrame({"id": [1, 2], "email": ["a@b.c", "d@e.f"]}), "Person", "id")
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.cypher("CREATE (:Person {id: 3, email: 'a@b.c'})")


def test_unique_constraint_survives_save_and_load(tmp_path):
    """The declaration persists and is live again after a reload — a constraint
    that vanished on reload would silently stop protecting the graph."""
    path = str(tmp_path / "g.kgl")
    kg = KnowledgeGraph()
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    kg.save(path)

    reloaded = kglite.load(path)
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        reloaded.cypher("CREATE (:Person {id: 2, email: 'a@b.c'})")
    assert _count(reloaded, "Person") == 1


def test_old_graph_without_constraints_still_loads(tmp_path):
    """Data-format compatibility: a graph saved with no constraint section loads
    with no constraints and stays permissive."""
    path = str(tmp_path / "g.kgl")
    kg = KnowledgeGraph()
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    kg.cypher("CREATE (:Person {id: 2, email: 'a@b.c'})")
    kg.save(path)

    reloaded = kglite.load(path)
    assert _count(reloaded, "Person") == 2


# ===========================================================================
# Write-time NOT NULL
# ===========================================================================


@pytest.mark.parametrize("mode", MODES)
def test_not_null_rejects_cypher_create_missing_the_property(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        kg.cypher("CREATE (:Person {id: 1})")
    assert _count(kg, "Person") == 0


@pytest.mark.parametrize("mode", MODES)
def test_not_null_accepts_create_supplying_the_property(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    assert _count(kg, "Person") == 1


@pytest.mark.parametrize("mode", MODES)
def test_not_null_rejects_setting_the_property_to_null(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        kg.cypher("MATCH (p:Person {id: 1}) SET p.email = null")


@pytest.mark.parametrize("mode", MODES)
def test_not_null_rejects_removing_the_property(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        kg.cypher("MATCH (p:Person {id: 1}) REMOVE p.email")


@pytest.mark.parametrize("mode", MODES)
def test_not_null_rejects_add_nodes_row_missing_the_property(mode, tmp_path):
    """The bulk path enforces NOT NULL too, and writes nothing when it fails."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        kg.add_nodes(pd.DataFrame({"id": [1], "name": ["a"]}), "Person", "id")
    assert _count(kg, "Person") == 0


@pytest.mark.parametrize("mode", MODES)
def test_not_null_rejects_add_nodes_row_with_null_cell(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    df = pd.DataFrame({"id": [1, 2], "email": ["a@b.c", None]})
    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        kg.add_nodes(df, "Person", "id")
    assert _count(kg, "Person") == 0


@pytest.mark.parametrize("mode", MODES)
def test_not_null_on_structural_id_is_a_noop(mode, tmp_path):
    """`id` / `title` are NodeData fields, always present — requiring them must
    not reject every write. Matches the offline validator's exemption."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["id", "title"]}}})
    kg.cypher("CREATE (:Person {id: 1, name: 'A'})")
    assert _count(kg, "Person") == 1


# ===========================================================================
# NOT NULL × auto-vivified provisional stubs
#
# The design decision under test: a stub is DEFERRED, not exempt. Vivification
# may create an incomplete placeholder, but the promotion write that supplies
# the real row is fully enforced, and an unpromoted stub stays reportable.
# ===========================================================================


@pytest.mark.parametrize("mode", MODES)
def test_edge_vivification_may_create_an_incomplete_stub(mode, tmp_path):
    """Enforcing NOT NULL on stub creation would break graph building outright:
    an edge list routinely names a node before its own row loads."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    kg.add_nodes(pd.DataFrame({"id": [1], "email": ["a@b.c"]}), "Person", "id")
    edges = pd.DataFrame({"src": [1], "dst": [99]})
    kg.add_connections(edges, "KNOWS", "Person", "src", "Person", "dst")
    # id 99 was vivified as a stub despite lacking the required property.
    assert _count(kg, "Person") == 2


@pytest.mark.parametrize("mode", MODES)
def test_promoting_a_stub_without_the_required_property_is_rejected(mode, tmp_path):
    """The promotion write is a normal write, so it is fully enforced — the
    escape hatch is the promotion flow, not a permanent exemption."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    kg.add_nodes(pd.DataFrame({"id": [1], "email": ["a@b.c"]}), "Person", "id")
    kg.add_connections(pd.DataFrame({"src": [1], "dst": [99]}), "KNOWS", "Person", "src", "Person", "dst")
    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        kg.add_nodes(pd.DataFrame({"id": [99], "name": ["late"]}), "Person", "id")


@pytest.mark.parametrize("mode", MODES)
def test_promoting_a_stub_with_the_required_property_succeeds(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    kg.add_nodes(pd.DataFrame({"id": [1], "email": ["a@b.c"]}), "Person", "id")
    kg.add_connections(pd.DataFrame({"src": [1], "dst": [99]}), "KNOWS", "Person", "src", "Person", "dst")
    kg.add_nodes(pd.DataFrame({"id": [99], "email": ["late@b.c"]}), "Person", "id")
    assert _count(kg, "Person") == 2


# ===========================================================================
# Generalized primary key (NODE KEY = unique + not null)
# ===========================================================================


@pytest.mark.parametrize("mode", MODES)
def test_primary_key_on_arbitrary_property_enforces_uniqueness(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "email"}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    with pytest.raises(Exception, match=NODE_KEY_ERROR):
        kg.cypher("CREATE (:Person {id: 2, email: 'a@b.c'})")
    assert _count(kg, "Person") == 1


@pytest.mark.parametrize("mode", MODES)
def test_primary_key_on_arbitrary_property_implies_not_null(mode, tmp_path):
    """A primary key is unique AND present — a node without it is rejected."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "email"}}})
    with pytest.raises(Exception, match=NODE_KEY_ERROR):
        kg.cypher("CREATE (:Person {id: 1})")


@pytest.mark.parametrize("mode", MODES)
def test_primary_key_on_arbitrary_property_enforced_in_bulk(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "email"}}})
    df = pd.DataFrame({"id": [1, 2], "email": ["a@b.c", "a@b.c"]})
    with pytest.raises(Exception, match=NODE_KEY_ERROR):
        kg.add_nodes(df, "Person", "id")
    assert _count(kg, "Person") == 0


@pytest.mark.parametrize("mode", MODES)
def test_primary_key_on_id_keeps_its_dedicated_message(mode, tmp_path):
    """`primary_key: 'id'` still routes through the O(1) id-index probe, so its
    established error text is unchanged — existing callers keep matching on it."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    kg.cypher("CREATE (:Person {id: 1})")
    with pytest.raises(Exception, match="duplicate primary key"):
        kg.cypher("CREATE (:Person {id: 1})")


# ===========================================================================
# Rollback interaction
#
# These assert the OBSERVABLE property — a statement that violates a constraint
# part-way through leaves the graph exactly as it was — not the mechanism. They
# pass against the clone-based checkpoint and must keep passing against any
# replacement rollback mechanism.
# ===========================================================================


@pytest.mark.parametrize("mode", MODES)
def test_rollback_mid_statement_unique_violation_leaves_graph_unchanged(mode, tmp_path):
    """A multi-row CREATE whose LAST row collides must not leave the earlier
    rows behind."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'taken@b.c'})")
    before = _snapshot(kg, "Person")

    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.cypher(
            "UNWIND [{i: 2, e: 'x@b.c'}, {i: 3, e: 'y@b.c'}, "
            "{i: 4, e: 'taken@b.c'}] AS row "
            "CREATE (:Person {id: row.i, email: row.e})"
        )

    assert _snapshot(kg, "Person") == before


@pytest.mark.parametrize("mode", MODES)
def test_rollback_mid_statement_not_null_violation_leaves_graph_unchanged(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"required": ["email"]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'a@b.c'})")
    before = _snapshot(kg, "Person")

    with pytest.raises(Exception, match=NOT_NULL_ERROR):
        kg.cypher("UNWIND [{i: 2, e: 'x@b.c'}, {i: 3, e: null}] AS row CREATE (:Person {id: row.i, email: row.e})")

    assert _snapshot(kg, "Person") == before


@pytest.mark.parametrize("mode", MODES)
def test_rollback_mid_statement_set_violation_leaves_graph_unchanged(mode, tmp_path):
    """A multi-node SET whose second node collides must not keep the first
    node's new value."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.cypher("CREATE (:Person {id: 1, email: 'one@b.c'})")
    kg.cypher("CREATE (:Person {id: 2, email: 'two@b.c'})")
    kg.cypher("CREATE (:Person {id: 3, email: 'three@b.c'})")
    before = _snapshot(kg, "Person")

    # Rewriting every email to a single value collides on the second node.
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.cypher("MATCH (p:Person) SET p.email = 'collapsed@b.c'")

    assert _snapshot(kg, "Person") == before


@pytest.mark.parametrize("mode", MODES)
def test_bulk_violation_writes_nothing_at_all(mode, tmp_path):
    """The bulk path validates the whole batch before storage, so a rejected
    load needs no rollback — assert the stronger property directly."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.add_nodes(pd.DataFrame({"id": [1], "email": ["a@b.c"]}), "Person", "id")
    before = _snapshot(kg, "Person")

    df = pd.DataFrame({"id": [2, 3, 4], "email": ["x@b.c", "y@b.c", "a@b.c"]})
    with pytest.raises(Exception, match=UNIQUE_ERROR):
        kg.add_nodes(df, "Person", "id")

    assert _snapshot(kg, "Person") == before


@pytest.mark.parametrize("mode", MODES)
def test_a_refused_bulk_load_records_no_observed_metadata(mode, tmp_path):
    """ "Writes nothing at all" includes the *schema* a load would have recorded.

    A refused batch used to leave the rejected column's observed type behind,
    so the next conforming load warned about a type mismatch against a schema
    the user never accepted — and `describe()` reported it as if the load had
    happened.
    """
    import warnings

    kg = _fresh(mode, tmp_path)
    kg.add_nodes(pd.DataFrame({"id": [1], "email": ["a@b.c"], "age": [30]}), "Person", "id")
    kg.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.email IS NOT NULL")
    before = kg.describe()

    # No email (refused by NOT NULL) *and* an `age` whose type disagrees with
    # the recorded one — the column that must not be recorded on the way out.
    with pytest.raises(Exception):
        kg.add_nodes(pd.DataFrame({"id": [2], "age": ["thirty"]}), "Person", "id")

    assert kg.describe() == before, "a refused load left observed metadata behind"

    # The user-visible consequence: the next conforming load is clean.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kg.add_nodes(pd.DataFrame({"id": [3], "email": ["c@b.c"], "age": [61]}), "Person", "id")
    mismatches = [w for w in caught if "Type mismatch" in str(w.message)]
    assert not mismatches, [str(w.message) for w in mismatches]


# ---------------------------------------------------------------------------
# verify_unique_constraints(): the audit for the paths that bypass enforcement
# ---------------------------------------------------------------------------


def _duplicate_via_ntriples(tmp_path):
    """A graph whose declared UNIQUE constraint is violated by data that never
    passed through enforcement.

    The N-Triples loader is one of the two documented bypasses
    (`docs/python/guides/primary-store.md`), and it is the reachable one from
    Python — the constraint is declared on an empty graph (so the declaration
    installs cleanly) and the duplicates arrive afterwards, behind its back.
    """
    path = tmp_path / "dup.nt"
    path.write_text(
        '<http://www.wikidata.org/entity/Q1> <http://schema.org/description> "dup"@en .\n'
        '<http://www.wikidata.org/entity/Q2> <http://schema.org/description> "dup"@en .\n',
        encoding="utf-8",
    )
    kg = KnowledgeGraph()
    kg.define_schema({"nodes": {"Entity": {"unique": ["description"]}}})
    kg.load_ntriples(str(path), languages=["en"])
    return kg


def test_verify_unique_constraints_is_empty_on_clean_data():
    kg = KnowledgeGraph()
    kg.define_schema({"nodes": {"Person": {"unique": [["email"]]}}})
    kg.add_nodes(pd.DataFrame({"id": [1, 2], "email": ["a@b.c", "b@b.c"]}), "Person", "id")
    assert kg.verify_unique_constraints() == []


def test_verify_unique_constraints_reports_a_bypassed_duplicate(tmp_path):
    kg = _duplicate_via_ntriples(tmp_path)

    # The data really is there and really is duplicated — otherwise the audit
    # below would be asserting nothing.
    rows = kg.cypher("MATCH (n:Entity) RETURN n.description AS d").to_df()
    assert rows["d"].tolist() == ["dup", "dup"]

    violations = kg.verify_unique_constraints()
    assert len(violations) == 1
    found = violations[0]
    assert found["constraint"] == "UNIQUE"
    assert found["node_type"] == "Entity"
    assert found["properties"] == ["description"]
    assert found["duplicate_tuples"] == 1
    assert found["sample"] == ["dup"]
    # The message is an *audit* message: the constraint already exists, so it
    # must not advise declaring it.
    assert "violated" in found["message"], found["message"]
    assert "before declaring" not in found["message"], found["message"]


def test_verify_unique_constraints_goes_quiet_once_the_data_is_fixed(tmp_path):
    kg = _duplicate_via_ntriples(tmp_path)
    assert kg.verify_unique_constraints()
    kg.cypher("MATCH (n:Entity) WHERE n.id = 2 DETACH DELETE n")
    assert kg.verify_unique_constraints() == []

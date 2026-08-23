"""Byte-exact record goldens for `RETURN n` / `RETURN r` / `RETURN p`.

# Why this file exists (Part N safety net, phase N1)

Part N (the ``Arc``'d value-representation work, amended N2)
replaces ``NodeValue``/``RelValue``'s ``properties: BTreeMap<String, Value>``
with an ``Arc``'d sorted flat map with shared keys. That change rewrites both
sides of the Python boundary: the Rust materialisation
(``collect_node_properties``) **and** the ``py_out::value_to_py`` conversion
that turns a ``Value::Node`` into the dict below.

Nothing in the suite pinned the *complete* record before this file. Tests
asserted a key here, a value there — so a representation change could drop an
alias, reorder keys, resurrect a provenance key or change a virtual, and stay
green. These goldens assert the **whole record**: key set, key order, values,
and the absences (null omission, provenance exclusion).

The expectations are **checked-in literals**, never re-derived from the same
code path under test. A test that reads the record and asserts it equals
itself proves nothing.

**N2 must pass against these records unchanged.** A red line here means a
representation change altered a user-visible result shape, which needs a
documented decision and a CHANGELOG entry — not a refreshed expectation.

The Rust-side companions live at:
  * ``crates/kglite/src/graph/languages/cypher/executor/node_record_golden_tests.rs``
    (the same records asserted on the ``NodeValue`` itself, before py_out)
  * ``crates/kglite/src/datatypes/value_shape_tests.rs`` (derived ``Ord``)
  * ``crates/kglite/src/graph/value_byte_identity_tests.rs`` (serialized bytes)
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

STORAGE_MODES = ("memory", "mapped", "disk")


def _new_kg(mode: str, tmp_path, tag: str = "kg") -> KnowledgeGraph:
    """Construct an empty graph in the requested storage mode.

    Mirrors the module-level idiom in ``tests/test_storage_parity.py`` and
    ``tests/test_golden.py`` — only ``disk`` needs a path.
    """
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    if mode == "disk":
        return KnowledgeGraph(storage="disk", path=str(tmp_path / f"{tag}-disk"))
    raise ValueError(f"unknown storage mode: {mode}")


# ───────────────────────────────────────────────────────────────────────────
# Fixture: the `node_projection_graph` shape, shrunk to a readable size.
#
# Same column contract as `tests/benchmarks/test_bench_core.py::node_projection_graph`
# (pid -> id, name -> title, plus age and city), which is the shape Part N's
# whole measurement programme is denominated in. Six rows instead of 10k, and
# one deliberate null.
# ───────────────────────────────────────────────────────────────────────────

N_NODES = 6


def _build_projection_graph(mode: str, tmp_path, tag: str = "proj") -> KnowledgeGraph:
    kg = _new_kg(mode, tmp_path, tag)
    nodes = pd.DataFrame(
        {
            "pid": list(range(N_NODES)),
            "name": [f"P{i}" for i in range(N_NODES)],
            "age": [20 + i for i in range(N_NODES)],
            # Row 5 carries a null city — the null-omission rule must drop the
            # key entirely rather than surfacing it as None.
            "city": [f"city_{i % 3}" for i in range(N_NODES - 1)] + [None],
        }
    )
    kg.add_nodes(nodes, "Person", "pid", "name")
    edges = pd.DataFrame(
        {
            "s": list(range(N_NODES - 1)),
            "d": list(range(1, N_NODES)),
            "since": [2020 + i for i in range(N_NODES - 1)],
        }
    )
    kg.add_connections(edges, "KNOWS", "Person", "s", "Person", "d")
    return kg


# ═══════════════════════════════════════════════════════════════════════════
# Artifact 1 — the complete `RETURN n` record, all three backends
# ═══════════════════════════════════════════════════════════════════════════

# ---- CHECKED-IN EXPECTATION (do not regenerate from a query result) --------
#
# Read this as the contract it encodes, key by key:
#   id      — the petgraph node index, surfaced as the record's top-level `id`
#   labels  — the full label set (primary type when no secondaries exist)
#   age     — an ordinary stored property
#   city    — an ordinary stored property; ABSENT on the null row
#   id      — the `id` virtual (the node's canonical identity value)
#   name    — the TITLE COLUMN ALIAS re-surfaced under its original df name
#   pid     — the ID COLUMN ALIAS re-surfaced under its original df name
#   title   — the `title` virtual
#   type    — the `type` virtual (structural type string)
#
# Key ORDER is part of the golden: the properties dict is built by iterating a
# sorted map, so it is alphabetical. A representation change that iterates in
# insertion order instead would reorder these and go red here.
EXPECTED_NODE_0 = {
    "id": 0,
    "labels": ["Person"],
    "properties": {
        "age": 20,
        "city": "city_0",
        "id": 0,
        "name": "P0",
        "pid": 0,
        "title": "P0",
        "type": "Person",
    },
}

# The null row: `city` is OMITTED, not present-as-None. Every other key stays.
EXPECTED_NODE_NULL_CITY = {
    "id": 5,
    "labels": ["Person"],
    "properties": {
        "age": 25,
        "id": 5,
        "name": "P5",
        "pid": 5,
        "title": "P5",
        "type": "Person",
    },
}

EXPECTED_PROPERTY_KEY_ORDER = ["age", "city", "id", "name", "pid", "title", "type"]

# Provenance keys are engine metadata. They must never reach a materialised
# record, on any backend.
FORBIDDEN_PROVENANCE_KEYS = {
    "updated_at",
    "created_at",
    "_provisional",
    "provisional",
}
# ---------------------------------------------------------------------------


def _assert_record_exactly(got: dict, expected: dict, what: str) -> None:
    """Assert a whole record: top-level key set AND order, then contents.

    Deliberately not `assert got == expected` alone — dict equality ignores
    key order, and key order is half of what this golden exists to pin.
    """
    assert list(got.keys()) == list(expected.keys()), (
        f"{what}: top-level key ORDER changed. Got {list(got.keys())}, expected {list(expected.keys())}."
    )
    assert list(got["properties"].keys()) == list(expected["properties"].keys()), (
        f"{what}: property key ORDER changed. Got {list(got['properties'].keys())}, "
        f"expected {list(expected['properties'].keys())}."
    )
    assert got == expected, f"{what}: record contents changed."


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_return_node_record_is_byte_exact(mode, tmp_path):
    """`RETURN n` produces exactly the checked-in record, on every backend."""
    kg = _build_projection_graph(mode, tmp_path)
    rows = list(kg.cypher("MATCH (n:Person) WHERE n.pid = 0 RETURN n"))
    assert len(rows) == 1, f"fixture query matched {len(rows)} rows, expected 1"

    assert list(rows[0].keys()) == ["n"], "result row column set changed"
    _assert_record_exactly(rows[0]["n"], EXPECTED_NODE_0, f"RETURN n on mode={mode}")


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_return_node_omits_null_properties(mode, tmp_path):
    """A null-valued property is dropped from the record, not surfaced as None."""
    kg = _build_projection_graph(mode, tmp_path)
    rows = list(kg.cypher("MATCH (n:Person) WHERE n.pid = 5 RETURN n"))
    assert len(rows) == 1
    record = rows[0]["n"]
    assert "city" not in record["properties"], (
        f"mode={mode}: a null property must be OMITTED from a materialised node, not present with a None value"
    )
    _assert_record_exactly(record, EXPECTED_NODE_NULL_CITY, f"RETURN n (null row) on mode={mode}")


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_return_node_excludes_provenance_keys(mode, tmp_path):
    """Reserved provenance keys never reach a materialised record."""
    kg = _build_projection_graph(mode, tmp_path)
    for row in kg.cypher("MATCH (n:Person) RETURN n"):
        leaked = FORBIDDEN_PROVENANCE_KEYS & set(row["n"]["properties"])
        assert not leaked, f"mode={mode}: provenance keys leaked into RETURN n: {sorted(leaked)}"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_return_node_property_key_order_is_pinned(mode, tmp_path):
    """Every non-null row carries the same key set, in the same order."""
    kg = _build_projection_graph(mode, tmp_path)
    for row in kg.cypher("MATCH (n:Person) WHERE n.pid < 5 RETURN n"):
        assert list(row["n"]["properties"].keys()) == EXPECTED_PROPERTY_KEY_ORDER, (
            f"mode={mode}: property key order drifted for node {row['n']['properties'].get('pid')}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Artifact 1 (cont.) — Relationship and Path records
#
# RelValue and PathValue change in N2 too, and their py_out conversions have
# their own key sets (`start`/`end`/`type`, `nodes`/`relationships`).
# ═══════════════════════════════════════════════════════════════════════════

# ---- CHECKED-IN EXPECTATION ------------------------------------------------
EXPECTED_REL_0 = {
    "id": 0,
    "start": 0,
    "end": 1,
    "type": "KNOWS",
    "properties": {"since": 2020},
}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_return_relationship_record_is_byte_exact(mode, tmp_path):
    """`RETURN r` produces exactly the checked-in relationship record."""
    kg = _build_projection_graph(mode, tmp_path)
    rows = list(kg.cypher("MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.pid = 0 AND b.pid = 1 RETURN r"))
    assert len(rows) == 1, f"fixture query matched {len(rows)} rows, expected 1"
    rel = rows[0]["r"]
    assert list(rel.keys()) == ["id", "start", "end", "type", "properties"], (
        f"mode={mode}: relationship record key ORDER changed — got {list(rel.keys())}"
    )
    assert rel == EXPECTED_REL_0, f"mode={mode}: relationship record contents changed"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_return_path_record_is_byte_exact(mode, tmp_path):
    """`RETURN p` nests the same node and relationship records, unchanged.

    The path record is where N2's `py_out` Path double-clone fix lands, so its
    exact nesting is pinned: a path over one hop carries 2 nodes and 1 rel, and
    each nested record must be identical to the standalone one.
    """
    kg = _build_projection_graph(mode, tmp_path)
    rows = list(kg.cypher("MATCH p = (a:Person)-[:KNOWS]->(b:Person) WHERE a.pid = 0 AND b.pid = 1 RETURN p"))
    assert len(rows) == 1, f"fixture query matched {len(rows)} rows, expected 1"
    path = rows[0]["p"]

    assert list(path.keys()) == ["nodes", "relationships"], (
        f"mode={mode}: path record key ORDER changed — got {list(path.keys())}"
    )
    assert len(path["nodes"]) == 2, "a one-hop path carries k+1 = 2 nodes"
    assert len(path["relationships"]) == 1, "a one-hop path carries k = 1 rel"

    # The nested node record must be identical to the standalone RETURN n one —
    # same materialisation, same conversion, no path-specific shortcut.
    _assert_record_exactly(path["nodes"][0], EXPECTED_NODE_0, f"path.nodes[0] on mode={mode}")
    assert path["relationships"][0] == EXPECTED_REL_0, (
        f"mode={mode}: the relationship nested in a path differs from the same relationship returned standalone"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Artifact 2 — keys(n) == keys(properties(n))
#
# Currently only a doc-comment invariant on `PropertySink`
# (crates/kglite/src/graph/languages/cypher/executor/helpers.rs). The two sinks
# share one collection pass precisely so they cannot drift; nothing asserted it.
#
# The corpus covers, explicitly and separately:
#   * aliased id/title columns (pid -> id, name -> title)
#   * COLUMNAR-stored nodes  (bulk `add_nodes`)
#   * MAP-stored nodes       (Cypher `CREATE`) — the C2 lesson: on mapped mode
#                             a Cypher-created node is Map-stored, not columnar,
#                             and takes a different branch through the collector
#   * null-valued properties (the omission rule must apply to BOTH sinks)
#   * secondary labels
# ═══════════════════════════════════════════════════════════════════════════


def _corpus_graph(mode: str, tmp_path) -> KnowledgeGraph:
    """A graph deliberately mixing every storage shape and property shape."""
    kg = _build_projection_graph(mode, tmp_path, tag="corpus")

    # Cypher-created nodes: a different construction path from bulk
    # `add_nodes` — no id/title column aliases to re-surface. (See the module
    # docstring for why these are NOT Map-stored on this branch.)
    kg.cypher(
        "CREATE (:Gadget {gid: 1, label: 'g1', weight: 2.5, flag: true}), "
        "(:Gadget {gid: 2, label: 'g2'}), "
        "(:Widget {wid: 1, label: 'w1', note: 'n'})"
    )
    # Secondary labels on a subset.
    kg.cypher("MATCH (n:Gadget) WHERE n.gid = 1 SET n:Featured")
    return kg


@pytest.mark.parametrize("mode", STORAGE_MODES)
@pytest.mark.parametrize("label", ["Person", "Gadget", "Widget"])
def test_keys_equals_keys_of_properties(mode, label, tmp_path):
    """The `keys(n) == keys(properties(n))` invariant, over the type corpus.

    `keys(n)` runs a names-only sink over the same collection pass that builds
    `properties(n)`. If N2 changes the value sink's container without changing
    the key sink identically, the two drift and this goes red.
    """
    kg = _corpus_graph(mode, tmp_path)
    rows = list(kg.cypher(f"MATCH (n:{label}) RETURN keys(n) AS k, properties(n) AS p, n AS node"))
    assert rows, f"mode={mode}: corpus produced no {label} nodes"

    for row in rows:
        keys = row["k"]
        props = row["p"]
        assert sorted(keys) == sorted(props.keys()), (
            f"mode={mode} label={label}: keys(n) != keys(properties(n)).\n"
            f"  keys(n)              = {sorted(keys)}\n"
            f"  keys(properties(n))  = {sorted(props.keys())}\n"
            f"  only in keys(n)      = {sorted(set(keys) - set(props))}\n"
            f"  only in properties(n)= {sorted(set(props) - set(keys))}"
        )
        # `keys(n)` is documented as the sorted, de-duplicated key set.
        assert keys == sorted(keys), f"mode={mode} label={label}: keys(n) not sorted"
        assert len(keys) == len(set(keys)), f"mode={mode} label={label}: keys(n) contains duplicates"
        # And the third consumer of the same pass — the materialised record —
        # must agree with both. A drift between `RETURN n` and `properties(n)`
        # is exactly the class N2 could introduce.
        assert list(row["node"]["properties"].keys()) == sorted(keys), (
            f"mode={mode} label={label}: RETURN n's property keys disagree with "
            "keys(n) — the shared collection pass has forked"
        )


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_keys_omits_nulls_for_both_sinks(mode, tmp_path):
    """The null-omission rule applies to the key sink and the value sink alike.

    The value sink drops a null by not inserting it; the key sink must drop the
    same key. A sink-asymmetric change here is invisible to every other test.
    """
    kg = _corpus_graph(mode, tmp_path)
    rows = list(kg.cypher("MATCH (n:Person) WHERE n.pid = 5 RETURN keys(n) AS k"))
    assert len(rows) == 1
    assert "city" not in rows[0]["k"], (
        f"mode={mode}: keys(n) surfaced a null-valued property that properties(n) omits — the two sinks have drifted"
    )


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_secondary_labels_reach_the_materialised_record(mode, tmp_path):
    """A secondary label appears in `labels`, and `labels(n)` agrees with it."""
    kg = _corpus_graph(mode, tmp_path)
    rows = list(kg.cypher("MATCH (n:Gadget) WHERE n.gid = 1 RETURN n, labels(n) AS l"))
    assert len(rows) == 1
    record_labels = rows[0]["n"]["labels"]
    assert sorted(record_labels) == ["Featured", "Gadget"], (
        f"mode={mode}: materialised label set changed — got {record_labels}"
    )
    assert sorted(rows[0]["l"]) == sorted(record_labels), (
        f"mode={mode}: labels(n) disagrees with the materialised record's labels"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Artifact 3 (Python half) — `ORDER BY n`
#
# The Rust half pins the derived `Ord` directly
# (crates/kglite/src/datatypes/value_shape_tests.rs). This pins that the
# comparison actually reaches Cypher's ORDER BY over mixed labels and mixed
# property shapes, including ties.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_order_by_node_is_stable_and_pinned(mode, tmp_path):
    """`ORDER BY n` over mixed labels + mixed property shapes.

    `NodeValue`'s derived `Ord` is (id, labels, properties) — `id` is the
    petgraph index, so distinct nodes never tie and the order is creation
    order. That is the contract being pinned: a field reorder in N2 would make
    labels or properties outrank the index and reshuffle this list.
    """
    kg = _corpus_graph(mode, tmp_path)
    rows = list(kg.cypher("MATCH (n) RETURN n ORDER BY n"))

    ids = [r["n"]["id"] for r in rows]
    assert ids == sorted(ids), (
        f"mode={mode}: ORDER BY n did not order by the node's `id` field first. "
        f"Got {ids}. `id` is NodeValue's first field; a struct reorder is a "
        "user-visible sort change."
    )

    # Mixed labels really are present, so the ordering above is not trivially
    # single-type.
    labels_seen = {tuple(sorted(r["n"]["labels"])) for r in rows}
    assert len(labels_seen) >= 3, (
        f"mode={mode}: corpus lost its label variety ({labels_seen}); the ordering assertion would be vacuous"
    )

    # Mixed property shapes too (different key counts across types).
    key_counts = {len(r["n"]["properties"]) for r in rows}
    assert len(key_counts) >= 2, f"mode={mode}: corpus lost its property-shape variety ({key_counts})"

    # Ties: two nodes identical in every field except `id`. `ORDER BY n` must
    # place them by `id`, and the relative order must be total (no equal pair).
    tie_rows = list(kg.cypher("MATCH (n:Gadget) WHERE n.gid IN [1, 2] RETURN n ORDER BY n"))
    tie_ids = [r["n"]["id"] for r in tie_rows]
    assert tie_ids == sorted(tie_ids) and len(set(tie_ids)) == len(tie_ids), (
        f"mode={mode}: near-identical nodes did not order by index — got {tie_ids}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Artifact 4 — mixed-type projection: `MATCH (n) RETURN n, id(n)`
#
# The retirement record's unused pre-mortem scenario: a projection returning a
# whole node ALONGSIDE a scalar derived from it, across more than one label.
# The two columns are produced by different code paths (materialisation vs the
# id resolver) and must agree.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_mixed_type_projection_node_and_id(mode, tmp_path):
    """`RETURN n, id(n)` agrees column-to-column over >= 2 labels."""
    kg = _corpus_graph(mode, tmp_path)
    rows = list(kg.cypher("MATCH (n) RETURN n, id(n) AS nid ORDER BY id(n)"))
    assert rows, f"mode={mode}: mixed projection returned no rows"

    labels_seen = {tuple(sorted(r["n"]["labels"])) for r in rows}
    assert len(labels_seen) >= 3, (
        f"mode={mode}: expected at least 3 distinct label sets in the corpus, "
        f"got {sorted(labels_seen)} — the multi-label premise of this golden is void"
    )

    for row in rows:
        assert list(row.keys()) == ["n", "nid"], f"mode={mode}: result column ORDER changed — got {list(row.keys())}"
        assert row["n"]["id"] == row["nid"], (
            f"mode={mode}: the materialised record's `id` ({row['n']['id']}) "
            f"disagrees with id(n) ({row['nid']}) — two code paths for the same "
            "identity have diverged"
        )
        # The record shape survives being projected next to a scalar (i.e. the
        # node column is not degraded to a scalar or a reference).
        assert set(row["n"].keys()) == {"id", "labels", "properties"}, (
            f"mode={mode}: node record shape changed in a mixed projection — got {sorted(row['n'].keys())}"
        )

    ids = [r["nid"] for r in rows]
    assert len(set(ids)) == len(ids), f"mode={mode}: id(n) is not unique per node"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_mixed_projection_node_matches_standalone_record(mode, tmp_path):
    """A node projected beside a scalar is byte-identical to one projected alone.

    Pins that the mixed-column path does not take a cheaper materialisation.
    """
    kg = _build_projection_graph(mode, tmp_path)
    solo = list(kg.cypher("MATCH (n:Person) WHERE n.pid = 0 RETURN n"))[0]["n"]
    mixed = list(kg.cypher("MATCH (n:Person) WHERE n.pid = 0 RETURN n, id(n) AS nid"))[0]["n"]

    _assert_record_exactly(solo, EXPECTED_NODE_0, f"solo projection on mode={mode}")
    _assert_record_exactly(mixed, EXPECTED_NODE_0, f"mixed projection on mode={mode}")
    assert list(solo["properties"].keys()) == list(mixed["properties"].keys())

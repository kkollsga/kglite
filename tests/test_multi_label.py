"""Multi-label node tests: CREATE (n:A:B), SET n:Label, REMOVE n:Label,
add_label / remove_label pymethods, labels() function, and save+load
round-trip.

Track C lands in 0.10.5. Reference design from
``docs/concepts/multi-label-rationale.md``; the mrmagooey/kglite fork
shipped a similar feature on its own branch.
"""

from pathlib import Path

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture
def g() -> KnowledgeGraph:
    return KnowledgeGraph()


# ─── CREATE (n:A:B) ────────────────────────────────────────────────────────


def test_create_single_label_unchanged(g):
    g.cypher("CREATE (n:Person {name: 'Alice'})")
    rows = g.cypher("MATCH (n:Person) RETURN n.name AS name").to_list()
    assert rows[0]["name"] == "Alice"


def test_create_multi_label_findable_by_primary(g):
    g.cypher("CREATE (n:Person:Director {name: 'Alice'})")
    rows = g.cypher("MATCH (n:Person) RETURN n.name AS name").to_list()
    assert rows[0]["name"] == "Alice"


def test_create_multi_label_findable_by_secondary(g):
    g.cypher("CREATE (n:Person:Director {name: 'Alice'})")
    rows = g.cypher("MATCH (n:Director) RETURN n.name AS name").to_list()
    assert rows[0]["name"] == "Alice"


def test_create_three_labels(g):
    g.cypher("CREATE (n:Animal:Pet:Dog {name: 'Rex'})")
    labels = g.cypher("MATCH (n:Animal) RETURN labels(n) AS labels").to_list()[0]["labels"]
    assert set(labels) == {"Animal", "Pet", "Dog"}
    # Findable by every label.
    for label in ("Animal", "Pet", "Dog"):
        rows = g.cypher(f"MATCH (n:{label}) RETURN n.name AS name").to_list()
        assert rows[0]["name"] == "Rex"


# ─── labels(n) ─────────────────────────────────────────────────────────────


def test_labels_single(g):
    g.cypher("CREATE (n:Person {name: 'Bob'})")
    rows = g.cypher("MATCH (n:Person) RETURN labels(n) AS labels").to_list()
    assert rows[0]["labels"] == ["Person"]


def test_labels_multi_order_primary_first(g):
    g.cypher("CREATE (n:Person:Director:Producer {name: 'Carol'})")
    rows = g.cypher("MATCH (n:Person) RETURN labels(n) AS labels").to_list()
    labels = rows[0]["labels"]
    # Primary first, insertion order for the rest.
    assert labels[0] == "Person"
    assert set(labels) == {"Person", "Director", "Producer"}


# ─── SET n:Label ───────────────────────────────────────────────────────────


def test_set_label_adds_secondary(g):
    g.cypher("CREATE (n:Person {name: 'Alice'})")
    g.cypher("MATCH (n:Person) SET n:Employee")
    rows = g.cypher("MATCH (n:Person) RETURN labels(n) AS labels").to_list()
    assert set(rows[0]["labels"]) == {"Person", "Employee"}


def test_set_label_idempotent(g):
    g.cypher("CREATE (n:Person {name: 'Alice'})")
    g.cypher("MATCH (n:Person) SET n:Employee")
    g.cypher("MATCH (n:Person) SET n:Employee")
    rows = g.cypher("MATCH (n:Person) RETURN labels(n) AS labels").to_list()
    labels = rows[0]["labels"]
    assert labels.count("Employee") == 1


def test_set_primary_label_is_noop(g):
    """Setting the primary label again must not duplicate it."""
    g.cypher("CREATE (n:Person {name: 'Alice'})")
    g.cypher("MATCH (n:Person) SET n:Person")
    rows = g.cypher("MATCH (n:Person) RETURN labels(n) AS labels").to_list()
    assert rows[0]["labels"].count("Person") == 1


def test_set_multiple_labels_one_clause(g):
    g.cypher("CREATE (n:Person {name: 'Dave'})")
    g.cypher("MATCH (n:Person) SET n:Reviewer:Manager")
    rows = g.cypher("MATCH (n:Person) RETURN labels(n) AS labels").to_list()
    assert set(rows[0]["labels"]) == {"Person", "Reviewer", "Manager"}


def test_match_and_intersect(g):
    """MATCH (n:A:B) is AND across labels."""
    g.cypher("CREATE (n:Person:Reviewer {name: 'Alice'})")
    g.cypher("CREATE (n:Person {name: 'Bob'})")
    g.cypher("CREATE (n:Reviewer {name: 'Carol'})")
    rows = g.cypher("MATCH (n:Person:Reviewer) RETURN n.name AS name").to_list()
    assert [r["name"] for r in rows] == ["Alice"]


# ─── REMOVE n:Label ────────────────────────────────────────────────────────


def test_remove_secondary_label(g):
    g.cypher("CREATE (n:Person:Director {name: 'Eve'})")
    g.cypher("MATCH (n:Person) REMOVE n:Director")
    rows = g.cypher("MATCH (n:Person) RETURN labels(n) AS labels").to_list()
    assert rows[0]["labels"] == ["Person"]


def test_remove_nonexistent_label_noop(g):
    g.cypher("CREATE (n:Person {name: 'Frank'})")
    # Should not error.
    g.cypher("MATCH (n:Person) REMOVE n:Ghost")
    rows = g.cypher("MATCH (n:Person) RETURN labels(n) AS labels").to_list()
    assert rows[0]["labels"] == ["Person"]


def test_remove_primary_label_errors(g):
    g.cypher("CREATE (n:Person {name: 'Grace'})")
    with pytest.raises(Exception, match=r"(?i)primary label"):
        g.cypher("MATCH (n:Person) REMOVE n:Person")


def test_set_then_remove_then_set_idempotence(g):
    """The idempotence invariant the rationale doc calls out."""
    g.cypher("CREATE (n:Person {name: 'Henry'})")
    g.cypher("MATCH (n:Person) SET n:VIP")
    g.cypher("MATCH (n:Person) REMOVE n:VIP")
    g.cypher("MATCH (n:Person) SET n:VIP")
    rows = g.cypher("MATCH (n:Person) RETURN labels(n) AS labels").to_list()
    labels = rows[0]["labels"]
    assert labels.count("VIP") == 1
    assert set(labels) == {"Person", "VIP"}


# ─── add_label / remove_label pymethods ───────────────────────────────────


def test_add_label_pymethod(g):
    df = pd.DataFrame({"id": ["a", "b", "c"], "name": ["A", "B", "C"]})
    g.add_nodes(df, "Agent", "id", "name")
    result = g.add_label("Agent", ["a", "b"], "Reviewer")
    assert result == {"labelled": 2, "skipped": 0}
    rows = g.cypher("MATCH (a:Reviewer) RETURN a.id AS id ORDER BY a.id").to_list()
    assert [r["id"] for r in rows] == ["a", "b"]


def test_add_label_pymethod_skips_unknown_ids(g):
    df = pd.DataFrame({"id": ["a"], "name": ["A"]})
    g.add_nodes(df, "Agent", "id", "name")
    result = g.add_label("Agent", ["a", "unknown"], "Reviewer")
    assert result == {"labelled": 1, "skipped": 1}


def test_add_label_pymethod_idempotent_returns_skipped(g):
    df = pd.DataFrame({"id": ["a"], "name": ["A"]})
    g.add_nodes(df, "Agent", "id", "name")
    g.add_label("Agent", ["a"], "Reviewer")
    result = g.add_label("Agent", ["a"], "Reviewer")
    assert result == {"labelled": 0, "skipped": 1}


def test_remove_label_pymethod(g):
    df = pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]})
    g.add_nodes(df, "Agent", "id", "name")
    g.add_label("Agent", ["a", "b"], "Reviewer")
    result = g.remove_label("Agent", ["a"], "Reviewer")
    assert result == {"removed": 1, "skipped": 0}
    rows = g.cypher("MATCH (a:Reviewer) RETURN a.id AS id").to_list()
    assert [r["id"] for r in rows] == ["b"]


def test_remove_label_pymethod_primary_errors(g):
    df = pd.DataFrame({"id": ["a"], "name": ["A"]})
    g.add_nodes(df, "Agent", "id", "name")
    with pytest.raises(Exception, match=r"(?i)primary label"):
        g.remove_label("Agent", ["a"], "Agent")


# ─── add_nodes(labels=[...]) batch kwarg ──────────────────────────────────


def test_add_nodes_with_labels_kwarg(g):
    df = pd.DataFrame({"id": ["a", "b", "c"], "name": ["A", "B", "C"]})
    g.add_nodes(df, "Agent", "id", "name", labels=["Reviewer"])
    rows = g.cypher("MATCH (a:Reviewer) RETURN a.id AS id ORDER BY a.id").to_list()
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    # Primary type still works.
    rows = g.cypher("MATCH (a:Agent) RETURN a.id AS id ORDER BY a.id").to_list()
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    # Combined AND-intersect.
    rows = g.cypher("MATCH (a:Agent:Reviewer) RETURN a.id AS id ORDER BY a.id").to_list()
    assert [r["id"] for r in rows] == ["a", "b", "c"]


def test_add_nodes_multiple_labels_kwarg(g):
    df = pd.DataFrame({"id": ["a"], "name": ["A"]})
    g.add_nodes(df, "Agent", "id", "name", labels=["Reviewer", "Senior"])
    rows = g.cypher("MATCH (a:Agent) RETURN labels(a) AS labels").to_list()
    assert set(rows[0]["labels"]) == {"Agent", "Reviewer", "Senior"}


def test_add_nodes_labels_kwarg_none_is_unchanged(g):
    """Default (no labels kwarg) preserves existing single-label behavior."""
    df = pd.DataFrame({"id": ["a"], "name": ["A"]})
    g.add_nodes(df, "Agent", "id", "name")  # no labels kwarg
    rows = g.cypher("MATCH (a:Agent) RETURN labels(a) AS labels").to_list()
    assert rows[0]["labels"] == ["Agent"]


# ─── save / load round-trip ───────────────────────────────────────────────


def test_save_load_preserves_secondary_labels(g, tmp_path: Path):
    g.cypher("CREATE (n:Person:Director {name: 'Alice'})")
    g.cypher("CREATE (n:Person {name: 'Bob'})")
    g.cypher("MATCH (n:Person {name: 'Bob'}) SET n:Manager")

    save_path = tmp_path / "round_trip.kgl"
    g.save(str(save_path))
    loaded = kglite.load(str(save_path))

    rows = loaded.cypher("MATCH (n:Person) RETURN n.name AS name, labels(n) AS labels ORDER BY n.name").to_list()
    by_name = {r["name"]: set(r["labels"]) for r in rows}
    assert by_name["Alice"] == {"Person", "Director"}
    assert by_name["Bob"] == {"Person", "Manager"}

    # Secondary-only match also works after reload.
    director_rows = loaded.cypher("MATCH (n:Director) RETURN n.name AS name").to_list()
    assert [r["name"] for r in director_rows] == ["Alice"]


def test_disk_save_load_preserves_secondary_labels(tmp_path: Path):
    """Disk mode: secondary labels persist across save+load via the
    `secondary_labels.bin.zst` sidecar added in 0.10.5."""
    import pandas as pd

    disk_dir = tmp_path / "disk_orig"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.add_nodes(
        pd.DataFrame({"id": ["a", "b", "c"], "name": ["A", "B", "C"]}),
        "Agent",
        "id",
        "name",
        labels=["Reviewer"],
    )
    g.cypher("MATCH (n:Agent {id:'a'}) SET n:Verified:Senior")
    g.add_label("Agent", ["c"], "OnCall")

    save_dir = tmp_path / "disk_saved"
    g.save(str(save_dir))

    loaded = kglite.load(str(save_dir))

    # MATCH by every secondary label survives the round-trip.
    rows = loaded.cypher("MATCH (n:Reviewer) RETURN n.id AS id ORDER BY id").to_list()
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    rows = loaded.cypher("MATCH (n:Verified) RETURN n.id AS id").to_list()
    assert [r["id"] for r in rows] == ["a"]
    rows = loaded.cypher("MATCH (n:Senior) RETURN n.id AS id").to_list()
    assert [r["id"] for r in rows] == ["a"]
    rows = loaded.cypher("MATCH (n:OnCall) RETURN n.id AS id").to_list()
    assert [r["id"] for r in rows] == ["c"]
    # AND-intersect across labels also round-trips.
    rows = loaded.cypher("MATCH (n:Agent:Reviewer:Verified) RETURN n.id AS id").to_list()
    assert [r["id"] for r in rows] == ["a"]
    # labels() returns the full set.
    rows = loaded.cypher("MATCH (n:Agent) RETURN labels(n) AS labels ORDER BY n.id").to_list()
    by_id = {i: set(r["labels"]) for i, r in zip(("a", "b", "c"), rows)}
    assert by_id["a"] == {"Agent", "Reviewer", "Verified", "Senior"}
    assert by_id["b"] == {"Agent", "Reviewer"}
    assert by_id["c"] == {"Agent", "Reviewer", "OnCall"}


def test_disk_save_load_no_secondaries_no_sidecar(tmp_path: Path):
    """Single-label disk graphs must not write the sidecar (zero-cost
    invariant for the kglite-docs / Sodir / Wikidata workloads that
    don't use multi-label)."""
    import pandas as pd

    disk_dir = tmp_path / "disk_orig_single"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.add_nodes(
        pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]}),
        "Agent",
        "id",
        "name",
    )

    save_dir = tmp_path / "disk_saved_single"
    g.save(str(save_dir))

    sidecar = save_dir / "secondary_labels.bin.zst"
    assert not sidecar.exists(), (
        f"secondary_labels.bin.zst should not be written for single-label disk graphs (found at {sidecar})"
    )

    loaded = kglite.load(str(save_dir))
    rows = loaded.cypher("MATCH (n:Agent) RETURN labels(n) AS labels ORDER BY n.id").to_list()
    for r in rows:
        assert r["labels"] == ["Agent"]


def test_disk_in_session_labels_and_match(tmp_path: Path):
    """Disk mode: in-session multi-label works regardless of
    persistence (confirms the in-session truth comes from the
    secondary_label_index, not from on-disk NodeData)."""
    import pandas as pd

    disk_dir = tmp_path / "disk_kg"
    g = kglite.KnowledgeGraph(storage="disk", path=str(disk_dir))
    g.add_nodes(
        pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]}),
        "Agent",
        "id",
        "name",
        labels=["Reviewer"],
    )
    # In-session: MATCH and labels() both find the secondary.
    rows = g.cypher("MATCH (n:Reviewer) RETURN n.id AS id ORDER BY id").to_list()
    assert [r["id"] for r in rows] == ["a", "b"]
    rows = g.cypher("MATCH (n:Agent) RETURN labels(n) AS labels ORDER BY n.id").to_list()
    for r in rows:
        assert set(r["labels"]) == {"Agent", "Reviewer"}
    # SET / add_label / REMOVE all work in-session on disk.
    g.cypher("MATCH (n:Agent {id:'a'}) SET n:Verified")
    rows = g.cypher("MATCH (n:Verified) RETURN n.id AS id").to_list()
    assert [r["id"] for r in rows] == ["a"]
    g.add_label("Agent", ["b"], "OnCall")
    rows = g.cypher("MATCH (n:OnCall) RETURN n.id AS id").to_list()
    assert [r["id"] for r in rows] == ["b"]


def test_mapped_full_multi_label(tmp_path: Path):
    """Mapped mode: every multi-label flow works end-to-end including
    save+load."""
    import pandas as pd

    g = kglite.KnowledgeGraph(storage="mapped")
    g.add_nodes(
        pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]}),
        "Agent",
        "id",
        "name",
        labels=["Reviewer"],
    )
    g.cypher("MATCH (n:Agent {id:'a'}) SET n:Verified")
    save_path = tmp_path / "mapped.kgl"
    g.save(str(save_path))
    loaded = kglite.load(str(save_path))
    rows = loaded.cypher("MATCH (n:Verified) RETURN n.id AS id").to_list()
    assert [r["id"] for r in rows] == ["a"]
    rows = loaded.cypher("MATCH (n:Agent) RETURN labels(n) AS labels ORDER BY n.id").to_list()
    assert set(rows[0]["labels"]) == {"Agent", "Reviewer", "Verified"}
    assert set(rows[1]["labels"]) == {"Agent", "Reviewer"}


def test_single_label_save_load_unchanged(g, tmp_path: Path):
    """Graphs without secondary labels should save/load identically to
    0.10.4 (modulo the wire-format shift; the digest already accounted
    for it). This is the kglite-docs / Sodir / Wikidata workload."""
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    g.add_nodes(df, "Item", "id", "name")
    save_path = tmp_path / "single.kgl"
    g.save(str(save_path))
    loaded = kglite.load(str(save_path))
    rows = loaded.cypher("MATCH (n:Item) RETURN labels(n) AS labels ORDER BY n.id").to_list()
    for r in rows:
        assert r["labels"] == ["Item"]


# ─── 0.10.6 read-path correctness (downstream regression report) ─────────────
#
# Background: a downstream library reported `MATCH (n:Item:Pending) RETURN
# count(n)` over-reporting after remove_label. Root cause was NOT a stale
# index (the index is correct) but read-side count/scan/expansion fast-paths
# that consulted only the primary `type_indices`. These tests pin the
# corrected behaviour against an independent Python oracle computed from the
# (verified-correct) labels() output — the differential harness alone can't,
# because the gates push optimised + naive onto the SAME general path.


def _oracle_labels(g: KnowledgeGraph) -> dict[str, set[str]]:
    """id -> set(labels) for every node, via the verified-correct labels()."""
    rows = g.cypher("MATCH (n) RETURN n.id AS id, labels(n) AS labels").to_list()
    return {r["id"]: set(r["labels"]) for r in rows}


def test_reporter_repro_count_after_remove_label(g):
    """The exact downstream repro: count by multi-label / secondary label
    must track labels(n) through add_label / remove_label."""
    g.add_nodes(pd.DataFrame([{"id": "n1"}, {"id": "n2"}]), "Item", "id", "id")
    g.add_label("Item", ["n1", "n2"], "Pending")

    def mc(label: str) -> int:
        return g.cypher(f"MATCH (n:Item:{label}) RETURN count(n) AS c").to_list()[0]["c"]

    assert mc("Pending") == 2
    assert g.remove_label("Item", ["n1"], "Pending") == {"removed": 1, "skipped": 0}
    assert mc("Pending") == 1  # the original over-report
    # secondary-only label, and primary unchanged
    assert g.cypher("MATCH (n:Pending) RETURN count(n) AS c").to_list()[0]["c"] == 1
    assert g.cypher("MATCH (n:Item) RETURN count(n) AS c").to_list()[0]["c"] == 2
    # labels(n) and WHERE 'X' IN labels(n) agree with the predicate match
    assert g.cypher("MATCH (n:Item) WHERE 'Pending' IN labels(n) RETURN count(n) AS c").to_list()[0]["c"] == 1


def test_reporter_repro_survives_save_load(g, tmp_path: Path):
    g.add_nodes(pd.DataFrame([{"id": "n1"}, {"id": "n2"}]), "Item", "id", "id")
    g.add_label("Item", ["n1", "n2"], "Pending")
    g.remove_label("Item", ["n1"], "Pending")
    p = tmp_path / "t.kgl"
    g.save(str(p))
    reloaded = kglite.load(str(p))
    assert reloaded.cypher("MATCH (n:Item:Pending) RETURN count(n) AS c").to_list()[0]["c"] == 1


def test_parity_count_and_rows_by_label(multi_label_graph):
    """MATCH (n:Label) count + rows == Python oracle over labels()."""
    g = multi_label_graph
    oracle = _oracle_labels(g)
    for label in ("Person", "Company", "VIP", "Staff", "Ghost"):
        expected_ids = sorted(i for i, labs in oracle.items() if label in labs)
        got_rows = sorted(r["id"] for r in g.cypher(f"MATCH (n:{label}) RETURN n.id AS id").to_list())
        assert got_rows == expected_ids, f":{label} rows {got_rows} != {expected_ids}"
        got_count = g.cypher(f"MATCH (n:{label}) RETURN count(n) AS c").to_list()[0]["c"]
        assert got_count == len(expected_ids), f":{label} count {got_count} != {len(expected_ids)}"


def test_parity_label_intersection(multi_label_graph):
    g = multi_label_graph
    oracle = _oracle_labels(g)
    expected = sorted(i for i, labs in oracle.items() if {"VIP", "Staff"} <= labs)
    got = sorted(r["id"] for r in g.cypher("MATCH (n:VIP:Staff) RETURN n.id AS id").to_list())
    assert got == expected
    assert g.cypher("MATCH (n:VIP:Staff) RETURN count(n) AS c").to_list()[0]["c"] == len(expected)


def test_parity_edge_aggregate_secondary_endpoint(multi_label_graph):
    """`(a)-[:KNOWS]->(b:VIP)` over a secondary-labelled endpoint == oracle.
    Exercises the gated aggregate fusions + the matcher expansion endpoint
    filter (both primary-only before 0.10.6)."""
    g = multi_label_graph
    oracle = _oracle_labels(g)
    edges = [(r["a"], r["b"]) for r in g.cypher("MATCH (a)-[:KNOWS]->(b) RETURN a.id AS a, b.id AS b").to_list()]
    expected: dict[str, int] = {}
    for a, b in edges:
        if "VIP" in oracle[b]:
            expected[a] = expected.get(a, 0) + 1
    got = {
        r["a"]: r["c"] for r in g.cypher("MATCH (a:Person)-[:KNOWS]->(b:VIP) RETURN a.id AS a, count(b) AS c").to_list()
    }
    assert got == expected


def test_parity_where_label_predicate(multi_label_graph):
    g = multi_label_graph
    oracle = _oracle_labels(g)
    expected = sorted(i for i, labs in oracle.items() if "Person" in labs and "VIP" in labs)
    got = sorted(r["id"] for r in g.cypher("MATCH (n:Person) WHERE n:VIP RETURN n.id AS id").to_list())
    assert got == expected


def test_delete_evicts_secondary_label(multi_label_graph, tmp_path: Path):
    """DETACH DELETE removes the node from secondary-label counts, and the
    repair survives save+load (pre-0.10.6 left a dangling index entry)."""
    g = multi_label_graph
    before = g.cypher("MATCH (n:VIP) RETURN count(n) AS c").to_list()[0]["c"]
    g.cypher("MATCH (n:Person {id:'P2'}) DETACH DELETE n")  # P2 was :VIP
    after = g.cypher("MATCH (n:VIP) RETURN count(n) AS c").to_list()[0]["c"]
    assert after == before - 1
    assert "P2" not in {r["id"] for r in g.cypher("MATCH (n:VIP) RETURN n.id AS id").to_list()}
    p = tmp_path / "del.kgl"
    g.save(str(p))
    reloaded = kglite.load(str(p))
    assert reloaded.cypher("MATCH (n:VIP) RETURN count(n) AS c").to_list()[0]["c"] == after


def test_fluent_select_include_secondary(multi_label_graph):
    g = multi_label_graph
    oracle = _oracle_labels(g)
    expected = sorted(i for i, labs in oracle.items() if "VIP" in labs)
    # primary-only select misses secondary-labelled nodes
    assert len(g.select("VIP")) == 0
    sel = g.select("VIP", include_secondary=True)
    got = sorted(sel.ids().tolist() if hasattr(sel.ids(), "tolist") else sel.ids())
    assert got == expected
    # existing primary select unchanged
    assert len(g.select("Person")) == sum(1 for labs in oracle.values() if "Person" in labs)


def test_gate_suppresses_fusion_on_multilabel(multi_label_graph):
    """Plan-shape: the aggregate fusion must NOT fire when the graph has
    secondary labels (it would mis-filter); it falls to the general path."""
    g = multi_label_graph
    ops = [
        r["operation"] for r in g.cypher("EXPLAIN MATCH (a:Person)-[:KNOWS]->(b:VIP) RETURN a.id, count(b)").to_list()
    ]
    assert not any("FusedMatch" in o for o in ops), ops


def test_single_label_still_fuses(social_graph):
    """Counterpart: on a single-label graph the same shape still fuses —
    proving the gate costs single-label graphs nothing."""
    ops = [
        r["operation"]
        for r in social_graph.cypher("EXPLAIN MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.id, count(b)").to_list()
    ]
    assert any("Fused" in o for o in ops), ops

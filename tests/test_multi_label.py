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

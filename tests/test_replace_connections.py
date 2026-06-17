"""replace_connections edge upsert — operator feedback B3 (2026-06-17).

`replace_connections` sets a source node's edges of one type to exactly the
supplied list (prune-then-add), in one call:

- only edges *of that type* from the *sources in the input* are pruned;
- edges of other types, and edges from untouched sources, survive;
- validation runs before any pruning (a bad DataFrame leaves the graph intact);
- it accepts the same `query` mode and options as `add_connections`.

Also covers the B6 `create_index(created=...)` honesty fix.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


def _graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2], "title": ["d1", "d2"]}), "Doc", "id", "title")
    g.add_nodes(
        pd.DataFrame({"id": ["A", "B", "C"], "title": ["A", "B", "C"]}),
        "Entity",
        "id",
        "title",
    )
    return g


def _mentions(g: KnowledgeGraph, doc_id: int) -> set[str]:
    rows = g.cypher(f"MATCH (d:Doc {{id: {doc_id}}})-[:MENTIONS]->(e:Entity) RETURN e.id AS e").to_list()
    return {r["e"] for r in rows}


def test_replace_sets_exact_edge_set():
    g = _graph()
    g.add_connections(
        pd.DataFrame({"s": [1, 1], "t": ["A", "B"]}),
        "MENTIONS",
        "Doc",
        "s",
        "Entity",
        "t",
    )
    assert _mentions(g, 1) == {"A", "B"}

    # Re-sync doc 1 to [B, C]: the stale 1->A edge is pruned, 1->C added.
    g.replace_connections(
        pd.DataFrame({"s": [1, 1], "t": ["B", "C"]}),
        "MENTIONS",
        "Doc",
        "s",
        "Entity",
        "t",
    )
    assert _mentions(g, 1) == {"B", "C"}


def test_replace_only_touches_sources_in_input():
    g = _graph()
    g.add_connections(
        pd.DataFrame({"s": [1, 2], "t": ["A", "A"]}),
        "MENTIONS",
        "Doc",
        "s",
        "Entity",
        "t",
    )
    assert _mentions(g, 1) == {"A"}
    assert _mentions(g, 2) == {"A"}

    # Re-sync only doc 1 — doc 2's edges must be untouched.
    g.replace_connections(
        pd.DataFrame({"s": [1], "t": ["C"]}),
        "MENTIONS",
        "Doc",
        "s",
        "Entity",
        "t",
    )
    assert _mentions(g, 1) == {"C"}
    assert _mentions(g, 2) == {"A"}


def test_replace_only_touches_named_connection_type():
    g = _graph()
    g.add_connections(pd.DataFrame({"s": [1], "t": ["A"]}), "MENTIONS", "Doc", "s", "Entity", "t")
    g.add_connections(pd.DataFrame({"s": [1], "t": ["B"]}), "CITES", "Doc", "s", "Entity", "t")

    g.replace_connections(pd.DataFrame({"s": [1], "t": ["C"]}), "MENTIONS", "Doc", "s", "Entity", "t")
    assert _mentions(g, 1) == {"C"}
    # The CITES edge from the same source survives.
    cites = g.cypher("MATCH (d:Doc {id: 1})-[:CITES]->(e:Entity) RETURN e.id AS e").to_list()
    assert {r["e"] for r in cites} == {"B"}


def test_replace_on_fresh_source_is_pure_add():
    g = _graph()
    # No existing MENTIONS — replace behaves as a plain add.
    report = g.replace_connections(
        pd.DataFrame({"s": [1, 1], "t": ["A", "B"]}),
        "MENTIONS",
        "Doc",
        "s",
        "Entity",
        "t",
    )
    assert report["connections_created"] == 2
    assert _mentions(g, 1) == {"A", "B"}


def test_replace_validates_before_pruning():
    g = _graph()
    g.add_connections(pd.DataFrame({"s": [1], "t": ["A"]}), "MENTIONS", "Doc", "s", "Entity", "t")
    # A bad target column name must error WITHOUT pruning the existing edge.
    with pytest.raises(Exception):
        g.replace_connections(
            pd.DataFrame({"s": [1], "wrong": ["B"]}),
            "MENTIONS",
            "Doc",
            "s",
            "Entity",
            "t",
        )
    assert _mentions(g, 1) == {"A"}


def test_replace_query_mode():
    g = _graph()
    g.add_connections(pd.DataFrame({"s": [1], "t": ["A"]}), "MENTIONS", "Doc", "s", "Entity", "t")
    # Query mode: re-point doc 1 at every Entity whose id != 'A'.
    g.replace_connections(
        None,
        "MENTIONS",
        "Doc",
        "src",
        "Entity",
        "tgt",
        query="MATCH (d:Doc {id: 1}), (e:Entity) WHERE e.id <> 'A' RETURN d.id AS src, e.id AS tgt",
    )
    assert _mentions(g, 1) == {"B", "C"}


def test_create_index_created_flag_is_honest():
    g = _graph()
    first = g.create_index("Doc", "title")
    assert first["created"] is True
    # Re-creating is idempotent but reports created=False.
    second = g.create_index("Doc", "title")
    assert second["created"] is False

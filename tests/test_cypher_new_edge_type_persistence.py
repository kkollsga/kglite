"""A relationship type introduced via Cypher CREATE/MERGE must register fully
and survive save()/reload (SimulatoRS, 0.12.1 critical data-loss report).

Root cause: Cypher edge CREATE added the new type to the lightweight
`connection_types` cache but not to `connection_type_metadata`. The columnar
`save()` consolidates edges *by registered connection type*, so a brand-new
type's edges were silently dropped on save (and queries warned "unknown
relationship type"). The node side already registered correctly — only the
edge type was lost.
"""

import pandas as pd

import kglite


def _connection_type_names(kg):
    return {c["type"] for c in kg.connection_types()}


def test_new_edge_type_is_registered_in_metadata():
    # The direct root-cause guard: creating an edge of a new type must make
    # that type visible in connection_types() (i.e. connection_type_metadata),
    # not just the internal cache.
    kg = kglite.KnowledgeGraph()
    kg.cypher("CREATE (:A {id: 1})")
    kg.cypher("CREATE (:B {id: 1})")
    assert "LINKS_TO" not in _connection_type_names(kg)
    kg.cypher("MATCH (a:A {id: 1}), (b:B {id: 1}) CREATE (a)-[:LINKS_TO]->(b)")
    assert "LINKS_TO" in _connection_type_names(kg)


def _multi_type_graph():
    """A graph built the way a real loader does — multiple node types and
    several edge types via add_connections — so its save() produces a
    populated columnar edge store (the condition under which an unregistered
    new edge type was dropped)."""
    kg = kglite.KnowledgeGraph()
    for t in ("Spec", "Algo", "Source", "Strategy"):
        kg.add_nodes(
            pd.DataFrame({"id": list(range(1, 6)), "title": [f"{t}{i}" for i in range(1, 6)]}),
            node_type=t,
            unique_id_field="id",
            node_title_field="title",
        )
    for etype, s, t in [("CITES", "Algo", "Source"), ("REFINES", "Strategy", "Algo"), ("DERIVES", "Spec", "Algo")]:
        kg.add_connections(
            pd.DataFrame({"s": [1, 2, 3], "t": [1, 2, 3]}),
            etype,
            s,
            "s",
            t,
            "t",
        )
    return kg


def test_new_edge_type_survives_save_reload(tmp_path):
    p = str(tmp_path / "g.kgl")
    _multi_type_graph().save(p)

    # Reload (now the edge store is columnar), then add a brand-new edge type
    # via Cypher — the exact agent-contract operation that was dropped.
    kg = kglite.load(p)
    kg.cypher("CREATE (:Task {id: 1})")
    kg.cypher("MATCH (t:Task {id: 1}), (s:Spec {id: 1}) CREATE (t)-[:IMPLEMENTS_SPEC]->(s)")
    assert kg.cypher("MATCH (:Task)-[:IMPLEMENTS_SPEC]->() RETURN count(*) AS c").to_dicts() == [{"c": 1}]

    kg.save(p)
    reloaded = kglite.load(p)
    # The edge (and its type) must survive the columnar save consolidation.
    assert "IMPLEMENTS_SPEC" in _connection_type_names(reloaded)
    assert reloaded.cypher("MATCH (:Task)-[:IMPLEMENTS_SPEC]->() RETURN count(*) AS c").to_dicts() == [{"c": 1}]


def test_new_edge_type_via_merge_survives_save_reload(tmp_path):
    p = str(tmp_path / "g.kgl")
    _multi_type_graph().save(p)
    kg = kglite.load(p)
    kg.cypher("CREATE (:Question {id: 1})")
    kg.cypher("MATCH (q:Question {id: 1}), (a:Algo {id: 1}) MERGE (q)-[:ABOUT]->(a)")
    kg.save(p)
    reloaded = kglite.load(p)
    assert reloaded.cypher("MATCH (:Question)-[:ABOUT]->() RETURN count(*) AS c").to_dicts() == [{"c": 1}]

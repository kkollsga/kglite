"""Edge confidence convention: `confidence = "inferred"` marks heuristic
edges (cross-language coupling, detected by best-effort source matching)
so consumers can keep them separate from parsed facts.

Convention:
- **extracted** (the default) — a parsed fact. The property is *absent* on
  these edges, so facts-only queries are `WHERE r.confidence IS NULL`.
- **inferred** — a heuristic edge (e.g. the cross-language `calls_service`
  edges); carries `confidence = "inferred"`.

This test pins that an `inferred` edge property is queryable and survives a
`.kgl` round-trip — the data-model guarantee the cross-language pass relies
on. (No core change is needed: the same edge-property mechanism that
carries `call_count` / `call_lines` on CALLS carries `confidence`.)
"""

import kglite


def test_edge_confidence_queryable_and_roundtrips(tmp_path):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Function {id:'a', title:'a'}), (:Function {id:'b', title:'b'})")
    g.cypher(
        "MATCH (a:Function {id:'a'}), (b:Function {id:'b'}) CREATE (a)-[:CALLS_SERVICE {confidence: 'inferred'}]->(b)"
    )
    got = g.cypher("MATCH ()-[r:CALLS_SERVICE]->() RETURN r.confidence AS c").to_list()
    assert got and got[0]["c"] == "inferred"

    path = str(tmp_path / "conf.kgl")
    g.save(path)
    g2 = kglite.load(path)
    n = g2.cypher("MATCH ()-[r:CALLS_SERVICE]->() WHERE r.confidence = 'inferred' RETURN count(*) AS n").to_list()[0][
        "n"
    ]
    assert n == 1

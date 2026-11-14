"""Regenerate the v8 and v9 container goldens under ``tests/fixtures/kgl_containers/``.

Unlike the v5 fixtures, these are written by the build under test: they pin
the byte layout of two containers this tree writes, so a change that alters
what an older reader sees fails ``tests/test_kgl_format_compat.py`` instead of
shipping. Regenerate only for a deliberate format change, in the same commit
as the change and its CHANGELOG entry.

The v9 file needs the topology and index sections, which a save writes above
the size floors. A debug build lets ``KGLITE_RAW_SECTION_FLOORS`` lower them
for this small graph; the sections written are the ones a large graph gets.

    KGLITE_RAW_SECTION_FLOORS=10,10 uv run --no-sync python tests/fixtures/build_container_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import kglite

OUT = Path(__file__).parent / "kgl_containers"

QUERIES = {
    "people": "MATCH (p:Person) RETURN count(p) AS c",
    "orgs": "MATCH (o:Org) RETURN count(o) AS c",
    "works_at": "MATCH (:Person)-[r:WORKS_AT]->(:Org) RETURN count(r) AS c",
    "knows": "MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r) AS c",
    "name_of_3": "MATCH (p:Person {id: 3}) RETURN p.name AS name",
    "since_of_1": "MATCH (:Person {id: 1})-[r:WORKS_AT]->(o:Org) RETURN r.since AS since, o.name AS org",
}
EXPECTED = {
    "people": [{"c": 12}],
    "orgs": [{"c": 3}],
    "works_at": [{"c": 12}],
    "knows": [{"c": 11}],
    "name_of_3": [{"name": "person-3"}],
    "since_of_1": [{"since": 2001, "org": "org-1"}],
}


def build() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher("UNWIND range(1, 12) AS i CREATE (:Person {id: i, name: 'person-' + toString(i), age: 20 + i})")
    g.cypher("UNWIND range(1, 3) AS i CREATE (:Org {id: i, name: 'org-' + toString(i)})")
    g.cypher(
        "UNWIND range(1, 12) AS i MATCH (p:Person {id: i}), (o:Org {id: (i - 1) % 3 + 1}) "
        "CREATE (p)-[:WORKS_AT {since: 2000 + i}]->(o)"
    )
    g.cypher("UNWIND range(1, 11) AS i MATCH (a:Person {id: i}), (b:Person {id: i + 1}) CREATE (a)-[:KNOWS]->(b)")
    return g


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    graph = build()
    for name, layout in (("v8", "compact"), ("v9", "auto")):
        path = OUT / f"{name}.kgl"
        graph.save(str(path), layout=layout)
        magic = path.read_bytes()[:4]
        print(name, path.stat().st_size, "bytes", magic)
    for key, query in QUERIES.items():
        assert graph.cypher(query).to_list() == EXPECTED[key], key

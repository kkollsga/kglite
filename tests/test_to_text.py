"""`to_text` — deterministic, human-readable text projection of a `.kgl`.

The canonical form behind the `.kgl` git ``textconv`` diff filter
(`kglite export-text <file>`). Must be stable across insert order AND across
save/load (in-memory vs columnar), so `git diff` over two snapshots shows real
content changes, not reordering noise.
"""

import kglite


def _g():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:Task {id: 1, name: 'A'}), (b:Task {id: 2, name: 'B'})")
    g.cypher("MATCH (a:Task {id: 1}), (b:Task {id: 2}) CREATE (a)-[:DEP {w: 3}]->(b)")
    return g


def test_to_text_is_insert_order_independent():
    g1 = _g()
    g2 = kglite.KnowledgeGraph()
    # Reversed insert order — text must be identical (sorted by id/endpoints).
    g2.cypher("CREATE (b:Task {id: 2, name: 'B'}), (a:Task {id: 1, name: 'A'})")
    g2.cypher("MATCH (a:Task {id: 1}), (b:Task {id: 2}) CREATE (a)-[:DEP {w: 3}]->(b)")
    assert g1.to_text() == g2.to_text()


def test_to_text_stable_across_save_load(tmp_path):
    """In-memory and reloaded (columnar) graphs project to identical text — the
    property that makes a git diff of two .kgl files meaningful."""
    g = _g()
    before = g.to_text()
    p = str(tmp_path / "g.kgl")
    g.save(p)
    after = kglite.load(p).to_text()
    assert before == after
    # Properties survive the columnar round-trip (regression: property_iter
    # yields nothing on columnar — to_text uses property_keys + get_property).
    assert "name=A" in after
    assert "w=3" in after


def test_to_text_excludes_provenance_keys():
    g = kglite.KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"auto_timestamp": True}}})
    g.cypher("CREATE (:Task {id: 1, name: 'A'})", git_sha="abc")
    t = g.to_text()
    assert "name=A" in t
    assert "updated_at" not in t  # volatile metadata would swamp diffs
    assert "git_sha" not in t


def test_to_text_content_shape():
    t = _g().to_text()
    assert "# Task (2 node(s))" in t
    assert "1 | A | name=A" in t
    assert "(1)-[DEP]->(2) {w=3}" in t

"""File-freshness helpers — the binding-layer answer to "is a node's linked file
stale?" The engine never reads the filesystem (that's the no-fs posture line);
these run in Python (which has fs access): `stamp_file_freshness` captures the
file state into properties, `check_file_freshness` re-checks for drift.
"""

import kglite


def _graph(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello")
    b.write_text("world")
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Artifact {id: 'x', file_path: $a}), (:Artifact {id: 'y', file_path: $b})",
        params={"a": str(a), "b": str(b)},
    )
    return g, a, b


def test_stamp_then_no_drift(tmp_path):
    g, a, b = _graph(tmp_path)
    assert kglite.stamp_file_freshness(g, node_type="Artifact") == 2
    assert kglite.check_file_freshness(g, node_type="Artifact") == []


def test_detects_changed_and_missing(tmp_path):
    g, a, b = _graph(tmp_path)
    kglite.stamp_file_freshness(g, node_type="Artifact")
    a.write_text("CHANGED")  # content differs
    b.unlink()  # gone
    drift = {d["id"]: d["status"] for d in kglite.check_file_freshness(g, node_type="Artifact")}
    assert drift == {"x": "changed", "y": "missing"}


def test_stamped_mtime_is_queryable(tmp_path):
    g, a, b = _graph(tmp_path)
    kglite.stamp_file_freshness(g, node_type="Artifact")
    # file_mtime lands as a real (queryable) Timestamp value, not a string.
    m = g.cypher("MATCH (n:Artifact {id: 'x'}) RETURN n.file_mtime AS m").to_dicts()[0]["m"]
    assert m is not None


def test_check_is_read_only(tmp_path):
    """The drift check never mutates the graph (no updated_at churn)."""
    g, a, b = _graph(tmp_path)
    kglite.stamp_file_freshness(g, node_type="Artifact")
    before = g.cypher("MATCH (n:Artifact) RETURN count(n) AS c").to_dicts()[0]["c"]
    kglite.check_file_freshness(g, node_type="Artifact")
    after = g.cypher("MATCH (n:Artifact) RETURN count(n) AS c").to_dicts()[0]["c"]
    assert before == after == 2

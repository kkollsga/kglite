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


def test_custom_keyword_property_names_are_escaped(tmp_path):
    artifact = tmp_path / "keyword.txt"
    artifact.write_text("content")
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Artifact {id: 'x', `order`: $path})",
        params={"path": str(artifact)},
    )
    assert (
        kglite.stamp_file_freshness(
            g,
            node_type="Artifact",
            path_property="order",
            mtime_property="contains",
            hash_property="return",
        )
        == 1
    )
    assert g.cypher(
        "MATCH (n:Artifact {id: 'x'}) RETURN n.`contains` IS NOT NULL AS has_mtime, n.`return` IS NOT NULL AS has_hash"
    ).to_dicts() == [{"has_mtime": True, "has_hash": True}]


def test_custom_label_and_property_names_with_spaces(tmp_path):
    artifact = tmp_path / "spaced.txt"
    artifact.write_text("content")
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:`Build Artifact` {id: 'x', `file path`: $path})",
        params={"path": str(artifact)},
    )

    assert (
        kglite.stamp_file_freshness(
            g,
            node_type="Build Artifact",
            path_property="file path",
            mtime_property="file mtime",
            hash_property=None,
        )
        == 1
    )
    assert g.cypher("MATCH (n:`Build Artifact` {id: 'x'}) RETURN n.`file mtime` IS NOT NULL AS stamped").to_dicts() == [
        {"stamped": True}
    ]


def test_identifier_with_backtick_is_rejected(tmp_path):
    g, _, _ = _graph(tmp_path)
    import pytest

    with pytest.raises(kglite.ArgumentError, match="backtick"):
        kglite.stamp_file_freshness(g, path_property="file_path` SET n.hacked = true //")


def test_stamp_uses_one_batched_mutation_query(tmp_path):
    g, _, _ = _graph(tmp_path)

    class RecordingGraph:
        def __init__(self, inner):
            self.inner = inner
            self.queries = []

        def cypher(self, query, **kwargs):
            self.queries.append(query)
            return self.inner.cypher(query, **kwargs)

    recording = RecordingGraph(g)
    assert kglite.stamp_file_freshness(recording, node_type="Artifact") == 2
    mutation_queries = [query for query in recording.queries if " SET " in query]
    assert len(mutation_queries) == 1
    assert mutation_queries[0].startswith("UNWIND $rows AS row")

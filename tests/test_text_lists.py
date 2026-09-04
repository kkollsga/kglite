"""Native lists compose with lexical indexing and ontology property types."""

import pytest

import kglite


@pytest.mark.parametrize("storage", ["memory", "mapped"])
def test_list_documents_match_joined_text_and_skip_whole_invalid_lists(storage):
    g = kglite.KnowledgeGraph(**({"storage": storage} if storage != "memory" else {}))
    values = [
        ["red", "fox fox", None],
        "red fox fox",
        [],
        [None, None],
        ["hidden", 42],
        ["hidden", ["nested"]],
        ["hidden", {"nested": "value"}],
        None,
    ]
    for i, value in enumerate(values):
        g.cypher("CREATE (:Doc {id: $id, body: $body})", params={"id": i, "body": value})
    assert g.build_text_index("Doc", "body") == {"indexed": 4, "skipped": 4, "terms": 2}
    rows = g.cypher("MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', 'fox') AS s ORDER BY id").to_list()
    assert rows[0]["s"] > 0
    assert rows[0]["s"] == pytest.approx(rows[1]["s"])
    assert [r["s"] for r in rows[2:]] == [0.0, 0.0, None, None, None, None]
    assert all(
        r["s"] in (0.0, None) for r in g.cypher("MATCH (d:Doc) RETURN text_bm25(d, 'body', 'hidden') AS s").to_list()
    )


@pytest.mark.parametrize("storage", ["memory", "mapped"])
@pytest.mark.parametrize("count", [1, 1501])
def test_list_refresh_fold_and_rebuild_preserve_semantics(storage, count, tmp_path):
    g = kglite.KnowledgeGraph(**({"storage": storage} if storage != "memory" else {}))
    g.cypher("UNWIND range(1, $count) AS i CREATE (:Doc {id: i, body: 'original'})", params={"count": count})
    g.build_text_index("Doc", "body", auto_refresh_limit=2000)
    for value, expected in [(["new", "needle"], 1), (None, 0), ([None, "needle"], 1)]:
        g.cypher("MATCH (d:Doc) SET d.body = $body", params={"body": value})
        # Persist a pending delta, then exercise exactly the same extraction
        # through incremental fold (one slot) or bulk refresh (1501 slots).
        path = tmp_path / "pending.kgl"
        g.save(str(path))
        loaded = kglite.load(str(path))
        for subject in [g, loaded]:
            rows = subject.cypher(
                "MATCH (d:Doc) WHERE text_bm25(d, 'body', 'needle') > 0 RETURN count(d) AS n"
            ).to_list()
            assert rows == [{"n": count * expected}]
            assert subject.cypher("SHOW INDEXES").to_list()[0]["delta"] == 0


@pytest.mark.parametrize("type_name", ["list", "array", "LiSt", "ARRAY"])
def test_ontology_list_type_validates_container_shape(type_name):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:Doc {id: 1}), (b:Doc {id: 2})")
    for value in [["text"], [], [1, "mixed", None], [["nested"]], "scalar", 4, None]:
        g.cypher("MATCH (a:Doc {id: 1}), (b:Doc {id: 2}) CREATE (a)-[:LINK {tags: $v}]->(b)", params={"v": value})
    g.define_ontology({"classes": {"Doc": {}}, "relationships": {"LINK": {"property_types": {"tags": type_name}}}})
    rows = g.cypher("CALL ontology_audit() YIELD rule, violations, total RETURN *").to_list()
    row = next(r for r in rows if r["rule"] == "LINK.property_types")
    assert (row["violations"], row["total"]) == (2, 7)

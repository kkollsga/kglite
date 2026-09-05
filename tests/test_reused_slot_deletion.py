"""Exact scan order, rollback and reader isolation after low-slot reuse."""

import pytest

import kglite


def rows(graph):
    return graph.cypher("MATCH (n:Item) RETURN n.id AS id, n.title AS title, elementId(n) AS slot").to_list()


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
@pytest.mark.parametrize("held_reader", [False, True])
def test_reused_tail_delete_rolls_back_and_preserves_order(tmp_path, storage, held_reader):
    path = str(tmp_path / "graph")
    options = {"storage": storage}
    if storage == "disk":
        options["path"] = path
    graph = kglite.KnowledgeGraph(**options)
    graph.set_auto_vacuum(None)
    graph.cypher("UNWIND range(0,7) AS i CREATE (:Item {id:toString(i),title:toString(i)})")
    original = rows(graph)
    assert [r["id"] for r in original] == [str(i) for i in range(8)]
    graph.cypher("MATCH (n:Item {id:'0'}) DELETE n")
    graph.cypher("CREATE (:Item {id:'100',title:'reused'})")
    before = rows(graph)
    assert before[:-1] == original[1:]
    assert (before[-1]["id"], before[-1]["title"]) == ("100", "reused")
    if storage != "disk":
        assert before[-1]["slot"] == original[0]["slot"]
    reader = graph.copy() if held_reader else None

    # The trailing expression fails after the real deletion has executed.
    with pytest.raises(Exception, match="duration|month|overflow"):
        graph.cypher(
            "MATCH (n:Item {id:'100'}) DELETE n CREATE (:Item {id:'999',title:toString(duration({months:2147483648}))})"
        )
    assert rows(graph) == before
    if reader is not None:
        assert rows(reader) == before

    graph.cypher("MATCH (n:Item {id:'100'}) DELETE n")
    assert rows(graph) == original[1:]
    if reader is not None:
        assert rows(reader) == before
    graph.save(path)
    loaded = kglite.load(path)
    assert rows(loaded) == original[1:]


def test_deleting_reused_tail_with_duplicate_logical_id_keeps_original():
    graph = kglite.KnowledgeGraph()
    graph.set_auto_vacuum(None)
    graph.cypher("UNWIND range(0,7) AS i CREATE (:Item {id:i,title:toString(i)})")
    original = rows(graph)
    graph.cypher("MATCH (n:Item {id:0}) DELETE n")
    graph.cypher("CREATE (:Item {id:3,title:'duplicate'})")
    before = rows(graph)
    assert before[-1] == {"id": 3, "title": "duplicate", "slot": original[0]["slot"]}
    assert sum(row["id"] == 3 for row in before) == 2
    graph.cypher("MATCH (n:Item {title:'duplicate'}) DELETE n")
    assert rows(graph) == original[1:]
    assert graph.cypher("MATCH (n:Item {id:3}) RETURN n.title").scalar() == "3"

"""An undirected self-loop binds one relationship while degree counts two."""

import pytest

import kglite


def graph_with_loops(storage, tmp_path):
    graph = kglite.KnowledgeGraph(storage=storage, path=str(tmp_path / "disk") if storage == "disk" else None)
    graph.cypher("CREATE(a:N{id:0}),(b:N{id:1}),(a)-[:R{k:0}]->(a),(a)-[:R{k:1}]->(a),(a)-[:R{k:2}]->(b)")
    return graph


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
@pytest.mark.parametrize("disabled", [False, True])
def test_self_loop_binding_identity_and_counts(storage, disabled, tmp_path):
    graph = graph_with_loops(storage, tmp_path)
    rows = graph.cypher(
        "MATCH(a:N)-[r:R]-(b:N) RETURN a.id AS a,r.k AS k,b.id AS b ORDER BY a,k,b",
        disable_optimizer=disabled,
    ).to_list()
    assert rows == [
        {"a": 0, "k": 0, "b": 0},
        {"a": 0, "k": 1, "b": 0},
        {"a": 0, "k": 2, "b": 1},
        {"a": 1, "k": 2, "b": 0},
    ]
    assert graph.cypher("MATCH(a:N)-[:R]-(b:N) RETURN count(*)", disable_optimizer=disabled).scalar() == 4
    assert graph.cypher("MATCH(a:N{id:0})-[:R]-(b:N) RETURN count(*)", disable_optimizer=disabled).scalar() == 3
    assert graph.cypher("MATCH(a:N)-[:R]-(b:N{id:0}) RETURN count(*)", disable_optimizer=disabled).scalar() == 3
    assert graph.cypher("MATCH(a:N)-[r:R]-(a) RETURN count(r)", disable_optimizer=disabled).scalar() == 2
    assert graph.cypher("MATCH(a:N{id:0}) RETURN degree(a)").scalar() == 5
    assert graph.cypher("MATCH(a:N)-[r:R]->(b:N) RETURN count(r)", disable_optimizer=disabled).scalar() == 3
    rows = graph.cypher(
        "MATCH(a:N) OPTIONAL MATCH(a)-[r:R]-(b:N) RETURN a.id AS id,count(r) AS n ORDER BY id",
        disable_optimizer=disabled,
    ).to_list()
    assert rows == [{"id": 0, "n": 3}, {"id": 1, "n": 1}]


@pytest.mark.parametrize("disabled", [False, True])
def test_self_loop_fixed_lowering_and_trail_uniqueness(disabled):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(a:N{id:0}),(a)-[:R{k:0}]->(a),(a)-[:R{k:1}]->(a)")
    for hops, expected in [(1, 2), (2, 2), (3, 0)]:
        assert (
            graph.cypher(
                f"MATCH p=(a:N)-[:R*{hops}..{hops}]-(b:N) RETURN count(*)", disable_optimizer=disabled
            ).scalar()
            == expected
        )
    paths = graph.cypher(
        "MATCH p=(a:N)-[r:R]-(b:N) RETURN r.k AS k,length(p) AS len ORDER BY k", disable_optimizer=disabled
    ).to_list()
    assert paths == [{"k": 0, "len": 1}, {"k": 1, "len": 1}]


@pytest.mark.parametrize("disabled", [False, True])
def test_self_loop_endpoint_predicate_pushdown(disabled):
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE(a:N{id:0}),(a)-[:R{k:0}]->(a)")
    for condition in ["startNode(r)=b", "endNode(r)=b", "startNode(r)=a", "endNode(r)=a"]:
        assert graph.cypher(
            f"MATCH(a:N)-[r:R]-(b:N) WHERE {condition} RETURN r.k AS k", disable_optimizer=disabled
        ).to_list() == [{"k": 0}]


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
@pytest.mark.parametrize("disabled", [False, True])
def test_endpoint_predicates_use_entity_slots_across_representations(storage, disabled, tmp_path):
    graph = kglite.KnowledgeGraph(storage=storage, path=str(tmp_path / "disk") if storage == "disk" else None)
    graph.cypher(
        "CREATE(a:N{id:0,title:'same',v:7}),(b:N{id:1,title:'same',v:7}),(c:M{id:0,title:'same',v:7}),(a)-[:R]->(b)"
    )
    row = graph.cypher(
        "MATCH(a:N)-[r:R]->(b:N) MATCH(c:M) RETURN "
        "startNode(r)=a AS start,a=startNode(r) AS reversed,endNode(r)=b AS end,"
        "startNode(r)=b AS different,startNode(r)=c AS same_user_id,"
        "[startNode(r)]=[a] AS nested_list,{n:endNode(r)}={n:b} AS nested_map,"
        "endNode(r) IN [null,a,b] AS member,startNode(r) IN [null,b] AS unknown,"
        "startNode(r)<>a AS ne,startNode(r)<>b AS different_ne",
        disable_optimizer=disabled,
    ).to_list()
    assert row == [
        {
            "start": True,
            "reversed": True,
            "end": True,
            "different": False,
            "same_user_id": False,
            "nested_list": True,
            "nested_map": True,
            "member": True,
            "unknown": None,
            "ne": False,
            "different_ne": True,
        }
    ]
    assert graph.cypher(
        "MATCH(a:N{id:0}) OPTIONAL MATCH(a)-[missing:ABSENT]->(b) "
        "RETURN startNode(missing)=a AS eq,a=endNode(missing) AS reversed",
        disable_optimizer=disabled,
    ).to_list() == [{"eq": None, "reversed": None}]

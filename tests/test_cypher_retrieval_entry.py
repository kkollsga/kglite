"""Initial HNSW retrieval preserves the existing materialized route's values."""

import pytest

from tests.test_retrieval_options import _graph

QUERY = (
    "MATCH (d:Doc) RETURN d.id AS id, d.title AS title, "
    "vector_score(d, 'summary_emb', [1.0,0.0]) AS s ORDER BY s DESC LIMIT 4"
)


def _same_routes(graph, query=QUERY):
    ordinary = graph.cypher(query)
    profiled = graph.cypher("PROFILE " + query)
    assert ordinary.to_list() == profiled.to_list()
    assert ordinary.diagnostics["retrieval"] == profiled.diagnostics["retrieval"]
    return ordinary.to_list()


@pytest.mark.parametrize("limit", [0, 1, 4, 400])
def test_whole_type_hnsw_matches_profiled_route(limit):
    rows = _same_routes(_graph(), QUERY.replace("LIMIT 4", f"LIMIT {limit}"))
    assert len(rows) == min(limit, 320)
    if rows:
        assert rows[0]["id"] == 0
        assert rows[0]["s"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "shape",
    [
        "missing",
        "secondary",
        "extra",
        "alternate",
        "property",
        "where",
        "duplicate",
        "correlated",
        "slots",
        "refresh",
        "stale",
        "exact",
        "node",
    ],
)
def test_entry_declines_do_not_change_rows(shape):
    graph = _graph()
    query = QUERY
    if shape == "missing":
        graph.cypher("CREATE (:Doc {id: 9001, title: 'missing', summary: 'missing'})")
    elif shape == "secondary":
        graph.cypher("CREATE (x:Other {id:9001, title:'secondary', summary:'secondary'}) SET x:Doc")
        graph.set_embeddings("Other", "summary", {9001: [0.0, 1.0]})
    elif shape == "extra":
        graph.cypher("MATCH (d:Doc) WHERE d.id < 10 SET d:Selected")
        query = query.replace("d:Doc", "d:Doc:Selected")
    elif shape == "alternate":
        graph.cypher("CREATE (:Other {id:9001, title:'other', summary:'other'})")
        graph.set_embeddings("Other", "summary", {9001: [0.0, 1.0]})
        query = query.replace("d:Doc", "d:Doc|Other")
    elif shape == "property":
        query = query.replace("d:Doc", "d:Doc {id:0}")
    elif shape == "where":
        query = query.replace("RETURN", "WHERE d.id < 10 RETURN")
    elif shape == "duplicate":
        query = query.replace("RETURN", "UNWIND [1,2] AS x RETURN")
    elif shape == "correlated":
        query = "UNWIND [1,2] AS x CALL { WITH x " + query + " } RETURN x, id, title, s ORDER BY x, s DESC"
    elif shape == "slots":
        graph.cypher("MATCH (d:Doc {id:0}) DETACH DELETE d")
        graph.cypher("CREATE (:Doc {id:9001, title:'reused', summary:'reused'})")
        graph.add_embeddings("Doc", "summary", {9001: [1.0, 0.0]})
        graph.build_vector_index("Doc", "summary")
    elif shape in {"refresh", "stale"}:
        graph.add_embeddings("Doc", "summary", {i: [0.0, 1.0] for i in range(1 if shape == "refresh" else 2)})
        # Refresh is a once-per-event path; consume it once before comparing
        # fixed index routes, then ensure the query still agrees afterwards.
        graph.cypher(query).to_list()
    elif shape == "exact":
        query = query.replace("[1.0,0.0])", "[1.0,0.0], {exact:true})")
    elif shape == "node":
        query = query.replace("d.title AS title", "d AS node")
    assert _same_routes(graph, query)


@pytest.mark.parametrize(
    "replacement,error", [("[1.0]", "dimension"), ("[1.0,0.0], {exact:1}", "exact"), ("[1.0,0.0], 'unknown'", "metric")]
)
def test_entry_retains_argument_errors(replacement, error):
    graph = _graph()
    for prefix in ["", "PROFILE "]:
        with pytest.raises(Exception, match=error):
            graph.cypher(prefix + QUERY.replace("[1.0,0.0]", replacement)).to_list()


def test_whole_type_entry_charges_work():
    graph = _graph()
    for prefix in ["", "PROFILE "]:
        with pytest.raises(Exception, match="(?i)(work|budget|limit)"):
            graph.cypher(prefix + QUERY, max_work_units=100)


def test_whole_store_equal_scores_keep_candidate_scan_order():
    graph = _graph(indexed=False)
    graph.set_embeddings("Doc", "summary", {i: [1.0, 0.0] for i in range(320)})
    graph.build_vector_index("Doc", "summary")
    rows = _same_routes(graph, QUERY.replace("LIMIT 4", "LIMIT 400"))
    # Approximate search can visit fewer nodes on duplicate-vector topology.
    # Preserve the established candidate set and its stable row-order ties.
    ids = [row["id"] for row in rows]
    assert ids and ids == sorted(set(ids))
    assert all(row["s"] == 1.0 for row in rows)
    exact = QUERY.replace("[1.0,0.0])", "[1.0,0.0], {exact:true})").replace("LIMIT 4", "LIMIT 400")
    assert [row["id"] for row in _same_routes(graph, exact)] == list(range(320))


def test_held_read_only_snapshot_keeps_retrieval_values_after_write():
    graph = _graph()
    snapshot = graph.freeze()
    before = _same_routes(snapshot)
    graph.cypher("MATCH (d:Doc {id:0}) SET d.title = 'changed'")
    assert _same_routes(snapshot) == before
    assert _same_routes(graph)[0]["title"] == "changed"

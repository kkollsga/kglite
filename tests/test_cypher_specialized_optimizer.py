"""Optimizer-on/off oracles for schema-dependent specialized operators."""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

SPECIALIZED_ORACLE_IDS = {
    "spatial_join",
    "vector_score_top_k",
    "text_score_top_k",
}

PLANNER_SHAPE_TRIGGER_IDS = {
    "reorder_match_clauses",
    "reorder_cyclic_pattern_edges",
    "optimize_pattern_start_node",
    "reorder_match_patterns",
}


class DeterministicEmbedder:
    dimension = 2
    model_id = "optimizer-oracle-v1"

    def embed(self, texts):
        vectors = {
            "alpha": [1.0, 0.0],
            "mixed": [0.8, 0.2],
            "beta": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


@pytest.fixture
def specialized_spatial_graph():
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2],
                "title": ["large", "small"],
                "geometry": [
                    "POLYGON((0 0,4 0,4 4,0 4,0 0))",
                    "POLYGON((1 1,2 1,2 2,1 2,1 1))",
                ],
            }
        ),
        "Area",
        "id",
        "title",
        column_types={"geometry": "geometry"},
    )
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": [10, 11, 12],
                "title": ["inside-both", "large-only", "outside"],
                "latitude": [1.5, 3.0, 8.0],
                "longitude": [1.5, 3.0, 8.0],
            }
        ),
        "Point",
        "id",
        "title",
        column_types={"latitude": "location.lat", "longitude": "location.lon"},
    )
    return graph


@pytest.fixture
def specialized_vector_graph():
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["alpha", "mixed", "beta"],
                "summary": ["alpha", "mixed", "beta"],
            }
        ),
        "Doc",
        "id",
        "title",
        ["summary"],
    )
    vectors = {1: [1.0, 0.0], 2: [0.8, 0.2], 3: [0.0, 1.0]}
    graph.set_embeddings("Doc", "summary", vectors)
    graph.set_embedder(DeterministicEmbedder())
    return graph


@pytest.fixture
def planner_shape_graph():
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": list(range(50)), "title": [f"b{i}" for i in range(50)]}),
        "Big",
        "id",
        "title",
    )
    graph.add_nodes(pd.DataFrame({"id": [100, 101], "title": ["s0", "s1"]}), "Small", "id", "title")
    graph.add_nodes(
        pd.DataFrame({"id": [200, 201, 202, 203], "title": ["m0", "m1", "m2", "m3"]}),
        "Mid",
        "id",
        "title",
    )
    graph.add_nodes(
        pd.DataFrame({"id": [10_000, 20_000], "title": ["common-anchor", "rare-anchor"]}),
        "Anchor",
        "id",
        "title",
    )
    graph.add_connections(
        pd.DataFrame({"src": list(range(50)), "dst": [100 + i % 2 for i in range(50)]}),
        "TO_SMALL",
        "Big",
        "src",
        "Small",
        "dst",
    )
    graph.add_connections(
        pd.DataFrame({"src": list(range(50)), "dst": [10_000] * 50}),
        "COMMON",
        "Big",
        "src",
        "Anchor",
        "dst",
    )
    graph.add_connections(
        pd.DataFrame({"src": [0], "dst": [20_000]}),
        "RARE",
        "Big",
        "src",
        "Anchor",
        "dst",
    )
    graph.add_connections(
        pd.DataFrame({"src": [100, 101], "dst": [200, 201]}),
        "S_TO_M",
        "Small",
        "src",
        "Mid",
        "dst",
    )
    graph.add_connections(
        pd.DataFrame({"src": [200, 201], "dst": [0, 1]}),
        "M_TO_B",
        "Mid",
        "src",
        "Big",
        "dst",
    )
    graph.rebuild_caches()
    return graph


PLANNER_SHAPE_TRIGGERS = {
    "reorder_match_clauses": (
        "MATCH (b)-[:COMMON]->({id: 10000}) MATCH (b)-[:RARE]->({id: 20000}) RETURN b.id AS id ORDER BY id"
    ),
    "reorder_cyclic_pattern_edges": (
        "MATCH (b:Big)-[:TO_SMALL]->(s:Small)-[:S_TO_M]->(m:Mid)-[:M_TO_B]->(b) RETURN b.id AS id ORDER BY id"
    ),
    "optimize_pattern_start_node": ("MATCH (b:Big)-[:TO_SMALL]->(s:Small) RETURN b.id AS b, s.id AS s ORDER BY b"),
    "reorder_match_patterns": ("MATCH (b:Big), (s:Small) RETURN b.id AS b, s.id AS s ORDER BY b, s"),
}


ORACLES = {
    "spatial_join": {
        "fixture": "specialized_spatial_graph",
        "pass": "fuse_spatial_join",
        "operator": "SpatialJoin",
        "query": (
            "MATCH (a:Area), (p:Point) WHERE contains(a, p) "
            "RETURN a.title AS area, p.title AS point ORDER BY area, point"
        ),
        "ids": [("large", "inside-both"), ("large", "large-only"), ("small", "inside-both")],
    },
    "vector_score_top_k": {
        "fixture": "specialized_vector_graph",
        "pass": "fuse_vector_score_order_limit",
        "disable": ["fuse_vector_score_order_limit", "fuse_order_by_top_k"],
        "operator": "FusedVectorScoreTopK",
        "query": (
            "MATCH (d:Doc) RETURN d.id AS id, "
            "vector_score(d, 'summary_emb', $query_vector) AS score "
            "ORDER BY score DESC LIMIT 2"
        ),
        "params": {"query_vector": [1.0, 0.0]},
        "ids": [1, 2],
    },
    "text_score_top_k": {
        "fixture": "specialized_vector_graph",
        "pass": "fuse_vector_score_order_limit",
        "disable": ["fuse_vector_score_order_limit", "fuse_order_by_top_k"],
        "operator": "FusedVectorScoreTopK",
        "query": (
            "MATCH (d:Doc) RETURN d.id AS id, text_score(d, 'summary', 'alpha') AS score ORDER BY score DESC LIMIT 2"
        ),
        "ids": [1, 2],
    },
}


def _plan(graph, query, *, params, disabled_passes=None):
    rows = graph.cypher(
        f"EXPLAIN {query}",
        params=params,
        disabled_passes=disabled_passes,
    ).to_list()
    return "\n".join(row["operation"] for row in rows)


@pytest.mark.parametrize("oracle_id", sorted(SPECIALIZED_ORACLE_IDS))
def test_specialized_optimizer_oracle(oracle_id, request):
    oracle = ORACLES[oracle_id]
    graph = request.getfixturevalue(oracle["fixture"])
    query = oracle["query"]
    params = oracle.get("params", {})
    pass_name = oracle["pass"]

    optimized = graph.cypher(query, params=params).to_list()
    disabled = oracle.get("disable", [pass_name])
    pass_disabled = graph.cypher(query, params=params, disabled_passes=disabled).to_list()
    optimizer_disabled = graph.cypher(query, params=params, disable_optimizer=True).to_list()
    assert optimized == pass_disabled == optimizer_disabled

    if oracle_id == "spatial_join":
        assert [(row["area"], row["point"]) for row in optimized] == oracle["ids"]
    else:
        assert [row["id"] for row in optimized] == oracle["ids"]
        assert optimized[0]["score"] == pytest.approx(1.0)

    enabled_plan = _plan(graph, query, params=params)
    disabled_plan = _plan(graph, query, params=params, disabled_passes=disabled)
    assert oracle["operator"] in enabled_plan
    assert oracle["operator"] not in disabled_plan
    assert f"OptimizerPass {pass_name}" in enabled_plan
    assert f"OptimizerPass {pass_name}" not in disabled_plan


def test_specialized_oracle_registry_is_complete():
    assert set(ORACLES) == SPECIALIZED_ORACLE_IDS
    assert set(PLANNER_SHAPE_TRIGGERS) == PLANNER_SHAPE_TRIGGER_IDS


@pytest.mark.parametrize("pass_name", sorted(PLANNER_SHAPE_TRIGGER_IDS))
def test_schema_dependent_pass_trigger(pass_name, planner_shape_graph):
    query = PLANNER_SHAPE_TRIGGERS[pass_name]
    optimized = planner_shape_graph.cypher(query).to_list()
    disabled = planner_shape_graph.cypher(query, disabled_passes=[pass_name]).to_list()
    assert optimized == disabled

    enabled_plan = _plan(planner_shape_graph, query, params={})
    disabled_plan = _plan(planner_shape_graph, query, params={}, disabled_passes=[pass_name])
    assert f"OptimizerPass {pass_name}" in enabled_plan
    assert f"OptimizerPass {pass_name}" not in disabled_plan


def test_vector_top_k_equal_scores_preserve_input_order():
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": [0, 1, 2], "title": ["a", "b", "c"], "summary": ["x", "y", "z"]}),
        "Doc",
        "id",
        "title",
        ["summary"],
    )
    graph.set_embeddings("Doc", "summary", {i: [1.0, 0.0] for i in range(3)})
    query = (
        "MATCH (d:Doc) RETURN d.id AS id, "
        "vector_score(d, 'summary_emb', [1.0, 0.0]) AS score "
        "ORDER BY score DESC LIMIT 2"
    )
    optimized = graph.cypher(query).to_list()
    naive = graph.cypher(query, disable_optimizer=True).to_list()
    assert [row["id"] for row in optimized] == [0, 1]
    assert optimized == naive


def test_generic_top_k_equal_keys_preserve_input_order():
    graph = kglite.KnowledgeGraph()
    query = "UNWIND [0, 1, 2] AS id RETURN id, 1 AS score ORDER BY score DESC LIMIT 2"
    optimized = graph.cypher(query).to_list()
    naive = graph.cypher(query, disable_optimizer=True).to_list()
    assert [row["id"] for row in optimized] == [0, 1]
    assert optimized == naive


def test_global_two_hop_count_fusion_matches_naive_with_self_loop():
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": [0, 1, 2], "title": ["a", "b", "c"]}), "N", "id", "title")
    graph.add_connections(
        pd.DataFrame({"src": [0, 1, 1], "dst": [1, 2, 1]}),
        "R",
        "N",
        "src",
        "N",
        "dst",
    )
    query = "MATCH (a:N)-[:R]->(b:N)-[:R]->(c:N) RETURN count(*) AS paths"

    optimized = graph.cypher(query).to_list()
    naive = graph.cypher(query, disabled_passes=["fuse_match_return_aggregate"]).to_list()

    assert optimized == naive == [{"paths": 4}]
    assert "FusedMatchReturnAggregate" in _plan(graph, query, params={})

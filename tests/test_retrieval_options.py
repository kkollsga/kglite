"""Explicit exact retrieval, selector independence and metric identity."""

import math

import pandas as pd
import pytest

import kglite


class QueryEmbedder:
    dimension = 2
    model_id = "retrieval-options-test"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _graph(indexed=True, metric="cosine"):
    g = kglite.KnowledgeGraph()
    rows = [
        {"id": i, "title": str(i), "summary": str(i), "x": math.cos(i / 50), "y": math.sin(i / 50)} for i in range(320)
    ]
    g.add_nodes(pd.DataFrame(rows), "Doc", "id", "title")
    g.set_embeddings("Doc", "summary", {r["id"]: [r["x"], r["y"]] for r in rows}, metric=metric)
    g.set_embedder(QueryEmbedder())
    if indexed:
        g.build_vector_index("Doc", "summary", auto_refresh_limit=1)
    return g


@pytest.mark.parametrize("function", ["vector_score", "text_score"])
@pytest.mark.parametrize("options", ["{exact:true}", "$options", "'cosine', {exact:true}"])
def test_exact_request_does_not_refresh_or_consume_index(function, options):
    g = _graph()
    g.add_embeddings("Doc", "summary", {0: [0.0, 1.0]})
    query = "[1.0,0.0]" if function == "vector_score" else "'query'"
    prop = "summary_emb" if function == "vector_score" else "summary"
    rows = g.cypher(
        f"MATCH (d:Doc) RETURN d.id AS id, {function}(d, '{prop}', {query}, {options}) AS s ORDER BY s DESC LIMIT 4",
        params={"options": {"exact": True}},
    ).to_list()
    assert len(rows) == 4
    assert g.cypher("SHOW INDEXES").to_list()[0]["delta"] == 1
    g.cypher("MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', [1.0,0.0]) AS s ORDER BY s DESC LIMIT 4")
    assert g.cypher("SHOW INDEXES").to_list()[0]["delta"] == 0


@pytest.mark.parametrize("options", ["{exact:1}", "{exact:'true'}", "{exact:null}", "{typo:true}", "'cosine', 1"])
def test_invalid_options_fail_for_indexed_and_scalar_paths(options):
    g = _graph()
    for tail in ["ORDER BY s DESC LIMIT 4", ""]:
        with pytest.raises(Exception, match="(?i)(options|exact)"):
            g.cypher(f"MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', [1.0,0.0], {options}) AS s {tail}").to_list()


def test_row_dependent_query_vector_is_never_frozen_at_first_row():
    g = _graph()
    rows = g.cypher(
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [d.x,d.y]) AS s ORDER BY s DESC LIMIT 320"
    ).to_list()
    assert len(rows) == 320
    assert all(r["s"] == pytest.approx(1.0, abs=1e-6) for r in rows)


@pytest.mark.parametrize("options", ["", ", {exact:true}", ", {exact:false}"])
def test_omitted_metric_uses_each_actual_node_store(options):
    g = kglite.KnowledgeGraph()
    for kind, metric in [("A", "cosine"), ("B", "dot_product")]:
        g.add_nodes(pd.DataFrame({"id": [1], "title": [kind], "summary": [kind]}), kind, "id", "title")
        g.set_embeddings(kind, "summary", {1: [2.0, 0.0]}, metric=metric)
    rows = g.cypher(
        f"MATCH (d) RETURN d.title AS title, vector_score(d, 'summary_emb', [1.0,0.0]{options}) AS s ORDER BY title"
    ).to_list()
    assert rows == [{"title": "A", "s": 1.0}, {"title": "B", "s": 2.0}]


@pytest.mark.parametrize("disable_optimizer", [False, True])
def test_json_vector_filter_preserves_metric_errors_and_options(disable_optimizer):
    g = _graph(indexed=False, metric="dot_product")
    g.add_embeddings("Doc", "summary", {0: [2.0, 0.0]})
    rows = g.cypher(
        "MATCH (d:Doc) WHERE vector_score(d, 'summary_emb', '[1.0,0.0]') > 1.5 RETURN d.id AS id",
        disable_optimizer=disable_optimizer,
    ).to_list()
    assert rows == [{"id": 0}]
    for property, query, tail, message in [
        ("missing_emb", "[1.0,0.0]", "", "(?i)embedding"),
        ("summary_emb", "[1.0]", "", "dimension"),
        ("summary_emb", "[1.0,0.0]", ", {exact:1}", "exact"),
    ]:
        with pytest.raises(Exception, match=message):
            g.cypher(
                f"MATCH (d:Doc) WHERE vector_score(d, '{property}', '{query}'{tail}) > 0 RETURN count(d) AS n",
                disable_optimizer=disable_optimizer,
            ).to_list()


@pytest.mark.parametrize("disable_optimizer", [False, True])
@pytest.mark.parametrize("property,vector", [("summary_emb", "[1.0]"), ("missing_emb", "[1.0,0.0]")])
def test_false_left_guard_does_not_evaluate_vector_errors(disable_optimizer, property, vector):
    g = _graph(indexed=False)
    result = g.cypher(
        "MATCH (d:Doc) WITH d WHERE d.id % 1 = 1 "
        f"AND vector_score(d, '{property}', '{vector}') > 0 RETURN count(d) AS n",
        disable_optimizer=disable_optimizer,
    )
    assert result.to_list() == [{"n": 0}]

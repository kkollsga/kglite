"""Exact entry must preserve scalar Values, stable ordering and route telemetry."""

import pytest

from kglite import KnowledgeGraph
from tests.test_retrieval_options import _graph

VECTORS = {0: [0.5, 0.0], 1: [0.0, 0.5], 2: [-0.5, 0.0], 3: [0.0, 0.0], 4: [0.5, 0.0]}
QUERY = "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [0.25,0.0]%s) AS s ORDER BY s DESC LIMIT %s"


def fixture(metric="cosine", indexed=False):
    graph = KnowledgeGraph()
    graph.set_auto_vacuum(None)
    for i in VECTORS:
        graph.cypher("CREATE (:Doc {id:$id, summary:'doc'})", params={"id": i})
    graph.set_embeddings("Doc", "summary", VECTORS, metric=metric)
    if indexed:
        graph.build_vector_index("Doc", "summary", auto_refresh_limit=1)
    return graph


def same_routes(graph, query, *, unfused=True):
    result = graph.cypher(query)
    profiled = graph.cypher("PROFILE " + query)
    assert result.to_list() == profiled.to_list()
    assert result.diagnostics["retrieval"] == profiled.diagnostics["retrieval"]
    if unfused:
        assert result.to_list() == graph.cypher(query, disable_optimizer=True).to_list()
    return result


@pytest.mark.parametrize(
    "metric,ids",
    [
        ("cosine", [0, 4, 1, 3, 2]),
        ("dot_product", [0, 4, 1, 3, 2]),
        ("euclidean", [0, 3, 4, 1, 2]),
        ("poincare", [3, 0, 4, 1, 2]),
    ],
)
@pytest.mark.parametrize("policy", ["auto", "exact", "indexed_exact"])
def test_exact_entry_keeps_scalar_metric_values_and_stable_ties(metric, ids, policy):
    indexed = policy == "indexed_exact"
    graph = fixture("cosine" if indexed else metric, indexed)
    options = f", '{metric}', {{exact:true}}" if indexed else (", {exact:true}" if policy == "exact" else "")
    result = same_routes(graph, QUERY % (options, 20))
    assert [row["id"] for row in result.to_list()] == ids
    info = result.diagnostics["retrieval"]
    assert len(info) == 1
    assert info[0]["actual_mode"] == "exact"
    assert info[0]["fallback_reason"] == ("no_index" if policy == "auto" else "forced_exact")
    assert info[0]["store"] == ("Doc.summary_emb" if policy == "auto" else None)
    if metric == "cosine":
        assert [row["s"] for row in result.to_list()] == [1.0, 1.0, 0.0, 0.0, -1.0]


@pytest.mark.parametrize("limit", [0, 1, 5, 50, 9_223_372_036_854_775_807])
def test_exact_entry_limits_and_zero_query(limit):
    graph = fixture()
    result = same_routes(graph, (QUERY % (", {exact:true}", limit)).replace("[0.25,0.0]", "[0.0,0.0]"))
    assert result.to_list() == [{"id": i, "s": 0.0} for i in range(min(limit, 5))]
    if not limit:
        assert result.diagnostics["retrieval"] == []


@pytest.mark.parametrize(
    "shape",
    ["missing", "secondary", "filter", "slots", "unwind", "correlated", "ascending", "multiple_keys", "nulls_last"],
)
def test_exact_entry_fallbacks_keep_rows_and_diagnostics(shape):
    graph = fixture()
    query = QUERY % (", {exact:true}", 5)
    if shape == "missing":
        graph.cypher("CREATE (:Doc {id:9, summary:'missing'})")
    elif shape == "secondary":
        graph.cypher("CREATE (d:Other {id:9, summary:'secondary'}) SET d:Doc")
        graph.set_embeddings("Other", "summary", {9: [0.0, 0.5]})
    elif shape == "filter":
        query = query.replace("RETURN", "WHERE d.id > 0 RETURN")
    elif shape == "slots":
        graph.cypher("MATCH (d:Doc {id:0}) DETACH DELETE d")
        graph.cypher("CREATE (:Doc {id:9, summary:'reused'})")
        graph.add_embeddings("Doc", "summary", {9: [0.5, 0.0]})
    elif shape == "unwind":
        query = query.replace("RETURN", "UNWIND [1,2] AS x RETURN")
    elif shape == "correlated":
        query = "UNWIND [1,2] AS x CALL { WITH x " + query + " } RETURN x, id, s ORDER BY x, s DESC"
    elif shape == "ascending":
        query = query.replace("DESC", "ASC")
    elif shape == "multiple_keys":
        query = query.replace("s DESC", "s DESC, id DESC")
    else:
        query = query.replace("s DESC", "s DESC NULLS LAST")
    result = same_routes(graph, query)
    if shape == "missing":
        assert result.to_list()[0] == {"id": 9, "s": None}


def test_forced_exact_uses_updated_store_without_refreshing_snapshot_index():
    graph = _graph()
    frozen = graph.freeze()
    query = (
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [1.0,0.0], "
        "{exact:true}) AS s ORDER BY s DESC LIMIT 4"
    )
    before = same_routes(frozen, query, unfused=False).to_list()
    graph.add_embeddings("Doc", "summary", {0: [0.0, 1.0]})
    assert same_routes(frozen, query, unfused=False).to_list() == before
    after = same_routes(graph, query).to_list()
    assert after != before
    assert graph.cypher("SHOW INDEXES").to_list()[0]["delta"] == 1


def test_exact_entry_keeps_independent_projection_call_sites():
    graph = fixture()
    query = (QUERY % (", {exact:true}", 5)).replace(
        "AS s ORDER", "AS s, vector_score(d, 'summary_emb', [0.0,0.25], {exact:true}) AS other ORDER"
    )
    result = same_routes(graph, query).to_list()
    assert result[0] == {"id": 0, "s": 1.0, "other": 0.0}
    assert next(row for row in result if row["id"] == 1)["other"] == 1.0


@pytest.mark.parametrize(
    "replacement,error",
    [
        ("[0.25]", "dimension"),
        ("[0.25,0.0], {exact:1}", "exact"),
        ("[0.25,0.0], 'unknown'", "metric"),
        ("'bad json'", "(?i)(parse|vector|list|json)"),
    ],
)
def test_exact_entry_argument_errors_stay_scalar_errors(replacement, error):
    graph = fixture()
    query = (QUERY % ("", 5)).replace("[0.25,0.0]", replacement)
    for prefix in ["", "PROFILE "]:
        with pytest.raises(Exception, match=error):
            graph.cypher(prefix + query)


def test_exact_entry_charges_match_work():
    graph = fixture()
    for prefix in ["", "PROFILE "]:
        with pytest.raises(Exception, match="(?i)(work|budget|limit)"):
            graph.cypher(prefix + QUERY % (", {exact:true}", 1), max_work_units=4)


def test_exact_entry_sorts_second_distinct_parameterized_call():
    graph = fixture()
    graph.cypher("MATCH (d:Doc) SET d.abstract = 'other'")
    graph.set_embeddings(
        "Doc", "abstract", {i: [0.0, 2.0] if i == 1 else [1.0, 0.0] for i in VECTORS}, metric="euclidean"
    )
    query = (
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', $first, 'cosine', "
        "{exact:true}) AS first, vector_score(d, 'abstract_emb', $second, $metric, {exact:true}) "
        "AS second ORDER BY second DESC LIMIT 1"
    )
    params = {"first": [1.0, 0.0], "second": [0.0, 1.0], "metric": "dot_product"}
    result = graph.cypher(query, params=params)
    profiled = graph.cypher("PROFILE " + query, params=params)
    expected = [{"id": 1, "first": 0.0, "second": 2.0}]
    assert result.to_list() == profiled.to_list() == expected
    assert graph.cypher(query, params=params, disable_optimizer=True).to_list() == expected
    assert result.diagnostics["retrieval"] == profiled.diagnostics["retrieval"]


@pytest.mark.parametrize("metric,ids", [("cosine", [0, 1]), ("dot_product", [0, 2]), ("euclidean", [0, 4])])
def test_exact_entry_nonfinite_scores_keep_classes_and_cutoff(metric, ids):
    import math

    graph = fixture()
    graph.set_embeddings(
        "Doc", "summary", {0: [1e30, 0.0], 1: [-1e30, 0.0], 2: [1e30, 1e30], 3: [0.0, 0.0], 4: [1e30, 0.0]}
    )
    query = (QUERY % (f", '{metric}', {{exact:true}}", 2)).replace("[0.25,0.0]", "[1e30,0.0]")

    def classified(result):
        def score(value):
            if math.isnan(value):
                return "nan"
            if math.isinf(value):
                return "positive_inf" if value > 0 else "negative_inf"
            return value, math.copysign(1, value)

        return [(row["id"], score(row["s"])) for row in result.to_list()]

    result = graph.cypher(query)
    profiled = graph.cypher("PROFILE " + query)
    assert classified(result) == classified(profiled) == classified(graph.cypher(query, disable_optimizer=True))
    assert [row["id"] for row in result.to_list()] == ids
    assert result.diagnostics["retrieval"] == profiled.diagnostics["retrieval"]
    expected = {"cosine": "nan", "dot_product": "positive_inf", "euclidean": (-0.0, -1.0)}[metric]
    assert [value for _, value in classified(result)] == [expected, expected]


def test_exact_entry_never_replaces_imported_scored_binding():
    graph = fixture()
    query = (
        "MATCH (d:Doc) WHERE d.id IN [0,2] CALL { WITH d MATCH (d:Doc) RETURN d.id AS id, "
        "vector_score(d, 'summary_emb', [0.25,0.0], {exact:true}) AS s ORDER BY s DESC LIMIT 1 } "
        "RETURN id, s ORDER BY id"
    )
    assert same_routes(graph, query).to_list() == [{"id": 0, "s": 1.0}, {"id": 2, "s": -1.0}]


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "expression",
    [
        "vector_score(d, 'missing_emb', [0.25,0.0], {exact:true})",
        "vector_score(d, 'summary_emb', [0.25], {exact:true})",
    ],
)
def test_exact_empty_and_zero_limit_keep_existing_argument_evaluation(empty, expression):
    graph = fixture()
    if empty:
        graph.cypher("MATCH (d:Doc) DETACH DELETE d")
    query = f"MATCH (d:Doc) RETURN {expression} AS s ORDER BY s DESC LIMIT {5 if empty else 0}"
    for prefix in ["", "PROFILE "]:
        if not empty:
            # Zero literal limits do not produce the positive-limit fused
            # consumer; the existing RETURN evaluates before LIMIT.
            with pytest.raises(Exception, match="(?i)(embedding|dimension)"):
                graph.cypher(prefix + query)
            continue
        result = graph.cypher(prefix + query)
        assert result.to_list() == []
        assert result.diagnostics["retrieval"] == []

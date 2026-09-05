"""Runtime retrieval telemetry describes execution, including nested queries."""

import pytest

from tests.test_retrieval_options import _graph

QUERY = "MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', [1.0,0.0]%s) AS s ORDER BY s DESC LIMIT 4"


@pytest.mark.parametrize("profile", ["", "PROFILE "])
@pytest.mark.parametrize(
    "indexed,options,mode,reason",
    [
        (True, "", "hnsw", None),
        (True, ", {exact:true}", "exact", "forced_exact"),
        (False, "", "exact", "no_index"),
        (True, ", 'euclidean'", "exact", "metric_mismatch"),
    ],
)
def test_requested_and_actual_retrieval(profile, indexed, options, mode, reason):
    result = _graph(indexed=indexed).cypher(profile + QUERY % options)
    records = result.diagnostics["retrieval"]
    assert len(records) == 1
    assert records[0]["requested_policy"] == ("exact" if "true" in options else "auto")
    assert records[0]["actual_mode"] == mode
    assert records[0]["fallback_reason"] == reason
    if mode == "hnsw":
        assert records[0]["store"] == "Doc.summary_emb"


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("stale", "stale_index"),
        ("filtered", "filtered_underfill"),
        ("row_dependent", "row_dependent_selectors"),
        ("ascending", "ordering_requires_exact"),
    ],
)
def test_exact_fallback_reason(scenario, reason):
    g = _graph()
    query = QUERY % ""
    if scenario == "stale":
        g.add_embeddings("Doc", "summary", {0: [0.0, 1.0], 1: [0.0, 1.0]})
    elif scenario == "filtered":
        query = query.replace("RETURN", "WHERE d.id = 160 RETURN")
    elif scenario == "row_dependent":
        query = query.replace("[1.0,0.0]", "[d.x,d.y]")
    else:
        query = query.replace("DESC", "ASC")
    records = g.cypher(query).diagnostics["retrieval"]
    assert len(records) == 1
    assert records[0]["actual_mode"] == "exact"
    assert records[0]["fallback_reason"] == reason


@pytest.mark.parametrize("shape", ["call", "correlated", "union", "mutation"])
def test_nested_retrieval_is_retained(shape):
    query = QUERY % ", {exact:true}"
    if shape == "call":
        query = f"CALL {{ {query} }} RETURN s"
    elif shape == "correlated":
        query = f"UNWIND [1,2] AS x CALL {{ WITH x {query} }} RETURN s"
    elif shape == "union":
        query += " UNION ALL RETURN 0 AS s"
    else:
        query = f"CALL {{ {query} }} CREATE (:Log {{id:s}}) RETURN s"
    records = _graph().cypher(query).diagnostics["retrieval"]
    assert len(records) == 1
    assert records[0]["actual_mode"] == "exact"
    assert records[0]["fallback_reason"] == "forced_exact"


def test_empty_and_explain_do_not_claim_execution():
    g = _graph()
    for query in [(QUERY % "").replace("LIMIT 4", "LIMIT 0"), (QUERY % "").replace("RETURN", "WHERE d.id < 0 RETURN")]:
        assert g.cypher(query).diagnostics["retrieval"] == []
    plan = g.cypher("EXPLAIN " + QUERY % ", {exact:true}")
    assert plan.diagnostics["retrieval"] == []
    assert any("requested=exact" in str(row) for row in plan.to_list())


@pytest.mark.parametrize("vector,order", [("[1.0,0.0]", "ASC"), ("[d.x,d.y]", "DESC")])
def test_literal_exact_policy_is_independent_of_query_vector_and_order(vector, order):
    query = (QUERY % ", {exact:true}").replace("[1.0,0.0]", vector).replace("DESC", order)
    records = _graph().cypher(query).diagnostics["retrieval"]
    assert records[0]["requested_policy"] == "exact"
    assert records[0]["actual_mode"] == "exact"


@pytest.mark.parametrize("scenario", ["unembedded", "duplicate_rows"])
def test_index_cannot_hide_rows_in_diagnostics(scenario):
    g = _graph()
    query = QUERY % ""
    if scenario == "unembedded":
        g.cypher("CREATE (:Doc {id:9001, title:'missing', summary:'missing'})")
    else:
        query = query.replace("RETURN", "UNWIND [1,2] AS x RETURN")
    result = g.cypher(query)
    assert result.diagnostics["retrieval"][0]["fallback_reason"] == "row_coverage"
    assert result.diagnostics["retrieval"][0]["actual_mode"] == "exact"
    assert _graph().cypher("RETURN 1 AS n").diagnostics["retrieval"] == []

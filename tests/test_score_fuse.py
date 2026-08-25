"""The `score_fuse()` Cypher scalar — hybrid retrieval in one query.

`tests/test_text_bm25.py` covers the keyword lane and `tests/test_vector_search.py`
the semantic one; this file covers combining them, and the contract around the
combined number: which signals reach the average, which weights are refused,
and what a row scores when one lane cannot see it.

The ranking *values* live in `tests/golden/` (the `hybrid_*` snapshots over the
12-document corpus). What is pinned here is the behaviour.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

# Two lanes that deliberately disagree. `b` is the paraphrase: its body shares
# no word with the keyword query, and its vector is the query's topic.
DOCS = pd.DataFrame(
    {
        "doc_id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "body": [
            "photosynthesis converts light into sugar",
            "how a plant turns sunshine into food",
            "the quick brown fox jumps over the lazy dog",
        ],
    }
)
VECTORS = {1: [1.0, 0.0], 2: [1.0, 0.0], 3: [0.0, 1.0]}
QUERY_VECTOR = [1.0, 0.0]


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(DOCS, "Doc", "doc_id", "name")
    g.build_text_index("Doc", "body")
    g.set_embeddings("Doc", "body", VECTORS)
    return g


def _one(g: kglite.KnowledgeGraph, expression: str, extra: str = ""):
    """Evaluate a constant `score_fuse` expression."""
    return g.cypher(f"RETURN {expression} AS s{extra}").to_list()[0]["s"]


# ── the marquee query ─────────────────────────────────────────────────────


def test_one_query_ranks_by_both_retrieval_lanes(graph) -> None:
    """Keyword lane, vector lane, fusion, ordering and a limit — the shape the
    whole retrieval lane exists to make possible, in a single statement."""
    rows = graph.cypher(
        "MATCH (d:Doc) "
        "RETURN d.name AS name, "
        "       text_bm25(d, 'body', $q) AS lexical, "
        "       vector_score(d, 'body_emb', $qv) AS semantic, "
        "       score_fuse(text_bm25(d, 'body', $q), vector_score(d, 'body_emb', $qv)) AS score "
        "ORDER BY score DESC LIMIT 2",
        params={"q": "photosynthesis sugar", "qv": QUERY_VECTOR},
    ).to_list()

    assert [row["name"] for row in rows] == ["a", "b"]
    # `b` shares no query word — the keyword lane searched it and found
    # nothing — and reaches second place on the vector lane alone.
    assert rows[1]["lexical"] == 0.0
    assert rows[1]["semantic"] == pytest.approx(1.0)
    assert rows[0]["score"] == pytest.approx((rows[0]["lexical"] + rows[0]["semantic"]) / 2)
    # `c` loses on both lanes and does not survive the LIMIT.
    assert len(rows) == 2


def test_weights_rebalance_the_two_lanes(graph) -> None:
    """BM25 is unbounded and cosine is not, so without weights the lexical
    lane's scale decides the ranking on its own."""
    ranked = graph.cypher(
        "MATCH (d:Doc) "
        "RETURN d.name AS name, "
        "       score_fuse(text_bm25(d, 'body', $q), vector_score(d, 'body_emb', $qv), $w) AS score "
        "ORDER BY score DESC",
        params={"q": "photosynthesis sugar", "qv": QUERY_VECTOR, "w": [0.0, 1.0]},
    ).to_list()

    # All weight on the vector lane: `a` and `b` share a vector and tie above
    # `c`, which the keyword lane had ranked second.
    assert {ranked[0]["name"], ranked[1]["name"]} == {"a", "b"}
    assert ranked[0]["score"] == pytest.approx(1.0)
    assert ranked[2]["name"] == "c"


def test_a_row_no_lane_can_see_is_null_not_zero(graph) -> None:
    """A node created after the index was built, with no embedding either: the
    keyword lane has no document for it and the vector lane has no vector, so
    there is nothing to rank it on."""
    graph.cypher("CREATE (:Doc {doc_id: 99, name: 'z'})")

    row = graph.cypher(
        "MATCH (d:Doc) WHERE d.name = 'z' "
        "RETURN score_fuse(text_bm25(d, 'body', $q), vector_score(d, 'body_emb', $qv)) AS score",
        params={"q": "photosynthesis", "qv": QUERY_VECTOR},
    ).to_list()[0]

    assert row["score"] is None


# ── the contract around the number ────────────────────────────────────────


def test_equal_weights_average_the_present_signals(graph) -> None:
    assert _one(graph, "score_fuse(1.0, 3.0)") == pytest.approx(2.0)
    assert _one(graph, "score_fuse(1.0, 2.0, 6.0)") == pytest.approx(3.0)


def test_an_absent_signal_leaves_the_average(graph) -> None:
    # Not 0.5: null means "this lane could not see the row", and averaging it
    # in as a zero would rank the row below one both lanes disliked.
    assert _one(graph, "score_fuse(1.0, null)") == pytest.approx(1.0)


def test_every_signal_absent_is_null(graph) -> None:
    assert _one(graph, "score_fuse(null, null)") is None


def test_a_trailing_list_weights_the_scores(graph) -> None:
    assert _one(graph, "score_fuse(1.0, 3.0, [3.0, 1.0])") == pytest.approx(1.5)
    # Weights are relative, so a scaled list ranks identically.
    assert _one(graph, "score_fuse(1.0, 3.0, [0.75, 0.25])") == pytest.approx(1.5)


def test_an_absent_signals_weight_leaves_the_denominator(graph) -> None:
    assert _one(graph, "score_fuse(null, 4.0, [3.0, 1.0])") == pytest.approx(4.0)


def test_a_wrong_length_weights_list_is_refused(graph) -> None:
    with pytest.raises(kglite.CypherExecutionError, match="2 weights for 3 scores"):
        _one(graph, "score_fuse(1.0, 2.0, 3.0, [1.0, 1.0])")


def test_a_negative_weight_is_refused(graph) -> None:
    with pytest.raises(kglite.CypherExecutionError, match="weight 2"):
        _one(graph, "score_fuse(1.0, 3.0, [1.0, -1.0])")


def test_a_non_numeric_score_is_refused(graph) -> None:
    with pytest.raises(kglite.CypherExecutionError, match="argument 2"):
        _one(graph, "score_fuse(1.0, 'high')")


def test_fewer_than_two_scores_is_refused(graph) -> None:
    with pytest.raises(kglite.CypherExecutionError, match="2 or more scores"):
        _one(graph, "score_fuse(1.0)")
    with pytest.raises(kglite.CypherExecutionError, match="2 or more scores"):
        _one(graph, "score_fuse(1.0, [1.0])")


def test_the_reciprocal_rank_fusion_recipe_runs(graph) -> None:
    """CYPHER.md documents window-function ranks + `score_fuse` in place of an
    `rrf()` scalar. A documented recipe is a contract."""
    ranked = graph.cypher(
        "MATCH (d:Doc) "
        "WITH d, rank() OVER (ORDER BY text_bm25(d, 'body', $q) DESC) AS lex_rank, "
        "        rank() OVER (ORDER BY vector_score(d, 'body_emb', $qv) DESC) AS vec_rank "
        "RETURN d.name AS name, score_fuse(1.0 / (60 + lex_rank), 1.0 / (60 + vec_rank)) AS score "
        "ORDER BY score DESC",
        params={"q": "photosynthesis sugar", "qv": QUERY_VECTOR},
    ).to_list()

    assert ranked[0]["name"] == "a"
    assert all(row["score"] is not None for row in ranked)


def test_score_fuse_is_listed_by_show_functions(graph) -> None:
    listed = {row["name"]: row for row in graph.cypher("SHOW FUNCTIONS YIELD name, signature").to_list()}
    assert "score_fuse" in listed
    assert listed["score_fuse"]["signature"].startswith("score_fuse(")

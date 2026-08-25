"""The `text_bm25()` Cypher scalar — the query half of the BM25 lifecycle.

`tests/test_text_index.py` covers building and dropping an index; this file
covers reading one: what a row scores, what an *unscoreable* row answers, and
what a query does when the index has fallen behind the graph
(release-train-0-16-10, decision 11a). The ranking *values* are pinned in
`tests/golden/` — what is pinned here is the contract around the number.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


def _docs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "body": [
                "the quick brown fox",
                "a quick brown marmoset appears",
                "slow green turtles",
            ],
        }
    )


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(_docs(), "Doc", "doc_id", "name")
    return g


def _scores(g: kglite.KnowledgeGraph, query: str = "quick fox"):
    view = g.cypher(
        "MATCH (d:Doc) RETURN d.name AS name, text_bm25(d, 'body', $q) AS s ORDER BY name",
        params={"q": query},
    )
    return {row["name"]: row["s"] for row in view.to_list()}, view.warnings


# ── the number, and the two ways there is none ────────────────────────────


def test_an_indexed_document_with_no_shared_term_scores_zero(graph) -> None:
    graph.build_text_index("Doc", "body")

    scores, warnings = _scores(graph)

    assert scores["a"] > 0.0
    # 'c' is indexed and shares no word with the query: searched, no match.
    assert scores["c"] == 0.0
    assert warnings == []


def test_a_node_with_no_document_scores_null_not_zero() -> None:
    """The distinction the whole surface turns on. `0.0` means the document
    was searched and did not match; `null` means it was never searched."""
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Doc {doc_id: 1, name: 'a', body: 'the quick brown fox'}) "
        "CREATE (:Doc {doc_id: 2, name: 'b', body: 'slow green turtles'}) "
        "CREATE (:Doc {doc_id: 3, name: 'c'})"
    )
    g.build_text_index("Doc", "body")

    scores, _ = _scores(g)

    assert scores["a"] > 0.0
    assert scores["b"] == 0.0
    assert scores["c"] is None, "no property, so no document, so no score"


def test_no_index_is_an_error_naming_build_text_index(graph) -> None:
    with pytest.raises(kglite.CypherExecutionError) as excinfo:
        _scores(graph)

    message = str(excinfo.value)
    assert "no text index on 'Doc.body'" in message
    assert "build_text_index('Doc', 'body')" in message


def test_the_error_lists_what_is_indexed_on_the_type(graph) -> None:
    graph.build_text_index("Doc", "name")

    with pytest.raises(kglite.CypherExecutionError, match="Indexed on 'Doc' today: name."):
        _scores(graph)


def test_a_null_query_scores_null_rather_than_erroring(graph) -> None:
    graph.build_text_index("Doc", "body")

    rows = graph.cypher("MATCH (d:Doc) RETURN text_bm25(d, 'body', null) AS s").to_list()

    assert [row["s"] for row in rows] == [None, None, None]


# ── the freshness contract (decision 11a) ─────────────────────────────────


def test_a_document_created_after_the_build_scores_without_a_rebuild(graph) -> None:
    """The end-to-end of the catch-up contract: a small delta is folded in at
    query entry, so nothing about the write path had to touch the index."""
    graph.build_text_index("Doc", "body")

    graph.cypher("CREATE (:Doc {doc_id: 4, name: 'd', body: 'another quick fox'})")

    scores, warnings = _scores(graph)
    assert scores["d"] > 0.0
    assert warnings == [], "a delta under the limit is caught up silently"
    # And the catch-up is recorded, not merely applied to this one query.
    text_row = [r for r in graph.cypher("SHOW INDEXES").to_list() if r["type"] == "FULLTEXT"][0]
    assert text_row["stale"] is False
    assert text_row["delta"] == 0


def test_a_delta_over_the_limit_scores_null_and_warns(graph) -> None:
    graph.build_text_index("Doc", "body", auto_refresh_limit=0)

    graph.cypher("CREATE (:Doc {doc_id: 4, name: 'd', body: 'another quick fox'})")

    scores, warnings = _scores(graph)
    assert scores["a"] > 0.0, "what the index does hold is still served"
    assert scores["d"] is None, "and what it does not hold is null, not zero"
    assert len(warnings) == 1, warnings
    assert "text index 'Doc.body' is stale" in warnings[0]
    assert "up to 1 documents" in warnings[0]
    assert "auto_refresh_limit of 0" in warnings[0]
    assert "build_text_index('Doc', 'body')" in warnings[0]
    # No refresh happened, so the index is still behind.
    text_row = [r for r in graph.cypher("SHOW INDEXES").to_list() if r["type"] == "FULLTEXT"][0]
    assert text_row["stale"] is True
    assert text_row["delta"] == 1


def test_a_read_only_graph_warns_instead_of_catching_up(graph) -> None:
    """A query may not write, and a catch-up is a write. The delta here is
    well under the limit, so the read-only flag is the only thing stopping it."""
    graph.build_text_index("Doc", "body")
    graph.cypher("CREATE (:Doc {doc_id: 4, name: 'd', body: 'another quick fox'})")
    graph.read_only(True)

    scores, warnings = _scores(graph)

    assert scores["d"] is None
    assert len(warnings) == 1, warnings
    assert "read-only" in warnings[0]
    text_row = [r for r in graph.cypher("SHOW INDEXES").to_list() if r["type"] == "FULLTEXT"][0]
    assert text_row["stale"] is True, "the index must not have been refreshed"


def test_a_reloaded_read_only_graph_behaves_the_same(graph, tmp_path) -> None:
    """The shape a user actually reaches read-only through: save, reload,
    lock. The index rides in the `.kgl` with its outstanding delta."""
    graph.build_text_index("Doc", "body")
    graph.cypher("CREATE (:Doc {doc_id: 4, name: 'd', body: 'another quick fox'})")
    path = tmp_path / "g.kgl"
    graph.save(str(path))

    loaded = kglite.load(str(path))
    loaded.read_only(True)

    scores, warnings = _scores(loaded)
    assert scores["a"] > 0.0
    assert scores["d"] is None
    assert any("read-only" in w for w in warnings), warnings


def test_editing_an_indexed_document_is_caught_up_too(graph) -> None:
    graph.build_text_index("Doc", "body")

    graph.cypher("MATCH (d:Doc) WHERE d.name = 'c' SET d.body = 'a quick fox indeed'")

    scores, _ = _scores(graph)
    assert scores["c"] > 0.0, "the rewritten document is re-read, not left stale"


# ── composition with ordinary Cypher ──────────────────────────────────────


def test_where_and_order_by_limit_pick_the_scalar_up(graph) -> None:
    graph.build_text_index("Doc", "body")

    matching = graph.cypher(
        "MATCH (d:Doc) WHERE text_bm25(d, 'body', 'quick') > 0 RETURN d.name AS name ORDER BY name"
    ).to_list()
    assert [row["name"] for row in matching] == ["a", "b"]

    top = graph.cypher(
        "MATCH (d:Doc) RETURN d.name AS name, text_bm25(d, 'body', 'fox') AS s ORDER BY s DESC LIMIT 1"
    ).to_list()
    assert top[0]["name"] == "a"


def test_two_calls_in_one_query_keep_their_own_queries(graph) -> None:
    """A hybrid query scores more than one thing; neither call may be answered
    with the other's query text."""
    graph.build_text_index("Doc", "body")
    graph.build_text_index("Doc", "name")

    row = graph.cypher(
        "MATCH (d:Doc) WHERE d.name = 'a' "
        "RETURN text_bm25(d, 'body', 'fox') AS body_score, "
        "text_bm25(d, 'name', 'a') AS name_score"
    ).to_list()[0]

    assert row["body_score"] > 0.0
    assert row["name_score"] > 0.0


def test_mapped_mode_ranks_identically(graph) -> None:
    graph.build_text_index("Doc", "body")
    memory, _ = _scores(graph)

    mapped = kglite.KnowledgeGraph(storage="mapped")
    mapped.add_nodes(_docs(), "Doc", "doc_id", "name")
    mapped.build_text_index("Doc", "body")

    assert _scores(mapped)[0] == memory


def test_show_functions_advertises_the_scalar(graph) -> None:
    rows = graph.cypher("SHOW FUNCTIONS").to_list()
    entry = [row for row in rows if row["name"] == "text_bm25"]

    assert len(entry) == 1, "the registry is what a client's autocomplete reads"
    assert entry[0]["category"] == "utility"

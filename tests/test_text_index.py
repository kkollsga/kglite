"""The BM25 text-index build surface — `build_text_index` / `drop_text_index`.

These tests cover lifecycle, reports, refusal and storage modes. Query ranking
is covered in test_text_bm25.py and list composition in test_text_lists.py.
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


def test_build_reports_documents_skips_and_vocabulary(graph) -> None:
    report = graph.build_text_index("Doc", "body")

    assert report == {"indexed": 3, "skipped": 0, "terms": 10}
    assert graph.has_text_index("Doc", "body")


def test_a_non_string_or_missing_property_is_skipped_not_indexed() -> None:
    # Built through Cypher, not add_nodes: a mixed-dtype pandas column is
    # stored as text by the columnar loader (it says so in a warning), so it
    # cannot produce the non-string value this test is about.
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Doc {doc_id: 1, body: 'real text'}) "
        "CREATE (:Doc {doc_id: 2, body: ''}) "
        "CREATE (:Doc {doc_id: 3, body: 42}) "
        "CREATE (:Doc {doc_id: 4})"
    )

    report = g.build_text_index("Doc", "body")

    # The empty string is a document (with no terms); the number and the null
    # are not — BM25 indexes text, and a stringified number is not text.
    assert report["indexed"] == 2
    assert report["skipped"] == 2


def test_a_rebuild_is_just_another_build(graph) -> None:
    graph.build_text_index("Doc", "body")
    before = graph.cypher("SHOW INDEXES").to_list()

    graph.build_text_index("Doc", "body")

    assert graph.cypher("SHOW INDEXES").to_list() == before, "a rebuild replaces, it does not add"


def test_drop_reports_whether_an_index_existed(graph) -> None:
    graph.build_text_index("Doc", "body")

    assert graph.drop_text_index("Doc", "body") is True
    assert graph.has_text_index("Doc", "body") is False
    assert graph.drop_text_index("Doc", "body") is False


def test_an_unknown_node_type_is_refused(graph) -> None:
    with pytest.raises(ValueError, match="Unknown node type 'Nope'"):
        graph.build_text_index("Nope", "body")


def test_a_misspelled_property_is_refused_rather_than_silently_empty(graph) -> None:
    with pytest.raises(ValueError, match="bdoy"):
        graph.build_text_index("Doc", "bdoy")
    assert not graph.has_text_index("Doc", "bdoy")


def test_a_non_string_only_property_is_refused(graph) -> None:
    """An index that can never match anything is a trap, not an empty index."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"doc_id": [1, 2], "name": ["a", "b"], "count": [10, 20]}),
        "Doc",
        "doc_id",
        "name",
    )
    with pytest.raises(ValueError, match="not indexable"):
        g.build_text_index("Doc", "count")


def test_the_indexed_column_may_be_a_title_alias(graph) -> None:
    """`name` is the type's title column, so indexing it indexes titles."""
    report = graph.build_text_index("Doc", "name")

    assert report["indexed"] == 3
    assert graph.has_text_index("Doc", "name")


def test_show_indexes_reports_the_text_index_as_fulltext(graph) -> None:
    graph.create_index("Doc", "body")
    graph.build_text_index("Doc", "body")

    rows = graph.cypher("SHOW INDEXES").to_list()
    by_type = {row["type"]: row for row in rows}

    assert set(by_type) == {"PROPERTY", "FULLTEXT"}
    text = by_type["FULLTEXT"]
    assert text["name"] == "Doc.body"
    assert text["labelsOrTypes"] == ["Doc"]
    assert text["properties"] == ["body"]
    assert text["state"] == "ONLINE"
    # The freshness columns are the text index's alone; a hash index is
    # maintained on every write and has no staleness to report.
    assert text["stale"] is False
    assert text["delta"] == 0
    assert by_type["PROPERTY"]["stale"] is None
    assert by_type["PROPERTY"]["delta"] is None
    # `unembedded` is the vector lane's column and is null on every other row.
    assert text["unembedded"] is None
    assert by_type["PROPERTY"]["unembedded"] is None
    # db.indexes() is the same collector, so the two must agree.
    assert (
        graph.cypher(
            "CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, state, stale, delta, unembedded"
        ).to_list()
        == rows
    )


def test_drop_index_ddl_removes_the_text_index_too(graph) -> None:
    """`SHOW INDEXES` prints one canonical name for both structures, so
    `DROP INDEX` on that name must not leave one of them behind and then
    report that the name does not exist."""
    graph.create_index("Doc", "body")
    graph.build_text_index("Doc", "body")

    graph.cypher("DROP INDEX Doc.body")

    assert not graph.has_index("Doc", "body")
    assert not graph.has_text_index("Doc", "body")
    assert graph.cypher("SHOW INDEXES").to_list() == []


def test_schema_lists_the_text_index_distinctly(graph) -> None:
    graph.build_text_index("Doc", "body")

    assert "Doc.body [text]" in graph.schema()["indexes"]


def test_deleting_a_node_does_not_leave_a_ghost_document(graph) -> None:
    """The reuse hazard, from Python: a freed node slot is handed to the next
    node created, and the index addresses documents by that slot."""
    graph.build_text_index("Doc", "body")

    graph.cypher("MATCH (d:Doc) WHERE d.doc_id = 2 DELETE d")
    graph.cypher("CREATE (:Doc {doc_id: 99, name: 'fresh', body: 'unrelated'})")

    # Nothing observable should have broken; the surviving index still reports.
    assert graph.has_text_index("Doc", "body")
    assert graph.cypher("SHOW INDEXES").to_list()[0]["name"] == "Doc.body"


def test_vacuum_drops_the_index(graph) -> None:
    graph.cypher("MATCH (d:Doc) WHERE d.doc_id = 1 DELETE d")
    graph.build_text_index("Doc", "body")

    graph.vacuum()

    assert not graph.has_text_index("Doc", "body"), (
        "vacuum renumbers every node, so every document would point at the wrong one"
    )


def test_mapped_mode_builds() -> None:
    g = kglite.KnowledgeGraph(storage="mapped")
    g.add_nodes(_docs(), "Doc", "doc_id", "name")

    assert g.build_text_index("Doc", "body")["indexed"] == 3


def test_disk_mode_refuses_and_names_the_modes_that_work(tmp_path) -> None:
    g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    g.add_nodes(_docs(), "Doc", "doc_id", "name")

    with pytest.raises(ValueError, match="disk-backed graph"):
        g.build_text_index("Doc", "body")


# ── ranking on a property with no index ───────────────────────────────────
#
# The documented fast path is `WHERE text_bm25(...) > 0 ... ORDER BY
# text_bm25(...) DESC LIMIT k`. It is served by a fused scan whose WHERE filter
# drops any row whose predicate will not evaluate — so on a property with no
# index it answered zero rows while the bare scalar raised. Reported downstream
# on 0.16.21.

UNINDEXED = "no text index on 'Doc.body'"


def test_the_scalar_names_the_call_that_builds_the_index(graph) -> None:
    with pytest.raises(kglite.KgError, match=UNINDEXED) as excinfo:
        graph.cypher("MATCH (d:Doc) RETURN text_bm25(d, 'body', 'quick') AS s")

    assert "build_text_index('Doc', 'body')" in str(excinfo.value)


def test_the_ranked_retrieval_shape_refuses_rather_than_answering_empty(graph) -> None:
    with pytest.raises(kglite.KgError, match=UNINDEXED):
        graph.cypher(
            "MATCH (d:Doc) WHERE text_bm25(d, 'body', 'quick') > 0 "
            "RETURN d.name AS n ORDER BY text_bm25(d, 'body', 'quick') DESC LIMIT 5"
        )


def test_a_count_over_the_same_predicate_refuses_rather_than_answering_zero(graph) -> None:
    with pytest.raises(kglite.KgError, match=UNINDEXED):
        graph.cypher("MATCH (d:Doc) WHERE text_bm25(d, 'body', 'quick') > 0 RETURN count(d) AS c")


def test_the_built_index_serves_the_same_shape(graph) -> None:
    """The refusal must be about the missing index, not about the shape."""
    graph.build_text_index("Doc", "body")

    rows = graph.cypher(
        "MATCH (d:Doc) WHERE text_bm25(d, 'body', 'quick') > 0 "
        "RETURN d.name AS n ORDER BY text_bm25(d, 'body', 'quick') DESC LIMIT 5"
    ).to_list()

    assert [row["n"] for row in rows] == ["a", "b"]


# ── catch-up (P10b) ───────────────────────────────────────────────────────
#
# The user-visible half of the freshness contract: how `SHOW INDEXES` reports
# what the graph has done to an index since it was built. The refresh itself is
# driven from the query path and asserted in the Rust tests, which can call it;
# what a user can see today is `stale` and `delta`.


def _text_row(graph: kglite.KnowledgeGraph) -> dict:
    rows = [row for row in graph.cypher("SHOW INDEXES").to_list() if row["type"] == "FULLTEXT"]
    assert len(rows) == 1, rows
    return rows[0]


def test_a_fresh_build_reports_itself_current(graph) -> None:
    graph.build_text_index("Doc", "body")

    row = _text_row(graph)
    assert row["stale"] is False
    assert row["delta"] == 0


def test_a_node_created_after_the_build_shows_up_as_a_delta(graph) -> None:
    graph.build_text_index("Doc", "body")

    graph.cypher("CREATE (:Doc {doc_id: 4, name: 'd', body: 'a later document'})")

    row = _text_row(graph)
    assert row["stale"] is True
    assert row["delta"] == 1


def test_setting_the_indexed_property_shows_up_and_another_property_does_not(graph) -> None:
    graph.build_text_index("Doc", "body")

    graph.cypher("MATCH (d:Doc) WHERE d.doc_id = 1 SET d.other = 'x'")
    assert _text_row(graph)["delta"] == 0, "'other' is not the indexed property"

    graph.cypher("MATCH (d:Doc) WHERE d.doc_id = 1 SET d.body = 'rewritten'")
    assert _text_row(graph)["stale"] is True
    assert _text_row(graph)["delta"] == 1


def test_bulk_ingest_of_another_node_type_leaves_the_index_current(graph) -> None:
    """The watermark steps over creations this index cannot hold, so loading an
    unrelated table does not make every index look stale."""
    graph.build_text_index("Doc", "body")

    graph.add_nodes(
        pd.DataFrame({"company_id": [1, 2, 3], "name": ["Acme", "Globex", "Initech"]}),
        "Company",
        "company_id",
        "name",
    )

    assert _text_row(graph)["stale"] is False
    assert _text_row(graph)["delta"] == 0


def test_deleting_a_node_is_not_staleness(graph) -> None:
    """A delete prunes the document at the delete, so nothing is outstanding —
    otherwise a bulk delete would push every index past its refresh limit."""
    graph.build_text_index("Doc", "body")

    graph.cypher("MATCH (d:Doc) WHERE d.doc_id = 1 DELETE d")

    assert _text_row(graph)["stale"] is False
    assert _text_row(graph)["delta"] == 0


def test_a_rebuild_clears_the_delta(graph) -> None:
    graph.build_text_index("Doc", "body")
    graph.cypher("CREATE (:Doc {doc_id: 4, name: 'd', body: 'a later document'})")
    assert _text_row(graph)["delta"] == 1

    report = graph.build_text_index("Doc", "body")

    assert report["indexed"] == 4
    assert _text_row(graph)["stale"] is False


def test_the_auto_refresh_limit_is_accepted_as_a_keyword(graph) -> None:
    """The limit itself has no Python accessor (`SHOW INDEXES` reports the
    delta, not the ceiling), so what this pins is the call shape."""
    assert graph.build_text_index("Doc", "body", auto_refresh_limit=5)["indexed"] == 3
    assert graph.build_text_index("Doc", "body", 5)["indexed"] == 3


# ── persistence ──────────────────────────────────────────────────────


def test_a_saved_graph_reloads_with_its_text_index(graph, tmp_path) -> None:
    """The index rides in the `.kgl` as its own skippable section, so a reload
    does not silently drop search off a graph that had it."""
    graph.build_text_index("Doc", "body")
    path = tmp_path / "g.kgl"
    graph.save(str(path))

    loaded = kglite.load(str(path))

    assert loaded.has_text_index("Doc", "body")
    row = _text_row(loaded)
    assert row["name"] == "Doc.body"
    assert row["type"] == "FULLTEXT"
    assert row["state"] == "ONLINE"
    assert row["stale"] is False
    assert row["delta"] == 0
    assert "Doc.body [text]" in loaded.schema()["indexes"]


def test_a_reloaded_index_remembers_what_it_has_yet_to_cover(graph, tmp_path) -> None:
    """A stale index must come back stale. Restoring the corpus alone would
    present it as current, and every document written before the save but
    after the build would be silently unsearchable."""
    graph.build_text_index("Doc", "body")
    graph.cypher("CREATE (:Doc {doc_id: 4, name: 'd', body: 'a later document'})")
    assert _text_row(graph)["delta"] == 1

    path = tmp_path / "g.kgl"
    graph.save(str(path))
    loaded = kglite.load(str(path))

    row = _text_row(loaded)
    assert row["stale"] is True
    assert row["delta"] == 1


def test_a_graph_with_no_text_index_reloads_without_one(graph, tmp_path) -> None:
    """The pre-0.16.10 file shape: no section written, nothing to read."""
    path = tmp_path / "g.kgl"
    graph.save(str(path))

    loaded = kglite.load(str(path))

    assert not loaded.has_text_index("Doc", "body")
    assert loaded.cypher("SHOW INDEXES").to_list() == []
    assert loaded.select("Doc").len() == 3


def test_a_reloaded_index_can_still_be_dropped_and_rebuilt(graph, tmp_path) -> None:
    graph.build_text_index("Doc", "body")
    path = tmp_path / "g.kgl"
    graph.save(str(path))
    loaded = kglite.load(str(path))

    assert loaded.drop_text_index("Doc", "body")
    assert not loaded.has_text_index("Doc", "body")
    assert loaded.build_text_index("Doc", "body")["indexed"] == 3

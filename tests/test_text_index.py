"""The BM25 text-index build surface — `build_text_index` / `drop_text_index`.

Ranking itself has no query surface in this release; what is asserted here is
the lifecycle a user can see: what a build reports, what it refuses, how it
shows up in `SHOW INDEXES`, and which storage modes serve it. Scoring
correctness lives in the Rust tests, against the reference oracle.
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
    # db.indexes() is the same collector, so the two must agree.
    assert (
        graph.cypher("CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, state").to_list()
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

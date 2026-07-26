"""`LOAD CSV` — grammar, binding, streaming, and the filesystem capability.

The Neo4j on-ramp clause. What matters here, in order:

1. **Correctness across batch boundaries.** `LOAD CSV` does not execute as a
   clause; it drives the rest of the pipeline over 1000-row batches. Any query
   whose result depends on rows it cannot see in its own batch must therefore
   either be recognised as unbatchable or produce the same answer anyway —
   `test_merge_dedupes_across_batch_boundaries` and
   `test_aggregates_see_the_whole_file` are the two that would catch a broken
   batching rule, and both use inputs larger than one batch on purpose.
2. **Bounded memory.** Peak RSS must not track file size for the ingest shape.
3. **The capability gate.** Reading local files is opt-in; the default is deny.

Every CSV is written under `tmp_path`.
"""

from __future__ import annotations

import csv as csv_module
from pathlib import Path

import pytest

from kglite import KnowledgeGraph

# The engine's batch size (`executor/load_csv.rs::BATCH_ROWS`). Tests that
# must cross a batch boundary derive their row counts from this so they keep
# doing so if the constant moves.
BATCH_ROWS = 1000


def write_csv(path: Path, rows: list[list[str]], header: list[str] | None = None, delimiter: str = ",") -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.writer(handle, delimiter=delimiter)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)
    return path


def people_csv(path: Path, count: int) -> Path:
    return write_csv(
        path,
        [[str(i), f"Person{i}", str(20 + (i % 50))] for i in range(count)],
        header=["id", "name", "age"],
    )


# ---------------------------------------------------------------------------
# Grammar and binding
# ---------------------------------------------------------------------------


def test_with_headers_binds_a_map(tmp_path: Path) -> None:
    path = people_csv(tmp_path / "people.csv", 3)
    g = KnowledgeGraph()
    g.cypher(
        f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row "
        "CREATE (:Person {id: toInteger(row.id), name: row.name, age: toInteger(row.age)})"
    )
    rows = g.cypher("MATCH (p:Person) RETURN p.name AS name ORDER BY name").to_list()
    assert [r["name"] for r in rows] == ["Person0", "Person1", "Person2"]


def test_without_headers_binds_a_zero_indexed_list(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "raw.csv", [["1", "Alice"], ["2", "Bob"]])
    g = KnowledgeGraph()
    g.cypher(f"LOAD CSV FROM 'file://{path}' AS row CREATE (:Person {{id: toInteger(row[0]), name: row[1]}})")
    rows = g.cypher("MATCH (p:Person) RETURN p.id AS id, p.name AS name ORDER BY id").to_list()
    assert rows == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


def test_no_headers_keeps_the_first_line_as_data(tmp_path: Path) -> None:
    """Without WITH HEADERS the header line is data — Neo4j behaves the same,
    and silently eating row 1 would be a data-loss bug."""
    path = write_csv(tmp_path / "raw.csv", [["id", "name"], ["1", "Alice"]])
    g = KnowledgeGraph()
    rows = g.cypher(f"LOAD CSV FROM 'file://{path}' AS row RETURN row[0] AS first").to_list()
    assert [r["first"] for r in rows] == ["id", "1"]


def test_fieldterminator_selects_the_delimiter(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "semi.csv",
        [["1", "Alice"], ["2", "Bob"]],
        header=["id", "name"],
        delimiter=";",
    )
    g = KnowledgeGraph()
    rows = g.cypher(
        f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row FIELDTERMINATOR ';' RETURN row.name AS name"
    ).to_list()
    assert [r["name"] for r in rows] == ["Alice", "Bob"]


def test_bare_filesystem_path_works_like_a_file_url(tmp_path: Path) -> None:
    path = people_csv(tmp_path / "people.csv", 2)
    g = KnowledgeGraph()
    rows = g.cypher(f"LOAD CSV WITH HEADERS FROM '{path}' AS row RETURN row.id AS id").to_list()
    assert len(rows) == 2


def test_source_can_come_from_a_parameter(tmp_path: Path) -> None:
    path = people_csv(tmp_path / "people.csv", 2)
    g = KnowledgeGraph()
    rows = g.cypher(
        "LOAD CSV WITH HEADERS FROM $path AS row RETURN row.name AS name",
        params={"path": str(path)},
    ).to_list()
    assert [r["name"] for r in rows] == ["Person0", "Person1"]


def test_empty_fields_bind_as_null(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "gaps.csv", [["1", "", "x"]], header=["a", "b", "c"])
    g = KnowledgeGraph()
    rows = g.cypher(
        f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row RETURN row.b IS NULL AS b_null, row.c AS c"
    ).to_list()
    assert rows == [{"b_null": True, "c": "x"}]


def test_short_rows_bind_missing_columns_as_null(tmp_path: Path) -> None:
    """A ragged CSV is a data problem, not a parse failure: Neo4j nulls the
    missing fields rather than aborting the whole load."""
    path = tmp_path / "ragged.csv"
    path.write_text("a,b,c\n1,2\n", encoding="utf-8")
    g = KnowledgeGraph()
    rows = g.cypher(
        f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row RETURN row.a AS a, row.c IS NULL AS c_null"
    ).to_list()
    assert rows == [{"a": "1", "c_null": True}]


def test_fields_stay_strings_until_converted(tmp_path: Path) -> None:
    """CSV carries no types. Guessing them would corrupt leading-zero ids, so
    values arrive as strings and conversion is explicit."""
    path = write_csv(tmp_path / "zip.csv", [["01234"]], header=["zip"])
    g = KnowledgeGraph()
    rows = g.cypher(
        f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row RETURN row.zip AS raw, toInteger(row.zip) AS converted"
    ).to_list()
    assert rows == [{"raw": "01234", "converted": 1234}]


# ---------------------------------------------------------------------------
# Batching correctness — the tests that matter most
# ---------------------------------------------------------------------------


def test_merge_dedupes_across_batch_boundaries(tmp_path: Path) -> None:
    """MERGE must see nodes created in *earlier* batches.

    The file repeats 5 distinct cities across 3 batches' worth of rows. If
    per-batch execution lost sight of earlier batches' writes, this would
    create up to 5 cities per batch instead of 5 in total.
    """
    rows = [[f"City{i % 5}"] for i in range(BATCH_ROWS * 3)]
    path = write_csv(tmp_path / "cities.csv", rows, header=["city"])
    g = KnowledgeGraph()
    g.cypher(f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row MERGE (:City {{id: row.city}})")
    count = g.cypher("MATCH (c:City) RETURN count(c) AS n").to_list()[0]["n"]
    assert count == 5


def test_every_row_of_a_multi_batch_file_is_ingested(tmp_path: Path) -> None:
    total = BATCH_ROWS * 2 + 7  # deliberately not a batch multiple
    path = people_csv(tmp_path / "many.csv", total)
    g = KnowledgeGraph()
    g.cypher(f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row CREATE (:Person {{id: toInteger(row.id)}})")
    count = g.cypher("MATCH (p:Person) RETURN count(p) AS n").to_list()[0]["n"]
    assert count == total


def test_aggregates_see_the_whole_file(tmp_path: Path) -> None:
    """`count(*)` is not batchable, so the driver must fall back to one pass
    over every row. A per-batch answer would return 3 rows of 1000, not 1 of
    3000 — this is the assertion that pins the batching rule."""
    total = BATCH_ROWS * 3
    path = people_csv(tmp_path / "many.csv", total)
    g = KnowledgeGraph()
    rows = g.cypher(f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row RETURN count(*) AS n").to_list()
    assert rows == [{"n": total}]


def test_order_by_and_limit_span_the_whole_file(tmp_path: Path) -> None:
    total = BATCH_ROWS * 2
    path = write_csv(
        tmp_path / "nums.csv",
        [[f"{i:06d}"] for i in range(total)],
        header=["n"],
    )
    g = KnowledgeGraph()
    rows = g.cypher(
        f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row RETURN row.n AS n ORDER BY n DESC LIMIT 2"
    ).to_list()
    assert [r["n"] for r in rows] == [f"{total - 1:06d}", f"{total - 2:06d}"]


def test_non_aggregating_return_streams_every_row(tmp_path: Path) -> None:
    total = BATCH_ROWS * 2 + 3
    path = people_csv(tmp_path / "many.csv", total)
    g = KnowledgeGraph()
    rows = g.cypher(f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row RETURN row.id AS id").to_list()
    assert len(rows) == total


def test_where_filters_rows_before_ingest(tmp_path: Path) -> None:
    path = people_csv(tmp_path / "people.csv", BATCH_ROWS * 2)
    g = KnowledgeGraph()
    g.cypher(
        f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row "
        "WITH row WHERE toInteger(row.age) > 60 "
        "CREATE (:Person {id: toInteger(row.id), age: toInteger(row.age)})"
    )
    ages = g.cypher("MATCH (p:Person) RETURN min(p.age) AS lo").to_list()[0]["lo"]
    assert ages > 60


def test_match_after_load_csv_joins_against_the_graph(tmp_path: Path) -> None:
    g = KnowledgeGraph()
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'}), (:Person {id: 2, name: 'Bob'})")
    path = write_csv(tmp_path / "edges.csv", [["1", "2"]], header=["src", "dst"])
    # Pattern-property values take variables, not function calls (a
    # pre-existing engine limitation — `UNWIND ['1'] AS s MATCH (a {id:
    # toInteger(s)})` fails the same way), so convert in a WITH first.
    g.cypher(
        f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row "
        "WITH toInteger(row.src) AS src, toInteger(row.dst) AS dst "
        "MATCH (a:Person {id: src}), (b:Person {id: dst}) "
        "CREATE (a)-[:KNOWS]->(b)"
    )
    edges = g.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r) AS n").to_list()[0]["n"]
    assert edges == 1


def test_empty_data_file_is_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / "headers_only.csv"
    path.write_text("id,name\n", encoding="utf-8")
    g = KnowledgeGraph()
    g.cypher(f"LOAD CSV WITH HEADERS FROM 'file://{path}' AS row CREATE (:Person {{id: row.id}})")
    assert g.cypher("MATCH (p:Person) RETURN count(p) AS n").to_list()[0]["n"] == 0


# ---------------------------------------------------------------------------
# Rejections — each names the construct and the route that works
# ---------------------------------------------------------------------------


def test_http_source_is_rejected_with_the_network_free_explanation() -> None:
    g = KnowledgeGraph()
    with pytest.raises(Exception) as excinfo:
        g.cypher("LOAD CSV FROM 'https://example.com/data.csv' AS row RETURN row")
    message = str(excinfo.value)
    assert "network-free" in message
    assert "file:///" in message
    # Not a syntax error: the statement was understood, the source was not.
    assert "syntax" not in message.lower()


def test_unknown_url_scheme_names_the_supported_sources() -> None:
    g = KnowledgeGraph()
    with pytest.raises(Exception) as excinfo:
        g.cypher("LOAD CSV FROM 's3://bucket/data.csv' AS row RETURN row")
    assert "`s3:` URL scheme" in str(excinfo.value)


def test_load_csv_must_lead_the_query(tmp_path: Path) -> None:
    path = people_csv(tmp_path / "people.csv", 1)
    g = KnowledgeGraph()
    with pytest.raises(Exception) as excinfo:
        g.cypher(f"MATCH (n) LOAD CSV FROM 'file://{path}' AS row RETURN row")
    assert "must be the first clause" in str(excinfo.value)


def test_missing_file_reports_the_path() -> None:
    g = KnowledgeGraph()
    with pytest.raises(Exception) as excinfo:
        g.cypher("LOAD CSV FROM 'file:///nonexistent/nope.csv' AS row RETURN row")
    message = str(excinfo.value)
    assert "cannot open" in message
    assert "nope.csv" in message


def test_multi_character_fieldterminator_is_rejected(tmp_path: Path) -> None:
    path = people_csv(tmp_path / "people.csv", 1)
    g = KnowledgeGraph()
    with pytest.raises(Exception) as excinfo:
        g.cypher(f"LOAD CSV FROM 'file://{path}' AS row FIELDTERMINATOR '::' RETURN row")
    assert "single-byte character" in str(excinfo.value)


def test_load_is_still_usable_as_an_ordinary_identifier() -> None:
    """`LOAD` is a soft keyword. Reserving it would break every graph holding a
    property or alias called `load`."""
    g = KnowledgeGraph()
    g.cypher("CREATE (:Meter {id: 1, load: 42})")
    rows = g.cypher("MATCH (m:Meter) RETURN m.load AS load").to_list()
    assert rows == [{"load": 42}]
    assert g.cypher("RETURN 1 AS load").to_list() == [{"load": 1}]


# ---------------------------------------------------------------------------
# EXPLAIN / PROFILE surface
# ---------------------------------------------------------------------------


def test_explain_names_the_load_csv_step(tmp_path: Path) -> None:
    path = people_csv(tmp_path / "people.csv", 1)
    g = KnowledgeGraph()
    rows = g.cypher(f"EXPLAIN LOAD CSV WITH HEADERS FROM 'file://{path}' AS row RETURN row.id AS id").to_list()
    operations = [r["operation"] for r in rows]
    assert any("LoadCsv" in op for op in operations)


def test_profile_reports_one_row_per_clause_not_per_batch(tmp_path: Path) -> None:
    """Batches are an implementation detail. PROFILE must sum them, not emit a
    row per batch per clause."""
    path = people_csv(tmp_path / "many.csv", BATCH_ROWS * 2)
    g = KnowledgeGraph()
    result = g.cypher(f"PROFILE LOAD CSV WITH HEADERS FROM 'file://{path}' AS row RETURN row.id AS id")
    profile = result.profile
    assert profile is not None
    clause_names = [entry["clause"] for entry in profile]
    # One LoadCsv entry and one Return entry — not four.
    assert sum(1 for name in clause_names if "LoadCsv" in name) == 1
    assert sum(1 for name in clause_names if "Return" in name) == 1

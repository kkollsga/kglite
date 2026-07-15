"""DEFINES edge determinism on duplicate-id entities.

Regression net for the 2026-07-15 codingest bug report: total edge counts
flapped across processes on repos where minified CSS/HTML repeats a
selector/element name on one line (entity ids embed file + line, so
same-line repeats produce duplicate ids and duplicate (file, entity)
DEFINES rows). Root cause was the composition of per-process-random
HashMap frame iteration with `add_connections`' initial-load fast path,
which skips edge-existence checks and leaves within-batch consolidation
to the caller.

The durable cross-process oracle lives in codingest's parity corpus; this
test asserts the in-tree consolidation invariant: no parallel duplicate
DEFINES edges, duplicate-id nodes preserved, repeated builds exact-equal.
"""

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402

# Single-line (minified) sources: repeats share the line number, so their
# qualified ids collide — the duplicate-id shape that triggered the bug.
MINIFIED_CSS = ".card{color:red}.card{padding:1em}#hero{margin:0}#hero{border:none}\n"
MINIFIED_HTML = (
    '<html><body><div class="card">one</div><div class="card">two</div>'
    '<span id="x">a</span><span id="x">b</span></body></html>\n'
)
PY_MODULE = """def alpha():
    return 1


def beta():
    return alpha()
"""


@pytest.fixture()
def dup_repo(tmp_path):
    (tmp_path / "app.min.css").write_text(MINIFIED_CSS)
    (tmp_path / "index.html").write_text(MINIFIED_HTML)
    (tmp_path / "app.py").write_text(PY_MODULE)
    return tmp_path


def _edge_counts(graph):
    rows = graph.cypher("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY t").to_dicts()
    return {row["t"]: row["n"] for row in rows}


def test_fixture_exercises_duplicate_id_path(dup_repo):
    graph = build(str(dup_repo))
    dup_groups = graph.cypher(
        "MATCH (n) WITH labels(n)[0] AS t, n.id AS id, count(*) AS c "
        "WHERE c > 1 RETURN t, count(*) AS groups ORDER BY t"
    ).to_dicts()
    assert dup_groups == [
        {"t": "Element", "groups": 1},
        {"t": "Selector", "groups": 2},
    ], "fixture must produce duplicate-id nodes or this test guards nothing"


def test_no_parallel_duplicate_defines_edges(dup_repo):
    graph = build(str(dup_repo))
    dupes = graph.cypher(
        "MATCH (a)-[r:DEFINES]->(b) WITH a, b, count(r) AS c WHERE c > 1 RETURN count(*) AS pairs"
    ).to_dicts()
    assert dupes == [{"pairs": 0}], (
        "duplicate (file, entity) DEFINES rows must consolidate onto one edge "
        "regardless of which type-pair hits the initial-load fast path first"
    )


def test_duplicate_id_nodes_preserved(dup_repo):
    # Consolidation must not collapse the duplicate-id *nodes* — only the
    # parallel edges onto them.
    graph = build(str(dup_repo))
    counts = {
        row["t"]: row["n"]
        for row in graph.cypher(
            "MATCH (n) WHERE n:Selector OR n:Element RETURN labels(n)[0] AS t, count(*) AS n"
        ).to_dicts()
    }
    assert counts == {"Selector": 4, "Element": 2}


def test_repeated_builds_agree_exactly(dup_repo):
    baseline = _edge_counts(build(str(dup_repo)))
    for _ in range(3):
        assert _edge_counts(build(str(dup_repo))) == baseline

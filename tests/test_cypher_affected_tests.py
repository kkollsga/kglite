"""CALL affected_tests({files: [...], max_depth?}) YIELD test_file, depth

Given a seed set of changed file paths, BFS over inbound IMPORTS edges
and yield the subset of reached File nodes whose `is_test` property is
true. Builds on the File→File IMPORTS edges added in 0.9.34.
"""

import pytest

from kglite import KnowledgeGraph


def _file_graph(files: list[tuple[str, bool]], imports: list[tuple[str, str]]) -> KnowledgeGraph:
    """Hand-build a code-schema File graph mirroring what a code-graph build
    emits for the `affected_tests` procedure: File nodes carrying an
    ``is_test`` flag, wired by File→File IMPORTS edges.

    Args:
        files: ``(path, is_test)`` for each File node.
        imports: ``(source_path, target_path)`` File→File IMPORTS edges,
            i.e. *source* imports *target*.
    """
    g = KnowledgeGraph()
    for path, is_test in files:
        filename = path.rsplit("/", 1)[-1]
        g.cypher(
            "CREATE (n:File {id: $id, name: $name, is_test: $is_test})",
            params={"id": path, "name": filename, "is_test": is_test},
        )
    for source, target in imports:
        g.cypher(
            "MATCH (a:File {id: $src}), (b:File {id: $tgt}) CREATE (a)-[:IMPORTS]->(b)",
            params={"src": source, "tgt": target},
        )
    return g


# ── Hand-built code-schema graphs ───────────────────────────────────


def test_direct_test_importer():
    """tests/ file that imports the changed file shows up at depth 1."""
    g = _file_graph(
        [("util.py", False), ("tests/test_util.py", True)],
        [("tests/test_util.py", "util.py")],
    )
    rows = g.cypher(
        "CALL affected_tests({files: ['util.py']}) YIELD test_file, depth RETURN test_file, depth ORDER BY test_file"
    ).to_list()
    assert rows == [{"test_file": "tests/test_util.py", "depth": 1}], rows


def test_transitive_test_importer():
    """Test file importing an importer of the seed: depth 2."""
    g = _file_graph(
        [("util.py", False), ("core.py", False), ("tests/test_core.py", True)],
        [("core.py", "util.py"), ("tests/test_core.py", "core.py")],
    )
    rows = g.cypher(
        "CALL affected_tests({files: ['util.py']}) YIELD test_file, depth RETURN test_file, depth ORDER BY test_file"
    ).to_list()
    assert rows == [{"test_file": "tests/test_core.py", "depth": 2}], rows


def test_max_depth_cuts_off_transitive():
    """max_depth=1 finds only direct importers; transitive ones drop."""
    g = _file_graph(
        [("util.py", False), ("core.py", False), ("tests/test_core.py", True)],
        [("core.py", "util.py"), ("tests/test_core.py", "core.py")],
    )
    rows = g.cypher(
        "CALL affected_tests({files: ['util.py'], max_depth: 1}) YIELD test_file RETURN test_file"
    ).to_list()
    assert rows == [], f"expected no tests at depth>1 with max_depth=1, got {rows}"


def test_seed_test_file_not_emitted():
    """If the seed itself is a test file, it must not appear in the output."""
    g = _file_graph([("tests/test_util.py", True)], [])
    rows = g.cypher("CALL affected_tests({files: ['tests/test_util.py']}) YIELD test_file RETURN test_file").to_list()
    assert rows == [], f"seed should not echo back, got {rows}"


def test_unknown_seed_paths_silently_skip():
    """Seed paths that don't match a File node produce no rows (no error)."""
    g = _file_graph(
        [("util.py", False), ("tests/test_util.py", True)],
        [("tests/test_util.py", "util.py")],
    )
    rows = g.cypher(
        "CALL affected_tests({files: ['nonexistent.py', 'util.py']}) YIELD test_file RETURN test_file"
    ).to_list()
    assert rows == [{"test_file": "tests/test_util.py"}], rows


def test_non_test_importers_excluded():
    """Only files with is_test=true appear; non-test importers are filtered out."""
    g = _file_graph(
        [("util.py", False), ("core.py", False)],
        [("core.py", "util.py")],
    )
    rows = g.cypher("CALL affected_tests({files: ['util.py']}) YIELD test_file RETURN test_file").to_list()
    assert rows == [], f"expected no test files in pure-src graph, got {rows}"


# ── Error handling ─────────────────────────────────────────────────


def test_missing_files_param_raises():
    """Calling without the required `files` parameter must error clearly."""
    from kglite import KnowledgeGraph

    kg = KnowledgeGraph()
    with pytest.raises(Exception) as exc:
        kg.cypher("CALL affected_tests({}) YIELD test_file RETURN test_file").to_list()
    msg = str(exc.value)
    assert "files" in msg, f"error should mention 'files' param: {msg}"


def test_empty_files_list_yields_zero_rows(file_imports_graph):
    """An empty `files: []` is not an error, just yields nothing."""
    rows = file_imports_graph.cypher("CALL affected_tests({files: []}) YIELD test_file RETURN test_file").to_list()
    assert rows == []


def test_yield_only_test_file(file_imports_graph):
    """Yielding only `test_file` (the common case) must work."""
    rows = file_imports_graph.cypher(
        "CALL affected_tests({files: ['src/util.py']}) YIELD test_file RETURN test_file ORDER BY test_file"
    ).to_list()
    assert rows == [
        {"test_file": "tests/test_a.py"},
        {"test_file": "tests/test_util.py"},
    ], rows


def test_unknown_yield_raises(file_imports_graph):
    """YIELD'ing a column the procedure doesn't expose must error."""
    with pytest.raises(Exception) as exc:
        file_imports_graph.cypher(
            "CALL affected_tests({files: ['src/util.py']}) YIELD bogus_col RETURN bogus_col"
        ).to_list()
    assert "bogus_col" in str(exc.value), f"expected error mentioning bogus_col, got {exc.value}"


# ── Synthetic-fixture coverage ─────────────────────────────────────


def test_synthetic_fixture_simple(file_imports_graph):
    """The synthetic fixture exercises the same path as the differential corpus."""
    rows = file_imports_graph.cypher(
        "CALL affected_tests({files: ['src/util.py']}) YIELD test_file, depth "
        "RETURN test_file, depth ORDER BY test_file"
    ).to_list()
    # src/util.py is imported by src/a.py, src/b.py, tests/test_util.py.
    # tests/test_a.py imports src/a.py (depth 2 from util.py).
    # Tests at depth 1: tests/test_util.py. Depth 2: tests/test_a.py (via src/a.py).
    assert rows == [
        {"test_file": "tests/test_a.py", "depth": 2},
        {"test_file": "tests/test_util.py", "depth": 1},
    ], rows


def test_synthetic_fixture_max_depth(file_imports_graph):
    rows = file_imports_graph.cypher(
        "CALL affected_tests({files: ['src/util.py'], max_depth: 1}) "
        "YIELD test_file, depth RETURN test_file ORDER BY test_file"
    ).to_list()
    assert rows == [{"test_file": "tests/test_util.py"}], rows

"""Cross-feature integration tests for the 0.9.34 / 0.9.35 code-tree additions.

Builds one Flask-shaped Python package that exercises every new
surface at once — File→File IMPORTS, DECORATES, Route extraction,
affected_tests, refresh_stats, explore — and asserts they coexist
correctly on the same graph. Per-feature tests live alongside each
step (`test_code_tree_file_imports.py`, `test_code_tree_routes.py`,
etc.); this file guards against silent interaction bugs between them.
"""

import pathlib
import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _build_flask_app(tmp_path: pathlib.Path) -> pathlib.Path:
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "util.py").write_text(
        textwrap.dedent(
            """
            def helper(x):
                return x + 1

            def cache(fn):
                return fn
            """
        )
    )
    (pkg / "handlers.py").write_text(
        textwrap.dedent(
            """
            from app.util import helper, cache

            class _App:
                def route(self, *a, **k): return lambda fn: fn
                def get(self, *a, **k): return lambda fn: fn

            app = _App()

            @app.route('/users')
            @cache
            def list_users():
                return helper(1)

            @app.get('/users/<int:id>')
            def user_detail(id):
                return helper(id)
            """
        )
    )
    tests = pkg / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "test_handlers.py").write_text(
        textwrap.dedent(
            """
            from app.handlers import list_users, user_detail

            def test_list():
                assert list_users() is not None
            """
        )
    )
    return pkg


def test_all_new_edges_coexist(tmp_path):
    """Build one Flask-shaped graph and assert every new edge type
    materialises without interference between passes."""
    pkg = _build_flask_app(tmp_path)
    g = build(str(pkg))

    # File → File IMPORTS (0.9.34)
    file_imports = g.cypher("MATCH (s:File)-[:IMPORTS]->(t:File) RETURN count(*) AS n").to_list()
    assert file_imports[0]["n"] >= 1, file_imports

    # DECORATES (0.9.34)
    decorates = g.cypher("MATCH (d:Function)-[:DECORATES]->(f:Function) RETURN d.name AS d, f.name AS f").to_list()
    # `cache` decorates `list_users`. `app.route`/`app.get` are method
    # decorators on the `_App` instance — `route`/`get` are class methods
    # so they may also surface depending on resolution.
    assert any(r["d"] == "cache" and r["f"] == "list_users" for r in decorates), decorates

    # Routes + HANDLES (0.9.34)
    routes = g.cypher(
        "MATCH (r:Route)-[:HANDLES]->(f:Function) "
        "RETURN r.framework AS fw, r.method AS m, r.path AS p, f.name AS h "
        "ORDER BY p, m"
    ).to_list()
    assert any(r["p"] == "/users" and r["h"] == "list_users" for r in routes), routes
    assert any(r["p"] == "/users/<int:id>" and r["h"] == "user_detail" for r in routes), routes


def test_affected_tests_finds_test_via_imports(tmp_path):
    """A change to util.py should surface tests/test_handlers.py via
    transitive IMPORTS (util.py ← handlers.py ← test_handlers.py)."""
    pkg = _build_flask_app(tmp_path)
    g = build(str(pkg))
    rows = g.cypher(
        "CALL affected_tests({files: ['util.py']}) YIELD test_file RETURN test_file ORDER BY test_file"
    ).to_list()
    paths = {r["test_file"] for r in rows}
    assert any("test_handlers.py" in p for p in paths), paths


def test_refresh_stats_includes_route_handles_triple(tmp_path):
    """refresh_stats output must enumerate the new HANDLES edge type
    once route extraction has populated it."""
    pkg = _build_flask_app(tmp_path)
    g = build(str(pkg))
    rows = g.cypher(
        "CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count RETURN src_type, edge_type, tgt_type, count"
    ).to_list()
    by_edge = {r["edge_type"] for r in rows}
    assert "HANDLES" in by_edge, by_edge
    assert "DECORATES" in by_edge, by_edge
    assert "IMPORTS" in by_edge, by_edge


def test_label_pair_counts_separates_handler_routes(tmp_path):
    """label_pair_counts should distinguish HANDLES targets if routes
    point at different Function subtypes (here all targets are Function
    so just one (Route, HANDLES, Function) row is expected)."""
    pkg = _build_flask_app(tmp_path)
    g = build(str(pkg))
    triples = {(s, e, t): c for (s, e, t, c) in g.label_pair_counts()}
    handles_pairs = [(s, e, t, c) for (s, e, t), c in triples.items() if e == "HANDLES"]
    assert handles_pairs, "no HANDLES triple in label_pair_counts"
    # Each Route maps to exactly one Function, so the count equals
    # the number of detected routes.
    assert all(t == "Function" for (_, _, t, _) in handles_pairs)


def test_explore_finds_route_handler(tmp_path):
    """explore() should surface a route handler when queried by name."""
    pkg = _build_flask_app(tmp_path)
    g = build(str(pkg))
    md = g.explore("user_detail", max_entities=5, include_source=False)
    assert "user_detail" in md, md


def test_nodes_path_includes_properties_on_code_tree(tmp_path):
    """nodes(p) dicts must carry every node property — verified here on
    a real code-tree graph. Function nodes carry branch_count,
    file_path, signature, etc.

    Phase A.1 / C2 — node dicts have the Bolt-shaped {id, labels,
    properties} structure; user properties are nested under `properties`."""
    pkg = _build_flask_app(tmp_path)
    g = build(str(pkg))
    rows = g.cypher("MATCH p = (a:Function {name: 'list_users'})-[:CALLS]->(b:Function) RETURN nodes(p) AS N").to_list()
    assert rows, "expected list_users → helper CALLS path"
    first = rows[0]["N"][0]
    # Phase A.1 shape: type is in properties (and labels carries the
    # type name as the canonical Neo4j-compatible field).
    assert first["labels"] == ["Function"]
    # Every Function node carries branch_count and file_path on the
    # graph property layer — they must appear in the path-node dict's
    # `properties` map.
    assert "branch_count" in first["properties"], f"missing branch_count: {first}"
    assert "file_path" in first["properties"], f"missing file_path: {first}"

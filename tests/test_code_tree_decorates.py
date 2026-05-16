"""Function -[DECORATES]-> Function edges.

The Python/TS/Java/C# parsers populate `FunctionInfo.decorators` with the
raw decorator strings. The builder pass resolves each to a target Function
in the project's function set (same bare-name lookup as CALLS), strips
call-args (`@app.route('/x')` → `app.route`), and emits one edge per
unambiguous match. Third-party decorators (e.g. `@pytest.fixture` where
`fixture` isn't in the parsed code) silently drop.
"""

import pathlib
import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _make_pkg(tmp_path, files: dict[str, str]) -> pathlib.Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for rel, content in files.items():
        fp = pkg / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))
    return pkg


def test_python_decorator_resolves_to_definition(tmp_path):
    """An in-project decorator is wired to its decoratee via DECORATES."""
    pkg = _make_pkg(
        tmp_path,
        {
            "deco.py": """
            def my_decorator(fn):
                def wrapper(*a, **kw):
                    return fn(*a, **kw)
                return wrapper
            """,
            "core.py": """
            from pkg.deco import my_decorator

            @my_decorator
            def hello():
                return 1
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (d:Function)-[r:DECORATES]->(f:Function)
        RETURN d.name AS d, f.name AS f, r.decorator_name AS n
        """
    ).to_list()
    assert any(r["d"] == "my_decorator" and r["f"] == "hello" for r in rows), (
        f"missing my_decorator → hello edge; got {rows}"
    )
    # The raw decorator literal is preserved.
    edge = next(r for r in rows if r["f"] == "hello")
    assert edge["n"] == "my_decorator", edge


def test_decorator_with_call_args_strips_to_bare_name(tmp_path):
    """`@my_route('/x', methods=['GET'])` resolves to `my_route` after arg-strip."""
    pkg = _make_pkg(
        tmp_path,
        {
            "deco.py": """
            def my_route(path, methods=None):
                def outer(fn):
                    return fn
                return outer
            """,
            "core.py": """
            from pkg.deco import my_route

            @my_route('/users/{id}', methods=['GET'])
            def get_user():
                return {}
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (d:Function)-[r:DECORATES]->(f:Function {name: 'get_user'})
        RETURN d.name AS d, r.decorator_name AS n
        """
    ).to_list()
    assert rows, "expected an edge"
    assert rows[0]["d"] == "my_route"
    # Raw form is preserved including the args.
    assert "my_route" in rows[0]["n"] and "/users" in rows[0]["n"], rows[0]


def test_dotted_decorator_resolves_to_terminal_name(tmp_path):
    """`@mod.helper` resolves on `helper` (last segment)."""
    pkg = _make_pkg(
        tmp_path,
        {
            "mod.py": """
            def helper(fn):
                return fn
            """,
            "core.py": """
            import pkg.mod as mod

            @mod.helper
            def f():
                return 1
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (d:Function)-[:DECORATES]->(f:Function {name: 'f'}) RETURN d.name AS d").to_list()
    assert rows == [{"d": "helper"}], rows


def test_third_party_decorator_silently_drops(tmp_path):
    """An unresolved decorator string (no Function with that bare name) emits no edge."""
    pkg = _make_pkg(
        tmp_path,
        {
            "core.py": """
            import pytest

            @pytest.fixture
            def setup_data():
                return {}
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH ()-[:DECORATES]->(f:Function {name: 'setup_data'}) RETURN count(*) AS n").to_list()
    assert rows == [{"n": 0}], f"expected no DECORATES edge for unresolved pytest.fixture, got {rows}"


def test_ambiguous_decorator_skipped(tmp_path):
    """Two functions sharing a name → the decorator string is ambiguous; no edge emitted."""
    pkg = _make_pkg(
        tmp_path,
        {
            "a.py": """
            def helper(fn):
                return fn
            """,
            "b.py": """
            def helper(fn):
                return fn
            """,
            "core.py": """
            from pkg.a import helper

            @helper
            def thing():
                return 1
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH ()-[:DECORATES]->(f:Function {name: 'thing'}) RETURN count(*) AS n").to_list()
    assert rows == [{"n": 0}], f"expected no edge under ambiguous-name policy, got {rows}"


def test_decorators_property_still_populated(tmp_path):
    """Regression — the existing `decorators` property on Function nodes
    must remain populated even after DECORATES edges land."""
    pkg = _make_pkg(
        tmp_path,
        {
            "deco.py": """
            def my_decorator(fn):
                return fn
            """,
            "core.py": """
            from pkg.deco import my_decorator

            @my_decorator
            def hello():
                return 1
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:Function {name: 'hello'}) RETURN f.decorators AS d").to_list()
    assert rows, "expected hello function"
    assert "my_decorator" in (rows[0]["d"] or ""), rows

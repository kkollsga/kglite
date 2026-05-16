"""Web-framework Route nodes + HANDLES edges.

Three frameworks ship in 0.9.34:
  - Flask: `@app.route('/x')` / `@app.get('/x')` decorators (+ blueprints).
  - FastAPI: `@app.get('/x')` / `@router.post('/x')` decorators.
  - Django: `urlpatterns = [path('users/', view), ...]` in urls.py.

Express and Axum need parser-side call-arg capture and ship later. The
per-framework module structure means each lands as one new file.
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


# ── Flask ──────────────────────────────────────────────────────────


def test_flask_route_decorator(tmp_path):
    pkg = _make_pkg(
        tmp_path,
        {
            "app.py": """
            class _Flask:
                def route(self, *a, **k): return lambda fn: fn
                def get(self, *a, **k): return lambda fn: fn

            app = _Flask()

            @app.route('/users')
            def list_users():
                return []

            @app.route('/users/<int:id>', methods=['GET', 'DELETE'])
            def user_detail(id):
                return {}
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (r:Route {framework: 'flask'})-[:HANDLES]->(f:Function)
        RETURN r.method AS method, r.path AS path, f.name AS handler
        ORDER BY path, method
        """
    ).to_list()
    # `@app.route('/users')` with no `methods=` becomes a single Route
    # tagged `ANY`. `@app.route('/users/<int:id>', methods=['GET','DELETE'])`
    # expands into two Route rows (one per method) — both linking to the
    # same handler so users querying "what handles DELETE /users/<id>"
    # works.
    assert {
        ("ANY", "/users", "list_users"),
        ("GET", "/users/<int:id>", "user_detail"),
        ("DELETE", "/users/<int:id>", "user_detail"),
    } <= {(r["method"], r["path"], r["handler"]) for r in rows}, rows


def test_flask_method_shortcut_decorators(tmp_path):
    pkg = _make_pkg(
        tmp_path,
        {
            "app.py": """
            class _Flask:
                def get(self, *a, **k): return lambda fn: fn
                def post(self, *a, **k): return lambda fn: fn

            app = _Flask()

            @app.get('/health')
            def health():
                return 'ok'

            @app.post('/upload')
            def upload():
                return 'ok'
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (r:Route)-[:HANDLES]->(f:Function)
        WHERE r.framework = 'flask'
        RETURN r.method AS m, r.path AS p, f.name AS h ORDER BY p
        """
    ).to_list()
    methods_by_path = {r["p"]: (r["m"], r["h"]) for r in rows}
    assert methods_by_path.get("/health") == ("GET", "health"), rows
    assert methods_by_path.get("/upload") == ("POST", "upload"), rows


def test_flask_blueprint_route(tmp_path):
    """Any `*.route` suffix matches — blueprint variable names vary in the wild."""
    pkg = _make_pkg(
        tmp_path,
        {
            "bp.py": """
            class _Blueprint:
                def route(self, *a, **k): return lambda fn: fn

            bp = _Blueprint()

            @bp.route('/blueprinted')
            def from_blueprint():
                return 'ok'
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        "MATCH (r:Route {framework: 'flask'})-[:HANDLES]->(f:Function) RETURN r.path AS p, f.name AS h"
    ).to_list()
    assert any(r["p"] == "/blueprinted" and r["h"] == "from_blueprint" for r in rows), rows


# ── FastAPI ───────────────────────────────────────────────────────


def test_fastapi_router_decorator(tmp_path):
    pkg = _make_pkg(
        tmp_path,
        {
            "api.py": """
            class _Router:
                def get(self, *a, **k): return lambda fn: fn
                def post(self, *a, **k): return lambda fn: fn

            router = _Router()

            @router.get('/items')
            def list_items():
                return []

            @router.post('/items')
            def create_item():
                return {}
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (r:Route {framework: 'fastapi'})-[:HANDLES]->(f:Function)
        RETURN r.method AS m, r.path AS p, f.name AS h ORDER BY p, m
        """
    ).to_list()
    expected = {
        ("GET", "/items", "list_items"),
        ("POST", "/items", "create_item"),
    }
    assert expected <= {(r["m"], r["p"], r["h"]) for r in rows}, rows


# ── Django ────────────────────────────────────────────────────────


def test_django_urlpatterns(tmp_path):
    pkg = _make_pkg(
        tmp_path,
        {
            "urls.py": """
            from django.urls import path

            def home(request):
                return None

            def about(request):
                return None

            urlpatterns = [path('home/', home), path('about/', about)]
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (r:Route {framework: 'django'})-[:HANDLES]->(f:Function)
        RETURN r.path AS p, f.name AS h ORDER BY p
        """
    ).to_list()
    # Both paths are linked to their handlers (path() args parse from
    # the value_preview; handler names resolve on bare-name lookup).
    paths = {(r["p"], r["h"]) for r in rows}
    assert ("home/", "home") in paths, rows
    assert ("about/", "about") in paths, rows


# ── Cross-framework / regression ─────────────────────────────────


def test_no_route_node_for_plain_decorator(tmp_path):
    """A function decorated with `@functools.cache` (or anything non-routing)
    must NOT emit a Route node."""
    pkg = _make_pkg(
        tmp_path,
        {
            "core.py": """
            import functools

            @functools.cache
            def compute():
                return 1
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (r:Route) RETURN count(r) AS n").to_list()
    assert rows == [{"n": 0}], rows


def test_no_route_nodes_without_routes(tmp_path):
    """A pure-function file emits zero Route nodes and zero HANDLES edges."""
    pkg = _make_pkg(
        tmp_path,
        {
            "core.py": """
            def add(a, b):
                return a + b
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (r:Route) RETURN count(r) AS n").to_list()
    assert rows == [{"n": 0}], rows
    rows = g.cypher("MATCH ()-[:HANDLES]->() RETURN count(*) AS n").to_list()
    assert rows == [{"n": 0}], rows

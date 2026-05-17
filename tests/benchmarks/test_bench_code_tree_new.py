"""Performance benchmarks for the 0.9.34 code_tree additions.

Covers the four new passes (File→File IMPORTS, DECORATES, routes,
explore) plus the affected_tests procedure and Swift parser. Numbers
ride alongside the existing core benchmarks under `make bench-save` /
`make bench-compare`.

The fixture is a synthetic Python package shaped like a small but
realistic application: a router file with Flask-style decorators, a
util module, a tests/ tree, and a handler chain. Big enough that the
new passes have something to do; small enough that the bench timing
isn't dominated by warm-up.
"""

import pathlib
import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _build_pkg(tmp_path: pathlib.Path) -> pathlib.Path:
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

            def trace(fn):
                return fn
            """
        )
    )
    (pkg / "handlers.py").write_text(
        textwrap.dedent(
            """
            from app.util import helper, cache, trace

            class _App:
                def route(self, *a, **k): return lambda fn: fn
                def get(self, *a, **k): return lambda fn: fn
                def post(self, *a, **k): return lambda fn: fn
                def put(self, *a, **k): return lambda fn: fn
                def delete(self, *a, **k): return lambda fn: fn

            app = _App()

            @app.route('/users')
            @cache
            def list_users():
                return helper(1)

            @app.get('/users/<int:id>')
            @trace
            def user_detail(id):
                return helper(id)

            @app.post('/users')
            def create_user():
                return helper(2)

            @app.put('/users/<int:id>')
            def update_user(id):
                return helper(id)

            @app.delete('/users/<int:id>')
            def delete_user(id):
                return helper(id)
            """
        )
    )
    (pkg / "urls.py").write_text(
        textwrap.dedent(
            """
            from django.urls import path
            from app.handlers import list_users

            urlpatterns = [path('home/', list_users)]
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

            def test_detail():
                assert user_detail(1) is not None
            """
        )
    )
    (tests / "test_util.py").write_text(
        textwrap.dedent(
            """
            from app.util import helper

            def test_helper():
                assert helper(1) == 2
            """
        )
    )
    return pkg


@pytest.fixture
def synthetic_pkg(tmp_path):
    return _build_pkg(tmp_path)


@pytest.fixture
def synthetic_graph(synthetic_pkg):
    return build(str(synthetic_pkg))


@pytest.mark.benchmark
def test_bench_code_tree_build(benchmark, tmp_path):
    """End-to-end `build()` — exercises every new pass in one shot."""
    counter = [0]

    def setup():
        counter[0] += 1
        sub = tmp_path / f"round_{counter[0]}"
        sub.mkdir()
        pkg = _build_pkg(sub)
        return (pkg,), {}

    benchmark.pedantic(lambda pkg: build(str(pkg)), setup=setup, rounds=5, iterations=1)


@pytest.mark.benchmark
def test_bench_affected_tests(benchmark, synthetic_graph):
    """`CALL affected_tests` over the synthetic package."""
    g = synthetic_graph
    benchmark(lambda: g.cypher("CALL affected_tests({files: ['util.py']}) YIELD test_file RETURN test_file").to_list())


@pytest.mark.benchmark
def test_bench_explore(benchmark, synthetic_graph, synthetic_pkg):
    """`explore()` end-to-end including source slicing."""
    g = synthetic_graph
    pkg_path = str(synthetic_pkg)
    benchmark(
        lambda: g.explore(
            "user_detail",
            max_entities=5,
            max_depth=2,
            include_source=True,
            source_roots=[pkg_path],
        )
    )


@pytest.mark.benchmark
def test_bench_route_query(benchmark, synthetic_graph):
    """Walking the Route → HANDLES → Function neighborhood."""
    g = synthetic_graph
    benchmark(
        lambda: g.cypher(
            "MATCH (r:Route)-[:HANDLES]->(f:Function) RETURN r.method AS m, r.path AS p, f.name AS h ORDER BY p, m"
        ).to_list()
    )


@pytest.mark.benchmark
def test_bench_decorates_query(benchmark, synthetic_graph):
    """Tracing decorators to their decoratees."""
    g = synthetic_graph
    benchmark(
        lambda: g.cypher(
            "MATCH (d:Function)-[:DECORATES]->(f:Function) RETURN d.name AS d, f.name AS f ORDER BY d, f"
        ).to_list()
    )

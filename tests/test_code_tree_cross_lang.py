"""Cross-language HTTP boundary edges: a client HTTP call links to the server
route handler sharing the normalized path, so impact analysis crosses the
client/server (and language) boundary. Edges are tagged confidence='inferred'.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _build(tmp_path):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "api.ts").write_text(
        textwrap.dedent(
            """
            export function loadUsers(): Promise<Response> {
              return fetch("/api/users");
            }
            export function loadUser(id: number): Promise<Response> {
              return fetch(`/api/users/${id}`);
            }
            """
        )
    )
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "app.py").write_text(
        textwrap.dedent(
            """
            from fastapi import FastAPI

            app = FastAPI()

            @app.get("/api/users")
            def list_users():
                return []

            @app.get("/api/users/{id}")
            def get_user(id: int):
                return {}
            """
        )
    )
    return build(str(tmp_path))


def test_client_call_connects_to_handler(tmp_path):
    g = _build(tmp_path)
    rows = g.cypher(
        "MATCH (f:Function)-[:CALLS_SERVICE]->(:Route)-[:HANDLES]->(h:Function) "
        "RETURN f.name AS client, h.name AS handler"
    ).to_list()
    pairs = {(r["client"], r["handler"]) for r in rows}
    assert ("loadUsers", "list_users") in pairs


def test_parameterized_route_matches_concrete_client_path(tmp_path):
    g = _build(tmp_path)
    rows = g.cypher(
        "MATCH (f:Function {name:'loadUser'})-[:CALLS_SERVICE]->(:Route)-[:HANDLES]->(h:Function) "
        "RETURN h.name AS handler"
    ).to_list()
    # `/api/users/${id}` (template-literal) → the `/api/users/{id}` route.
    assert any(r["handler"] == "get_user" for r in rows)


def test_cross_language_edges_tagged_inferred(tmp_path):
    g = _build(tmp_path)
    rows = g.cypher("MATCH ()-[r:CALLS_SERVICE]->() RETURN r.confidence AS c").to_list()
    assert rows, "expected CALLS_SERVICE edges"
    assert all(r["c"] == "inferred" for r in rows)


def test_no_routes_means_no_cross_edges(tmp_path):
    # A plain library with no web routes: the pass is a no-op, no Route nodes.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text("def f():\n    return 1\n")
    g = build(str(tmp_path))
    n = g.cypher("MATCH ()-[r:CALLS_SERVICE]->() RETURN count(*) AS n").to_list()[0]["n"]
    assert n == 0

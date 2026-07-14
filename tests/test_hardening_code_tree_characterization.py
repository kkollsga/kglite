"""Canonical insertion-order fixture for the code-tree loader refactor."""

from __future__ import annotations

import textwrap

from kglite.code_tree import build


def test_code_tree_node_insertion_order_and_edge_membership_are_stable(tmp_path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "model.py").write_text(
        textwrap.dedent(
            """
            LIMIT = 3

            def helper(value):
                return value + LIMIT

            class Worker:
                def run(self, value):
                    return helper(value)
            """
        )
    )

    graph = build(str(package))
    nodes = graph.cypher("MATCH (n) RETURN labels(n)[0] AS kind, n.id AS id").to_list()
    edges = graph.cypher(
        "MATCH (source)-[edge]->(target) RETURN type(edge) AS kind, source.id AS source, target.id AS target"
    ).to_list()

    assert nodes == [
        {"kind": "File", "id": "__init__.py"},
        {"kind": "File", "id": "model.py"},
        {"kind": "Module", "id": "pkg"},
        {"kind": "Module", "id": "pkg.model"},
        {"kind": "Function", "id": "pkg.model.helper"},
        {"kind": "Function", "id": "pkg.model.Worker.run"},
        {"kind": "Class", "id": "pkg.model.Worker"},
        {"kind": "Constant", "id": "pkg.model.LIMIT"},
    ]
    expected_edges = [
        {"kind": "DEFINES", "source": "model.py", "target": "pkg.model.Worker.run"},
        {"kind": "DEFINES", "source": "model.py", "target": "pkg.model.helper"},
        {"kind": "DEFINES", "source": "model.py", "target": "pkg.model.LIMIT"},
        {"kind": "DEFINES", "source": "model.py", "target": "pkg.model.Worker"},
        {"kind": "HAS_FILE", "source": "pkg", "target": "__init__.py"},
        {"kind": "HAS_SUBMODULE", "source": "pkg", "target": "pkg.model"},
        {"kind": "HAS_FILE", "source": "pkg.model", "target": "model.py"},
        {"kind": "CALLS", "source": "pkg.model.Worker.run", "target": "pkg.model.helper"},
        {"kind": "HAS_METHOD", "source": "pkg.model.Worker", "target": "pkg.model.Worker.run"},
    ]

    def edge_key(edge):
        return (edge["kind"], edge["source"], edge["target"])

    assert sorted(edges, key=edge_key) == sorted(expected_edges, key=edge_key)

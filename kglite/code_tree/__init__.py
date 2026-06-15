"""Parse codebases into KGLite knowledge graphs.

All functionality is implemented in Rust (with bundled tree-sitter
grammars) and exposed through the native `_kglite_code_tree` submodule.
No optional dependencies are required.

Usage::

    from kglite.code_tree import build

    graph = build("/path/to/project")

Pass ``include_docs=True`` to also ingest the repo's markdown as ``:Doc``
nodes and link them to the code they mention::

    graph = build("/path/to/project", include_docs=True)
    graph.cypher(
        "MATCH (d:Doc)-[:MENTIONS]->(c) RETURN d.title, labels(c)[0], c.name"
    )
"""

from kglite._kglite_code_tree import build, read_manifest, repo_tree

__all__ = ["build", "read_manifest", "repo_tree"]

"""Parse codebases into KGLite knowledge graphs.

All functionality is implemented in Rust (with bundled tree-sitter
grammars). No optional dependencies are required.

**Public API:** ``kglite.code_tree.build`` (here) or the top-level
``kglite.build_code_tree``. The native ``kglite._kglite_code_tree`` module is
an internal implementation detail — import from ``kglite.code_tree`` or the
top level, never the underscore-prefixed module (it may change without notice).

Usage::

    from kglite.code_tree import build   # or: import kglite; kglite.build_code_tree(...)

    graph = build("/path/to/project")

Pass ``include_docs=True`` to also ingest the repo's markdown as ``:Doc``
nodes and link them to the code they mention::

    graph = build("/path/to/project", include_docs=True)
    graph.cypher(
        "MATCH (d:Doc)-[:MENTIONS]->(c) RETURN d.title, labels(c)[0], c.name"
    )

Pass ``rev`` (a git tag, branch, or SHA) to build a codebase as it existed at
that revision without disturbing the working tree — its tracked files are
materialized via ``git archive`` into a tempdir and built there::

    old = build("/path/to/repo", rev="v1.0")   # committed content at v1.0
    now = build("/path/to/repo")               # current working tree
"""

from kglite._kglite_code_tree import build, read_manifest, repo_tree

__all__ = ["build", "read_manifest", "repo_tree"]

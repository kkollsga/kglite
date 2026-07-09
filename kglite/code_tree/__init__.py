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

Compare two such graphs structurally (what functions/classes/constants were
added, removed, moved, or changed between two revisions) with ``diff``::

    delta = diff(old, now)
    print(delta["summary"])

Alternatively, pass ``revs=[...]`` (oldest → newest, mutually exclusive with
``rev``) to merge N revisions into a single **multi-rev graph** — one node per
entity across revs, each node carrying native list props ``revs: [str]``
(revisions it appears in) + ``rev_fp: [int]`` (per-rev shape fingerprint), and
each edge carrying ``revs: [str]``. Ordinary properties report the newest rev
(newest-wins). Scope a query to one rev with ``WHERE 'v2' IN n.revs`` (an
unscoped query spans all revs), and use ``CALL rev_diff({from, to})`` for
deltas::

    g = build("/path/to/repo", revs=["v1.0", "v2.0", "HEAD"])
    g.cypher("MATCH (n:Function) WHERE 'v2.0' IN n.revs RETURN n.name")
    g.cypher("CALL rev_diff({from: 'v1.0', to: 'HEAD'}) "
             "YIELD bucket, type, qualified_name RETURN *")
"""

from kglite._kglite_code_tree import build, read_manifest, repo_tree
from kglite.code_tree._diff import diff, match_entities

__all__ = ["build", "diff", "match_entities", "read_manifest", "repo_tree"]

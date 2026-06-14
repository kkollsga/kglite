"""Ingest Open Knowledge Format (OKF) bundles into KGLite knowledge graphs.

OKF is a directory of markdown files with YAML frontmatter, cross-linked by
markdown links (Google's Open Knowledge Format — also Claude memory dirs,
skills folders, and Obsidian vaults). This is a **read-only, partial** loader,
conceptually ``code_tree`` for prose knowledge instead of source code: each
concept becomes a node carrying its frontmatter as properties plus a
``file_path`` pointer; links become typed edges. The body is read on demand and
is not stored unless ``with_body=True``.

The result is a normal :class:`~kglite.KnowledgeGraph`, so every Cypher feature,
algorithm (``CALL leiden`` / ``pagerank``), and structural rule (``orphan_node``)
works over it — which is exactly what OKF itself lacks.

All functionality is implemented in Rust and exposed through the native
``_kglite_okf`` submodule (its YAML parser is gated behind the engine's ``okf``
feature, enabled in the wheel).

Usage::

    from kglite import okf

    g = okf.build("path/to/bundle")            # strict OKF markdown links
    g = okf.build("path/to/memory", dialect="obsidian")  # also [[wikilinks]]

    # Now query it like any graph:
    g.cypher("MATCH (n) WHERE NOT (n)--() RETURN n.concept_id")   # orphans
    g.cypher("CALL leiden() YIELD node, community RETURN community, count(*)")
"""

from kglite._kglite_okf import build

__all__ = ["build"]

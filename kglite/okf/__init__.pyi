"""Type stubs for kglite.okf — Open Knowledge Format bundle ingestion."""

from __future__ import annotations

from kglite import KnowledgeGraph

def build(
    path: str,
    *,
    dialect: str | None = ...,
    require_frontmatter: bool = ...,
    with_body: bool = ...,
    embed: bool = ...,
) -> KnowledgeGraph:
    """Build a :class:`~kglite.KnowledgeGraph` from an OKF bundle directory.

    A bundle is a directory tree of markdown files with YAML frontmatter,
    cross-linked by markdown links. Each non-reserved ``.md`` file becomes one
    node (label from the frontmatter ``type``, or ``Concept`` when absent; id =
    the bundle-relative path minus ``.md``); frontmatter keys become node
    properties (``tags`` and nested maps are JSON-encoded); markdown links become
    typed edges. Link types are inferred most-specific-first: an explicit link
    title (``[x](/y.md "JOINS_WITH")``) → the enclosing section header
    (``# Citations`` → ``CITES``) → ``LINKS_TO``. Links to not-yet-written
    concepts become ``_provisional`` stub nodes
    (``MATCH (n {_provisional: true})``).

    The graph is enriched by default with synthesized nodes that densify it:
    ``Tag`` nodes (``(:Concept)-[:TAGGED]->(:Tag)`` from ``tags``), ``Source``
    nodes (external ``http(s)`` links → ``(:Concept)-[:CITES]->(:Source)``), and
    ``Folder`` nodes (the directory tree, ``(:Folder)-[:CONTAINS]->`` concepts /
    subfolders, with each directory's ``index.md`` enriching its Folder).

    Ingestion is *partial*: the markdown body is not stored unless ``with_body``
    is set — each node keeps a ``file_path`` pointer instead.

    Args:
        path: Bundle root directory.
        dialect: ``"okf"`` (default) for strict markdown links, or
            ``"loose"`` / ``"obsidian"`` to also resolve ``[[wikilinks]]`` and
            tolerate concepts with no frontmatter ``type``.
        require_frontmatter: When ``True`` (default), only ``.md`` files with a
            YAML frontmatter block are ingested — the discriminator between
            *structured* knowledge (OKF concepts, Claude memories) and plain
            markdown (READMEs, notes). Point at a parent of many projects to
            sweep out only the structured files across all of them. Set ``False``
            to ingest every ``.md`` (vault-style). Node labels fall back
            ``type`` → ``metadata.type`` → ``Concept`` and titles ``title`` →
            ``name`` → file stem, so Claude memories land as ``:feedback`` /
            ``:project`` / etc. with their ``name`` as title.
        with_body: Store each concept's markdown body as a ``body`` property
            (off by default — bodies are read on demand).
        embed: Reserved for the opt-in embedder pass (body vectors for
            ``text_score``); not yet wired.

    Returns:
        A KnowledgeGraph of the bundle.

    Raises:
        RuntimeError: If the bundle path does not exist or is not a directory.
    """

def source(path: str) -> str:
    """Read a concept's markdown body on demand (frontmatter stripped).

    Pairs with partial ingestion: nodes store a ``file_path`` (relative to the
    bundle root), not the body. Join it with the root and pass it here to fetch
    the prose once a query has narrowed to a single concept. A file with no
    frontmatter returns its whole content.

    Args:
        path: Path to the concept's ``.md`` file.

    Returns:
        The markdown body, with any leading YAML frontmatter removed.

    Raises:
        RuntimeError: If the file cannot be read.
    """

"""KGLite - A high-performance graph database library with Python bindings written in Rust."""

from .blueprint import (  # noqa: E402  (must override star-import from .kglite)
    from_blueprint,
    from_records,
)
from .kglite import *  # noqa: F401, F403
from .kglite import (  # explicit re-exports — names listed in __all__ below
    ArgumentError,
    ConnectionNotFoundError,
    CypherError,
    CypherExecutionError,
    CypherSyntaxError,
    CypherTimeoutError,
    CypherTypeMismatchError,
    ExprError,
    FileError,
    FileFormatError,
    FileIoError,
    FrozenGraph,
    InternalError,
    KgError,
    KnowledgeGraph,
    MissingArgumentError,
    NodeNotFoundError,
    PropertyNotFoundError,
    ResultIter,
    ResultView,
    SchemaError,
    Transaction,
    ValidationError,
    __version__,
    cypher_pass_names,
    from_bytes,
    load,
)


class Agg:
    """Aggregation expression builders for ``add_properties()``.

    Each method returns the string expression that ``add_properties()``
    already understands, making the DSL discoverable via autocomplete.

    Example::

        from kglite import Agg

        graph.select('Well').traverse('HAS_BLOCK').add_properties({
            'Block': {'well_count': Agg.count(), 'avg_depth': Agg.mean('depth')}
        })
    """

    @staticmethod
    def count() -> str:
        """Count leaf nodes per ancestor — ``count(*)``."""
        return "count(*)"

    @staticmethod
    def sum(prop: str) -> str:
        """Sum a numeric property across leaves — ``sum(prop)``."""
        return f"sum({prop})"

    @staticmethod
    def mean(prop: str) -> str:
        """Arithmetic mean of a numeric property — ``mean(prop)``."""
        return f"mean({prop})"

    @staticmethod
    def min(prop: str) -> str:
        """Minimum value of a numeric property — ``min(prop)``."""
        return f"min({prop})"

    @staticmethod
    def max(prop: str) -> str:
        """Maximum value of a numeric property — ``max(prop)``."""
        return f"max({prop})"

    @staticmethod
    def std(prop: str) -> str:
        """Sample standard deviation of a numeric property — ``std(prop)``."""
        return f"std({prop})"

    @staticmethod
    def collect(prop: str) -> str:
        """Comma-separated string of property values — ``collect(prop)``."""
        return f"collect({prop})"


class Spatial:
    """Spatial compute expression builders for ``add_properties()``.

    Each method returns the string keyword that ``add_properties()``
    already understands for spatial computations.

    Example::

        from kglite import Spatial

        graph.select('Well').compare('Structure', 'contains') \\
            .add_properties({
                'Well': {'dist': Spatial.distance(), 'a': Spatial.area()}
            })
    """

    @staticmethod
    def distance() -> str:
        """Geodesic distance between leaf and ancestor (meters)."""
        return "distance"

    @staticmethod
    def area() -> str:
        """Area of ancestor geometry (square meters)."""
        return "area"

    @staticmethod
    def perimeter() -> str:
        """Perimeter of ancestor geometry (meters)."""
        return "perimeter"

    @staticmethod
    def centroid_lat() -> str:
        """Latitude of ancestor geometry centroid."""
        return "centroid_lat"

    @staticmethod
    def centroid_lon() -> str:
        """Longitude of ancestor geometry centroid."""
        return "centroid_lon"


def build_code_tree(
    path: str,
    **kwargs,
) -> "KnowledgeGraph":
    """Parse a codebase at ``path`` into a :class:`KnowledgeGraph`.

    The stable, public entry point for code-graph building (tree-sitter
    grammars are bundled in the Rust extension — no extra to install).
    Equivalent to :func:`kglite.code_tree.build`; prefer either of these over
    the internal ``kglite._kglite_code_tree`` module, which is an
    implementation detail and may change without notice.

    Pass ``include_docs=True`` to also ingest markdown as ``:Doc`` nodes linked
    to the code they mention. See :func:`kglite.code_tree.build` for the full
    keyword set.
    """
    from .code_tree import build as _build

    return _build(path, **kwargs)


def repo_tree(
    repo: str,
    **kwargs,
) -> "KnowledgeGraph":
    """Clone a GitHub repository and build a knowledge graph from its source code.

    Convenience re-export of :func:`kglite.code_tree.repo_tree`. Tree-sitter
    grammars are bundled in the Rust extension — no extra to install.
    """
    from .code_tree import repo_tree as _repo_tree

    return _repo_tree(repo, **kwargs)


def graphgen(
    scale: str = "medium",
    *,
    persons: "int | None" = None,
    seed: int = 1234,
    knows_per: int = 8,
    degree_dist: str = "zipf",
    zipf_exp: float = 1.6,
    out: "str | None" = None,
):
    """Generate a synthetic org/social knowledge graph (bundled, no extra deps).

    A seed-deterministic graph of Person/Company/Project/Skill/City nodes +
    KNOWS/WORKS_AT/CONTRIBUTES_TO/HAS_SKILL/OWNS/DEPENDS_ON/LOCATED_IN edges —
    handy for demos, tests, and benchmarks.

    - ``out=None`` (default): build and return a :class:`KnowledgeGraph`, ready
      to query. Best for small/medium scales (needs ``pandas``).
    - ``out=DIR``: stream one CSV per type + ``manifest.json`` into ``DIR`` in
      **bounded memory** (millions of nodes at flat RAM); returns a stats dict
      ``{'nodes', 'edges', 'out'}``. Load later with your own pipeline, or with
      any engine — every backend that reads the same bytes gets the same graph.

    Args:
        scale: ``tiny`` | ``small`` | ``medium`` (default) | ``large`` |
            ``huge`` | ``xhuge`` — sets the Person count (everything else scales
            off it). Ignored if ``persons`` is given.
        persons: Exact Person count (overrides ``scale``).
        seed: Deterministic seed.
        knows_per: Average KNOWS out-degree per person.
        degree_dist: ``'zipf'`` (default; high-degree hubs — realistic, makes
            multi-hop traversal interesting) or ``'uniform'``.
        zipf_exp: Zipf skew exponent (>1 → stronger hubs).
        out: Output directory for streaming mode, or ``None`` to return a graph.

    Example::

        g = kglite.graphgen("medium")                  # a graph to query now
        g.cypher("MATCH (p:Person)-[:KNOWS]->(f) RETURN count(f)")

        kglite.graphgen("huge", out="/tmp/g")           # stream 5M persons to CSV
    """
    from ._graphgen import generate

    return generate(
        scale,
        persons=persons,
        seed=seed,
        knows_per=knows_per,
        degree_dist=degree_dist,
        zipf_exp=zipf_exp,
        out=out,
    )


def from_networkx(
    nx_graph,
    *,
    default_node_type: str = "Node",
    default_edge_type: str = "RELATED",
) -> "KnowledgeGraph":
    """Build a :class:`KnowledgeGraph` from a ``networkx`` graph.

    Convenience re-export of :func:`kglite.networkx_interop.from_networkx`.
    The inverse is :meth:`KnowledgeGraph.to_networkx`.
    Requires the ``networkx`` package: ``pip install networkx``.
    """
    from .networkx_interop import from_networkx as _from_networkx

    return _from_networkx(
        nx_graph,
        default_node_type=default_node_type,
        default_edge_type=default_edge_type,
    )


def to_neo4j(
    graph: "KnowledgeGraph",
    uri: str,
    **kwargs,
) -> dict:
    """Push graph data to a Neo4j database.

    Convenience re-export of :func:`kglite.neo4j_export.to_neo4j`.
    Requires the ``neo4j`` package: ``pip install neo4j``.
    """
    from .neo4j_export import to_neo4j as _to_neo4j

    return _to_neo4j(graph, uri, **kwargs)


def outline(
    graph: "KnowledgeGraph",
    root,
    edge: str,
    *,
    max_depth: int | None = None,
    body: str | None = None,
) -> str:
    """Render the spanning tree from ``root`` along ``edge`` as a nested outline.

    A *projection* of the graph — the "open and skim" view a graph otherwise
    lacks. The engine's ``CALL outline`` yields the tree structure (node, depth,
    parent_id); this renders it as an indented markdown-style outline. Follows
    outgoing ``edge``-typed edges from the node whose id is ``root``; each node
    appears once (at first discovery, so a DAG renders as a tree). Nodes are
    labelled by title (falling back to id). ``max_depth`` bounds the descent.
    Pass ``body="<prop>"`` to indent each node's prose property under its bullet
    (this is the "markdown body" view — prose lives in a plain string property,
    not a special engine field).

    Example::

        print(kglite.outline(g, "epic-1", "DEPENDS_ON", body="notes"))
        # - Build the API
        #   The public REST surface.
        #   - Define the schema
        #   - Write the handlers

    Returns:
        The outline text (empty string if ``root`` has no node).
    """
    md = "" if max_depth is None else ", max_depth: $md"
    body_col = f", node.`{body}` AS body" if body else ""
    q = (
        f"CALL outline({{root: $root, edge: $edge{md}}}) "
        "YIELD node, depth, parent_id "
        f"RETURN node.id AS id, node.title AS title, parent_id AS parent_id{body_col}"
    )
    params = {"root": root, "edge": edge}
    if max_depth is not None:
        params["md"] = max_depth
    rows = graph.cypher(q, params=params).to_dicts()
    if not rows:
        return ""

    children: dict = {}
    root_id = None
    for r in rows:
        children.setdefault(r["parent_id"], []).append(r)
        if r["parent_id"] is None:
            root_id = r["id"]

    label = {r["id"]: (r["title"] if r["title"] is not None else r["id"]) for r in rows}
    bodies = {r["id"]: r.get("body") for r in rows} if body else {}
    lines: list = []

    def _walk(node_id, depth: int) -> None:
        lines.append("  " * depth + f"- {label.get(node_id, node_id)}")
        prose = bodies.get(node_id)
        if prose:
            for prose_line in str(prose).splitlines():
                lines.append("  " * (depth + 1) + prose_line)
        for child in sorted(children.get(node_id, []), key=lambda x: str(x["id"])):
            _walk(child["id"], depth + 1)

    if root_id is not None:
        _walk(root_id, 0)
    return "\n".join(lines)


def _nodes_with_path(graph, node_type, path_property, extra=""):
    label = f":{node_type}" if node_type else ""
    return graph.cypher(
        f"MATCH (n{label}) WHERE n.{path_property} IS NOT NULL RETURN n.id AS id, n.{path_property} AS path{extra}"
    ).to_dicts()


def stamp_file_freshness(
    graph: "KnowledgeGraph",
    *,
    node_type: str | None = None,
    path_property: str = "file_path",
    mtime_property: str = "file_mtime",
    hash_property: str | None = "content_hash",
    base_dir: str | None = None,
) -> int:
    """Capture each node's linked-file state into properties — the binding-layer
    answer to "auto-stamp file freshness" (the engine never reads the
    filesystem). For every node carrying ``path_property``, stat the file and SET
    ``mtime_property`` (a Timestamp) and, unless ``hash_property`` is None, its
    sha256. A missing file sets both to null. Run after a build/write; pair with
    :func:`check_file_freshness` to detect later drift.

    Args:
        node_type: restrict to one type (default: all nodes with the property).
        base_dir: prefix for relative ``path_property`` values.

    Returns:
        The number of nodes stamped.
    """
    import datetime
    import hashlib
    import os
    import pathlib

    rows = _nodes_with_path(graph, node_type, path_property)
    label = f":{node_type}" if node_type else ""
    for r in rows:
        full = os.path.join(base_dir, r["path"]) if base_dir else r["path"]
        sets: dict = {mtime_property: None}
        if hash_property:
            sets[hash_property] = None
        if os.path.isfile(full):
            sets[mtime_property] = datetime.datetime.fromtimestamp(os.path.getmtime(full))
            if hash_property:
                sets[hash_property] = hashlib.sha256(pathlib.Path(full).read_bytes()).hexdigest()
        assignments = ", ".join(f"n.`{k}` = ${k}" for k in sets)
        graph.cypher(
            f"MATCH (n{label}) WHERE n.id = $_id SET {assignments}",
            params={"_id": r["id"], **sets},
        )
    return len(rows)


def check_file_freshness(
    graph: "KnowledgeGraph",
    *,
    node_type: str | None = None,
    path_property: str = "file_path",
    hash_property: str | None = "content_hash",
    base_dir: str | None = None,
) -> list:
    """Read-only drift check (binding layer): for each node with ``path_property``,
    re-stat the file and compare against the stored ``hash_property`` (from
    :func:`stamp_file_freshness`). Returns the drifted nodes as
    ``[{"id", "path", "status"}]`` where ``status`` is ``"missing"`` (the file is
    gone — the stale-Artifact-pointing-at-a-deleted-crate case) or ``"changed"``
    (its sha256 differs from what was stamped). Nodes that still match are
    omitted. Replaces an ad-hoc ``os.path.exists`` gate; never mutates the graph.
    """
    import hashlib
    import os
    import pathlib

    extra = f", n.{hash_property} AS _hash" if hash_property else ""
    rows = _nodes_with_path(graph, node_type, path_property, extra)
    drift: list = []
    for r in rows:
        full = os.path.join(base_dir, r["path"]) if base_dir else r["path"]
        if not os.path.isfile(full):
            drift.append({"id": r["id"], "path": r["path"], "status": "missing"})
            continue
        if hash_property and r.get("_hash") is not None:
            if hashlib.sha256(pathlib.Path(full).read_bytes()).hexdigest() != r["_hash"]:
                drift.append({"id": r["id"], "path": r["path"], "status": "changed"})
    return drift


__all__ = [
    "__version__",
    "KnowledgeGraph",
    "FrozenGraph",
    "Transaction",
    "ResultView",
    "ResultIter",
    "load",
    "from_bytes",
    "cypher_pass_names",
    "from_blueprint",
    "from_records",
    "build_code_tree",
    "repo_tree",
    "graphgen",
    "outline",
    "stamp_file_freshness",
    "check_file_freshness",
    "to_neo4j",
    "from_networkx",
    "Agg",
    "Spatial",
    # Phase A.2 / C1 — typed exception classes. See
    # docs/explanation/error-handling.md for the hierarchy.
    "KgError",
    "CypherError",
    "CypherSyntaxError",
    "CypherTimeoutError",
    "CypherExecutionError",
    "CypherTypeMismatchError",
    "SchemaError",
    "ValidationError",
    "ExprError",
    "NodeNotFoundError",
    "ConnectionNotFoundError",
    "PropertyNotFoundError",
    "FileError",
    "FileFormatError",
    "FileIoError",
    "ArgumentError",
    "MissingArgumentError",
    "InternalError",
]

# Eager submodule bind so `import kglite; kglite.code_tree.build(...)` works
# without a separate `from kglite import code_tree`. Placed after the extension
# import above (it registers `kglite._kglite_code_tree`, which the code_tree
# package re-exports) and kept out of the top import block on purpose.
from . import code_tree  # noqa: E402, F401

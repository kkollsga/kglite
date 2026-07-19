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
    InternerCollisionError,
    KgError,
    KnowledgeGraph,
    MissingArgumentError,
    NodeNotFoundError,
    PropertyNotFoundError,
    ResultIter,
    ResultView,
    SchemaError,
    Session,
    Transaction,
    ValidationError,
    __version__,
    cypher_pass_names,
    from_bytes,
    load,
    load_rdf,
    open,
    open_session,
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


def _cypher_identifier(name: str, *, kind: str) -> str:
    """Quote an interpolated Cypher identifier or reject an unrepresentable one.

    KGLite's tokenizer accepts backtick-delimited identifiers but does not yet
    implement doubled-backtick escaping. Rejecting an embedded backtick keeps
    these convenience helpers injection-safe while still supporting spaces,
    punctuation, and keyword identifiers.
    """
    if not isinstance(name, str) or not name:
        raise ArgumentError(f"{kind} must be a non-empty string")
    if "`" in name:
        raise ArgumentError(f"{kind} cannot contain a backtick")
    return f"`{name}`"


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
    body_col = f", node.{_cypher_identifier(body, kind='body property')} AS body" if body else ""
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

    # Iterative depth-first walk — CALL outline yields unbounded depth, so
    # a recursive renderer would hit RecursionError past ~1000 levels.
    if root_id is not None:
        stack: list = [(root_id, 0)]
        while stack:
            node_id, depth = stack.pop()
            lines.append("  " * depth + f"- {label.get(node_id, node_id)}")
            prose = bodies.get(node_id)
            if prose:
                for prose_line in str(prose).splitlines():
                    lines.append("  " * (depth + 1) + prose_line)
            # Push in reverse so children pop in sorted order.
            for child in sorted(children.get(node_id, []), key=lambda x: str(x["id"]), reverse=True):
                stack.append((child["id"], depth + 1))
    return "\n".join(lines)


def _nodes_with_path(graph, node_type, path_property, extra=""):
    label = f":{_cypher_identifier(node_type, kind='node type')}" if node_type else ""
    path_ident = _cypher_identifier(path_property, kind="path property")
    return graph.cypher(
        f"MATCH (n{label}) WHERE n.{path_ident} IS NOT NULL "
        f"RETURN labels(n)[0] AS node_type, n.id AS id, n.{path_ident} AS path{extra}"
    )


_FILE_SNAPSHOT_CHUNK_BYTES = 1024 * 1024
_FILE_SNAPSHOT_ATTEMPTS = 3
_FILE_SNAPSHOT_CACHE_SIZE = 4096
_FRESHNESS_UPDATE_BATCH_SIZE = 1000


def _resolved_file_path(path, base_dir):
    import pathlib

    candidate = pathlib.Path(base_dir, path) if base_dir else pathlib.Path(path)
    return candidate.expanduser().resolve(strict=False)


def _stat_signature(info):
    """Fields that must remain fixed while one descriptor is being read."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _mtime_utc_ns(mtime_ns):
    """Lossless, timezone-explicit filesystem timestamp (RFC 3339, 9 digits)."""
    import datetime

    seconds, nanos = divmod(mtime_ns, 1_000_000_000)
    utc = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    return f"{utc:%Y-%m-%dT%H:%M:%S}.{nanos:09d}Z"


def _snapshot_file(path, *, include_hash):
    """Read a stable regular-file snapshot through one open descriptor.

    A rename can leave an open descriptor pointing at the old file, while an
    in-place writer can change bytes during hashing. Both cases retry from a
    fresh descriptor; after the bounded retry budget they fail explicitly.
    """
    import hashlib
    import os
    import stat

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    for _attempt in range(_FILE_SNAPSHOT_ATTEMPTS):
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            return None

        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                return None

            digest = hashlib.sha256() if include_hash else None
            if digest is not None:
                while chunk := os.read(descriptor, _FILE_SNAPSHOT_CHUNK_BYTES):
                    digest.update(chunk)

            after = os.fstat(descriptor)
            try:
                at_path = os.stat(path)
            except FileNotFoundError:
                continue

            descriptor_stable = _stat_signature(before) == _stat_signature(after)
            path_still_names_descriptor = (after.st_dev, after.st_ino) == (at_path.st_dev, at_path.st_ino)
            path_metadata_matches = _stat_signature(after) == _stat_signature(at_path)
            if descriptor_stable and path_still_names_descriptor and path_metadata_matches:
                return {
                    "mtime": _mtime_utc_ns(after.st_mtime_ns),
                    "hash": digest.hexdigest() if digest is not None else None,
                }
        finally:
            os.close(descriptor)

    raise RuntimeError(f"File remained unstable across {_FILE_SNAPSHOT_ATTEMPTS} snapshot attempts: {path}")


def _cached_file_snapshot(cache, resolved, *, include_hash):
    """Reuse recent snapshots without retaining one entry per graph node."""
    if resolved in cache:
        cache.move_to_end(resolved)
        return cache[resolved]
    snapshot = _snapshot_file(resolved, include_hash=include_hash)
    cache[resolved] = snapshot
    if len(cache) > _FILE_SNAPSHOT_CACHE_SIZE:
        cache.popitem(last=False)
    return snapshot


def _freshness_chunks(rows, *, base_dir, include_hash, batch_size):
    """Yield bounded mutation batches while caching duplicate resolved paths."""
    from collections import OrderedDict

    snapshots = OrderedDict()
    batch = []
    for row in rows:
        resolved = _resolved_file_path(row["path"], base_dir)
        snapshot = _cached_file_snapshot(snapshots, resolved, include_hash=include_hash)
        batch.append(
            {
                "node_type": row["node_type"],
                "id": row["id"],
                "mtime": snapshot["mtime"] if snapshot is not None else None,
                "hash": snapshot["hash"] if snapshot is not None else None,
            }
        )
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def stamp_file_freshness(
    graph: "KnowledgeGraph",
    *,
    node_type: str | None = None,
    path_property: str = "file_path",
    mtime_property: str = "file_mtime",
    hash_property: str | None = "content_hash",
    base_dir: str | None = None,
    batch_size: int = _FRESHNESS_UPDATE_BATCH_SIZE,
) -> int:
    """Capture each node's linked-file state into properties — the binding-layer
    answer to "auto-stamp file freshness" (the engine never reads the
    filesystem). For every node carrying ``path_property``, snapshot the file
    through one descriptor and SET ``mtime_property`` (a nanosecond UTC RFC 3339
    string) and, unless ``hash_property`` is None, its sha256. A missing file
    sets both to null. Run after a build/write; pair with
    :func:`check_file_freshness` to detect later drift.

    Args:
        node_type: restrict to one type (default: all nodes with the property).
        base_dir: prefix for relative ``path_property`` values.
        batch_size: maximum rows passed to each transaction query. All batches
            still commit atomically.

    Returns:
        The number of nodes stamped.
    """
    if batch_size <= 0:
        raise ArgumentError("batch_size must be greater than zero")
    rows = _nodes_with_path(graph, node_type, path_property)
    assignments = f"n.{_cypher_identifier(mtime_property, kind='mtime property')} = row.mtime"
    if hash_property is not None:
        assignments += f", n.{_cypher_identifier(hash_property, kind='hash property')} = row.hash"
    query = f"UNWIND $rows AS row MATCH (n) WHERE labels(n)[0] = row.node_type AND n.id = row.id SET {assignments}"

    stamped = 0
    with graph.begin() as transaction:
        for batch in _freshness_chunks(
            rows,
            base_dir=base_dir,
            include_hash=hash_property is not None,
            batch_size=batch_size,
        ):
            transaction.cypher(query, params={"rows": batch})
            stamped += len(batch)
    return stamped


def check_file_freshness(
    graph: "KnowledgeGraph",
    *,
    node_type: str | None = None,
    path_property: str = "file_path",
    mtime_property: str = "file_mtime",
    hash_property: str | None = "content_hash",
    base_dir: str | None = None,
) -> list:
    """Read-only drift check (binding layer): for each node with ``path_property``,
    snapshot the file and compare against the stored ``hash_property`` (from
    :func:`stamp_file_freshness`). Returns the drifted nodes as
    ``[{"id", "path", "status"}]`` where ``status`` is ``"missing"`` (the file is
    gone — the stale-Artifact-pointing-at-a-deleted-crate case) or ``"changed"``
    (its sha256 differs from what was stamped). Nodes that still match are
    omitted. Replaces an ad-hoc ``os.path.exists`` gate; never mutates the graph.
    """
    if hash_property is not None:
        extra = f", n.{_cypher_identifier(hash_property, kind='hash property')} AS _hash"
    else:
        extra = f", n.{_cypher_identifier(mtime_property, kind='mtime property')} AS _mtime"
    rows = _nodes_with_path(graph, node_type, path_property, extra)
    drift: list = []
    from collections import OrderedDict

    snapshots = OrderedDict()
    for r in rows:
        resolved = _resolved_file_path(r["path"], base_dir)
        snapshot = _cached_file_snapshot(snapshots, resolved, include_hash=hash_property is not None)
        if snapshot is None:
            drift.append({"id": r["id"], "path": r["path"], "status": "missing"})
            continue
        stored = r.get("_hash") if hash_property is not None else r.get("_mtime")
        current = snapshot["hash"] if hash_property is not None else snapshot["mtime"]
        if stored is None or current != stored:
            drift.append({"id": r["id"], "path": r["path"], "status": "changed"})
    return drift


__all__ = [
    "__version__",
    "KnowledgeGraph",
    "FrozenGraph",
    "Transaction",
    "Session",
    "ResultView",
    "ResultIter",
    "load",
    "load_rdf",
    "open",
    "open_session",
    "from_bytes",
    "cypher_pass_names",
    "from_blueprint",
    "from_records",
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
    "InternerCollisionError",
    "InternalError",
]

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

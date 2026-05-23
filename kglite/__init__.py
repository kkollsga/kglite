"""KGLite - A high-performance graph database library with Python bindings written in Rust."""

from .blueprint import from_blueprint  # noqa: E402  (must override star-import from .kglite)
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


def repo_tree(
    repo: str,
    **kwargs,
) -> "KnowledgeGraph":
    """Clone a GitHub repository and build a knowledge graph from its source code.

    Convenience re-export of :func:`kglite.code_tree.repo_tree`.
    Requires the ``[code-tree]`` extra: ``pip install kglite[code-tree]``.
    """
    from .code_tree import repo_tree as _repo_tree

    return _repo_tree(repo, **kwargs)


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
    "Transaction",
    "ResultView",
    "ResultIter",
    "load",
    "cypher_pass_names",
    "from_blueprint",
    "repo_tree",
    "to_neo4j",
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

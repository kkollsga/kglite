"""Build a KGLite graph from a networkx graph.

The reverse direction (``KnowledgeGraph.to_networkx()``) lives in Rust
(``crates/kglite-py/src/graph/pyapi/networkx.rs``). This module is the
import side — pure Python, bulk-loading via the DataFrame fast paths
(``add_nodes`` / ``add_connections``) so it stays O(n + e).
"""

from __future__ import annotations

from collections import defaultdict
import typing

if typing.TYPE_CHECKING:
    from . import KnowledgeGraph


def from_networkx(
    nx_graph: typing.Any,
    *,
    default_node_type: str = "Node",
    default_edge_type: str = "RELATED",
) -> "KnowledgeGraph":
    """Build a :class:`KnowledgeGraph` from a ``networkx`` graph.

    Accepts ``Graph`` / ``DiGraph`` / ``MultiGraph`` / ``MultiDiGraph``.
    Undirected edges (from ``Graph`` / ``MultiGraph``) become a single
    directed edge each, in the orientation networkx yields them.

    Round-trip with :meth:`KnowledgeGraph.to_networkx`: nodes carrying a
    ``node_type`` attribute are grouped by that type, the networkx node
    key becomes the node ``id``, and a ``title`` attribute (if present)
    becomes the node title (otherwise the id is used). Edges carrying a
    ``connection_type`` attribute use it as the edge type; for a
    ``MultiDiGraph`` the edge key produced by :meth:`to_networkx` is the
    connection type, so parallel edges of different types survive.

    Plain networkx graphs (no ``node_type`` / ``connection_type`` attrs)
    get ``default_node_type`` and ``default_edge_type``.

    Requires the ``networkx`` and ``pandas`` packages.

    Args:
        nx_graph: A networkx graph instance.
        default_node_type: Node type for nodes lacking a ``node_type`` attr.
        default_edge_type: Edge type for edges lacking a ``connection_type`` attr.

    Returns:
        A new :class:`KnowledgeGraph`.

    Example::

        import kglite, networkx as nx

        nxg = nx.karate_club_graph()
        g = kglite.from_networkx(nxg)
    """
    try:
        import networkx  # noqa: F401 — presence check; we only use the duck-typed nx_graph
    except ImportError:
        raise ImportError(
            "The 'networkx' package is required for from_networkx(). Install with: pip install networkx"
        ) from None
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "The 'pandas' package is required for from_networkx(). Install with: pip install pandas"
        ) from None

    from . import KnowledgeGraph

    g = KnowledgeGraph()

    # -- Group nodes by node_type, collecting a row dict per node. --
    # The networkx node key becomes the `id`. `title` attr -> title col;
    # remaining attrs (minus node_type/title/id) become node properties.
    nodes_by_type: dict[str, list[dict]] = defaultdict(list)
    for key, attrs in nx_graph.nodes(data=True):
        ntype = attrs.get("node_type", default_node_type)
        row: dict[str, typing.Any] = {"id": key}
        title = attrs.get("title", key)
        row["title"] = title
        for k, v in attrs.items():
            if k in ("node_type", "title", "id"):
                continue
            row[k] = v
        nodes_by_type[str(ntype)].append(row)

    for ntype, rows in nodes_by_type.items():
        df = pd.DataFrame(rows)
        g.add_nodes(df, ntype, "id", "title")

    # -- Build an index of key -> node_type so edge endpoints can name
    # their source/target types (add_connections needs both). --
    type_of_node: dict[typing.Any, str] = {}
    for key, attrs in nx_graph.nodes(data=True):
        type_of_node[key] = str(attrs.get("node_type", default_node_type))

    # -- Group edges by (connection_type, source_type, target_type). --
    # add_connections is keyed on a single (src_type, edge_type, tgt_type)
    # triple, so we bucket accordingly. Each bucket carries its own
    # property columns.
    EdgeKey = typing.Tuple[str, str, str]
    edges_by_key: dict[EdgeKey, list[dict]] = defaultdict(list)
    is_multigraph = nx_graph.is_multigraph()
    edge_iter = nx_graph.edges(keys=True, data=True) if is_multigraph else nx_graph.edges(data=True)
    for rec in edge_iter:
        if is_multigraph:
            u, v, ekey, attrs = rec
        else:
            u, v, attrs = rec
            ekey = None
        ctype = attrs.get("connection_type")
        if ctype is None:
            # MultiDiGraph from to_networkx() uses connection_type as the
            # edge key; fall back to it, then to the default.
            ctype = ekey if (is_multigraph and isinstance(ekey, str)) else default_edge_type
        stype = type_of_node.get(u, default_node_type)
        ttype = type_of_node.get(v, default_node_type)
        row = {"src": u, "tgt": v}
        for k, val in attrs.items():
            if k == "connection_type":
                continue
            row[k] = val
        edges_by_key[(str(ctype), stype, ttype)].append(row)

    for (ctype, stype, ttype), rows in edges_by_key.items():
        df = pd.DataFrame(rows)
        # add_connections only stores property columns named in `columns`;
        # everything except the src/tgt id fields becomes an edge property.
        prop_cols = [c for c in df.columns if c not in ("src", "tgt")]
        g.add_connections(
            df,
            ctype,
            stype,
            "src",
            ttype,
            "tgt",
            columns=prop_cols or None,
        )

    return g

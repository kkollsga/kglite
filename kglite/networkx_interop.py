"""Build a KGLite graph from a networkx graph.

The reverse direction (``KnowledgeGraph.to_networkx()``) lives in Rust
(``crates/kglite-py/src/graph/pyapi/networkx.rs``). This module is the
import side — pure Python, bulk-loading via the DataFrame fast paths
(``add_nodes`` / ``add_connections``) so it stays O(n + e).
"""

from __future__ import annotations

from collections import defaultdict
import numbers
import typing

if typing.TYPE_CHECKING:
    from . import KnowledgeGraph

# Node attributes that carry identity rather than data; never re-imported as
# properties (`to_networkx` writes node_type/title last so they always win).
_IDENTITY_ATTRS = ("node_type", "title", "id")


def _is_type_id_key(key: typing.Any, attrs: typing.Any) -> bool:
    """Whether ``key`` is the ``(node_type, id)`` tuple that
    ``to_networkx(node_key="type_id")`` emits.

    The first element must equal the node's own ``node_type`` attribute. A
    foreign tuple-labelled graph (``nx.grid_2d_graph`` coordinates, say)
    carries no such attribute, so the detection cannot misfire on it.
    """
    return isinstance(key, tuple) and len(key) == 2 and "node_type" in attrs and key[0] == str(attrs["node_type"])


def _is_representable_id(value: typing.Any) -> bool:
    """Whether the bulk loader can store ``value`` in a node-id column.

    Ids are stored as integers or text. Booleans and byte strings coerce;
    a float only survives when it is a whole number (``1.0`` does, ``1.5``
    does not); everything else — tuples, sets, ``NaN``, arbitrary objects —
    is dropped row-by-row by ``add_nodes``, which is what the caller must be
    told about instead of receiving a smaller graph.
    """
    if isinstance(value, (str, bytes, bytearray, bool)):
        return True
    if isinstance(value, numbers.Integral):
        return True
    if isinstance(value, numbers.Real):
        try:
            return float(value).is_integer()
        except (OverflowError, ValueError, TypeError):
            return False
    return False


def _mixed_key_shapes_error(type_id_key: typing.Any, offender: typing.Any, attrs: typing.Any) -> Exception:
    """The graph holds both export tuple keys and something else."""
    from . import ArgumentError

    if isinstance(offender, tuple) and len(offender) == 2 and "node_type" in attrs:
        detail = (
            f"{offender!r} is a 2-tuple whose 'node_type' attribute says "
            f"{str(attrs['node_type'])!r} — an export key always repeats the "
            f"node's own type, so this key and this attribute disagree"
        )
    else:
        detail = f"{offender!r} (type {type(offender).__name__}) is not a (node_type, id) 2-tuple"
    return ArgumentError(
        f"from_networkx(): the graph mixes node-key shapes. {type_id_key!r} is the "
        f"(node_type, id) tuple key that to_networkx(node_key='type_id') emits, but "
        f"{detail}. Importing only the half that matches would silently drop or "
        f"mistype the rest, so the whole call is refused: give every node the same "
        f"key shape."
    )


def _node_key_mode(nx_graph: typing.Any) -> str:
    """Decide, for the WHOLE graph, how its node keys are shaped.

    Returns ``"type_id"`` when every node key is a ``to_networkx`` export
    tuple and ``"id"`` when none is. A graph that mixes the two raises: a
    partial import is exactly the silent shrinkage this guard exists to stop.
    """
    first_type_id: typing.Any = None
    first_plain: typing.Any = None
    first_plain_attrs: typing.Any = None
    seen_type_id = seen_plain = False
    for key, attrs in nx_graph.nodes(data=True):
        if _is_type_id_key(key, attrs):
            if not seen_type_id:
                first_type_id, seen_type_id = key, True
        elif not seen_plain:
            first_plain, first_plain_attrs, seen_plain = key, attrs, True
        if seen_type_id and seen_plain:
            raise _mixed_key_shapes_error(first_type_id, first_plain, first_plain_attrs)
    return "type_id" if seen_type_id else "id"


def _reject_unrepresentable_ids(nx_graph: typing.Any, key_mode: str) -> None:
    """Refuse ids the id column cannot store, before any row is loaded.

    ``add_nodes`` warn-and-drops such rows, which turned a whole tuple-keyed
    graph into a silently empty one. The drop machinery stays (other callers
    rely on it); this importer simply stops feeding it garbage. On the
    ``type_id`` path the id is the tuple's second element, so that is what
    gets checked.
    """
    from . import ArgumentError

    count = 0
    sample: typing.Any = None
    sample_key: typing.Any = None
    for key in nx_graph.nodes():
        node_id = key[1] if key_mode == "type_id" else key
        if not _is_representable_id(node_id):
            count += 1
            if count == 1:
                sample, sample_key = node_id, key
    if count == 0:
        return

    if key_mode == "type_id":
        detail = f"the id half of node key {sample_key!r} is {sample!r} (type {type(sample).__name__})"
        hint = ""
    else:
        detail = f"the first is {sample!r} (type {type(sample).__name__})"
        hint = (
            (
                " A tuple key is unwrapped as a (node_type, id) pair only when it is a 2-tuple "
                "whose first element equals the node's own 'node_type' attribute — the shape "
                "to_networkx(node_key='type_id') emits. These keys are not that shape."
            )
            if isinstance(sample, tuple)
            else ""
        )
    raise ArgumentError(
        f"from_networkx(): {count} of {nx_graph.number_of_nodes()} node keys cannot be "
        f"stored as a node id — {detail}. A node id must be an integer or a string.{hint} "
        f"Relabel the nodes before importing, e.g. "
        f"nx.convert_node_labels_to_integers(nx_graph) or "
        f"nx.relabel_nodes(nx_graph, {{old: new, ...}})."
    )


def _collect_nodes(
    nx_graph: typing.Any, key_mode: str, default_node_type: str
) -> tuple[dict[str, list[dict]], dict[typing.Any, str]]:
    """Group nodes into per-type row dicts, and index key -> node_type so the
    edge pass can name both endpoint types.

    On the ``type_id`` path the type is already in the key, so the index is
    left empty rather than duplicating it.
    """
    nodes_by_type: dict[str, list[dict]] = defaultdict(list)
    type_of_node: dict[typing.Any, str] = {}
    for key, attrs in nx_graph.nodes(data=True):
        if key_mode == "type_id":
            ntype, node_id = key[0], key[1]
        else:
            ntype, node_id = str(attrs.get("node_type", default_node_type)), key
            type_of_node[key] = ntype
        row: dict[str, typing.Any] = {"id": node_id, "title": attrs.get("title", node_id)}
        for k, v in attrs.items():
            if k in _IDENTITY_ATTRS:
                continue
            row[k] = v
        nodes_by_type[ntype].append(row)
    return nodes_by_type, type_of_node


def _collect_edges(
    nx_graph: typing.Any,
    key_mode: str,
    type_of_node: dict[typing.Any, str],
    default_node_type: str,
    default_edge_type: str,
) -> dict[tuple[str, str, str], list[dict]]:
    """Group edges by (connection_type, source_type, target_type).

    ``add_connections`` is keyed on a single (src_type, edge_type, tgt_type)
    triple, so we bucket accordingly. Each bucket carries its own property
    columns.
    """
    edges_by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
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
        if key_mode == "type_id":
            (stype, src), (ttype, tgt) = u, v
        else:
            stype = type_of_node.get(u, default_node_type)
            ttype = type_of_node.get(v, default_node_type)
            src, tgt = u, v
        row = {"src": src, "tgt": tgt}
        for k, val in attrs.items():
            if k == "connection_type":
                continue
            row[k] = val
        edges_by_key[(str(ctype), stype, ttype)].append(row)
    return edges_by_key


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

    A graph exported with ``to_networkx(node_key="type_id")`` is detected
    and unwrapped automatically: its keys are ``(node_type, id)`` tuples
    whose first element repeats the node's own ``node_type`` attribute, so
    the id and both endpoint types come straight from the keys. Detection
    is per graph and all-or-nothing — a graph mixing tuple keys with plain
    ones raises rather than importing the half it understands.

    Plain networkx graphs (no ``node_type`` / ``connection_type`` attrs)
    get ``default_node_type`` and ``default_edge_type``. Node keys must be
    storable as ids (integers or strings); a key that is not — a foreign
    tuple label, a fractional float — raises before anything is loaded,
    rather than being dropped row by row into a smaller graph.

    Requires the ``networkx`` and ``pandas`` packages.

    Args:
        nx_graph: A networkx graph instance.
        default_node_type: Node type for nodes lacking a ``node_type`` attr.
        default_edge_type: Edge type for edges lacking a ``connection_type`` attr.

    Returns:
        A new :class:`KnowledgeGraph`.

    Raises:
        ArgumentError: A node key cannot be stored as an id, or the graph
            mixes ``(node_type, id)`` export keys with other key shapes.

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

    # Settle the key shape and reject unusable ids BEFORE the first
    # add_nodes call, so a refusal never leaves a half-built graph behind.
    key_mode = _node_key_mode(nx_graph)
    _reject_unrepresentable_ids(nx_graph, key_mode)

    g = KnowledgeGraph()

    nodes_by_type, type_of_node = _collect_nodes(nx_graph, key_mode, default_node_type)
    for ntype, rows in nodes_by_type.items():
        df = pd.DataFrame(rows)
        g.add_nodes(df, ntype, "id", "title")

    edges_by_key = _collect_edges(nx_graph, key_mode, type_of_node, default_node_type, default_edge_type)
    for (ctype, stype, ttype), rows in edges_by_key.items():
        df = pd.DataFrame(rows)
        g.add_connections(
            df,
            ctype,
            stype,
            "src",
            ttype,
            "tgt",
        )

    return g

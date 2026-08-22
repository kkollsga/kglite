//! WAL replay — apply recovered [`MutationOp`]s to a `DirGraph`.
//!
//! The inverse of the capture seam ([`crate::graph::storage::recording`]).
//! On open, the engine loads the `.kgl` checkpoint, then calls
//! [`apply_frames`] with the WAL frames recovered by
//! [`crate::graph::wal::recover`] to fold back every mutation committed
//! since the checkpoint.
//!
//! ## Reuse, not reimplementation
//!
//! Like [`crate::graph::mutation::extend`], replay routes upserts through
//! `maintain::add_nodes` / `add_connections` (the single source of truth
//! for schema/interner extension, id-indexing, and edge dedup) and node
//! removals through `maintain::detach_delete_nodes`. Replaying never
//! touches the storage layer directly except for the one thing those
//! helpers don't expose — removing a single edge by identity.
//!
//! ## Net-state fold (why not per-frame)
//!
//! The WAL is a **redo log**: the recovered graph is the fold of every op
//! with `lsn > checkpoint_version` over the snapshot. Applying that
//! frame-by-frame is correct but quadratic — each frame's `add_nodes` call
//! rebuilds the type's id-index over a growing graph, so replaying N
//! single-row frames is O(N · graph). Instead we **fold all ops into a net
//! per-entity state first** (last write wins per `(node_type, id)` /
//! `(conn, src, tgt)` — `Upsert` or `Remove`), then apply that net state in
//! a handful of bulk calls (one `add_nodes` per node type, one
//! `add_connections` per edge group), rebuilding each index once.
//!
//! This is sound because the ops are **identity-keyed and idempotent**: the
//! final value of an entity depends only on its last op, not on the path
//! there. Folding then applying reaches the same final state as a
//! frame-by-frame replay, and replaying twice is still harmless.
//!
//! Apply order — node upserts → label sets → edge upserts → edge removes →
//! node removes — respects referential integrity (endpoints exist before
//! their edges; a removed node's edges go with it via detach). An edge whose
//! endpoint is net-removed is dropped from the edge-upsert batch (its
//! node-remove will detach it anyway), and so is a label set.
//!
//! Labels fold in their own map rather than riding on `UpsertNode`, because
//! in the live graph properties and labels are independent state: neither
//! `SET n.x = 1` nor `SET n:B` disturbs the other. See [`LabelNet`].
//!
//! ## Recovery is value-faithful
//!
//! The bulk calls above take a [`DataFrame`], whose columns are singly
//! typed — so a property logged as `Int64` on one node and `String` on
//! another would come back as two strings. Replay is recovery, not a load:
//! a mixed-type property is legal in a live graph, and losing a value's
//! type across a crash is unrecoverable. So each group's columns are split
//! by [`split_faithful_columns`] and only the ones the frame carries
//! unchanged ride it; the rest are written value by value afterwards
//! ([`apply_exact_node_props`] / [`apply_exact_edge_props`]).
//!
//! The two *fixed* columns — a node's `id`/`title`, an edge's endpoint ids —
//! cannot be held back, because the bulk calls address rows by them. A group
//! whose fixed columns are mixed is split by shape into one bulk call each
//! ([`partition_by_fixed_shapes`]) instead. Endpoint ids make the stakes
//! plainest: `add_connections` vivifies a stub for an id it cannot find, so a
//! stringified endpoint id does not merely lose an edge, it *invents a node*.

use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;

use crate::datatypes::{DataFrame, Value};
use crate::graph::mutation::maintain::{add_connections, add_nodes, detach_delete_nodes};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::{GraphRead, GraphWrite};
use crate::graph::wal::{MutationOp, WalFrame};

/// Logical node identity: `(node_type, id)`.
type NodeKey = (String, Value);
/// Logical edge identity: `(conn_type, src_type, src_id, tgt_type, tgt_id)`.
type EdgeKey = (String, String, Value, String, Value);
/// One row headed for a bulk call: the two fixed cells (a node's id/title,
/// an edge's src/tgt id) plus that row's property payload.
type UpsertRow = (Value, Value, HashMap<String, Value>);

/// Net state of a node after folding: an upsert (title + props) or a remove.
enum NodeNet {
    Upsert {
        title: Value,
        props: Vec<(String, Value)>,
    },
    Remove,
}
/// Net state of an edge after folding.
enum EdgeNet {
    Upsert { props: Vec<(String, Value)> },
    Remove,
}

/// Net secondary-label set per node, folded independently of `NodeNet`.
///
/// Labels are *not* part of a node's property payload — in the live graph
/// `SET n.x = 1` does not touch labels and `SET n:B` does not touch
/// properties — so an `UpsertNode` must not be allowed to clobber a label
/// set logged before it. Keeping them in their own last-write-wins map
/// reproduces that independence regardless of the order the two op kinds
/// appear in the log.
type LabelNet = HashMap<NodeKey, Vec<String>>;

/// Fold every frame with `lsn > after_lsn` into net per-entity state and
/// apply it to `graph` in bulk. Returns the highest `lsn` folded in (or
/// `after_lsn` if none), so the caller can set the recovered graph version.
pub fn apply_frames(
    graph: &mut DirGraph,
    frames: &[WalFrame],
    after_lsn: u64,
) -> Result<u64, String> {
    graph
        .prepare_disk_mutation()
        .map_err(|e| format!("disk mutation lease failed: {e}"))?;
    let mut nodes: HashMap<NodeKey, NodeNet> = HashMap::new();
    let mut edges: HashMap<EdgeKey, EdgeNet> = HashMap::new();
    let mut labels: LabelNet = HashMap::new();
    let mut max_lsn = after_lsn;
    let mut any = false;

    for frame in frames {
        if frame.lsn <= after_lsn {
            continue;
        }
        any = true;
        max_lsn = max_lsn.max(frame.lsn);
        for op in &frame.ops {
            match op {
                MutationOp::UpsertNode {
                    node_type,
                    id,
                    title,
                    properties,
                } => {
                    nodes.insert(
                        (node_type.clone(), id.clone()),
                        NodeNet::Upsert {
                            title: title.clone(),
                            props: properties.clone(),
                        },
                    );
                }
                MutationOp::RemoveNode { node_type, id } => {
                    nodes.insert((node_type.clone(), id.clone()), NodeNet::Remove);
                }
                MutationOp::UpsertEdge {
                    conn_type,
                    src_type,
                    src_id,
                    tgt_type,
                    tgt_id,
                    properties,
                } => {
                    edges.insert(
                        (
                            conn_type.clone(),
                            src_type.clone(),
                            src_id.clone(),
                            tgt_type.clone(),
                            tgt_id.clone(),
                        ),
                        EdgeNet::Upsert {
                            props: properties.clone(),
                        },
                    );
                }
                MutationOp::RemoveEdge {
                    conn_type,
                    src_type,
                    src_id,
                    tgt_type,
                    tgt_id,
                } => {
                    edges.insert(
                        (
                            conn_type.clone(),
                            src_type.clone(),
                            src_id.clone(),
                            tgt_type.clone(),
                            tgt_id.clone(),
                        ),
                        EdgeNet::Remove,
                    );
                }
                MutationOp::SetNodeLabels {
                    node_type,
                    id,
                    labels: set,
                } => {
                    labels.insert((node_type.clone(), id.clone()), set.clone());
                }
            }
        }
    }

    if any {
        apply_net(graph, nodes, edges, labels)?;
    }
    Ok(max_lsn)
}

/// Apply folded net state in bulk, one phase per entity concern. See the
/// module docs for why this order is the referentially-safe one.
fn apply_net(
    graph: &mut DirGraph,
    nodes: HashMap<NodeKey, NodeNet>,
    edges: HashMap<EdgeKey, EdgeNet>,
    labels: LabelNet,
) -> Result<(), String> {
    // Node identities scheduled for removal — used to skip work whose
    // subject won't exist once phase 5 runs.
    let removed_nodes: HashSet<NodeKey> = nodes
        .iter()
        .filter(|(_, v)| matches!(v, NodeNet::Remove))
        .map(|(k, _)| k.clone())
        .collect();

    apply_node_upserts(graph, &nodes)?;
    apply_label_sets(graph, &labels, &removed_nodes);
    apply_edge_upserts(graph, &edges, &removed_nodes)?;
    apply_edge_removes(graph, &edges);
    apply_node_removes(graph, &nodes);
    Ok(())
}

/// Phase 1 — node upserts, grouped by node_type, one `add_nodes` each so
/// the type's id-index is rebuilt once rather than per row.
///
/// Property columns the frame cannot carry without retyping their values
/// (see [`split_faithful_columns`]) are held back from the `DataFrame` and
/// written one exact `Value` at a time afterwards — after, because
/// `add_nodes` runs in `replace` mode and clears a row's properties before
/// applying the frame's.
fn apply_node_upserts(
    graph: &mut DirGraph,
    nodes: &HashMap<NodeKey, NodeNet>,
) -> Result<(), String> {
    let mut node_groups: HashMap<&str, NodeRows> = HashMap::new();
    for ((node_type, id), net) in nodes {
        if let NodeNet::Upsert { title, props } = net {
            let g = node_groups.entry(node_type.as_str()).or_default();
            for (k, _) in props {
                g.note_column(k);
            }
            g.rows
                .push((id.clone(), title.clone(), props.iter().cloned().collect()));
        }
    }
    for (node_type, group) in node_groups {
        let (framed, exact) = split_faithful_columns(&group.columns, &group.rows);
        // `id` and `title` cannot be held back — `add_nodes` addresses rows by
        // them — so a type whose ids or titles differ in type across nodes is
        // split into one bulk call per shape instead. Only such a type pays
        // the extra call; the usual one keeps a single `add_nodes`.
        if fixed_columns_are_faithful(&group.rows) {
            upsert_node_rows(graph, node_type, &framed, &exact, &group.rows)?;
        } else {
            for part in partition_by_fixed_shapes(&group.rows) {
                upsert_node_rows(graph, node_type, &framed, &exact, &part)?;
            }
        }
    }
    Ok(())
}

/// One bulk `add_nodes` over rows that share an id and title shape, plus the
/// held-back columns those rows carry.
fn upsert_node_rows(
    graph: &mut DirGraph,
    node_type: &str,
    framed: &[String],
    exact: &[String],
    rows: &[UpsertRow],
) -> Result<(), String> {
    // Declare the held-back columns *before* the bulk create, not after. The
    // create is what builds the type's `ColumnStore`, and it types each column
    // from the type's declared metadata — so a column whose declaration arrives
    // afterwards is instead typed from whichever exact value happens to be
    // written into it first. For a property that mixes an int and a float that
    // is the difference between a `Mixed` column that returns both values as
    // logged and a `Float64` column that promotes the int on the way in
    // (`int_and_float_under_one_property_do_not_promote`). Latent on
    // mapped/disk before construction became columnar; universal after.
    declare_exact_node_columns(graph, node_type, exact, rows);
    let df = build_dataframe(&["id", "title"], framed, rows)?;
    add_nodes(
        graph,
        df,
        node_type.to_string(),
        "id".to_string(),
        Some("title".to_string()),
        Some("replace".to_string()),
    )?;
    apply_exact_node_props(graph, node_type, exact, rows);
    Ok(())
}

/// Declare each held-back column on the node type — its own value's type when
/// the column is single-typed (a `Point`, say, which has no frame column but
/// one clear type), `"mixed"` when it is not — and register its key on the
/// type's schema so the store the bulk create builds carries a column of that
/// type from the start.
///
/// Skipping the metadata half would also leave a replayed property out of the
/// type's catalogue, which on a disk graph decides what `save()` persists.
fn declare_exact_node_columns(
    graph: &mut DirGraph,
    node_type: &str,
    exact: &[String],
    rows: &[UpsertRow],
) {
    if exact.is_empty() {
        return;
    }
    let mut declared: HashMap<String, String> = HashMap::new();
    for col in exact {
        declared.insert(
            col.clone(),
            declared_type_name(rows.iter().filter_map(|(_, _, p)| p.get(col))),
        );
    }
    graph.upsert_node_type_metadata(node_type, declared);
    // `get_or_intern`, never `InternedKey::from_str`: a hash-only key cannot be
    // resolved back to a string at save time, and the property vanishes on
    // reload.
    let keys: Vec<_> = exact
        .iter()
        .map(|col| graph.interner.get_or_intern(col))
        .collect();
    graph.ensure_type_schema_keys(node_type, &keys);
}

/// Write the held-back columns onto each node with their logged `Value`
/// untouched, through the same per-property setter the Cypher `SET`
/// fallback uses.
fn apply_exact_node_props(
    graph: &mut DirGraph,
    node_type: &str,
    exact: &[String],
    rows: &[UpsertRow],
) {
    if exact.is_empty() {
        return;
    }
    // The declaration ran before the bulk create — see
    // `declare_exact_node_columns` for why the order is load-bearing.
    for (id, _, props) in rows {
        let Some(idx) = graph.lookup_by_id(node_type, id) else {
            continue;
        };
        for col in exact {
            match props.get(col) {
                None | Some(Value::Null) => continue,
                Some(value) => {
                    // `get_or_intern`, never `InternedKey::from_str`: a
                    // hash-only key cannot be resolved back to a string at
                    // save time, and the property vanishes on reload.
                    let key = graph.interner.get_or_intern(col);
                    graph.ensure_type_schema_keys(node_type, &[key]);
                    GraphWrite::set_node_property(&mut graph.graph, idx, key, value.clone());
                }
            }
        }
    }
}

/// Phase 2 — secondary-label sets, applied through the `DirGraph` choke
/// points so `secondary_label_index` and `has_secondary_labels` stay
/// canonical (a direct map write would desynchronise the fast-skip flag).
///
/// Each op carries the node's **whole** label set, so this reconciles
/// rather than adds: labels the checkpoint holds but the log does not are
/// removed. That is what makes a `REMOVE n:Label` recoverable, and what
/// keeps a re-replay idempotent. Runs after phase 1 so a node created by
/// this same replay is already present.
fn apply_label_sets(graph: &mut DirGraph, labels: &LabelNet, removed_nodes: &HashSet<NodeKey>) {
    for (key @ (node_type, id), target) in labels {
        if removed_nodes.contains(key) {
            continue;
        }
        let Some(idx) = graph.lookup_by_id(node_type, id) else {
            continue;
        };
        for stale in graph.secondary_label_names(idx) {
            if !target.contains(&stale) {
                let key = graph.interner.get_or_intern(&stale);
                // Only errors when `stale` is the primary type, which
                // `secondary_label_names` never yields.
                let _ = graph.remove_node_label(idx, key);
            }
        }
        for label in target {
            let key = graph.interner.get_or_intern(label);
            graph.add_node_label(idx, key);
        }
    }
}

/// Phase 3 — edge upserts, grouped by `(conn, src_type, tgt_type)`.
fn apply_edge_upserts(
    graph: &mut DirGraph,
    edges: &HashMap<EdgeKey, EdgeNet>,
    removed_nodes: &HashSet<NodeKey>,
) -> Result<(), String> {
    let mut edge_groups: HashMap<(&str, &str, &str), EdgeRows> = HashMap::new();
    for ((conn, src_type, src_id, tgt_type, tgt_id), net) in edges {
        if let EdgeNet::Upsert { props } = net {
            // Skip if either endpoint is being removed — the node-remove
            // detaches any such edge anyway, and add_connections would fail
            // on a missing endpoint.
            if removed_nodes.contains(&(src_type.clone(), src_id.clone()))
                || removed_nodes.contains(&(tgt_type.clone(), tgt_id.clone()))
            {
                continue;
            }
            let g = edge_groups
                .entry((conn.as_str(), src_type.as_str(), tgt_type.as_str()))
                .or_default();
            for (k, _) in props {
                g.note_column(k);
            }
            g.rows.push((
                src_id.clone(),
                tgt_id.clone(),
                props.iter().cloned().collect(),
            ));
        }
    }
    for ((conn, src_type, tgt_type), group) in edge_groups {
        let (framed, exact) = split_faithful_columns(&group.columns, &group.rows);
        // The endpoint-id columns get the same treatment as a node's `id`,
        // and for a sharper reason: `add_connections` *vivifies* a stub for an
        // endpoint id it cannot find, so a stringified `2` does not merely
        // lose an edge — it invents a node under the id `"2"`.
        let key = EdgeGroup {
            conn,
            src_type,
            tgt_type,
        };
        if fixed_columns_are_faithful(&group.rows) {
            upsert_edge_rows(graph, key, &framed, &exact, &group.rows)?;
        } else {
            for part in partition_by_fixed_shapes(&group.rows) {
                upsert_edge_rows(graph, key, &framed, &exact, &part)?;
            }
        }
    }
    Ok(())
}

/// The `(conn, src_type, tgt_type)` an edge group is keyed by.
#[derive(Clone, Copy)]
struct EdgeGroup<'a> {
    conn: &'a str,
    src_type: &'a str,
    tgt_type: &'a str,
}

/// One bulk `add_connections` over rows whose endpoint ids share a shape.
fn upsert_edge_rows(
    graph: &mut DirGraph,
    group: EdgeGroup<'_>,
    framed: &[String],
    exact: &[String],
    rows: &[UpsertRow],
) -> Result<(), String> {
    let df = build_dataframe(&["src_id", "tgt_id"], framed, rows)?;
    add_connections(
        graph,
        df,
        group.conn.to_string(),
        group.src_type.to_string(),
        "src_id".to_string(),
        group.tgt_type.to_string(),
        "tgt_id".to_string(),
        None,
        None,
        Some("replace".to_string()),
    )?;
    apply_exact_edge_props(graph, group, exact, rows);
    Ok(())
}

/// The edge twin of [`apply_exact_node_props`] — write the held-back
/// columns straight onto the `EdgeData`, as the Cypher `SET` path on an
/// edge binding does.
fn apply_exact_edge_props(
    graph: &mut DirGraph,
    group: EdgeGroup<'_>,
    exact: &[String],
    rows: &[UpsertRow],
) {
    let EdgeGroup {
        conn,
        src_type,
        tgt_type,
    } = group;
    if exact.is_empty() {
        return;
    }
    let mut declared: HashMap<String, String> = HashMap::new();
    for col in exact {
        declared.insert(
            col.clone(),
            declared_type_name(rows.iter().filter_map(|(_, _, p)| p.get(col))),
        );
    }
    graph.upsert_connection_type_metadata(conn, src_type, tgt_type, declared);

    let conn_key = InternedKey::from_str(conn);
    for (src_id, tgt_id, props) in rows {
        let (Some(src), Some(tgt)) = (
            graph.lookup_by_id(src_type, src_id),
            graph.lookup_by_id(tgt_type, tgt_id),
        ) else {
            continue;
        };
        let Some(eidx) = graph
            .graph
            .edges_connecting(src, tgt)
            .find(|er| er.weight().connection_type == conn_key)
            .map(|er| er.id())
        else {
            continue;
        };
        for col in exact {
            match props.get(col) {
                None | Some(Value::Null) => continue,
                Some(value) => {
                    let key = graph.interner.get_or_intern(col);
                    if let Some(edge) = GraphWrite::edge_weight_mut(&mut graph.graph, eidx) {
                        match edge.properties.iter_mut().find(|(ek, _)| *ek == key) {
                            Some((_, existing)) => *existing = value.clone(),
                            None => edge.properties.push((key, value.clone())),
                        }
                    }
                }
            }
        }
    }
}

/// Split a group's property columns into the ones a `DataFrame` carries
/// unchanged and the ones it would retype.
///
/// `DataFrame` columns are singly typed: `from_cypher_rows` promotes each
/// column to one `ColumnType` and rewrites every cell into it. That is the
/// documented, wanted behaviour for the load paths that share the builder —
/// but replay is not a load, it is *recovery*, and a mixed `Int64`/`String`
/// property replaying as two strings (or an `Int64`/`Float64` one replaying
/// as two floats) is type loss no re-query can undo. A live graph is allowed
/// mixed types under one property — `Value` is a sum type and the columnar
/// store demotes such a column to `Mixed` — so recovery must be allowed them
/// too.
///
/// A column is *faithful* when every value it carries has an exact column
/// shape and they all share it; `Null` carries no type and is ignored (the
/// frame fills absent cells with `Null`, which `add_nodes` skips). Anything
/// else — a mix, or a variant with no dense column at all such as `Point` —
/// is returned as `exact` for its caller to write value by value.
fn split_faithful_columns(columns: &[String], rows: &[UpsertRow]) -> (Vec<String>, Vec<String>) {
    let mut framed = Vec::with_capacity(columns.len());
    let mut exact = Vec::new();
    for col in columns {
        if column_is_faithful(rows.iter().filter_map(|(_, _, p)| p.get(col))) {
            framed.push(col.clone());
        } else {
            exact.push(col.clone());
        }
    }
    (framed, exact)
}

/// Would every value here survive `DataFrame::from_cypher_rows` unchanged?
/// True when they all share one exact column shape. `Null` carries no type
/// and is skipped — the frame stores it as a null cell in any column, and
/// the loaders skip null cells.
fn column_is_faithful<'a>(values: impl Iterator<Item = &'a Value>) -> bool {
    let mut shape: Option<FrameShape> = None;
    for value in values {
        if matches!(value, Value::Null) {
            continue;
        }
        match (frame_shape(value), shape) {
            (None, _) => return false,
            (Some(s), None) => shape = Some(s),
            (Some(s), Some(seen)) if s == seen => {}
            (Some(_), Some(_)) => return false,
        }
    }
    true
}

/// Do a group's two fixed columns — a node's `id`/`title`, an edge's
/// `src_id`/`tgt_id` — survive the frame unchanged? They cannot be held back
/// like a property (the bulk calls address rows by them), so a `false` here
/// means the rows must be split by shape instead.
fn fixed_columns_are_faithful(rows: &[UpsertRow]) -> bool {
    column_is_faithful(rows.iter().map(|(a, _, _)| a))
        && column_is_faithful(rows.iter().map(|(_, b, _)| b))
}

/// What a fixed cell contributes to its column's shape.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FixedKind {
    /// No type evidence — a null cell in any column, so it fits any part.
    Null,
    Shaped(FrameShape),
    /// A value with no exact column (a `Point` id). Kept in its own part so
    /// it cannot drag a typed column to text with it.
    Shapeless,
}

fn fixed_kind(value: &Value) -> FixedKind {
    match value {
        Value::Null => FixedKind::Null,
        other => match frame_shape(other) {
            Some(shape) => FixedKind::Shaped(shape),
            None => FixedKind::Shapeless,
        },
    }
}

/// Split rows so that each part's two fixed columns are singly shaped, and
/// therefore ride the frame unchanged. A null cell joins the first part it
/// fits — it is a null in every column shape, so which part carries it
/// cannot change what is stored.
///
/// Only reached for a group [`fixed_columns_are_faithful`] rejected, so the
/// clone it costs is paid by a mixed-id type alone.
fn partition_by_fixed_shapes(rows: &[UpsertRow]) -> Vec<Vec<UpsertRow>> {
    type PartKey = (FixedKind, FixedKind);
    let fits = |part: FixedKind, row: FixedKind| {
        part == FixedKind::Null || row == FixedKind::Null || part == row
    };
    let mut parts: Vec<(PartKey, Vec<UpsertRow>)> = Vec::new();
    for row in rows {
        let key = (fixed_kind(&row.0), fixed_kind(&row.1));
        let slot = parts
            .iter_mut()
            .find(|(k, _)| fits(k.0, key.0) && fits(k.1, key.1));
        match slot {
            Some((k, part)) => {
                // The part's shape is whichever half first showed one.
                if k.0 == FixedKind::Null {
                    k.0 = key.0;
                }
                if k.1 == FixedKind::Null {
                    k.1 = key.1;
                }
                part.push(row.clone());
            }
            None => parts.push((key, vec![row.clone()])),
        }
    }
    parts.into_iter().map(|(_, part)| part).collect()
}

/// The `DataFrame` column shapes that store a `Value` without changing it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameShape {
    UniqueId,
    Int64,
    Float64,
    String,
    Boolean,
    DateTime,
    Timestamp,
    List,
    Map,
}

/// The shape a lone `value` would round-trip through, or `None` when the
/// frame has no exact column for it (`Point`, `Duration`, the query-time
/// graph-entity variants) and would render it as text.
fn frame_shape(value: &Value) -> Option<FrameShape> {
    Some(match value {
        Value::UniqueId(_) => FrameShape::UniqueId,
        Value::Int64(_) => FrameShape::Int64,
        Value::Float64(_) => FrameShape::Float64,
        Value::String(_) => FrameShape::String,
        Value::Boolean(_) => FrameShape::Boolean,
        Value::DateTime(_) => FrameShape::DateTime,
        Value::Timestamp(_) => FrameShape::Timestamp,
        Value::List(_) => FrameShape::List,
        Value::Map(_) => FrameShape::Map,
        _ => return None,
    })
}

/// The type name to declare for a held-back column: the values' own type
/// when they agree, `"mixed"` when they don't — the string the columnar
/// store and the streaming writer already use for a heterogeneous column.
fn declared_type_name<'a>(values: impl Iterator<Item = &'a Value>) -> String {
    let mut seen: Option<&'static str> = None;
    for value in values {
        if matches!(value, Value::Null) {
            continue;
        }
        match seen {
            None => seen = Some(value.type_name()),
            Some(name) if name == value.type_name() => {}
            Some(_) => return "mixed".to_string(),
        }
    }
    seen.unwrap_or("mixed").to_string()
}

/// Phase 4 — edge removes by logical identity. The one thing the
/// `maintain::*` helpers don't expose, so it reaches the storage layer.
fn apply_edge_removes(graph: &mut DirGraph, edges: &HashMap<EdgeKey, EdgeNet>) {
    let mut removed_edges = 0usize;
    for ((conn, src_type, src_id, tgt_type, tgt_id), net) in edges {
        if !matches!(net, EdgeNet::Remove) {
            continue;
        }
        let (Some(src), Some(tgt)) = (
            graph.lookup_by_id(src_type, src_id),
            graph.lookup_by_id(tgt_type, tgt_id),
        ) else {
            continue;
        };
        let conn_key = InternedKey::from_str(conn);
        let eidx = graph
            .graph
            .edges_connecting(src, tgt)
            .find(|er| er.weight().connection_type == conn_key)
            .map(|er| er.id());
        if let Some(eidx) = eidx {
            GraphWrite::remove_edge(&mut graph.graph, eidx);
            removed_edges += 1;
        }
    }
    if removed_edges > 0 {
        graph.invalidate_edge_type_counts_cache();
        graph.connection_types.clear();
    }
}

/// Phase 5 — node removes (detach incident edges + index cleanup). Last,
/// so every earlier phase could still resolve identities it needed.
fn apply_node_removes(graph: &mut DirGraph, nodes: &HashMap<NodeKey, NodeNet>) {
    let mut to_delete: HashSet<NodeIndex> = HashSet::new();
    for ((node_type, id), net) in nodes {
        if matches!(net, NodeNet::Remove) {
            if let Some(idx) = graph.lookup_by_id(node_type, id) {
                to_delete.insert(idx);
            }
        }
    }
    if !to_delete.is_empty() {
        detach_delete_nodes(graph, &to_delete);
    }
}

/// Accumulator for one node_type's upsert rows.
#[derive(Default)]
struct NodeRows {
    columns: Vec<String>,
    seen: std::collections::HashSet<String>,
    rows: Vec<UpsertRow>,
}

/// Accumulator for one (conn, src_type, tgt_type)'s upsert rows.
#[derive(Default)]
struct EdgeRows {
    columns: Vec<String>,
    seen: std::collections::HashSet<String>,
    rows: Vec<UpsertRow>,
}

impl NodeRows {
    fn note_column(&mut self, name: &str) {
        if self.seen.insert(name.to_string()) {
            self.columns.push(name.to_string());
        }
    }
}
impl EdgeRows {
    fn note_column(&mut self, name: &str) {
        if self.seen.insert(name.to_string()) {
            self.columns.push(name.to_string());
        }
    }
}

/// Build a `DataFrame` with `[fixed... , props...]` columns. The two
/// leading fixed cells (id/title or src_id/tgt_id) ride in the row tuple;
/// absent property cells are filled `Null` (skip-on-null in add_nodes).
fn build_dataframe(
    fixed: &[&str],
    prop_columns: &[String],
    rows: &[UpsertRow],
) -> Result<DataFrame, String> {
    let mut columns: Vec<String> = fixed.iter().map(|s| s.to_string()).collect();
    columns.extend(prop_columns.iter().cloned());

    let out_rows: Vec<Vec<Value>> = rows
        .iter()
        .map(|(a, b, props)| {
            let mut row = Vec::with_capacity(columns.len());
            row.push(a.clone());
            row.push(b.clone());
            for col in prop_columns {
                row.push(props.get(col).cloned().unwrap_or(Value::Null));
            }
            row
        })
        .collect();

    DataFrame::from_cypher_rows(columns, out_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::storage::GraphRead;

    fn frame(lsn: u64, ops: Vec<MutationOp>) -> WalFrame {
        WalFrame { lsn, ops }
    }

    fn upsert_node(id: i64, title: &str, props: Vec<(&str, Value)>) -> MutationOp {
        MutationOp::UpsertNode {
            node_type: "Person".into(),
            id: Value::Int64(id),
            title: Value::String(title.into()),
            properties: props.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        }
    }

    fn knows(src: i64, tgt: i64) -> MutationOp {
        MutationOp::UpsertEdge {
            conn_type: "KNOWS".into(),
            src_type: "Person".into(),
            src_id: Value::Int64(src),
            tgt_type: "Person".into(),
            tgt_id: Value::Int64(tgt),
            properties: vec![],
        }
    }

    fn prop(g: &mut DirGraph, id: i64, key: &str) -> Option<Value> {
        let idx = g.lookup_by_id("Person", &Value::Int64(id))?;
        g.graph
            .node_view(idx)
            .and_then(|n| n.get_field_ref(key).map(|c| c.into_owned()))
    }

    #[test]
    fn replays_upserts_and_edge() {
        let mut g = DirGraph::new();
        let frames = vec![frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
                upsert_node(2, "Bob", vec![]),
                knows(1, 2),
            ],
        )];
        let max = apply_frames(&mut g, &frames, 0).unwrap();
        assert_eq!(max, 1);
        assert_eq!(g.graph.node_count(), 2);
        assert_eq!(g.graph.edge_count(), 1);
        assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(30)));
    }

    #[test]
    fn later_upsert_replaces_properties() {
        let mut g = DirGraph::new();
        let frames = vec![
            frame(
                1,
                vec![upsert_node(1, "Alice", vec![("age", Value::Int64(30))])],
            ),
            frame(
                2,
                vec![upsert_node(1, "Alice", vec![("age", Value::Int64(41))])],
            ),
        ];
        apply_frames(&mut g, &frames, 0).unwrap();
        assert_eq!(
            g.graph.node_count(),
            1,
            "same (type,id) is upserted, not duplicated"
        );
        assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(41)));
    }

    #[test]
    fn remove_node_deletes_it_and_its_edges() {
        let mut g = DirGraph::new();
        let frames = vec![
            frame(
                1,
                vec![
                    upsert_node(1, "Alice", vec![]),
                    upsert_node(2, "Bob", vec![]),
                    knows(1, 2),
                ],
            ),
            frame(
                2,
                vec![MutationOp::RemoveNode {
                    node_type: "Person".into(),
                    id: Value::Int64(2),
                }],
            ),
        ];
        apply_frames(&mut g, &frames, 0).unwrap();
        assert_eq!(g.graph.node_count(), 1);
        assert_eq!(
            g.graph.edge_count(),
            0,
            "incident edge removed with the node"
        );
        assert!(g.lookup_by_id("Person", &Value::Int64(2)).is_none());
    }

    /// Recovery replays a node removal through `detach_delete_nodes`, so the
    /// embedding prune rides along: a `.kgl` saved before the delete plus a
    /// WAL carrying it must not reload a graph whose store still holds the
    /// removed node's vector — the freed index is handed to the next node
    /// created and would inherit it.
    #[test]
    fn replayed_node_removal_prunes_the_embedding_store() {
        let mut g = DirGraph::new();
        apply_frames(
            &mut g,
            &[frame(
                1,
                vec![
                    upsert_node(1, "Alice", vec![]),
                    upsert_node(2, "Bob", vec![]),
                ],
            )],
            0,
        )
        .unwrap();
        let report = crate::graph::embeddings::set_embeddings(
            &mut g,
            "Person",
            "name",
            None,
            [
                (Value::Int64(1), vec![1.0f32, 0.0]),
                (Value::Int64(2), vec![0.0, 1.0]),
            ],
        )
        .expect("seed embeddings");
        assert_eq!(report.embeddings_stored, 2);
        let doomed = g
            .lookup_by_id("Person", &Value::Int64(2))
            .expect("Bob is present");

        apply_frames(
            &mut g,
            &[frame(
                2,
                vec![MutationOp::RemoveNode {
                    node_type: "Person".into(),
                    id: Value::Int64(2),
                }],
            )],
            1,
        )
        .unwrap();

        let store = g
            .embeddings
            .get(&("Person".to_string(), "name_emb".to_string()))
            .expect("store");
        assert_eq!(store.len(), 1);
        assert_eq!(store.get_embedding(doomed.index()), None);
        assert_eq!(store.validate_shape(), Ok(()));
    }

    #[test]
    fn remove_edge_keeps_endpoints() {
        let mut g = DirGraph::new();
        let frames = vec![
            frame(
                1,
                vec![
                    upsert_node(1, "Alice", vec![]),
                    upsert_node(2, "Bob", vec![]),
                    knows(1, 2),
                ],
            ),
            frame(
                2,
                vec![MutationOp::RemoveEdge {
                    conn_type: "KNOWS".into(),
                    src_type: "Person".into(),
                    src_id: Value::Int64(1),
                    tgt_type: "Person".into(),
                    tgt_id: Value::Int64(2),
                }],
            ),
        ];
        apply_frames(&mut g, &frames, 0).unwrap();
        assert_eq!(g.graph.node_count(), 2, "endpoints survive an edge remove");
        assert_eq!(g.graph.edge_count(), 0);
    }

    #[test]
    fn frames_at_or_below_checkpoint_are_skipped() {
        let mut g = DirGraph::new();
        let frames = vec![
            frame(1, vec![upsert_node(1, "Old", vec![])]),
            frame(2, vec![upsert_node(2, "New", vec![])]),
        ];
        // Checkpoint already folded in lsn 1; only replay lsn 2.
        let max = apply_frames(&mut g, &frames, 1).unwrap();
        assert_eq!(max, 2);
        assert!(g.lookup_by_id("Person", &Value::Int64(1)).is_none());
        assert!(g.lookup_by_id("Person", &Value::Int64(2)).is_some());
    }

    /// Secondary labels a node carries in `labels(n)` order. The exact
    /// list, not a set: `DirGraph::node_labels` promises primary-first then
    /// name-sorted, and replay must not degrade that to arbitrary order.
    fn labels_of(g: &mut DirGraph, id: i64) -> Vec<String> {
        let idx = g
            .lookup_by_id("Person", &Value::Int64(id))
            .expect("node must exist");
        g.node_labels(idx)
            .into_iter()
            .map(|k| g.interner.resolve(k).to_string())
            .collect()
    }

    fn set_labels(id: i64, labels: &[&str]) -> MutationOp {
        MutationOp::SetNodeLabels {
            node_type: "Person".into(),
            id: Value::Int64(id),
            labels: labels.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// The regression this op exists for: before `SetNodeLabels`, a node's
    /// properties survived replay and its secondary labels silently did
    /// not.
    #[test]
    fn replay_restores_secondary_labels_in_exact_order() {
        let mut g = DirGraph::new();
        let frames = vec![frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
                // Logged unsorted on purpose: ordering is replay's job.
                set_labels(1, &["Manager", "Employee"]),
            ],
        )];
        apply_frames(&mut g, &frames, 0).unwrap();

        assert_eq!(
            labels_of(&mut g, 1),
            vec!["Person", "Employee", "Manager"],
            "primary first, then secondaries sorted by name"
        );
        assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(30)));
        assert!(g.has_secondary_labels, "fast-skip flag must be set");
        // The label index is the candidate source for `MATCH (n:Employee)`.
        assert_eq!(g.nodes_with_label("Employee").len(), 1);
    }

    /// A whole-set op reconciles: labels present in the checkpoint but
    /// absent from the log are removed, which is what makes `REMOVE
    /// n:Label` recoverable.
    #[test]
    fn replay_removes_labels_the_log_dropped() {
        let mut g = DirGraph::new();
        apply_frames(
            &mut g,
            &[frame(
                1,
                vec![upsert_node(1, "Alice", vec![]), set_labels(1, &["A", "B"])],
            )],
            0,
        )
        .unwrap();
        assert_eq!(labels_of(&mut g, 1), vec!["Person", "A", "B"]);

        // A later frame carries only "B" — "A" was removed in the session.
        apply_frames(&mut g, &[frame(2, vec![set_labels(1, &["B"])])], 1).unwrap();
        assert_eq!(labels_of(&mut g, 1), vec!["Person", "B"]);
        assert!(
            g.nodes_with_label("A").is_empty(),
            "the dropped label must leave no index residue"
        );
    }

    /// Emptying the set clears the fast-skip flag, so a graph whose last
    /// label was removed pays no secondary-label scan cost after recovery.
    #[test]
    fn replay_to_an_empty_label_set_clears_the_flag() {
        let mut g = DirGraph::new();
        apply_frames(
            &mut g,
            &[
                frame(
                    1,
                    vec![upsert_node(1, "Alice", vec![]), set_labels(1, &["A"])],
                ),
                frame(2, vec![set_labels(1, &[])]),
            ],
            0,
        )
        .unwrap();
        assert_eq!(labels_of(&mut g, 1), vec!["Person"]);
        assert!(!g.has_secondary_labels);
    }

    /// Labels and properties are independent state: an `UpsertNode` logged
    /// after a label set (a later `SET n.age = …`) must not wipe the
    /// labels, in either fold order.
    #[test]
    fn property_upsert_does_not_clobber_labels() {
        for reversed in [false, true] {
            let mut ops = vec![
                upsert_node(1, "Alice", vec![]),
                set_labels(1, &["Employee"]),
                upsert_node(1, "Alice", vec![("age", Value::Int64(41))]),
            ];
            if reversed {
                ops.swap(1, 2);
            }
            let mut g = DirGraph::new();
            apply_frames(&mut g, &[frame(1, ops)], 0).unwrap();
            assert_eq!(
                labels_of(&mut g, 1),
                vec!["Person", "Employee"],
                "{reversed}"
            );
            assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(41)), "{reversed}");
        }
    }

    /// A node deleted later in the log must not be resurrected by its own
    /// label op.
    #[test]
    fn label_set_for_a_removed_node_is_skipped() {
        let mut g = DirGraph::new();
        let frames = vec![frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                set_labels(1, &["Employee"]),
                MutationOp::RemoveNode {
                    node_type: "Person".into(),
                    id: Value::Int64(1),
                },
            ],
        )];
        apply_frames(&mut g, &frames, 0).unwrap();
        assert_eq!(g.graph.node_count(), 0);
        assert!(g.nodes_with_label("Employee").is_empty());
    }

    #[test]
    fn replaying_labels_twice_is_idempotent() {
        let frames = vec![frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                set_labels(1, &["Employee", "Manager"]),
            ],
        )];
        let mut g = DirGraph::new();
        apply_frames(&mut g, &frames, 0).unwrap();
        apply_frames(&mut g, &frames, 0).unwrap();
        assert_eq!(labels_of(&mut g, 1), vec!["Person", "Employee", "Manager"]);
        assert_eq!(
            g.nodes_with_label("Employee").len(),
            1,
            "no duplicate bucket entry"
        );
    }

    /// Replay must work on a `mapped` graph, not only the heap default.
    /// Asserted here rather than from Python because the storage mode is not
    /// observable through the Python surface — a silent downgrade to memory
    /// would make an end-to-end mapped test pass vacuously.
    ///
    /// It works for a structural reason worth pinning: `MappedGraph` mutates
    /// the same petgraph `StableDiGraph` as `MemoryGraph` and differs only in
    /// its derived mmap-backed indexes, so `apply_frames`' `maintain::*` calls
    /// reach it unchanged.
    #[test]
    fn replays_onto_a_mapped_graph() {
        use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
        let mut g = new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap();
        assert!(g.graph.is_mapped(), "fixture must really be mapped");

        let frames = vec![
            frame(
                1,
                vec![
                    upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
                    upsert_node(2, "Bob", vec![]),
                    knows(1, 2),
                    set_labels(1, &["Employee"]),
                ],
            ),
            frame(
                2,
                vec![MutationOp::RemoveNode {
                    node_type: "Person".into(),
                    id: Value::Int64(2),
                }],
            ),
        ];
        apply_frames(&mut g, &frames, 0).unwrap();

        assert!(g.graph.is_mapped(), "replay must not switch the backend");
        assert_eq!(g.graph.node_count(), 1);
        assert_eq!(g.graph.edge_count(), 0, "edge went with the removed node");
        assert_eq!(labels_of(&mut g, 1), vec!["Person", "Employee"]);
        assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(30)));
    }

    /// A property whose values differ in type across nodes must replay with
    /// every value's type intact. Folding routes a whole node_type's rows
    /// through one `DataFrame`, whose columns are singly-typed, so a mixed
    /// column used to resolve to `String` (or `Float64` for an int/float
    /// mix) and rewrite every cell in it.
    #[test]
    fn mixed_typed_property_keeps_every_value_type() {
        use chrono::NaiveDate;
        let date = NaiveDate::from_ymd_opt(2020, 1, 2).unwrap();
        let cases: Vec<(i64, Value)> = vec![
            (1, Value::Int64(1)),
            (2, Value::String("two".into())),
            (3, Value::Float64(3.5)),
            (4, Value::Boolean(true)),
            (5, Value::DateTime(date)),
        ];
        let mut g = DirGraph::new();
        let frames: Vec<WalFrame> = cases
            .iter()
            .enumerate()
            .map(|(i, (id, v))| {
                frame(
                    i as u64 + 1,
                    vec![upsert_node(*id, "n", vec![("mixedish", v.clone())])],
                )
            })
            .collect();
        apply_frames(&mut g, &frames, 0).unwrap();
        for (id, expected) in &cases {
            assert_eq!(
                prop(&mut g, *id, "mixedish").as_ref(),
                Some(expected),
                "node {id}"
            );
        }
    }

    /// The framed half of the split: a singly-typed property still rides the
    /// bulk `DataFrame`, and every shape it carries round-trips exactly. Without
    /// this the mixed-type tests above could pass vacuously by routing
    /// everything down the per-value path.
    #[test]
    fn single_typed_properties_keep_their_types_through_the_frame() {
        use chrono::NaiveDate;
        let props = vec![
            ("i", Value::Int64(7)),
            ("f", Value::Float64(0.5)),
            ("s", Value::String("x".into())),
            ("b", Value::Boolean(true)),
            (
                "d",
                Value::DateTime(NaiveDate::from_ymd_opt(2020, 1, 2).unwrap()),
            ),
            ("l", Value::List(vec![Value::Int64(1), Value::Int64(2)])),
        ];
        let mut g = DirGraph::new();
        apply_frames(
            &mut g,
            &[frame(1, vec![upsert_node(1, "a", props.clone())])],
            0,
        )
        .unwrap();
        for (key, expected) in props {
            assert_eq!(prop(&mut g, 1, key).as_ref(), Some(&expected), "{key}");
        }
        // …and each one is a frame column, not a per-value write: the type's
        // metadata carries the DataFrame's name for it, never "mixed".
        let meta = g.get_node_type_metadata("Person").cloned().unwrap();
        assert!(
            !meta.values().any(|t| t == "mixed"),
            "single-typed columns must stay framed: {meta:?}"
        );
    }

    /// The narrower numeric case: an int and a float under one property must
    /// not promote the int to a float.
    #[test]
    fn int_and_float_under_one_property_do_not_promote() {
        let mut g = DirGraph::new();
        apply_frames(
            &mut g,
            &[frame(
                1,
                vec![
                    upsert_node(1, "a", vec![("n", Value::Int64(2))]),
                    upsert_node(2, "b", vec![("n", Value::Float64(2.5))]),
                ],
            )],
            0,
        )
        .unwrap();
        assert_eq!(prop(&mut g, 1, "n"), Some(Value::Int64(2)));
        assert_eq!(prop(&mut g, 2, "n"), Some(Value::Float64(2.5)));
    }

    /// Same defect class, single-typed: a `Point` has no columnar shape, so
    /// the frame column falls back to `String` and the value replays as WKT
    /// text.
    #[test]
    fn point_property_survives_replay_as_a_point() {
        let mut g = DirGraph::new();
        apply_frames(
            &mut g,
            &[frame(
                1,
                vec![upsert_node(
                    1,
                    "a",
                    vec![(
                        "loc",
                        Value::Point {
                            lat: 59.9,
                            lon: 10.7,
                        },
                    )],
                )],
            )],
            0,
        )
        .unwrap();
        assert_eq!(
            prop(&mut g, 1, "loc"),
            Some(Value::Point {
                lat: 59.9,
                lon: 10.7
            })
        );
    }

    /// A mixed property folded together with a later op on the same node:
    /// the faithless column is applied after the bulk upsert, so the last
    /// write must still win and the rest of the row must survive.
    #[test]
    fn mixed_property_folds_with_later_ops_on_the_same_node() {
        let mut g = DirGraph::new();
        apply_frames(
            &mut g,
            &[
                frame(
                    1,
                    vec![
                        upsert_node(1, "a", vec![("m", Value::Int64(1))]),
                        upsert_node(2, "b", vec![("m", Value::String("two".into()))]),
                    ],
                ),
                frame(
                    2,
                    vec![upsert_node(
                        1,
                        "a",
                        vec![("m", Value::Boolean(false)), ("age", Value::Int64(41))],
                    )],
                ),
            ],
            0,
        )
        .unwrap();
        assert_eq!(prop(&mut g, 1, "m"), Some(Value::Boolean(false)));
        assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(41)));
        assert_eq!(prop(&mut g, 2, "m"), Some(Value::String("two".into())));
        assert_eq!(
            g.get_node_type_metadata("Person").unwrap().get("m"),
            Some(&"mixed".to_string()),
            "a heterogeneous property is declared 'mixed', not left undeclared"
        );
    }

    /// Identity is a value too. Two nodes of one type whose ids differ in
    /// type share the frame's `id` column, which cannot be held back the way
    /// a property can — the rows are split into one bulk call each instead.
    #[test]
    fn nodes_whose_ids_differ_in_type_keep_their_ids() {
        let mut g = DirGraph::new();
        let string_id = MutationOp::UpsertNode {
            node_type: "Person".into(),
            id: Value::String("x".into()),
            title: Value::String("b".into()),
            properties: vec![("tag".to_string(), Value::String("str-id".into()))],
        };
        apply_frames(
            &mut g,
            &[frame(
                1,
                vec![
                    upsert_node(1, "a", vec![("tag", Value::String("int-id".into()))]),
                    string_id,
                ],
            )],
            0,
        )
        .unwrap();
        assert_eq!(g.graph.node_count(), 2);
        let idx = g
            .lookup_by_id("Person", &Value::Int64(1))
            .expect("the integer id must still be an integer");
        assert_eq!(g.graph.get_node_id(idx), Some(Value::Int64(1)));
        let idx = g
            .lookup_by_id("Person", &Value::String("x".into()))
            .expect("the string id must survive alongside it");
        assert_eq!(g.graph.get_node_id(idx), Some(Value::String("x".into())));
    }

    /// The title column has the same shape as the id column and the same
    /// exposure.
    #[test]
    fn nodes_whose_titles_differ_in_type_keep_their_titles() {
        let mut g = DirGraph::new();
        let numeric_title = MutationOp::UpsertNode {
            node_type: "Person".into(),
            id: Value::Int64(2),
            title: Value::Int64(5),
            properties: vec![],
        };
        apply_frames(
            &mut g,
            &[frame(1, vec![upsert_node(1, "a", vec![]), numeric_title])],
            0,
        )
        .unwrap();
        let title = |g: &mut DirGraph, id: i64| {
            let idx = g.lookup_by_id("Person", &Value::Int64(id)).unwrap();
            g.graph.get_node_title(idx)
        };
        assert_eq!(title(&mut g, 1), Some(Value::String("a".into())));
        assert_eq!(title(&mut g, 2), Some(Value::Int64(5)));
    }

    /// An edge's endpoints are addressed by those same ids, so a mixed-id
    /// node type must not cost the edges that reach it.
    #[test]
    fn edges_reach_endpoints_whose_ids_differ_in_type() {
        let mut g = DirGraph::new();
        let string_node = MutationOp::UpsertNode {
            node_type: "Person".into(),
            id: Value::String("x".into()),
            title: Value::String("b".into()),
            properties: vec![],
        };
        let edge = MutationOp::UpsertEdge {
            conn_type: "KNOWS".into(),
            src_type: "Person".into(),
            src_id: Value::Int64(1),
            tgt_type: "Person".into(),
            tgt_id: Value::String("x".into()),
            properties: vec![],
        };
        apply_frames(
            &mut g,
            &[frame(
                1,
                vec![upsert_node(1, "a", vec![]), string_node, knows(1, 2), edge],
            )],
            0,
        )
        .unwrap();
        // Nodes: 1, "x", and the id-2 stub `knows(1, 2)` vivifies — three,
        // not four. A `tgt_id` column holding both `2` and `"x"` renders the
        // integer endpoint as `"2"`, which matches nothing and vivifies a
        // *second* stub under a string id.
        assert_eq!(g.graph.node_count(), 3, "no stub under a stringified id");
        assert_eq!(g.graph.edge_count(), 2, "both edges land");
        let src = g.lookup_by_id("Person", &Value::Int64(1)).unwrap();
        for tgt_id in [Value::Int64(2), Value::String("x".into())] {
            let tgt = g
                .lookup_by_id("Person", &tgt_id)
                .unwrap_or_else(|| panic!("endpoint {tgt_id:?} must exist"));
            assert!(
                g.graph.find_edge(src, tgt).is_some(),
                "the edge to {tgt_id:?} must connect that node"
            );
        }
    }

    /// Same on a mapped graph — the per-value path must reach the mmap-backed
    /// backend as the bulk one does.
    #[test]
    fn mixed_typed_property_keeps_its_types_on_a_mapped_graph() {
        use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
        let mut g = new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap();
        assert!(g.graph.is_mapped(), "fixture must really be mapped");
        apply_frames(
            &mut g,
            &[frame(
                1,
                vec![
                    upsert_node(1, "a", vec![("m", Value::Int64(1))]),
                    upsert_node(2, "b", vec![("m", Value::String("two".into()))]),
                ],
            )],
            0,
        )
        .unwrap();
        assert!(g.graph.is_mapped(), "replay must not switch the backend");
        assert_eq!(prop(&mut g, 1, "m"), Some(Value::Int64(1)));
        assert_eq!(prop(&mut g, 2, "m"), Some(Value::String("two".into())));
    }

    /// Edge properties fold through the same `DataFrame` builder.
    #[test]
    fn mixed_typed_edge_property_keeps_every_value_type() {
        let mut g = DirGraph::new();
        let knows_with = |src: i64, tgt: i64, v: Value| MutationOp::UpsertEdge {
            conn_type: "KNOWS".into(),
            src_type: "Person".into(),
            src_id: Value::Int64(src),
            tgt_type: "Person".into(),
            tgt_id: Value::Int64(tgt),
            properties: vec![("w".to_string(), v)],
        };
        apply_frames(
            &mut g,
            &[frame(
                1,
                vec![
                    upsert_node(1, "a", vec![]),
                    upsert_node(2, "b", vec![]),
                    upsert_node(3, "c", vec![]),
                    knows_with(1, 2, Value::Int64(7)),
                    knows_with(1, 3, Value::String("heavy".into())),
                ],
            )],
            0,
        )
        .unwrap();
        let w = |g: &mut DirGraph, src: i64, tgt: i64| -> Option<Value> {
            let s = g.lookup_by_id("Person", &Value::Int64(src))?;
            let t = g.lookup_by_id("Person", &Value::Int64(tgt))?;
            let e = g.graph.find_edge(s, t)?;
            g.graph
                .edge_weight(e)?
                .properties
                .iter()
                .find(|(k, _)| *k == InternedKey::from_str("w"))
                .map(|(_, v)| v.clone())
        };
        assert_eq!(w(&mut g, 1, 2), Some(Value::Int64(7)));
        assert_eq!(w(&mut g, 1, 3), Some(Value::String("heavy".into())));
    }

    #[test]
    fn replaying_twice_is_idempotent() {
        let frames = vec![frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
                upsert_node(2, "Bob", vec![]),
                knows(1, 2),
            ],
        )];
        let mut g = DirGraph::new();
        apply_frames(&mut g, &frames, 0).unwrap();
        apply_frames(&mut g, &frames, 0).unwrap(); // replay again
        assert_eq!(g.graph.node_count(), 2, "idempotent — no duplicate nodes");
        assert_eq!(g.graph.edge_count(), 1, "idempotent — no duplicate edge");
    }
}

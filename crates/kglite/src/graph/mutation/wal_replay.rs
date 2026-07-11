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
//! Apply order — node upserts → edge upserts → edge removes → node removes —
//! respects referential integrity (endpoints exist before their edges; a
//! removed node's edges go with it via detach). An edge whose endpoint is
//! net-removed is dropped from the edge-upsert batch (its node-remove will
//! detach it anyway).

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
            }
        }
    }

    if any {
        apply_net(graph, nodes, edges)?;
    }
    Ok(max_lsn)
}

/// Apply folded net state in bulk. See the module docs for the ordering
/// rationale.
fn apply_net(
    graph: &mut DirGraph,
    nodes: HashMap<NodeKey, NodeNet>,
    edges: HashMap<EdgeKey, EdgeNet>,
) -> Result<(), String> {
    // Node identities scheduled for removal — used to drop edge upserts whose
    // endpoint won't exist.
    let removed_nodes: HashSet<NodeKey> = nodes
        .iter()
        .filter(|(_, v)| matches!(v, NodeNet::Remove))
        .map(|(k, _)| k.clone())
        .collect();

    // ── Phase 1: node upserts, grouped by node_type, one add_nodes each ──
    let mut node_groups: HashMap<&str, NodeRows> = HashMap::new();
    for ((node_type, id), net) in &nodes {
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
        let df = build_dataframe(&["id", "title"], &group.columns, &group.rows)?;
        add_nodes(
            graph,
            df,
            node_type.to_string(),
            "id".to_string(),
            Some("title".to_string()),
            Some("replace".to_string()),
        )?;
    }

    // ── Phase 2: edge upserts, grouped by (conn, src_type, tgt_type) ──
    let mut edge_groups: HashMap<(&str, &str, &str), EdgeRows> = HashMap::new();
    for ((conn, src_type, src_id, tgt_type, tgt_id), net) in &edges {
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
        let df = build_dataframe(&["src_id", "tgt_id"], &group.columns, &group.rows)?;
        add_connections(
            graph,
            df,
            conn.to_string(),
            src_type.to_string(),
            "src_id".to_string(),
            tgt_type.to_string(),
            "tgt_id".to_string(),
            None,
            None,
            Some("replace".to_string()),
        )?;
    }

    // ── Phase 3: edge removes ─────────────────────────────────────────
    let mut removed_edges = 0usize;
    for ((conn, src_type, src_id, tgt_type, tgt_id), net) in &edges {
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

    // ── Phase 4: node removes (detach incident edges + index cleanup) ─
    let mut to_delete: HashSet<NodeIndex> = HashSet::new();
    for ((node_type, id), net) in &nodes {
        if matches!(net, NodeNet::Remove) {
            if let Some(idx) = graph.lookup_by_id(node_type, id) {
                to_delete.insert(idx);
            }
        }
    }
    if !to_delete.is_empty() {
        detach_delete_nodes(graph, &to_delete);
    }

    Ok(())
}

/// Accumulator for one node_type's upsert rows.
#[derive(Default)]
struct NodeRows {
    columns: Vec<String>,
    seen: std::collections::HashSet<String>,
    rows: Vec<(Value, Value, HashMap<String, Value>)>,
}

/// Accumulator for one (conn, src_type, tgt_type)'s upsert rows.
#[derive(Default)]
struct EdgeRows {
    columns: Vec<String>,
    seen: std::collections::HashSet<String>,
    rows: Vec<(Value, Value, HashMap<String, Value>)>,
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
    rows: &[(Value, Value, HashMap<String, Value>)],
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
            .node_weight(idx)
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

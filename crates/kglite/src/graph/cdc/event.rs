//! CDC event vocabulary and its derivation from the write-capture buffer.
//!
//! An event describes **one entity's net change in one commit**: its logical
//! identity, whether the commit created, updated or deleted it, and — for the
//! first two — the entity's state *after* the commit. Before-images are
//! deliberately out of scope for v1 (a documented divergence from Neo4j's
//! `{before, after}` shape); nothing in this module keeps a pre-image.
//!
//! ## Why derive rather than reuse `MutationOp`
//!
//! The WAL's [`MutationOp`](crate::graph::wal::MutationOp) is add-or-replace
//! by design — that is what makes replay idempotent — so it cannot say whether
//! a write *created* an entity. The capture buffer can:
//! [`CaptureOrigin`] records which `GraphWrite` method ran. Deriving events
//! from the raw buffer therefore keeps create/update fidelity without touching
//! the on-disk WAL format (`WAL_FORMAT_VERSION` is unmoved by this module).

use crate::datatypes::Value;
use crate::graph::schema::StringInterner;
use crate::graph::storage::recording::{CaptureOrigin, RawOp};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

/// What happened to an entity in one commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdcEventKind {
    /// The entity did not exist before the commit.
    Create,
    /// An existing entity's properties, title or labels changed.
    Update,
    /// The entity no longer exists. Carries identity only — v1 keeps no
    /// before-image, so there is no final property set to report.
    Delete,
}

impl CdcEventKind {
    /// Stable lowercase wire name (`"create"` / `"update"` / `"delete"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            CdcEventKind::Create => "create",
            CdcEventKind::Update => "update",
            CdcEventKind::Delete => "delete",
        }
    }
}

/// A node's state after the commit.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeState {
    pub title: Value,
    /// Secondary labels only, ordered as `DirGraph::node_labels` orders them.
    /// The primary type is the event's `node_type` and is never listed here.
    pub labels: Vec<String>,
    pub properties: Vec<(String, Value)>,
}

/// An edge's state after the commit.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeState {
    pub properties: Vec<(String, Value)>,
}

/// The entity an event is about, by logical identity — never by
/// `NodeIndex`/`EdgeIndex`, which are storage addresses that a consumer
/// outside the process cannot resolve and that a reload does not preserve.
///
/// `after` is `Some` for creates and updates, `None` for deletes.
#[derive(Debug, Clone, PartialEq)]
pub enum CdcChange {
    Node {
        node_type: String,
        id: Value,
        after: Option<NodeState>,
    },
    Edge {
        conn_type: String,
        src_type: String,
        src_id: Value,
        tgt_type: String,
        tgt_id: Value,
        after: Option<EdgeState>,
    },
}

impl CdcChange {
    /// `"node"` or `"edge"` — the element discriminator a consumer routes on.
    pub fn element(&self) -> &'static str {
        match self {
            CdcChange::Node { .. } => "node",
            CdcChange::Edge { .. } => "edge",
        }
    }
}

/// One published change, with the sequence number a cursor addresses it by.
///
/// `seq` is assigned by [`CdcLog`](super::CdcLog) at append time and is
/// monotonic within an epoch; it is *not* meaningful across epochs (see
/// [`super::CdcLog::epoch`]).
#[derive(Debug, Clone, PartialEq)]
pub struct CdcEvent {
    pub seq: u64,
    pub kind: CdcEventKind,
    pub change: CdcChange,
}

/// An event before the log assigns it a sequence number.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingEvent {
    pub kind: CdcEventKind,
    pub change: CdcChange,
}

/// Collapse key for upserts: the storage address, which is stable for the
/// lifetime of one capture batch and cheap to hash. Logical identity is *not*
/// usable as a key — `Value` is `PartialEq` only, deliberately (it carries
/// floats).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SlotKey {
    Node(u32),
    Edge(u32),
}

/// A staged entry, before final-state resolution.
enum Entry {
    /// An upsert that a later op in the same batch superseded: the entry moved
    /// to the end of the order (see the ordering rule in [`events_from_raw`]).
    Vacated,
    Upsert {
        key: SlotKey,
        kind: CdcEventKind,
    },
    RemoveNode {
        node_type: String,
        id: Value,
    },
    RemoveEdge {
        conn_type: String,
        src_type: String,
        src_id: Value,
        tgt_type: String,
        tgt_id: Value,
    },
}

/// Derive the CDC events for one commit from its drained capture buffer.
///
/// `graph`/`interner`/`secondary_labels` are read exactly as
/// [`resolve_ops`](crate::graph::storage::recording::resolve_ops) reads them —
/// **final** post-commit state — so the two views of a commit cannot disagree
/// about what it did.
///
/// ## Collapse and ordering, pinned
///
/// - **One event per entity per commit.** Repeated writes to the same entity
///   in one commit (`CREATE (n) SET n.x = 1`, or a batch that touches a row
///   twice) collapse into a single event carrying the final state. The kind is
///   `Create` if *any* of the collapsed ops was a create, else `Update`.
/// - **Deletes never collapse into an upsert.** A remove is its own event, so
///   a delete-then-recreate of the same logical identity — what
///   `replace_connections` does for every edge of a reloaded relationship —
///   publishes a `delete` followed by a `create`, which is the truthful signal
///   for a consumer that mirrors state.
/// - **Order is by each entity's *last* capture in the batch.** Anything else
///   can invert a delete and the recreate that follows it when petgraph reuses
///   a freed index, publishing "created then deleted" for an entity that
///   exists.
/// - An upsert for an entity the same commit later removed resolves to nothing
///   and is dropped; the remove already carries the outcome.
pub(super) fn events_from_raw(
    raw: &[RawOp],
    graph: &impl GraphRead,
    interner: &StringInterner,
    secondary_labels: impl Fn(NodeIndex) -> Vec<String>,
) -> Vec<PendingEvent> {
    let mut entries: Vec<Entry> = Vec::with_capacity(raw.len());
    let mut slots: HashMap<SlotKey, usize> = HashMap::new();

    for op in raw {
        match op {
            RawOp::UpsertNode(idx, origin) => stage_upsert(
                &mut entries,
                &mut slots,
                SlotKey::Node(idx.index() as u32),
                *origin,
            ),
            RawOp::SetNodeLabels(idx) => stage_upsert(
                &mut entries,
                &mut slots,
                SlotKey::Node(idx.index() as u32),
                // A label change is a change to an existing node; a node
                // created with labels also has its `add_node` create op.
                CaptureOrigin::Update,
            ),
            RawOp::UpsertEdge(eidx, origin) => stage_upsert(
                &mut entries,
                &mut slots,
                SlotKey::Edge(eidx.index() as u32),
                *origin,
            ),
            RawOp::RemoveNode { node_type, id } => entries.push(Entry::RemoveNode {
                node_type: interner.resolve(*node_type).to_string(),
                id: id.clone(),
            }),
            RawOp::RemoveEdge {
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
            } => entries.push(Entry::RemoveEdge {
                conn_type: interner.resolve(*conn_type).to_string(),
                src_type: interner.resolve(*src_type).to_string(),
                src_id: src_id.clone(),
                tgt_type: interner.resolve(*tgt_type).to_string(),
                tgt_id: tgt_id.clone(),
            }),
        }
    }

    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let event = match entry {
            Entry::Vacated => continue,
            Entry::Upsert { key, kind } => {
                match resolve_upsert(key, graph, interner, &secondary_labels) {
                    Some(change) => PendingEvent { kind, change },
                    // Removed later in the same commit: its remove entry carries
                    // the outcome.
                    None => continue,
                }
            }
            Entry::RemoveNode { node_type, id } => PendingEvent {
                kind: CdcEventKind::Delete,
                change: CdcChange::Node {
                    node_type,
                    id,
                    after: None,
                },
            },
            Entry::RemoveEdge {
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
            } => PendingEvent {
                kind: CdcEventKind::Delete,
                change: CdcChange::Edge {
                    conn_type,
                    src_type,
                    src_id,
                    tgt_type,
                    tgt_id,
                    after: None,
                },
            },
        };
        out.push(event);
    }
    out
}

/// Stage an upsert, collapsing onto any earlier one for the same entity and
/// moving the merged entry to the end of the order.
fn stage_upsert(
    entries: &mut Vec<Entry>,
    slots: &mut HashMap<SlotKey, usize>,
    key: SlotKey,
    origin: CaptureOrigin,
) {
    let kind = match origin {
        CaptureOrigin::Create => CdcEventKind::Create,
        CaptureOrigin::Update => CdcEventKind::Update,
    };
    let kind = match slots.get(&key) {
        Some(&at) => {
            let previous = std::mem::replace(&mut entries[at], Entry::Vacated);
            match previous {
                // "Created in this commit" survives any number of later
                // property writes: the consumer must be told the entity is new.
                Entry::Upsert {
                    kind: CdcEventKind::Create,
                    ..
                } => CdcEventKind::Create,
                _ => kind,
            }
        }
        None => kind,
    };
    slots.insert(key, entries.len());
    entries.push(Entry::Upsert { key, kind });
}

/// Read an upserted entity's identity and after-state out of the final graph,
/// or `None` if it no longer exists.
fn resolve_upsert(
    key: SlotKey,
    graph: &impl GraphRead,
    interner: &StringInterner,
    secondary_labels: &impl Fn(NodeIndex) -> Vec<String>,
) -> Option<CdcChange> {
    match key {
        SlotKey::Node(raw) => {
            let idx = NodeIndex::new(raw as usize);
            let node = graph.node_view(idx)?;
            Some(CdcChange::Node {
                node_type: node.node_type_str(interner).to_string(),
                id: node.id().into_owned(),
                after: Some(NodeState {
                    title: node.title().into_owned(),
                    labels: secondary_labels(idx),
                    properties: node.properties_cloned(interner).into_iter().collect(),
                }),
            })
        }
        SlotKey::Edge(raw) => {
            let eidx = petgraph::graph::EdgeIndex::new(raw as usize);
            let (a, b) = graph.edge_endpoints(eidx)?;
            let edge = graph.edge_weight(eidx)?;
            let (src_type, src_id) = logical_node(graph, a, interner)?;
            let (tgt_type, tgt_id) = logical_node(graph, b, interner)?;
            Some(CdcChange::Edge {
                conn_type: edge.connection_type_str(interner).to_string(),
                src_type,
                src_id,
                tgt_type,
                tgt_id,
                after: Some(EdgeState {
                    properties: edge.properties_cloned(interner).into_iter().collect(),
                }),
            })
        }
    }
}

/// Resolve a node index to its logical `(node_type, id)`, or `None` if the
/// node is gone.
fn logical_node(
    graph: &impl GraphRead,
    idx: NodeIndex,
    interner: &StringInterner,
) -> Option<(String, Value)> {
    let node = graph.node_view(idx)?;
    Some((
        node.node_type_str(interner).to_string(),
        node.id().into_owned(),
    ))
}

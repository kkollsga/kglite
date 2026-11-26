//! WAL-only logical touches share the raw sequence with CDC, so rollback has
//! one truncation boundary. Final-state normalization never resolves a reused
//! slot without checking its captured logical identity.
use super::*;
use crate::graph::core::iterators::GraphEdgeRef;
use crate::graph::edge_embeddings::{EdgeEmbeddingKey, EdgeEmbeddingStore};
use crate::graph::wal::{
    EdgeEmbeddingStoreState, EdgeGroupEmbeddingPatchWal, EdgeGroupMemberPatchWal,
    EdgeGroupStoreWalState, EdgeVectorCellPatchWal, EdgeVectorWalState,
};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// The id is in its `canonical_id` spelling: a statement that widens a
/// UniqueId column re-spells every id already touched in the transaction, and
/// a raw key would no longer match the node it named.
type NodeKey = (InternedKey, Value);
type GroupKey = (InternedKey, NodeKey, NodeKey);

fn node_key(node_type: InternedKey, id: &Value) -> NodeKey {
    (
        node_type,
        crate::graph::schema::canonical_id(id).into_owned(),
    )
}

/// Whether `idx` still holds the logical node `key` names.
fn holds(graph: &impl GraphRead, idx: NodeIndex, key: &NodeKey) -> bool {
    graph.node_type_of(idx) == Some(key.0)
        && graph
            .get_node_id(idx)
            .is_some_and(|id| crate::graph::schema::canonical_id(&id).as_ref() == &key.1)
}

impl<G: GraphRead> RecordingGraph<G> {
    /// Re-key the node types named by the buffered ops after a bulk type
    /// rename that bypassed the capture layer (`rename_map` maps old
    /// interned-key values to new ones).
    ///
    /// Ops that carry only an index resolve against the final graph and need
    /// nothing; the ops that captured a logical `(type, id)` identity at touch
    /// time would otherwise name endpoints under a type that no longer exists
    /// and fail to replay.
    pub(crate) fn retype_captured_ops(&mut self, rename_map: &HashMap<u64, u64>) {
        let retype = |key: &mut InternedKey| {
            if let Some(&new) = rename_map.get(&key.as_u64()) {
                *key = InternedKey::from_u64(new);
            }
        };
        for op in &mut self.ops {
            match op {
                RawOp::RemoveNode { node_type, .. } | RawOp::WalNode { node_type, .. } => {
                    retype(node_type)
                }
                RawOp::RemoveEdge {
                    src_type, tgt_type, ..
                }
                | RawOp::WalGroup {
                    src_type, tgt_type, ..
                } => {
                    retype(src_type);
                    retype(tgt_type);
                }
                RawOp::WalRelationshipEmbeddingBase(touch) => {
                    retype(&mut touch.src_type);
                    retype(&mut touch.tgt_type);
                }
                _ => {}
            }
        }
    }

    pub(crate) fn note_wal_node_identity(
        &mut self,
        idx: NodeIndex,
        node_type: InternedKey,
        id: Value,
        reset: bool,
    ) {
        if self.wal_owner {
            self.ops.push(RawOp::WalNode {
                idx,
                node_type,
                id,
                reset,
            });
        }
    }

    pub(super) fn note_wal_node(&mut self, idx: NodeIndex, reset: bool) {
        if !self.wal_owner {
            return;
        }
        let identity = self
            .inner
            .node_type_of(idx)
            .zip(self.inner.get_node_id(idx));
        if let Some((kind, id)) = identity {
            self.note_wal_node_identity(idx, kind, id, reset);
        }
    }

    pub(crate) fn note_wal_group(&mut self, idx: EdgeIndex) {
        if !self.wal_owner {
            return;
        }
        let Some((source, target)) = self.inner.edge_endpoints(idx) else {
            return;
        };
        let Some(kind) = self.inner.edge_weight(idx).map(|e| e.connection_type) else {
            return;
        };
        self.note_wal_group_of(source, target, kind);
    }

    /// [`note_wal_group`](Self::note_wal_group) for a group named directly
    /// rather than through one of its members.
    ///
    /// `add_edge` calls this **before** inserting, because the base a patch is
    /// digested against is the group as of before the commit. Noting the group
    /// through the new edge afterwards snapshots a base that already contains
    /// it, and replay — which reconstructs the base from the prior commit —
    /// then reads the frame as a base-digest mismatch and refuses the reopen.
    pub(crate) fn note_wal_group_of(
        &mut self,
        source: NodeIndex,
        target: NodeIndex,
        kind: InternedKey,
    ) {
        if !self.wal_owner {
            return;
        }
        let Some((src_type, src_id)) = self
            .inner
            .node_type_of(source)
            .zip(self.inner.get_node_id(source))
        else {
            return;
        };
        let Some((tgt_type, tgt_id)) = self
            .inner
            .node_type_of(target)
            .zip(self.inner.get_node_id(target))
        else {
            return;
        };
        let mut base_members = if self.capture_edge_embedding_bases {
            group_members(&self.inner, source, target, kind)
                .into_iter()
                .map(|edge| (edge.id(), edge.weight().properties.to_vec()))
                .collect()
        } else {
            Vec::new()
        };
        base_members.sort_unstable_by_key(|(edge, _)| edge.index());
        self.ops.push(RawOp::WalGroup {
            source,
            target,
            conn_type: kind,
            src_type,
            src_id,
            tgt_type,
            tgt_id,
            base_members,
        });
    }

    pub(crate) fn note_wal_edge_embedding_store(&mut self, conn_type: &str, text_column: &str) {
        if self.wal_owner {
            self.ops.push(RawOp::WalRelationshipEmbeddingStore {
                conn_type: conn_type.to_string(),
                text_column: text_column.to_string(),
            });
        }
    }

    pub(crate) fn note_wal_edge_embedding_base(&mut self, touch: EdgeEmbeddingBaseTouch) {
        if self.wal_owner {
            self.ops
                .push(RawOp::WalRelationshipEmbeddingBase(Box::new(touch)));
        }
    }

    pub(super) fn note_wal_incident_groups(&mut self, idx: NodeIndex) {
        if !self.wal_owner {
            return;
        }
        let edges: HashSet<_> = self
            .inner
            .edges_directed(idx, Direction::Outgoing)
            .chain(self.inner.edges_directed(idx, Direction::Incoming))
            .map(|e| e.id())
            .collect();
        for edge in edges {
            self.note_wal_group(edge);
        }
    }
}

#[derive(Default)]
struct NodeTouch {
    idx: Option<NodeIndex>,
    reset: bool,
}

type GroupTouch = (
    NodeIndex,
    NodeIndex,
    Vec<(EdgeIndex, Vec<(InternedKey, Value)>)>,
);

#[derive(Default)]
struct WalTouches {
    // Insertion-ordered, so one copy of each key serves both the lookup and
    // the emission order; a separate order list doubled the key (and its id
    // allocation) per touched node.
    nodes: IndexMap<NodeKey, NodeTouch>,
    groups: IndexMap<GroupKey, GroupTouch>,
    declarations: Vec<MutationOp>,
    embedding_stores: Vec<(String, String)>,
    seen_embedding_stores: HashSet<(String, String)>,
    removed_groups: HashSet<GroupKey>,
    embedding_bases: HashMap<GroupKey, EdgeEmbeddingBaseTouch>,
}

impl WalTouches {
    fn capture(&mut self, op: &RawOp) {
        match op {
            RawOp::Declaration(op) => self.declarations.push((**op).clone()),
            RawOp::WalNode {
                idx,
                node_type,
                id,
                reset,
            } => {
                let touch = self.nodes.entry(node_key(*node_type, id)).or_default();
                touch.idx = Some(*idx);
                touch.reset |= *reset;
            }
            RawOp::RemoveNode { node_type, id, .. } => {
                let touch = self.nodes.entry(node_key(*node_type, id)).or_default();
                touch.idx = None;
                touch.reset = true;
            }
            RawOp::RemoveEdge {
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
                ..
            } => {
                self.removed_groups.insert((
                    *conn_type,
                    node_key(*src_type, src_id),
                    node_key(*tgt_type, tgt_id),
                ));
            }
            RawOp::WalGroup {
                source,
                target,
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
                base_members,
            } => {
                let key = (
                    *conn_type,
                    node_key(*src_type, src_id),
                    node_key(*tgt_type, tgt_id),
                );
                let touch = self
                    .groups
                    .entry(key)
                    .or_insert_with(|| (*source, *target, base_members.clone()));
                touch.0 = *source;
                touch.1 = *target;
            }
            RawOp::WalRelationshipEmbeddingStore {
                conn_type,
                text_column,
            } => {
                let key = (conn_type.clone(), text_column.clone());
                if self.seen_embedding_stores.insert(key.clone()) {
                    self.embedding_stores.push(key);
                }
            }
            RawOp::WalRelationshipEmbeddingBase(touch) => self.merge_embedding_base(touch),
            _ => {}
        }
    }

    fn merge_embedding_base(&mut self, touch: &EdgeEmbeddingBaseTouch) {
        let key = (
            touch.conn_type,
            node_key(touch.src_type, &touch.src_id),
            node_key(touch.tgt_type, &touch.tgt_id),
        );
        let Some(base) = self.embedding_bases.get_mut(&key) else {
            self.embedding_bases.insert(key, touch.clone());
            return;
        };
        for prior in &touch.prior_cells {
            if !base.prior_cells.iter().any(|existing| {
                existing.edge == prior.edge && existing.text_column == prior.text_column
            }) {
                base.prior_cells.push(prior.clone());
            }
        }
    }
}

pub(super) fn resolve(
    raw: &[RawOp],
    graph: &impl GraphRead,
    interner: &StringInterner,
    labels: impl Fn(NodeIndex) -> Vec<String>,
    edge_embeddings: Option<&HashMap<EdgeEmbeddingKey, EdgeEmbeddingStore>>,
    out: &mut dyn FnMut(MutationOp),
) {
    let mut touches = WalTouches::default();
    for op in raw {
        touches.capture(op);
    }
    for op in std::mem::take(&mut touches.declarations) {
        out(op);
    }
    emit_store_ops(out, &touches, edge_embeddings);
    emit_node_ops(out, &touches, graph, interner, labels);
    for key in touches.groups.keys() {
        emit_group_ops(out, key, &touches, graph, interner, edge_embeddings);
    }
}

fn emit_store_ops(
    out: &mut dyn FnMut(MutationOp),
    touches: &WalTouches,
    edge_embeddings: Option<&HashMap<EdgeEmbeddingKey, EdgeEmbeddingStore>>,
) {
    for (conn_type, text_column) in &touches.embedding_stores {
        let state = edge_embeddings
            .and_then(|stores| {
                stores.get(&crate::graph::edge_embeddings::edge_store_key(
                    conn_type,
                    text_column,
                ))
            })
            .map_or(EdgeEmbeddingStoreState::Absent, |store| {
                EdgeEmbeddingStoreState::Present {
                    dimension: store.dimension(),
                    metric: store.metric().map(str::to_string),
                    model_id: store.model_id().map(str::to_string),
                }
            });
        out(MutationOp::SetEdgeEmbeddingStore {
            conn_type: conn_type.clone(),
            text_column: text_column.clone(),
            state,
        });
    }
}

fn emit_node_ops(
    out: &mut dyn FnMut(MutationOp),
    touches: &WalTouches,
    graph: &impl GraphRead,
    interner: &StringInterner,
    labels: impl Fn(NodeIndex) -> Vec<String>,
) {
    for (key, touch) in &touches.nodes {
        let idx = touch.idx.filter(|idx| holds(graph, *idx, key));
        if let Some(node) = idx.and_then(|idx| graph.node_view(idx)) {
            out(MutationOp::ReplaceNodeState {
                node_type: interner.resolve(key.0).into(),
                // The stored spelling, so replay recreates the column it had.
                id: node.id().into_owned(),
                title: node.title().into_owned(),
                properties: node.properties_cloned(interner).into_iter().collect(),
                labels: labels(idx.expect("live node")),
                reset: touch.reset,
            });
        } else {
            out(MutationOp::RemoveNode {
                node_type: interner.resolve(key.0).into(),
                id: key.1.clone(),
            });
        }
    }
}

fn endpoint(
    graph: &impl GraphRead,
    touches: &WalTouches,
    key: &NodeKey,
    hint: NodeIndex,
) -> Option<NodeIndex> {
    let idx = touches
        .nodes
        .get(key)
        .map_or(Some(hint), |touch| touch.idx)?;
    holds(graph, idx, key).then_some(idx)
}

fn current_group(
    graph: &impl GraphRead,
    interner: &StringInterner,
    touches: &WalTouches,
    key: &GroupKey,
) -> (Vec<EdgeIndex>, Vec<Vec<(String, Value)>>) {
    let (source, target, _) = &touches.groups[key];
    let Some((source, target)) =
        endpoint(graph, touches, &key.1, *source).zip(endpoint(graph, touches, &key.2, *target))
    else {
        return (Vec::new(), Vec::new());
    };
    let mut members: Vec<_> = group_members(graph, source, target, key.0)
        .into_iter()
        .map(|edge| {
            (
                edge.id(),
                edge.weight()
                    .properties_cloned(interner)
                    .into_iter()
                    .collect(),
            )
        })
        .collect();
    members.sort_unstable_by_key(|(edge, _)| edge.index());
    members.into_iter().unzip()
}

fn emit_group_ops(
    out: &mut dyn FnMut(MutationOp),
    key: &GroupKey,
    touches: &WalTouches,
    graph: &impl GraphRead,
    interner: &StringInterner,
    edge_embeddings: Option<&HashMap<EdgeEmbeddingKey, EdgeEmbeddingStore>>,
) {
    let (member_slots, edges) = current_group(graph, interner, touches, key);
    out(MutationOp::ReplaceEdgeGroup {
        conn_type: interner.resolve(key.0).into(),
        src_type: interner.resolve(key.1 .0).into(),
        src_id: key.1 .1.clone(),
        tgt_type: interner.resolve(key.2 .0).into(),
        tgt_id: key.2 .1.clone(),
        edges: edges.clone(),
    });
    let Some(all_stores) = edge_embeddings else {
        return;
    };
    let conn_type = interner.resolve(key.0);
    let matching = matching_stores(all_stores, conn_type);
    let metadata_touched = touches
        .seen_embedding_stores
        .iter()
        .any(|(kind, _)| kind == conn_type);
    if matching.is_empty() && !metadata_touched {
        return;
    }
    let base_members = &touches.groups[key].2;
    let can_patch = !matching.is_empty()
        && base_members
            .iter()
            .map(|(edge, _)| *edge)
            .eq(member_slots.iter().copied())
        && !touches.removed_groups.contains(key)
        && (!metadata_touched || touches.embedding_bases.contains_key(key));
    let op = if can_patch {
        patch_group(key, touches, interner, edges, &member_slots, &matching)
    } else {
        full_group(
            key,
            interner,
            member_slots.len(),
            owned_store_state(&matching, &member_slots),
        )
    };
    out(op);
}

fn matching_stores<'a>(
    stores: &'a HashMap<EdgeEmbeddingKey, EdgeEmbeddingStore>,
    conn_type: &str,
) -> Vec<(&'a str, &'a EdgeEmbeddingStore)> {
    let mut matching: Vec<_> = stores
        .iter()
        .filter(|((kind, _), _)| kind == conn_type)
        .map(|((_, property), store)| {
            (
                crate::graph::embeddings::text_column_of(property)
                    .expect("edge embedding keys are canonical"),
                store,
            )
        })
        .collect();
    matching.sort_unstable_by_key(|(name, _)| *name);
    matching
}

fn borrowed_store_state<'a>(
    stores: &[(&'a str, &'a EdgeEmbeddingStore)],
    member_slots: &[EdgeIndex],
) -> Vec<crate::graph::mutation::wal_replay::edge_embedding_delta::BorrowedEdgeGroupStore<'a>> {
    stores
        .iter()
        .map(|(name, store)| {
            let members = member_slots
                .iter()
                .map(|edge| {
                    store.get(*edge).map(|vector| {
            crate::graph::mutation::wal_replay::edge_embedding_delta::BorrowedEdgeVectorCell {
                vector, text_hash: store.text_hash(*edge),
            }
        })
                })
                .collect();
            crate::graph::mutation::wal_replay::edge_embedding_delta::BorrowedEdgeGroupStore {
                text_column: name,
                members,
            }
        })
        .collect()
}

/// The base a `PatchEdgeGroupEmbeddings` digests, which is the group as of
/// **before** this commit — so it is scoped to the stores that existed before
/// it. A store created in the same commit is in `stores` (the final set) but
/// not in `base.base_stores`, and including it digests a column replay's base
/// cannot have, which decodes as `patch base digest mismatch` and fails the
/// reopen permanently. With no base touch at all there is nothing to scope
/// against, so the final set stands.
fn borrowed_base_store_state<'a>(
    stores: &[(&'a str, &'a EdgeEmbeddingStore)],
    member_slots: &[EdgeIndex],
    base: Option<&'a EdgeEmbeddingBaseTouch>,
) -> Vec<crate::graph::mutation::wal_replay::edge_embedding_delta::BorrowedEdgeGroupStore<'a>> {
    stores
        .iter()
        .filter(|(name, _)| {
            base.is_none_or(|base| base.base_stores.iter().any(|existing| existing == name))
        })
        .map(|(name, store)| {
            let members = member_slots
                .iter()
                .map(|edge| {
                    let prior = base.and_then(|base| {
                        base.prior_cells.iter().find(|prior| {
                            prior.edge == *edge && prior.text_column == *name
                        })
                    });
                    if let Some(prior) = prior {
                        prior.state.as_ref().map(|state| {
                            crate::graph::mutation::wal_replay::edge_embedding_delta::BorrowedEdgeVectorCell {
                                vector: &state.vector,
                                text_hash: state.text_hash,
                            }
                        })
                    } else {
                        store.get(*edge).map(|vector| {
                            crate::graph::mutation::wal_replay::edge_embedding_delta::BorrowedEdgeVectorCell {
                                vector,
                                text_hash: store.text_hash(*edge),
                            }
                        })
                    }
                })
                .collect();
            crate::graph::mutation::wal_replay::edge_embedding_delta::BorrowedEdgeGroupStore {
                text_column: name,
                members,
            }
        })
        .collect()
}

fn owned_store_state(
    stores: &[(&str, &EdgeEmbeddingStore)],
    member_slots: &[EdgeIndex],
) -> Vec<EdgeGroupStoreWalState> {
    stores
        .iter()
        .map(|(name, store)| EdgeGroupStoreWalState {
            text_column: (*name).to_string(),
            members: member_slots
                .iter()
                .map(|edge| {
                    store.get(*edge).map(|vector| EdgeVectorWalState {
                        vector: vector.to_vec(),
                        text_hash: store.text_hash(*edge),
                    })
                })
                .collect(),
        })
        .collect()
}

fn full_group(
    key: &GroupKey,
    interner: &StringInterner,
    member_count: usize,
    stores: Vec<EdgeGroupStoreWalState>,
) -> MutationOp {
    MutationOp::ReplaceEdgeGroupEmbeddings {
        conn_type: interner.resolve(key.0).into(),
        src_type: interner.resolve(key.1 .0).into(),
        src_id: key.1 .1.clone(),
        tgt_type: interner.resolve(key.2 .0).into(),
        tgt_id: key.2 .1.clone(),
        member_count,
        stores,
    }
}

fn patch_group(
    key: &GroupKey,
    touches: &WalTouches,
    interner: &StringInterner,
    result_properties: Vec<Vec<(String, Value)>>,
    member_slots: &[EdgeIndex],
    stores: &[(&str, &EdgeEmbeddingStore)],
) -> MutationOp {
    let base = touches.embedding_bases.get(key);
    let base_properties: Vec<_> = touches.groups[key]
        .2
        .iter()
        .map(|(_, properties)| {
            properties
                .iter()
                .map(|(name, value)| (interner.resolve(*name).to_string(), value.clone()))
                .collect()
        })
        .collect();
    let borrowed = borrowed_store_state(stores, member_slots);
    let borrowed_base = borrowed_base_store_state(stores, member_slots, base);
    let store_names: Vec<_> = stores.iter().map(|(name, _)| (*name).to_string()).collect();
    let base_store_names: Vec<_> = borrowed_base
        .iter()
        .map(|store| store.text_column.to_string())
        .collect();
    let members = member_slots
        .iter()
        .enumerate()
        .map(|(ordinal, edge)| EdgeGroupMemberPatchWal::Prior {
            prior_ordinal: ordinal as u32,
            cells: stores
                .iter()
                .map(|(name, store)| patch_cell(base, *edge, name, store))
                .collect(),
        })
        .collect();
    let base_digest =
        crate::graph::mutation::wal_replay::edge_embedding_delta::digest_borrowed_group_state(
            &base_properties,
            &borrowed_base,
        )
        .expect("captured values are encodable");
    let result_digest =
        crate::graph::mutation::wal_replay::edge_embedding_delta::digest_borrowed_group_state(
            &result_properties,
            &borrowed,
        )
        .expect("captured values are encodable");
    MutationOp::PatchEdgeGroupEmbeddings {
        conn_type: interner.resolve(key.0).into(),
        src_type: interner.resolve(key.1 .0).into(),
        src_id: key.1 .1.clone(),
        tgt_type: interner.resolve(key.2 .0).into(),
        tgt_id: key.2 .1.clone(),
        patch: EdgeGroupEmbeddingPatchWal {
            base_digest,
            result_digest,
            base_stores: base_store_names,
            stores: store_names,
            members,
        },
    }
}

fn patch_cell(
    base: Option<&EdgeEmbeddingBaseTouch>,
    edge: EdgeIndex,
    store_name: &str,
    store: &EdgeEmbeddingStore,
) -> EdgeVectorCellPatchWal {
    let final_vector = store.get(edge);
    let final_hash = store.text_hash(edge);
    let prior = base.and_then(|base| {
        base.prior_cells
            .iter()
            .find(|prior| prior.edge == edge && prior.text_column == store_name)
            .map(|prior| prior.state.as_ref())
    });
    if let Some(prior) = prior {
        if prior.map(|state| (state.vector.as_slice(), state.text_hash))
            == final_vector.map(|vector| (vector, final_hash))
        {
            return EdgeVectorCellPatchWal::Keep;
        }
        return final_vector.map_or(EdgeVectorCellPatchWal::Clear, |vector| {
            EdgeVectorCellPatchWal::Replace(EdgeVectorWalState {
                vector: vector.to_vec(),
                text_hash: final_hash,
            })
        });
    }
    if base.is_some_and(|base| !base.base_stores.iter().any(|name| name == store_name)) {
        return final_vector.map_or(EdgeVectorCellPatchWal::Clear, |vector| {
            EdgeVectorCellPatchWal::Replace(EdgeVectorWalState {
                vector: vector.to_vec(),
                text_hash: final_hash,
            })
        });
    }
    EdgeVectorCellPatchWal::Keep
}

fn group_members<'a>(
    graph: &'a impl GraphRead,
    source: NodeIndex,
    target: NodeIndex,
    kind: InternedKey,
) -> Vec<GraphEdgeRef<'a>> {
    const PROBE_LIMIT: usize = 32;
    let matches = |edge: &GraphEdgeRef<'_>| {
        edge.source() == source && edge.target() == target && edge.connection_type() == kind
    };
    let mut outgoing = graph.edges_directed(source, Direction::Outgoing);
    let mut members = Vec::new();
    for _ in 0..PROBE_LIMIT {
        let Some(edge) = outgoing.next() else {
            return members;
        };
        if matches(&edge) {
            members.push(edge);
        }
    }
    // Only exhaustion proves that a bounded incoming probe is the whole group.
    // Otherwise discard it and resume the already-consumed outgoing iterator.
    let mut incoming = graph.edges_directed(target, Direction::Incoming);
    let mut incoming_members = Vec::new();
    for _ in 0..PROBE_LIMIT {
        let Some(edge) = incoming.next() else {
            return incoming_members;
        };
        if matches(&edge) {
            incoming_members.push(edge);
        }
    }
    members.extend(outgoing.filter(matches));
    members
}

#[cfg(test)]
mod adjacency_probe_tests {
    use super::*;
    use crate::graph::schema::{GraphBackend, MappedGraph};
    use crate::graph::storage::disk::graph::DiskGraph;
    use std::sync::Arc;

    type GroupMember = (usize, Vec<(String, Value)>);

    fn verify_group(
        mut graph: GraphBackend,
        outgoing_extra: usize,
        incoming_extra: usize,
        loops: bool,
    ) {
        let mut interner = StringInterner::new();
        let nodes: Vec<_> = (0..100)
            .map(|id| {
                graph.add_node(NodeData::new(
                    Value::Int64(id),
                    Value::String(id.to_string()),
                    "N".into(),
                    HashMap::new(),
                    &mut interner,
                ))
            })
            .collect();
        let source = nodes[0];
        let target = nodes[usize::from(!loops)];
        let mut add = |graph: &mut GraphBackend, a, b, kind: &str, value| {
            graph.add_edge(
                a,
                b,
                EdgeData::new(
                    kind.into(),
                    HashMap::from([("v".into(), value)]),
                    &mut interner,
                ),
            )
        };
        // Equal parallel maps must remain separate members, including loops.
        add(&mut graph, source, target, "R", Value::Int64(1));
        add(&mut graph, source, target, "R", Value::Int64(1));
        let old = add(&mut graph, source, target, "R", Value::Int64(99));
        graph.remove_edge(old);
        add(
            &mut graph,
            source,
            target,
            "R",
            Value::String("typed".into()),
        );
        add(&mut graph, source, target, "OTHER", Value::Int64(8));
        for &peer in nodes.iter().skip(2).take(outgoing_extra) {
            add(&mut graph, source, peer, "R", Value::Int64(7));
        }
        for &peer in nodes.iter().skip(50).take(incoming_extra) {
            add(&mut graph, peer, target, "R", Value::Int64(6));
        }
        if !loops {
            add(&mut graph, target, source, "R", Value::Int64(5));
        }
        graph.flush_pending_writes();
        let kind = InternedKey::from_str("R");
        let mut expected: Vec<GroupMember> = graph
            .edges_connecting(source, target)
            .filter(|e| e.connection_type() == kind)
            .map(|e| {
                (
                    e.id().index(),
                    e.weight()
                        .properties_cloned(&interner)
                        .into_iter()
                        .collect(),
                )
            })
            .collect();
        expected.sort_unstable_by_key(|(slot, _)| *slot);
        assert_eq!(expected.len(), 3);
        assert_eq!(
            expected
                .iter()
                .filter(|(_, p)| p == &vec![("v".into(), Value::Int64(1))])
                .count(),
            2
        );
        assert_eq!(
            expected
                .iter()
                .filter(|(_, p)| p == &vec![("v".into(), Value::String("typed".into()))])
                .count(),
            1
        );
        let mut actual: Vec<GroupMember> = group_members(&graph, source, target, kind)
            .into_iter()
            .map(|e| {
                (
                    e.id().index(),
                    e.weight()
                        .properties_cloned(&interner)
                        .into_iter()
                        .collect(),
                )
            })
            .collect();
        actual.sort_unstable_by_key(|(slot, _)| *slot);
        assert_eq!(
            actual, expected,
            "outgoing={outgoing_extra}, incoming={incoming_extra}, loops={loops}"
        );
        assert!(group_members(&graph, source, target, InternedKey::from_str("ABSENT")).is_empty());
    }

    #[test]
    fn adjacency_probe_preserves_groups_across_direction_boundaries_and_backends() {
        // Four group/other-type edges put these degrees below, at, and above32.
        for (outgoing, incoming) in [
            (0, 40),
            (27, 40),
            (28, 0),
            (29, 0),
            (40, 27),
            (40, 28),
            (40, 29),
            (40, 40),
        ] {
            verify_group(GraphBackend::new(), outgoing, incoming, false);
            verify_group(
                GraphBackend::Mapped(Arc::new(MappedGraph::new())),
                outgoing,
                incoming,
                false,
            );
            let dir = tempfile::tempdir().unwrap();
            verify_group(
                GraphBackend::Disk(Box::new(DiskGraph::new_at_path(dir.path()).unwrap())),
                outgoing,
                incoming,
                false,
            );
        }
    }

    #[test]
    fn adjacency_probe_keeps_self_loops_once_on_selected_complete_direction() {
        for (outgoing, incoming) in [(0, 40), (40, 0), (40, 40)] {
            verify_group(GraphBackend::new(), outgoing, incoming, true);
            verify_group(
                GraphBackend::Mapped(Arc::new(MappedGraph::new())),
                outgoing,
                incoming,
                true,
            );
            let dir = tempfile::tempdir().unwrap();
            verify_group(
                GraphBackend::Disk(Box::new(DiskGraph::new_at_path(dir.path()).unwrap())),
                outgoing,
                incoming,
                true,
            );
        }
    }
}

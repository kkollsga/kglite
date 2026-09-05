//! WAL-only logical touches share the raw sequence with CDC, so rollback has
//! one truncation boundary. Final-state normalization never resolves a reused
//! slot without checking its captured logical identity.
use super::*;
use crate::graph::core::iterators::GraphEdgeRef;
use std::collections::HashSet;

type NodeKey = (InternedKey, Value);
type GroupKey = (InternedKey, NodeKey, NodeKey);

impl<G: GraphRead> RecordingGraph<G> {
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

    pub(super) fn note_wal_group(&mut self, idx: EdgeIndex) {
        if !self.wal_owner {
            return;
        }
        let Some((source, target)) = self.inner.edge_endpoints(idx) else {
            return;
        };
        let Some(kind) = self.inner.edge_weight(idx).map(|e| e.connection_type) else {
            return;
        };
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
        self.ops.push(RawOp::WalGroup {
            source,
            target,
            conn_type: kind,
            src_type,
            src_id,
            tgt_type,
            tgt_id,
        });
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

fn remember<'a, K: Eq + std::hash::Hash + Clone, V>(
    map: &'a mut HashMap<K, V>,
    order: &mut Vec<K>,
    key: K,
    initial: impl FnOnce() -> V,
) -> &'a mut V {
    if !map.contains_key(&key) {
        order.push(key.clone());
    }
    map.entry(key).or_insert_with(initial)
}

pub(super) fn resolve(
    raw: &[RawOp],
    graph: &impl GraphRead,
    interner: &StringInterner,
    labels: impl Fn(NodeIndex) -> Vec<String>,
) -> Vec<MutationOp> {
    let mut nodes: HashMap<NodeKey, NodeTouch> = HashMap::new();
    let mut node_order = Vec::new();
    let mut groups: HashMap<GroupKey, (NodeIndex, NodeIndex)> = HashMap::new();
    let mut group_order = Vec::new();
    for op in raw {
        match op {
            RawOp::WalNode {
                idx,
                node_type,
                id,
                reset,
            } => {
                let touch = remember(
                    &mut nodes,
                    &mut node_order,
                    (*node_type, id.clone()),
                    NodeTouch::default,
                );
                touch.idx = Some(*idx);
                touch.reset |= *reset;
            }
            RawOp::RemoveNode { node_type, id, .. } => {
                let touch = remember(
                    &mut nodes,
                    &mut node_order,
                    (*node_type, id.clone()),
                    NodeTouch::default,
                );
                touch.idx = None;
                touch.reset = true;
            }
            RawOp::WalGroup {
                source,
                target,
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
            } => {
                let key = (
                    *conn_type,
                    (*src_type, src_id.clone()),
                    (*tgt_type, tgt_id.clone()),
                );
                *remember(&mut groups, &mut group_order, key, || (*source, *target)) =
                    (*source, *target);
            }
            _ => {}
        }
    }
    let matches = |idx: NodeIndex, key: &NodeKey| {
        graph.node_type_of(idx) == Some(key.0) && graph.get_node_id(idx).as_ref() == Some(&key.1)
    };
    let endpoint = |key: &NodeKey, hint: NodeIndex| {
        let idx = nodes.get(key).map_or(Some(hint), |touch| touch.idx)?;
        matches(idx, key).then_some(idx)
    };
    let mut out = Vec::with_capacity(nodes.len() + groups.len());
    for key in node_order {
        let touch = &nodes[&key];
        let idx = touch.idx.filter(|idx| matches(*idx, &key));
        if let Some(node) = idx.and_then(|idx| graph.node_view(idx)) {
            out.push(MutationOp::ReplaceNodeState {
                node_type: interner.resolve(key.0).into(),
                id: key.1,
                title: node.title().into_owned(),
                properties: node.properties_cloned(interner).into_iter().collect(),
                labels: labels(idx.expect("live node")),
                reset: touch.reset,
            });
        } else {
            out.push(MutationOp::RemoveNode {
                node_type: interner.resolve(key.0).into(),
                id: key.1,
            });
        }
    }
    for key in group_order {
        let (source, target) = groups[&key];
        let mut edges = Vec::new();
        if let (Some(source), Some(target)) = (endpoint(&key.1, source), endpoint(&key.2, target)) {
            // Physical slots provide a deterministic order within this commit;
            // replay preserves every map, including equal parallel members.
            let mut members: Vec<_> = group_members(graph, source, target, key.0)
                .into_iter()
                .map(|e| {
                    (
                        e.id().index(),
                        e.weight().properties_cloned(interner).into_iter().collect(),
                    )
                })
                .collect();
            members.sort_unstable_by_key(|(idx, _)| *idx);
            edges.extend(members.into_iter().map(|(_, props)| props));
        }
        out.push(MutationOp::ReplaceEdgeGroup {
            conn_type: interner.resolve(key.0).into(),
            src_type: interner.resolve(key.1 .0).into(),
            src_id: key.1 .1,
            tgt_type: interner.resolve(key.2 .0).into(),
            tgt_id: key.2 .1,
            edges,
        });
    }
    out
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
        let _query = graph.begin_query();
        let kind = InternedKey::from_str("R");
        let mut expected: Vec<GroupMember> = graph
            .edges_connecting(source, target)
            .filter(|e| e.weight().connection_type == kind)
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

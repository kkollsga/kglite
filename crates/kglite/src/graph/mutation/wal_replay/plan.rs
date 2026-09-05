use std::collections::{HashMap, HashSet};

use crate::datatypes::Value;
use crate::graph::wal::{MutationOp, WalFrame};

pub(super) type NodeKey = (String, Value);
pub(super) type EdgeKey = (String, String, Value, String, Value);
pub(super) type Properties = Vec<(String, Value)>;

#[derive(Default)]
pub(super) struct NodeState {
    pub row: Option<(Value, Properties)>,
    pub removed: bool,
    pub reset: bool,
    pub generation: u64,
    pub labels: Option<Vec<String>>,
}

pub(super) struct EdgeState {
    pub properties: Option<Properties>,
    pub reset: bool,
    pub group: Option<Vec<Properties>>,
    source_generation: u64,
    target_generation: u64,
}

#[derive(Default)]
pub(super) struct ReplayPlan {
    pub nodes: Vec<(NodeKey, NodeState)>,
    node_slots: HashMap<NodeKey, usize>,
    pub edges: Vec<(EdgeKey, EdgeState)>,
    edge_slots: HashMap<EdgeKey, usize>,
    pub max_lsn: u64,
}

impl ReplayPlan {
    pub fn fold(frames: &[WalFrame], after: u64) -> Self {
        let mut plan = Self {
            max_lsn: after,
            ..Self::default()
        };
        for frame in frames.iter().filter(|frame| frame.lsn > after) {
            plan.max_lsn = plan.max_lsn.max(frame.lsn);
            for op in &frame.ops {
                plan.fold_op(op);
            }
        }
        plan
    }

    fn node_mut(&mut self, key: NodeKey) -> &mut NodeState {
        let slot = *self.node_slots.entry(key.clone()).or_insert_with(|| {
            let slot = self.nodes.len();
            self.nodes.push((key, NodeState::default()));
            slot
        });
        &mut self.nodes[slot].1
    }

    fn fold_op(&mut self, op: &MutationOp) {
        match op {
            MutationOp::ReplaceNodeState {
                node_type,
                id,
                title,
                properties,
                labels,
                reset,
            } => {
                let node = self.node_mut((node_type.clone(), id.clone()));
                if *reset {
                    node.reset = true;
                    node.generation += 1;
                }
                node.row = Some((title.clone(), properties.clone()));
                node.removed = false;
                node.labels = Some(labels.clone());
            }
            MutationOp::ReplaceEdgeGroup {
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
                edges,
            } => {
                let key = (
                    conn_type.clone(),
                    src_type.clone(),
                    src_id.clone(),
                    tgt_type.clone(),
                    tgt_id.clone(),
                );
                self.fold_edge(key.clone(), None);
                let slot = self.edge_slots[&key];
                self.edges[slot].1.group = Some(edges.clone());
            }
            MutationOp::UpsertNode {
                node_type,
                id,
                title,
                properties,
            } => {
                let node = self.node_mut((node_type.clone(), id.clone()));
                node.row = Some((title.clone(), properties.clone()));
                node.removed = false;
            }
            MutationOp::RemoveNode { node_type, id } => {
                let node = self.node_mut((node_type.clone(), id.clone()));
                node.row = None;
                node.removed = true;
                node.reset = true;
                node.generation += 1;
                node.labels = None;
            }
            MutationOp::SetNodeLabels {
                node_type,
                id,
                labels,
            } => {
                self.node_mut((node_type.clone(), id.clone())).labels = Some(labels.clone());
            }
            MutationOp::UpsertEdge {
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
                properties,
            } => {
                self.fold_edge(
                    (
                        conn_type.clone(),
                        src_type.clone(),
                        src_id.clone(),
                        tgt_type.clone(),
                        tgt_id.clone(),
                    ),
                    Some(properties.clone()),
                );
            }
            MutationOp::RemoveEdge {
                conn_type,
                src_type,
                src_id,
                tgt_type,
                tgt_id,
            } => {
                self.fold_edge(
                    (
                        conn_type.clone(),
                        src_type.clone(),
                        src_id.clone(),
                        tgt_type.clone(),
                        tgt_id.clone(),
                    ),
                    None,
                );
            }
        }
    }

    fn fold_edge(&mut self, key: EdgeKey, properties: Option<Properties>) {
        let prior_reset = self
            .edge_slots
            .get(&key)
            .is_some_and(|&slot| self.edges[slot].1.reset);
        let state = EdgeState {
            reset: properties.is_none() || prior_reset,
            group: None,
            properties,
            source_generation: self.generation(&(key.1.clone(), key.2.clone())),
            target_generation: self.generation(&(key.3.clone(), key.4.clone())),
        };
        if let Some(&slot) = self.edge_slots.get(&key) {
            self.edges[slot].1 = state;
        } else {
            self.edge_slots.insert(key.clone(), self.edges.len());
            self.edges.push((key, state));
        }
    }

    fn generation(&self, key: &NodeKey) -> u64 {
        self.node_slots
            .get(key)
            .map_or(0, |&slot| self.nodes[slot].1.generation)
    }

    pub fn node_removed(&self, key: &NodeKey) -> bool {
        self.node_slots
            .get(key)
            .is_some_and(|&slot| self.nodes[slot].1.removed)
    }

    pub fn edge_survives(&self, key: &EdgeKey, state: &EdgeState) -> bool {
        let source = (key.1.clone(), key.2.clone());
        let target = (key.3.clone(), key.4.clone());
        !self.node_removed(&source)
            && !self.node_removed(&target)
            && self.generation(&source) == state.source_generation
            && self.generation(&target) == state.target_generation
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    pub fn node_types(&self) -> HashSet<String> {
        self.nodes
            .iter()
            .map(|(key, _)| key.0.clone())
            .chain(
                self.edges
                    .iter()
                    .flat_map(|(key, _)| [key.1.clone(), key.3.clone()]),
            )
            .collect()
    }

    pub fn edge_types(&self) -> HashSet<String> {
        self.edges.iter().map(|(key, _)| key.0.clone()).collect()
    }
}

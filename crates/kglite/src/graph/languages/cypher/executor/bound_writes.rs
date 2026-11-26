//! The elements a `SET` clause moved a declared validity bound of, judged
//! once the clause has applied every item — so `SET n.vf = …, n.vt = …` is
//! judged as the interval it leaves, not item by item. A refusal fails the
//! statement, whose checkpoint undoes the clause; an element left with an
//! empty interval is counted into the statement's warning.

use crate::graph::features::temporal::{
    check_stored_edge, check_stored_node, declarations::node_names_bound, edge_property_is_bound,
    EmptyIntervals,
};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use petgraph::graph::{EdgeIndex, NodeIndex};

#[derive(Default)]
pub(super) struct BoundWrites {
    nodes: Vec<NodeIndex>,
    edges: Vec<EdgeIndex>,
}

impl BoundWrites {
    /// Note a node whose `property` was written, when that names a bound of a
    /// declaration on one of its labels.
    pub(super) fn node_property(&mut self, graph: &DirGraph, idx: NodeIndex, property: &str) {
        if node_names_bound(graph, idx, property) {
            self.nodes.push(idx);
        }
    }

    /// Note a node that gained `label`, when `label` is declared: its stored
    /// bounds now answer to that declaration.
    pub(super) fn node_label(&mut self, graph: &DirGraph, idx: NodeIndex, label: &str) {
        if graph.temporal.node(label).is_some() {
            self.nodes.push(idx);
        }
    }

    /// Note a relationship whose `property` was written, when that names a
    /// bound of a declaration on its type.
    pub(super) fn edge_property(&mut self, graph: &DirGraph, edge: EdgeIndex, property: &str) {
        if graph.temporal.is_empty() {
            return;
        }
        let rel_type = { graph.graph.edge_weight(edge).map(|e| e.connection_type) };
        let named = rel_type
            .and_then(|key| graph.interner.try_resolve(key))
            .is_some_and(|rel_type| edge_property_is_bound(graph, rel_type, property));
        if named {
            self.edges.push(edge);
        }
    }

    /// Refuse the clause when a noted element now breaks its declaration;
    /// count those it leaves empty into `empty`.
    pub(super) fn check(
        mut self,
        graph: &DirGraph,
        empty: &mut EmptyIntervals,
    ) -> Result<(), String> {
        if self.nodes.is_empty() && self.edges.is_empty() {
            return Ok(());
        }
        self.nodes.sort_unstable();
        self.nodes.dedup();
        for idx in self.nodes {
            check_stored_node(graph, idx, empty)?;
        }
        self.edges.sort_unstable();
        self.edges.dedup();
        for edge in self.edges {
            check_stored_edge(graph, edge, empty)?;
        }
        Ok(())
    }
}

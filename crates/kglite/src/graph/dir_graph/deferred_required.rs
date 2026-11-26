//! Statement-end judging of NOT NULL / NODE KEY presence.
//!
//! Neo4j checks existence and key constraints when a statement's changes are
//! committed, not as each clause lands, so `CREATE (n:L) SET n = $props` is
//! valid on a type that requires a property `$props` supplies, and a
//! statement may null a required property and restore it later. The Cypher
//! write path therefore skips the presence check per write, notes the entity
//! it touched here, and judges the *stored* state once the clauses have run
//! (`execute_mutable_with_csv`). A violation fails the statement, and the
//! statement's own rollback undoes every write in it. Type and UNIQUE checks
//! stay per write: a value that is wrong or taken is wrong immediately.
//!
//! Bulk loaders (`add_nodes`, `add_connections`) build each row complete
//! before it is written, so they judge presence per row against the finished
//! row and do not route through here.

use petgraph::graph::{EdgeIndex, NodeIndex};

use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::ontology::predicates::title_is_synthesised;
use crate::graph::storage::interner::InternedKey;
use crate::graph::storage::NodeView;
use crate::graph::storage::{GraphRead, GraphWrite};

impl DirGraph {
    /// Note a node of `node_type` that a Cypher write created or changed,
    /// when the type requires any property. One lookup for a type that does
    /// not.
    #[inline]
    pub(crate) fn note_required_node(&mut self, node_type: &str, idx: NodeIndex) {
        if self.has_required_fields(node_type) {
            self.required_touched.push(idx);
        }
    }

    /// [`Self::note_required_node`] for a relationship of `rel_type`.
    #[inline]
    pub(crate) fn note_required_edge(&mut self, rel_type: &str, idx: EdgeIndex) {
        if self.type_has_rel_not_null_constraints(rel_type) {
            self.required_touched_edges.push(idx);
        }
    }

    /// Forget what an earlier, failed statement left noted.
    pub(crate) fn clear_required_touched(&mut self) {
        self.required_touched.clear();
        self.required_touched_edges.clear();
    }

    /// The statement-end judge: required properties first, then the enforced
    /// ontology rules, over what the statement touched.
    pub(crate) fn judge_statement_writes(
        &mut self,
        tally: &mut crate::graph::ontology::node_gate::Tally,
    ) -> Result<(), String> {
        self.judge_required_touched()?;
        self.judge_touched_writes(tally)
    }

    /// Judge, and drain, the entities touched since the last call against
    /// their stored state. The first violation is parked on the typed side
    /// channel and returned. An entity deleted since it was touched is
    /// skipped.
    pub(crate) fn judge_required_touched(&mut self) -> Result<(), String> {
        if self.required_touched.is_empty() && self.required_touched_edges.is_empty() {
            return Ok(());
        }
        let mut nodes = std::mem::take(&mut self.required_touched);
        let mut edges = std::mem::take(&mut self.required_touched_edges);
        // Disk stages writes in a mut-cache that reads do not see.
        GraphWrite::flush_pending_writes(&mut self.graph);
        nodes.sort_unstable();
        nodes.dedup();
        edges.sort_unstable();
        edges.dedup();
        let found = {
            let graph: &DirGraph = self;
            let mut found = None;
            for idx in nodes {
                let Some(view) = graph.graph.node_view(idx) else {
                    continue;
                };
                let node_type = graph.interner.resolve(view.node_type());
                if let Err(violation) = graph.check_required_fields(node_type, |property| {
                    stored_required_value(graph, &view, node_type, property)
                }) {
                    found = Some(*violation);
                    break;
                }
            }
            if found.is_none() {
                for idx in edges {
                    let Some(weight) = graph.graph.edge_weight(idx) else {
                        continue;
                    };
                    let rel_type = graph.interner.resolve(weight.connection_type);
                    let read = |property: &str| -> Option<Value> {
                        let key = InternedKey::from_str(property);
                        weight
                            .properties
                            .iter()
                            .find(|(k, _)| *k == key)
                            .map(|(_, value)| value.clone())
                    };
                    if let Err(violation) = graph.rel_required_violation(rel_type, &read) {
                        found = Some(*violation);
                        break;
                    }
                }
            }
            found
        };
        match found {
            Some(violation) => Err(self.record_constraint_violation(violation)),
            None => Ok(()),
        }
    }
}

/// `property` of the stored node as the presence check reads it: the identity
/// fields under both spellings, everything else off the property store, `None`
/// for absent or null.
///
/// The literal `title` is present whenever the node has a title, since every
/// write mints one. A *declared* title spelling (`node_title_field='name'`)
/// asks the caller for a value, so there the engine-minted fallback does not
/// count — the rule `CREATE` applied before presence moved to statement end.
fn stored_required_value(
    graph: &DirGraph,
    view: &NodeView<'_>,
    node_type: &str,
    property: &str,
) -> Option<Value> {
    let value = match (property, graph.resolve_alias(node_type, property)) {
        ("id", _) | (_, "id") => Some(view.id().into_owned()),
        ("title", _) => Some(view.title().into_owned()),
        (_, "title") => {
            let (id, title) = (view.id(), view.title());
            (!title_is_synthesised(graph, node_type, &id, &title)).then(|| title.into_owned())
        }
        (_, field) => view
            .get(InternedKey::from_str(field))
            .map(|v| v.into_owned()),
    };
    value.filter(|v| !matches!(v, Value::Null))
}

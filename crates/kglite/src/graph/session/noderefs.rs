//! Output-only endpoint materialisation; expression evaluation keeps identity.

use std::borrow::Cow;

use crate::datatypes::values::{NodeValue, PathValue, RelValue};
use crate::datatypes::{DataFrame, PropMap, Value};
use crate::graph::schema::GraphBackend;
use crate::graph::storage::GraphRead;

/// Resolve internal node references recursively to titles from the executing view.
///
/// `execute_read`/`execute_mut` and lazy result materialisation apply this before
/// exposing values. Raw executor consumers can call it themselves. Entity IDs,
/// labels, relationship endpoints and property keys are unchanged. Missing
/// nodes and cyclic title references resolve to NULL.
pub fn resolve_noderefs(graph: &GraphBackend, rows: &mut [Vec<Value>]) {
    let _arena_guard = graph.begin_query();
    let mut resolver = Resolver {
        graph,
        active: Vec::new(),
    };
    for row in rows {
        for value in row {
            if let Some(resolved) = resolver.value(value) {
                *value = resolved;
            }
        }
    }
}

/// Resolve endpoint references in a borrowed output value using this graph view.
///
/// Values without endpoint references stay borrowed. Only output materialisation
/// should call this: stored values, predicate identity and grouping are unchanged.
/// Missing nodes and cyclic title references resolve to NULL.
pub fn resolve_noderef_value<'v>(graph: &GraphBackend, value: &'v Value) -> Cow<'v, Value> {
    let _arena_guard = graph.begin_query();
    let mut resolver = Resolver {
        graph,
        active: Vec::new(),
    };
    resolver
        .value(value)
        .map(Cow::Owned)
        .unwrap_or(Cow::Borrowed(value))
}

/// Replace endpoint references in values about to become stored properties.
///
/// Resolution uses the statement or transfer source view supplied by the
/// caller. Values without endpoint references retain their allocations;
/// missing nodes and cyclic title chains become NULL.
pub(crate) fn snapshot_property_values<'a>(
    graph: &GraphBackend,
    values: impl IntoIterator<Item = &'a mut Value>,
) {
    let _arena_guard = graph.begin_query();
    let mut resolver = Resolver {
        graph,
        active: Vec::new(),
    };
    for value in values {
        if let Some(resolved) = resolver.value(value) {
            *value = resolved;
        }
    }
}

pub(crate) fn property_value_needs_snapshot(value: &Value) -> bool {
    match value {
        Value::NodeRef(_) => true,
        Value::List(items) => items.iter().any(property_value_needs_snapshot),
        Value::Map(properties) => properties.values().any(property_value_needs_snapshot),
        Value::Node(node) => node
            .properties
            .iter()
            .any(|(_, value)| property_value_needs_snapshot(value)),
        Value::Relationship(rel) => rel
            .properties
            .iter()
            .any(|(_, value)| property_value_needs_snapshot(value)),
        Value::Path(path) => {
            path.nodes.iter().any(|node| {
                node.properties
                    .iter()
                    .any(|(_, value)| property_value_needs_snapshot(value))
            }) || path.rels.iter().any(|rel| {
                rel.properties
                    .iter()
                    .any(|(_, value)| property_value_needs_snapshot(value))
            })
        }
        _ => false,
    }
}

/// Snapshot endpoint references in property-bearing DataFrame cells while
/// preserving the columns that identify nodes or edge endpoints.
pub(crate) fn snapshot_dataframe_properties(
    graph: &GraphBackend,
    frame: &mut DataFrame,
    identity_columns: &[&str],
) {
    let _arena_guard = graph.begin_query();
    let mut resolver = Resolver {
        graph,
        active: Vec::new(),
    };
    frame.map_container_cells(
        |name| !identity_columns.contains(&name),
        |value| resolver.value(value),
    );
}

struct Resolver<'a> {
    graph: &'a GraphBackend,
    active: Vec<u32>,
}

impl Resolver<'_> {
    fn value(&mut self, value: &Value) -> Option<Value> {
        match value {
            Value::NodeRef(id) => Some(self.title(*id)),
            Value::List(items) => map_changed(items, |item| self.value(item)).map(Value::List),
            Value::Map(properties) => self.properties(properties).map(Value::Map),
            Value::Node(node) => {
                if node.properties.iter().all(|(_, value)| {
                    matches!(
                        value,
                        Value::UniqueId(_)
                            | Value::Int64(_)
                            | Value::Float64(_)
                            | Value::String(_)
                            | Value::Boolean(_)
                            | Value::DateTime(_)
                            | Value::Point { .. }
                            | Value::Null
                            | Value::Duration { .. }
                            | Value::Timestamp(_)
                    )
                }) {
                    return None;
                }
                self.node(node).map(|node| Value::Node(Box::new(node)))
            }
            Value::Relationship(rel) => self
                .relationship(rel)
                .map(|rel| Value::Relationship(Box::new(rel))),
            Value::Path(path) => {
                let nodes = map_changed(&path.nodes, |node| self.node(node));
                let rels = map_changed(&path.rels, |rel| self.relationship(rel));
                if nodes.is_none() && rels.is_none() {
                    return None;
                }
                Some(Value::Path(Box::new(PathValue {
                    nodes: nodes.unwrap_or_else(|| path.nodes.clone()),
                    rels: rels.unwrap_or_else(|| path.rels.clone()),
                })))
            }
            _ => None,
        }
    }

    fn title(&mut self, id: u32) -> Value {
        // Stored title references can form chains independent of AST depth.
        // Grow only on this reference path; ordinary output values need no guard.
        stacker::maybe_grow(128 * 1024, 4 * 1024 * 1024, || self.title_inner(id))
    }

    fn title_inner(&mut self, id: u32) -> Value {
        if self.active.contains(&id) {
            return Value::Null;
        }
        let index = petgraph::graph::NodeIndex::new(id as usize);
        let title = self
            .graph
            .node_view(index)
            .map(|node| node.title().into_owned())
            .unwrap_or(Value::Null);
        self.active.push(id);
        let value = self.value(&title).unwrap_or(title);
        self.active.pop();
        value
    }

    fn properties(&mut self, properties: &PropMap) -> Option<PropMap> {
        let mut changed: Option<PropMap> = None;
        for (key, value) in properties {
            if let Some(resolved) = self.value(value) {
                // Ordinary maps retain their shared backing; copy only on a
                // changed property, never merely to inspect its values.
                changed
                    .get_or_insert_with(|| properties.clone())
                    .insert(key, resolved);
            }
        }
        changed
    }

    fn node(&mut self, node: &NodeValue) -> Option<NodeValue> {
        self.properties(&node.properties)
            .map(|properties| NodeValue {
                id: node.id,
                labels: node.labels.clone(),
                properties,
            })
    }

    fn relationship(&mut self, rel: &RelValue) -> Option<RelValue> {
        self.properties(&rel.properties).map(|properties| RelValue {
            id: rel.id,
            start_id: rel.start_id,
            end_id: rel.end_id,
            rel_type: rel.rel_type.clone(),
            properties,
        })
    }
}

fn map_changed<T: Clone>(items: &[T], mut resolve: impl FnMut(&T) -> Option<T>) -> Option<Vec<T>> {
    let mut changed: Option<Vec<T>> = None;
    for (index, item) in items.iter().enumerate() {
        if let Some(value) = resolve(item) {
            changed.get_or_insert_with(|| items.to_vec())[index] = value;
        }
    }
    changed
}

#[cfg(test)]
mod direct_output_tests {
    use super::*;
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::session::{execute_mut, ExecuteOptions};
    use std::collections::HashMap;

    #[test]
    fn single_value_output_preserves_borrowing_and_stored_reference_identity() {
        let mut graph = DirGraph::new();
        execute_mut(
            &mut graph,
            "CREATE (:Item {id:1,title:'Alpha'})",
            &ExecuteOptions::eager(&HashMap::new()),
        )
        .unwrap();
        let plain = Value::List(vec![Value::String("plain".into()), Value::Int64(7)]);
        assert!(
            matches!(resolve_noderef_value(&graph.graph, &plain), Cow::Borrowed(value) if std::ptr::eq(value, &plain))
        );
        let stored = Value::Map(PropMap::from_pairs(vec![(
            "nested".into(),
            Value::List(vec![Value::NodeRef(0)]),
        )]));
        let expected = Value::Map(PropMap::from_pairs(vec![(
            "nested".into(),
            Value::List(vec![Value::String("Alpha".into())]),
        )]));
        assert_eq!(
            resolve_noderef_value(&graph.graph, &stored).as_ref(),
            &expected
        );
        assert_eq!(
            stored,
            Value::Map(PropMap::from_pairs(vec![(
                "nested".into(),
                Value::List(vec![Value::NodeRef(0)])
            )]))
        );
        assert_eq!(
            resolve_noderef_value(&graph.graph, &Value::NodeRef(999)).as_ref(),
            &Value::Null
        );
    }
}

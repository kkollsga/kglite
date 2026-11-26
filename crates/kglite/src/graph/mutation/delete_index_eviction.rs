//! Evicting deleted nodes from the value-keyed indexes in O(deleted).
//!
//! A delete used to sweep every bucket of every property, range and composite
//! index on the node's type: `retain_members` flattens the layer stack (an
//! O(index) copy while a fork is shared) and filters every member. Measured at
//! 500k nodes of one type that is 1.5 ms for an autocommit delete and 11.4 ms
//! inside a transaction, for a statement that removes a handful of nodes.
//!
//! The sweep is only needed to *find* the buckets. A node is filed under the
//! value an index rebuild would read from it, so reading that value *before*
//! the node is removed names the one bucket per index it sits in. This module
//! resolves those buckets up front ([`DoomedBuckets::resolve`]), lets the
//! journal capture exactly them, and evicts exactly them afterwards.
//!
//! Each resolved index is **verified**: every doomed node must really be in the
//! bucket its value names. An index that fails (stale, or a value space the
//! read cannot reproduce) is left out of the plan and takes the full sweep, so
//! a surprise costs the old time and never a stale member. A type whose delete
//! is large against its size skips resolution altogether, because one linear
//! sweep beats that many point reads.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use petgraph::graph::NodeIndex;

use crate::datatypes::values::Value;
use crate::graph::schema::{CompositeIndexKey, CompositeValue, DirGraph, IndexKey};
use crate::graph::storage::undo::BucketId;

/// Point reads per doomed node worth paying against one pass over the type.
/// A resolved index costs one read per doomed node; the sweep costs one visit
/// per member, which is cheaper per element but not by more than this.
const POINT_READ_WEIGHT: usize = 8;

/// The buckets of each resolved index that hold a doomed node.
#[derive(Default)]
pub(super) struct DoomedBuckets {
    property: HashMap<IndexKey, Vec<Value>>,
    range: HashMap<IndexKey, Vec<Value>>,
    composite: HashMap<CompositeIndexKey, Vec<CompositeValue>>,
}

impl DoomedBuckets {
    /// Resolve the buckets for every index on a type whose delete is small
    /// against its size. Must run while the doomed nodes are still readable.
    pub(super) fn resolve(
        graph: &mut DirGraph,
        doomed_by_type: &HashMap<String, Vec<NodeIndex>>,
    ) -> Self {
        let mut plan = Self::default();
        for (node_type, doomed) in doomed_by_type {
            let members = graph.type_indices.get(node_type).map_or(0, |m| m.len());
            if doomed.len().saturating_mul(POINT_READ_WEIGHT) > members {
                continue;
            }
            plan.resolve_single(graph, node_type, doomed);
            plan.resolve_composite(graph, node_type, doomed);
        }
        plan
    }

    fn resolve_single(&mut self, graph: &mut DirGraph, node_type: &str, doomed: &[NodeIndex]) {
        let keys: Vec<IndexKey> = graph
            .property_indices
            .keys()
            .chain(graph.range_indices.keys())
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        for key in keys {
            let mut values: HashSet<Value> = HashSet::new();
            let mut covered = true;
            for node in doomed {
                let Some(value) = graph.indexed_value(node_type, &key.1, *node) else {
                    continue;
                };
                let in_property = graph
                    .property_indices
                    .get(&key)
                    .is_none_or(|index| index.get(&value).is_some_and(|b| b.contains(node)));
                let in_range = graph
                    .range_indices
                    .get(&key)
                    .is_none_or(|index| index.get(&value).is_some_and(|b| b.contains(node)));
                if !(in_property && in_range) {
                    covered = false;
                    break;
                }
                values.insert(value);
            }
            if !covered {
                continue;
            }
            let values: Vec<Value> = values.into_iter().collect();
            if graph.property_indices.contains_key(&key) {
                self.property.insert(key.clone(), values.clone());
            }
            if graph.range_indices.contains_key(&key) {
                self.range.insert(key, values);
            }
        }
    }

    fn resolve_composite(&mut self, graph: &mut DirGraph, node_type: &str, doomed: &[NodeIndex]) {
        let keys: Vec<CompositeIndexKey> = graph
            .composite_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .cloned()
            .collect();
        for key in keys {
            let mut values: HashSet<CompositeValue> = HashSet::new();
            let mut covered = true;
            for node in doomed {
                let tuple: Vec<Value> = key
                    .1
                    .iter()
                    .map(|p| {
                        graph
                            .indexed_value(node_type, p, *node)
                            .unwrap_or(Value::Null)
                    })
                    .collect();
                // An all-null tuple is never filed.
                if tuple.iter().all(|v| matches!(v, Value::Null)) {
                    continue;
                }
                let tuple = CompositeValue(tuple);
                let filed = graph
                    .composite_indices
                    .get(&key)
                    .and_then(|index| index.get(&tuple))
                    .is_some_and(|bucket| bucket.contains(node));
                if !filed {
                    covered = false;
                    break;
                }
                values.insert(tuple);
            }
            if covered {
                self.composite.insert(key, values.into_iter().collect());
            }
        }
    }

    /// Whether the sweep may skip this index: its buckets were resolved.
    pub(super) fn covers_property(&self, key: &IndexKey) -> bool {
        self.property.contains_key(key)
    }

    pub(super) fn covers_range(&self, key: &IndexKey) -> bool {
        self.range.contains_key(key)
    }

    pub(super) fn covers_composite(&self, key: &CompositeIndexKey) -> bool {
        self.composite.contains_key(key)
    }

    /// The `(bucket, members)` pairs the undo journal must capture before the
    /// eviction: only buckets that actually hold a doomed node.
    pub(super) fn evictions<S: BuildHasher>(
        &self,
        graph: &DirGraph,
        nodes_to_delete: &HashSet<NodeIndex, S>,
    ) -> Vec<(BucketId, Vec<NodeIndex>)> {
        let holds = |members: &Vec<NodeIndex>| members.iter().any(|m| nodes_to_delete.contains(m));
        let mut out = Vec::new();
        for (key, values) in &self.property {
            let Some(index) = graph.property_indices.get(key) else {
                continue;
            };
            for value in values {
                if let Some(members) = index.get(value).filter(|m| holds(m)) {
                    out.push((
                        BucketId::PropertyValue {
                            key: key.clone(),
                            value: value.clone(),
                        },
                        members.clone(),
                    ));
                }
            }
        }
        for (key, values) in &self.range {
            let Some(index) = graph.range_indices.get(key) else {
                continue;
            };
            for value in values {
                if let Some(members) = index.get(value).filter(|m| holds(m)) {
                    out.push((
                        BucketId::RangeValue {
                            key: key.clone(),
                            value: value.clone(),
                        },
                        members.clone(),
                    ));
                }
            }
        }
        for (key, values) in &self.composite {
            let Some(index) = graph.composite_indices.get(key) else {
                continue;
            };
            for value in values {
                if let Some(members) = index.get(value).filter(|m| holds(m)) {
                    out.push((
                        BucketId::CompositeTuple {
                            key: key.clone(),
                            value: value.clone(),
                        },
                        members.clone(),
                    ));
                }
            }
        }
        out
    }

    /// Sweep every bucket of the indexes on `node_type` that were *not*
    /// resolved: the fallback for a stale index and for a delete too large for
    /// point reads. The B-tree range sweep prunes buckets it empties, as it
    /// always has, so `lookup_range` never hands out a tombstoned `NodeIndex`.
    pub(super) fn sweep_unresolved<S: BuildHasher>(
        &self,
        graph: &mut DirGraph,
        node_type: &str,
        nodes_to_delete: &HashSet<NodeIndex, S>,
    ) {
        let prop_keys: Vec<_> = graph
            .property_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .filter(|key| !self.covers_property(key))
            .cloned()
            .collect();
        for key in prop_keys {
            if let Some(value_map) = graph.property_indices.get_mut(&key) {
                value_map.retain_members(|idx| !nodes_to_delete.contains(idx));
            }
        }
        let comp_keys: Vec<_> = graph
            .composite_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .filter(|key| !self.covers_composite(key))
            .cloned()
            .collect();
        for key in comp_keys {
            if let Some(value_map) = graph.composite_indices.get_mut(&key) {
                value_map.retain_members(|idx| !nodes_to_delete.contains(idx));
            }
        }
        let range_keys: Vec<_> = graph
            .range_indices
            .keys()
            .filter(|(nt, _)| nt == node_type)
            .filter(|key| !self.covers_range(key))
            .cloned()
            .collect();
        for key in range_keys {
            if let Some(value_map) = graph.range_indices.get_mut(&key) {
                value_map.retain_members_pruning_empty(|idx| !nodes_to_delete.contains(idx));
            }
        }
    }

    /// Evict the doomed nodes from the resolved buckets. Emptied property and
    /// composite buckets stay (the sweep kept them too); emptied range buckets
    /// are dropped, as the sweep's pruning variant dropped them.
    pub(super) fn evict<S: BuildHasher>(
        &self,
        graph: &mut DirGraph,
        nodes_to_delete: &HashSet<NodeIndex, S>,
    ) {
        for (key, values) in &self.property {
            if let Some(index) = graph.property_indices.get_mut(key) {
                for value in values {
                    if let Some(members) = index.get_mut(value) {
                        members.retain(|m| !nodes_to_delete.contains(m));
                    }
                }
            }
        }
        for (key, values) in &self.range {
            if let Some(index) = graph.range_indices.get_mut(key) {
                for value in values {
                    let emptied = match index.get_mut(value) {
                        Some(members) => {
                            members.retain(|m| !nodes_to_delete.contains(m));
                            members.is_empty()
                        }
                        None => false,
                    };
                    if emptied {
                        index.remove(value);
                    }
                }
            }
        }
        for (key, values) in &self.composite {
            if let Some(index) = graph.composite_indices.get_mut(key) {
                for value in values {
                    if let Some(members) = index.get_mut(value) {
                        members.retain(|m| !nodes_to_delete.contains(m));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::session::execute::{execute_mut, ExecuteOptions};

    fn run(graph: &mut DirGraph, query: &str) {
        let params = HashMap::new();
        execute_mut(graph, query, &ExecuteOptions::eager(&params))
            .unwrap_or_else(|e| panic!("{query}: {e}"));
    }

    /// Every bucket of every value-keyed index on `T`, members sorted, empty
    /// buckets dropped (a rebuild never files one).
    fn buckets(graph: &DirGraph) -> Vec<(String, String, Vec<usize>)> {
        let sorted = |members: &Vec<NodeIndex>| {
            let mut ids: Vec<usize> = members.iter().map(|m| m.index()).collect();
            ids.sort();
            ids
        };
        let mut out = Vec::new();
        for (key, index) in &graph.property_indices {
            for (value, members) in index.iter().filter(|(_, m)| !m.is_empty()) {
                out.push((
                    format!("p:{}", key.1),
                    format!("{value:?}"),
                    sorted(members),
                ));
            }
        }
        for (key, index) in &graph.range_indices {
            for (value, members) in index.iter().filter(|(_, m)| !m.is_empty()) {
                out.push((
                    format!("r:{}", key.1),
                    format!("{value:?}"),
                    sorted(members),
                ));
            }
        }
        for (key, index) in &graph.composite_indices {
            for (value, members) in index.iter().filter(|(_, m)| !m.is_empty()) {
                out.push((
                    format!("c:{:?}", key.1),
                    format!("{value:?}"),
                    sorted(members),
                ));
            }
        }
        out.sort();
        out
    }

    fn fixture(rows: usize) -> DirGraph {
        let mut graph = DirGraph::new();
        run(
            &mut graph,
            &format!(
                "UNWIND range(1, {rows}) AS i CREATE (:T {{id: i, k: 'k' + toString(i), \
                 g: i % 7, a: i % 5, b: i % 3}})"
            ),
        );
        run(
            &mut graph,
            "CREATE CONSTRAINT FOR (n:T) REQUIRE n.k IS UNIQUE",
        );
        run(&mut graph, "CREATE INDEX FOR (n:T) ON (n.g)");
        run(&mut graph, "CREATE RANGE INDEX FOR (n:T) ON (n.a)");
        run(&mut graph, "CREATE INDEX FOR (n:T) ON (n.a, n.b)");
        graph
    }

    #[test]
    fn a_small_delete_resolves_every_index_and_leaves_what_a_rebuild_would() {
        let mut graph = fixture(400);
        let doomed: HashMap<String, Vec<NodeIndex>> = HashMap::from([(
            "T".to_string(),
            [3i64, 77, 150]
                .iter()
                .map(|id| {
                    graph
                        .lookup_by_id_readonly("T", &Value::Int64(*id))
                        .unwrap()
                })
                .collect(),
        )]);
        let plan = DoomedBuckets::resolve(&mut graph, &doomed);
        for key in graph.property_indices.keys() {
            assert!(plan.covers_property(key), "{key:?} must be resolved");
        }
        assert!(graph.range_indices.keys().all(|key| plan.covers_range(key)));
        assert!(graph
            .composite_indices
            .keys()
            .all(|key| plan.covers_composite(key)));

        run(
            &mut graph,
            "MATCH (n:T) WHERE n.id IN [3, 77, 150] DETACH DELETE n",
        );
        let targeted = buckets(&graph);
        graph.refresh_indexes_for_type("T");
        assert_eq!(
            targeted,
            buckets(&graph),
            "targeted eviction disagrees with a rebuild"
        );
        // The three tuples are free, the rest still held.
        run(
            &mut graph,
            "CREATE (:T {id: 1000, k: 'k3', g: 0, a: 0, b: 0})",
        );
        assert!(graph
            .check_unique_claims(
                &graph.unique_claims("T", |p| (p == "k").then(|| Value::String("k4".into()))),
                None
            )
            .is_err());
    }

    #[test]
    fn a_delete_large_against_its_type_takes_the_sweep() {
        let mut graph = fixture(40);
        let doomed: HashMap<String, Vec<NodeIndex>> = HashMap::from([(
            "T".to_string(),
            (1..=20i64)
                .map(|id| graph.lookup_by_id_readonly("T", &Value::Int64(id)).unwrap())
                .collect(),
        )]);
        let plan = DoomedBuckets::resolve(&mut graph, &doomed);
        assert!(!plan.covers_property(&("T".to_string(), "k".to_string())));
        run(&mut graph, "MATCH (n:T) WHERE n.id <= 20 DETACH DELETE n");
        let swept = buckets(&graph);
        graph.refresh_indexes_for_type("T");
        assert_eq!(swept, buckets(&graph));
    }

    /// A stale index is not trusted: a node missing from the bucket its value
    /// names leaves that index out of the plan, so the sweep still cleans it.
    #[test]
    fn an_index_that_does_not_hold_the_node_is_left_to_the_sweep() {
        let mut graph = fixture(400);
        let node = graph.lookup_by_id_readonly("T", &Value::Int64(10)).unwrap();
        graph
            .property_indices
            .get_mut(&("T".to_string(), "g".to_string()))
            .unwrap()
            .get_mut(&Value::Int64(3))
            .unwrap()
            .retain(|m| *m != node);
        let doomed = HashMap::from([("T".to_string(), vec![node])]);
        let plan = DoomedBuckets::resolve(&mut graph, &doomed);
        assert!(!plan.covers_property(&("T".to_string(), "g".to_string())));
        assert!(plan.covers_property(&("T".to_string(), "k".to_string())));
    }
}

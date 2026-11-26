//! Carrying relationship embedding stores between graphs: `.kgle` export and
//! import, and `copy_embeddings_with_relationships_from`.
//!
//! A relationship has no stable identity of its own — stores are keyed by
//! physical edge slots, which are graph-local. A carried vector is therefore
//! addressed by `(source type, source id, target type, target id, relationship
//! type)`. That address is unambiguous only when exactly one relationship of
//! the type connects those endpoints. A **parallel group** (two or more) is
//! carried only under a caller-named key property whose value is unique within
//! the group; without one the carry is refused by name — type, endpoints and
//! member count — and nothing is installed. Member order never enters the
//! identity, so a target graph that built the same group in another order
//! still receives each vector on the right member.
//!
//! Installation replaces the whole store, like a node import, through the same
//! WAL-capturing, undo-journalled steps the other store writes use, carrying
//! each vector's source-text hash and the store's model id.

use std::collections::{BTreeSet, HashMap};

use petgraph::graph::{EdgeIndex, NodeIndex};
use serde::{Deserialize, Serialize};

use super::{
    capture_wal_edge_embedding_bases, edge_store_key, live_edges_of_type,
    note_wal_edge_embedding_changes, validate_metric, EdgeEmbeddingStore,
};
use crate::datatypes::values::Value;
use crate::graph::embedding_validation::validate_finite_vector;
use crate::graph::embeddings::text_column_of;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;

/// Caller-named key property per relationship type (`{'SUPPORTS': 'uid'}`).
pub type RelationshipKeys = HashMap<String, String>;

/// One relationship store as carried: portable addresses instead of slots.
/// This is the `.kgle` v4 record, so field order is the wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CarriedEdgeStore {
    pub(crate) relationship_type: String,
    pub(crate) text_column: String,
    pub(crate) dimension: usize,
    pub(crate) metric: Option<String>,
    pub(crate) model_id: Option<String>,
    /// The key property the exporter used for this type, if any. An import
    /// uses it unless the caller names another.
    pub(crate) key_property: Option<String>,
    pub(crate) entries: Vec<CarriedEdgeEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CarriedEdgeEntry {
    pub(crate) source_type: String,
    pub(crate) source_id: Value,
    pub(crate) target_type: String,
    pub(crate) target_id: Value,
    /// The member's key value, when a key property was named for the type and
    /// the member carries it.
    pub(crate) key: Option<Value>,
    pub(crate) vector: Vec<f32>,
    pub(crate) text_hash: Option<u64>,
}

/// What a relationship carry did, summed over stores.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RelationshipCarryStats {
    /// Stores installed on the target graph.
    pub stores: usize,
    /// Vectors installed.
    pub carried: usize,
    /// Vectors whose relationship the target graph does not have.
    pub skipped: usize,
    /// Stores that had vectors but matched no relationship, so were not
    /// installed.
    pub dropped_stores: usize,
    /// `(relationship type, text column)` of each installed store.
    pub installed: Vec<(String, String)>,
}

/// A carried store resolved against the target graph, ready to install.
pub(crate) struct ResolvedEdgeStore {
    store: CarriedEdgeStore,
    members: Vec<(EdgeIndex, usize)>,
    skipped: usize,
}

pub(super) fn describe_group(
    graph: &DirGraph,
    relationship_type: &str,
    source: NodeIndex,
    target: NodeIndex,
    members: usize,
) -> String {
    format!(
        "{members} '{relationship_type}' relationships connect ({}) to ({})",
        super::describe_endpoint(graph, source),
        super::describe_endpoint(graph, target)
    )
}

fn refusal(store: &str, group: &str, reason: &str, relationship_type: &str) -> String {
    format!(
        "Relationship embedding store '{store}' cannot be carried: {group}, and {reason}. \
         A parallel group is carried only under a key property whose value is unique within \
         the group: pass relationship_keys={{'{relationship_type}': '<property>'}}."
    )
}

/// The live members of `relationship_type` from `source` to `target`, in slot
/// order.
///
/// Walks the source's outgoing and the target's incoming edges in lockstep
/// and answers from whichever list ends first, so the cost is twice the
/// smaller degree. `edges_connecting` walks the source's list alone, which
/// made resolving every relationship off one hub quadratic: 200k rows `Hub ->
/// Doc` took 31 s against 0.09 s for the node twin.
pub(super) fn group_members(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    relationship_type: &str,
) -> Vec<EdgeIndex> {
    use petgraph::Direction::{Incoming, Outgoing};
    let conn = InternedKey::from_str(relationship_type);
    let mut outgoing = graph
        .graph
        .edges_directed_filtered(source, Outgoing, Some(conn));
    let mut incoming = graph
        .graph
        .edges_directed_filtered(target, Incoming, Some(conn));
    let (mut from_source, mut from_target) = (Vec::new(), Vec::new());
    let mut members = loop {
        match outgoing.next() {
            None => break from_source,
            Some(edge) if edge.target() == target && edge.connection_type() == conn => {
                from_source.push(edge.id())
            }
            Some(_) => {}
        }
        match incoming.next() {
            None => break from_target,
            Some(edge) if edge.source() == source && edge.connection_type() == conn => {
                from_target.push(edge.id())
            }
            Some(_) => {}
        }
    };
    members.sort_unstable_by_key(|edge| edge.index());
    members
}

pub(super) fn key_value(graph: &DirGraph, edge: EdgeIndex, property: &str) -> Option<Value> {
    graph
        .graph
        .edge_weight(edge)?
        .get_property(property)
        .filter(|value| !matches!(value, Value::Null))
        .cloned()
}

/// Every member's key value, refusing a member without one or a repeated value.
pub(super) fn group_keys(
    graph: &DirGraph,
    members: &[EdgeIndex],
    property: &str,
) -> Result<Vec<(EdgeIndex, Value)>, String> {
    let mut keyed: Vec<(EdgeIndex, Value)> = Vec::with_capacity(members.len());
    for &edge in members {
        let value = key_value(graph, edge, property)
            .ok_or_else(|| format!("a member has no '{property}' value"))?;
        if keyed.iter().any(|(_, seen)| *seen == value) {
            return Err(format!(
                "'{property}' repeats the value {value} within the group"
            ));
        }
        keyed.push((edge, value));
    }
    Ok(keyed)
}

/// Extract every relationship store of `src` as carried stores, refusing any
/// store with a vector on a parallel-group member that no usable key names.
pub(crate) fn extract_edge_stores(
    src: &DirGraph,
    keys: &RelationshipKeys,
) -> Result<Vec<CarriedEdgeStore>, String> {
    let mut stores: Vec<_> = src.edge_embeddings.iter().collect();
    stores.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let mut carried = Vec::with_capacity(stores.len());
    for ((relationship_type, name), store) in stores {
        carried.push(extract_one(src, relationship_type, name, store, keys)?);
    }
    Ok(carried)
}

fn extract_one(
    src: &DirGraph,
    relationship_type: &str,
    name: &str,
    store: &EdgeEmbeddingStore,
    keys: &RelationshipKeys,
) -> Result<CarriedEdgeStore, String> {
    let key_property = keys.get(relationship_type).cloned();
    let text_column = text_column_of(name).unwrap_or(name).to_string();
    let label = format!("{relationship_type}.{text_column}");
    let mut checked: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut entries = Vec::with_capacity(store.len());
    for edge in store.edges() {
        let Some((source, target)) = src.graph.edge_endpoints(edge) else {
            continue;
        };
        let members = group_members(src, source, target, relationship_type);
        if members.len() > 1 && checked.insert((source.index(), target.index())) {
            let group = describe_group(src, relationship_type, source, target, members.len());
            let Some(property) = key_property.as_deref() else {
                let reason = format!("relationship_keys names no key for '{relationship_type}'");
                return Err(refusal(&label, &group, &reason, relationship_type));
            };
            group_keys(src, &members, property)
                .map_err(|reason| refusal(&label, &group, &reason, relationship_type))?;
        }
        let (Some(source_view), Some(target_view)) =
            (src.graph.node_view(source), src.graph.node_view(target))
        else {
            continue;
        };
        let Some(vector) = store.get(edge) else {
            continue;
        };
        entries.push(CarriedEdgeEntry {
            source_type: source_view.node_type_str(&src.interner).to_string(),
            source_id: source_view.id().into_owned(),
            target_type: target_view.node_type_str(&src.interner).to_string(),
            target_id: target_view.id().into_owned(),
            key: key_property
                .as_deref()
                .and_then(|property| key_value(src, edge, property)),
            vector: vector.to_vec(),
            text_hash: store.text_hash(edge),
        });
    }
    Ok(CarriedEdgeStore {
        relationship_type: relationship_type.to_string(),
        text_column,
        dimension: store.dimension(),
        metric: store.metric().map(str::to_owned),
        model_id: store.model_id().map(str::to_owned),
        key_property,
        entries,
    })
}

/// Validate every carried store's shape before anything is installed.
pub(crate) fn validate_carried_edge_stores(stores: &[CarriedEdgeStore]) -> Result<(), String> {
    for store in stores {
        let label = format!("{}.{}", store.relationship_type, store.text_column);
        if store.dimension == 0 {
            return Err(format!(
                "Relationship embedding store '{label}' has dimension 0"
            ));
        }
        validate_metric(store.metric.as_deref())
            .map_err(|error| format!("Relationship embedding store '{label}': {error}"))?;
        for (index, entry) in store.entries.iter().enumerate() {
            if entry.vector.len() != store.dimension {
                return Err(format!(
                    "Invalid embedding in relationship store '{label}' at entry {index}: \
                     expected width {}, got {}",
                    store.dimension,
                    entry.vector.len()
                ));
            }
            validate_finite_vector(&entry.vector).map_err(|error| {
                format!(
                    "Invalid embedding in relationship store '{label}' at entry {index}: {error}"
                )
            })?;
        }
    }
    Ok(())
}

/// Resolve carried stores against `dst`. A parallel group on the target side
/// that cannot be told apart by the key is refused by name before anything is
/// installed; an entry whose relationship `dst` lacks is counted as skipped.
pub(crate) fn resolve_edge_stores(
    dst: &mut DirGraph,
    stores: Vec<CarriedEdgeStore>,
    keys: &RelationshipKeys,
) -> Result<Vec<ResolvedEdgeStore>, String> {
    validate_carried_edge_stores(&stores)?;
    let endpoint_types: BTreeSet<String> = stores
        .iter()
        .flat_map(|store| store.entries.iter())
        .flat_map(|entry| [entry.source_type.clone(), entry.target_type.clone()])
        .collect();
    for node_type in &endpoint_types {
        dst.build_id_index(node_type);
    }
    let dst: &DirGraph = dst;
    let resolved = stores
        .into_iter()
        .map(|store| resolve_one(dst, store, keys))
        .collect::<Result<Vec<_>, _>>();
    resolved
}

fn resolve_one(
    dst: &DirGraph,
    store: CarriedEdgeStore,
    keys: &RelationshipKeys,
) -> Result<ResolvedEdgeStore, String> {
    let key_property = keys
        .get(&store.relationship_type)
        .cloned()
        .or_else(|| store.key_property.clone());
    let label = format!("{}.{}", store.relationship_type, store.text_column);
    let mut members = Vec::with_capacity(store.entries.len());
    let mut claimed: HashMap<usize, usize> = HashMap::new();
    let mut skipped = 0usize;
    let mut group_cache: HashMap<(usize, usize), Vec<(EdgeIndex, Value)>> = HashMap::new();
    for (index, entry) in store.entries.iter().enumerate() {
        let endpoints = (
            dst.lookup_by_id_readonly(&entry.source_type, &entry.source_id),
            dst.lookup_by_id_readonly(&entry.target_type, &entry.target_id),
        );
        let (Some(source), Some(target)) = endpoints else {
            skipped += 1;
            continue;
        };
        let group = group_members(dst, source, target, &store.relationship_type);
        let matched = match group.as_slice() {
            [] => None,
            [only] => match (&entry.key, key_property.as_deref()) {
                (Some(wanted), Some(property)) => match key_value(dst, *only, property) {
                    Some(found) if found != *wanted => None,
                    _ => Some(*only),
                },
                _ => Some(*only),
            },
            _ => {
                let described =
                    describe_group(dst, &store.relationship_type, source, target, group.len());
                let (Some(wanted), Some(property)) = (&entry.key, key_property.as_deref()) else {
                    let reason = "the carried vector names no key value to choose a member by";
                    return Err(refusal(
                        &label,
                        &described,
                        reason,
                        &store.relationship_type,
                    ));
                };
                let slot = (source.index(), target.index());
                if let std::collections::hash_map::Entry::Vacant(vacant) = group_cache.entry(slot) {
                    let keyed = group_keys(dst, &group, property).map_err(|reason| {
                        refusal(&label, &described, &reason, &store.relationship_type)
                    })?;
                    vacant.insert(keyed);
                }
                group_cache[&slot]
                    .iter()
                    .find(|(_, value)| value == wanted)
                    .map(|(edge, _)| *edge)
            }
        };
        let Some(edge) = matched else {
            skipped += 1;
            continue;
        };
        if claimed.insert(edge.index(), index).is_some() {
            return Err(format!(
                "Relationship embedding store '{label}' cannot be carried: two carried vectors \
                 resolve to the same relationship of this graph."
            ));
        }
        members.push((edge, index));
    }
    Ok(ResolvedEdgeStore {
        store,
        members,
        skipped,
    })
}

/// Install resolved stores, replacing any same-named relationship store.
/// A store whose vectors all missed is not installed (`dropped_stores`), as a
/// node import does not install an unmatched store.
pub(crate) fn install_edge_stores(
    dst: &mut DirGraph,
    resolved: Vec<ResolvedEdgeStore>,
) -> Result<RelationshipCarryStats, String> {
    let mut stats = RelationshipCarryStats::default();
    for resolved in resolved {
        stats.skipped += resolved.skipped;
        if resolved.members.is_empty() {
            if !resolved.store.entries.is_empty() {
                stats.dropped_stores += 1;
            }
            continue;
        }
        stats.carried += resolved.members.len();
        stats.stores += 1;
        stats.installed.push((
            resolved.store.relationship_type.clone(),
            resolved.store.text_column.clone(),
        ));
        install_one(dst, resolved)?;
    }
    if stats.stores > 0 {
        dst.bump_version();
    }
    Ok(stats)
}

fn install_one(dst: &mut DirGraph, resolved: ResolvedEdgeStore) -> Result<(), String> {
    let store = &resolved.store;
    let mut successor = EdgeEmbeddingStore::new(store.dimension, store.metric.as_deref());
    successor.set_wal_metadata(
        store.dimension,
        store.metric.clone(),
        store.model_id.clone(),
    )?;
    for &(edge, index) in &resolved.members {
        let entry = &store.entries[index];
        successor.install_wal_vector(edge, &entry.vector, entry.text_hash);
    }
    // The whole store is replaced, so every live member of the type is a
    // touched group: the WAL records each group's final state, prior members
    // included.
    let affected = live_edges_of_type(dst, &store.relationship_type)?;
    capture_wal_edge_embedding_bases(
        dst,
        &store.relationship_type,
        &store.text_column,
        affected.iter().copied(),
    )?;
    let key = edge_store_key(&store.relationship_type, &store.text_column);
    let prior = dst.edge_embeddings.insert(key.clone(), successor);
    if let Some(journal) = dst.graph.undo_journal_mut() {
        journal.note_edge_embedding_store_replaced(key, prior);
    }
    note_wal_edge_embedding_changes(
        dst,
        &store.relationship_type,
        &store.text_column,
        true,
        affected,
    );
    Ok(())
}

/// Relationship half of a graph-to-graph embedding copy.
pub(crate) fn resolve_copy(
    dst: &mut DirGraph,
    src: &DirGraph,
    keys: &RelationshipKeys,
) -> Result<Vec<ResolvedEdgeStore>, String> {
    let carried = extract_edge_stores(src, keys)?;
    resolve_edge_stores(dst, carried, keys)
}

/// `(stores_copied, vectors_copied, vectors_skipped)` for node stores plus the
/// relationship carry, as returned by
/// [`DirGraph::copy_embeddings_with_relationships_from`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EmbeddingCopyReport {
    pub stores_copied: usize,
    pub vectors_copied: usize,
    pub vectors_skipped: usize,
    pub relationships: RelationshipCarryStats,
}

impl DirGraph {
    /// [`copy_embeddings_from`](Self::copy_embeddings_from) plus every
    /// relationship embedding store of `src`, addressed by endpoint ids and,
    /// for parallel groups, the key property `keys` names for the type.
    ///
    /// Atomic with respect to refusal: an ambiguous relationship (a parallel
    /// group without a usable key, on either side) is refused by name before
    /// any node or relationship store is written.
    pub fn copy_embeddings_with_relationships_from(
        &mut self,
        src: &DirGraph,
        keys: &RelationshipKeys,
    ) -> Result<EmbeddingCopyReport, String> {
        let resolved = resolve_copy(self, src, keys)?;
        let (stores_copied, vectors_copied, vectors_skipped) = self.copy_embeddings_from(src);
        let relationships = install_edge_stores(self, resolved)?;
        Ok(EmbeddingCopyReport {
            stores_copied,
            vectors_copied,
            vectors_skipped,
            relationships,
        })
    }
}

#[cfg(test)]
#[path = "edge_embedding_carry_tests.rs"]
mod tests;

/// One relationship's stored vector, addressed by its endpoints — a row of
/// [`relationship_embeddings`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RelationshipEmbedding {
    /// The source node's type.
    pub source_type: String,
    /// The source node's id.
    pub source_id: Value,
    /// The target node's type.
    pub target_type: String,
    /// The target node's id.
    pub target_id: Value,
    /// The relationship's value of the key property named for its type, when
    /// one was named and the relationship carries it.
    pub key: Option<Value>,
    /// The stored vector.
    pub vector: Vec<f32>,
}

/// Every vector in the `(relationship_type, text_column)` store, addressed by
/// endpoint ids — the relationship twin of reading a node store by node id,
/// and the shape an edge list (`edge_index`) plus an edge-feature matrix
/// (`edge_attr`) is built from.
///
/// Rows are ordered by source (type, id), then target (type, id), then key
/// (when one is named), then relationship slot, so the order is stable for a
/// given graph. Several relationships of the type between the same two nodes
/// (a parallel group) are all returned; naming a key property for the type in
/// `keys` tells them apart, and a named key that is missing on a member or
/// repeats within a group is refused by name rather than returned ambiguous.
/// Refused when there is no such store.
pub fn relationship_embeddings(
    graph: &DirGraph,
    relationship_type: &str,
    text_column: &str,
    keys: &RelationshipKeys,
) -> Result<Vec<RelationshipEmbedding>, String> {
    let label = format!("{relationship_type}.{text_column}");
    let store = graph
        .edge_embeddings
        .get(&edge_store_key(relationship_type, text_column))
        .ok_or_else(|| {
            crate::graph::embedding_hints::missing_store_error(
                graph,
                crate::graph::embedding_inventory::EmbeddingEntity::Relationship,
                relationship_type,
                text_column,
                crate::graph::embedding_hints::Surface::Method,
            )
        })?;
    let key_property = keys.get(relationship_type).map(String::as_str);
    let mut checked: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut rows: Vec<(usize, RelationshipEmbedding)> = Vec::with_capacity(store.len());
    for edge in store.edges() {
        let Some((source, target)) = graph.graph.edge_endpoints(edge) else {
            continue;
        };
        if let Some(property) = key_property {
            if checked.insert((source.index(), target.index())) {
                let members = group_members(graph, source, target, relationship_type);
                if members.len() > 1 {
                    group_keys(graph, &members, property).map_err(|reason| {
                        let group =
                            describe_group(graph, relationship_type, source, target, members.len());
                        format!(
                            "Relationship embedding store '{label}': {group}, and {reason}; the \
                             key named in relationship_keys must be unique within each group"
                        )
                    })?;
                }
            }
        }
        let (Some(source_view), Some(target_view), Some(vector)) = (
            graph.graph.node_view(source),
            graph.graph.node_view(target),
            store.get(edge),
        ) else {
            continue;
        };
        rows.push((
            edge.index(),
            RelationshipEmbedding {
                source_type: source_view.node_type_str(&graph.interner).to_string(),
                source_id: source_view.id().into_owned(),
                target_type: target_view.node_type_str(&graph.interner).to_string(),
                target_id: target_view.id().into_owned(),
                key: key_property.and_then(|property| key_value(graph, edge, property)),
                vector: vector.to_vec(),
            },
        ));
    }
    rows.sort_by(|(left_slot, left), (right_slot, right)| {
        (
            &left.source_type,
            &left.source_id,
            &left.target_type,
            &left.target_id,
            &left.key,
        )
            .cmp(&(
                &right.source_type,
                &right.source_id,
                &right.target_type,
                &right.target_id,
                &right.key,
            ))
            .then(left_slot.cmp(right_slot))
    });
    Ok(rows.into_iter().map(|(_, row)| row).collect())
}

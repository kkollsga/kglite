//! Relationship property indexes — the equality index over one property of one
//! relationship type, the edge twin of `property_indices`.
//!
//! `rel_property_indices: (rel_type, property) -> value -> [edge]`, each a
//! [`LayeredIndex`], so a fork shares the buckets and the statement undo
//! journal reverses an edit by the same `BucketAppended` / `BucketRemoved`
//! entries the node families use ([`BucketId::RelPropertyValue`]).
//!
//! # Members are edge slots, stored as `NodeIndex`
//!
//! [`LayeredIndex`] buckets hold `NodeIndex`; an edge slot is a bare `usize`
//! with the same free-list identity as a node slot. [`edge_slot`] /
//! [`slot_edge`] are the only places the two are converted, so a bucket member
//! is never read as a node outside this module and the matcher's anchor.
//!
//! # Maintenance
//!
//! Every edge write path files into the index through one of three entry
//! points, and each is a single `is_empty` check on a graph that declares no
//! relationship index:
//!
//! - [`DirGraph::file_created_edge`] after an edge is added,
//! - [`DirGraph::evict_removed_edge`] before an edge is removed (it reads the
//!   values it must evict), reached from `remove_edge_with_embeddings`, the
//!   graph-level removal choke point,
//! - [`DirGraph::rel_index_capture`] before, and [`DirGraph::refresh_rel_indexes`]
//!   after, any in-place property write.
//!
//! WAL replay and `vacuum` renumber or rewrite edges wholesale, so both end in
//! `reindex()`, which rebuilds every relationship index from the live edges.
//!
//! # Exactness
//!
//! The matcher uses the index only to pick start nodes and re-checks the edge
//! pattern on every expansion, so a stale entry costs a wasted probe, never a
//! wrong row; a *missing* entry would. Enforcement (`UNIQUE`) therefore never
//! trusts a bucket member: it re-reads the member's live value
//! ([`DirGraph::rel_unique_occupant`]).
//!
//! # Storage modes
//!
//! Memory and mapped graphs only: a disk graph's edge ids are a frozen CSR
//! plus a process-local overlay, so a slot-keyed index would need its own
//! persistence story. Declaring one there is refused with a message that says
//! so ([`DirGraph::refuse_rel_index_on_disk`]).

use petgraph::graph::{EdgeIndex, NodeIndex};

use super::index_layer::LayeredIndex;
use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::schema::{IndexKey, InternedKey};
use crate::graph::storage::undo::BucketId;
use crate::graph::storage::GraphRead;
use crate::graph::wal::PropertyIndexKind;

/// The bucket member standing for `edge`.
#[inline]
pub(crate) fn edge_slot(edge: EdgeIndex) -> NodeIndex {
    NodeIndex::new(edge.index())
}

/// The edge a bucket member stands for.
#[inline]
pub(crate) fn slot_edge(slot: NodeIndex) -> EdgeIndex {
    EdgeIndex::new(slot.index())
}

/// The relationship index state of a [`DirGraph`].
#[derive(Debug, Clone, Default)]
pub(crate) struct RelIndexState {
    /// `(rel_type, property) -> value -> [edge slot]`; rebuilt from `index_keys`
    /// on load.
    pub(crate) indices: std::collections::HashMap<IndexKey, LayeredIndex<Value>>,
    /// Persisted keys of the indexes a user declared; one only a constraint
    /// owns is derived from `unique`.
    pub(crate) index_keys: Vec<IndexKey>,
    /// Declared relationship UNIQUE constraints `(rel_type, property)`.
    pub(crate) unique: std::collections::BTreeSet<(String, String)>,
    /// Slots a user declared independently of the constraint that owns them;
    /// they survive `DROP CONSTRAINT`.
    pub(crate) adopted: std::collections::HashSet<IndexKey>,
}

/// An edge's indexed values as a write found them, taken by
/// [`DirGraph::rel_index_capture`] and settled by
/// [`DirGraph::refresh_rel_indexes`].
#[derive(Debug)]
pub(crate) struct RelIndexPreImage {
    edge: EdgeIndex,
    values: Vec<(IndexKey, Option<Value>)>,
}

impl DirGraph {
    /// Whether any relationship index exists — the fast-out every edge write
    /// path takes first.
    #[inline]
    pub(crate) fn has_rel_property_indices(&self) -> bool {
        !self.rel.indices.is_empty()
    }

    /// Whether `(rel_type, property)` has a built index.
    pub(crate) fn has_rel_property_index(&self, rel_type: &str, property: &str) -> bool {
        self.rel
            .indices
            .contains_key(&(rel_type.to_string(), property.to_string()))
    }

    /// Refuse a relationship index on a disk graph.
    pub(crate) fn refuse_rel_index_on_disk(&self, what: &str) -> Result<(), String> {
        if self.graph.is_disk() {
            return Err(format!(
                "{what} on a relationship is not available on a storage='disk' graph: a disk \
                 graph's relationship ids are a frozen CSR plus a process-local overlay, so an \
                 index keyed by them cannot be kept or persisted. Use a memory or mapped graph, \
                 or query the relationship property unindexed."
            ));
        }
        Ok(())
    }

    /// Build the equality index over `property` of `rel_type`, replacing any
    /// existing one. Returns the number of distinct values indexed.
    ///
    /// The raw build: no refusals. A user's request goes through the DDL
    /// executor, which refuses a disk graph and a duplicate first; a load and
    /// `reindex()` rebuild a recorded declaration through here.
    pub(crate) fn build_rel_property_index(&mut self, rel_type: &str, property: &str) -> usize {
        self.materialize_indexes();
        let mut index: std::collections::HashMap<Value, Vec<NodeIndex>> =
            std::collections::HashMap::new();
        let type_key = InternedKey::from_str(rel_type);
        let property_key = InternedKey::from_str(property);
        {
            for edge in self.graph.edge_references() {
                if edge.connection_type() != type_key {
                    continue;
                }
                let id = edge.id();
                if let Some(value) = self.graph.get_edge_property(id, property_key) {
                    index.entry(value).or_default().push(edge_slot(id));
                }
            }
        }
        let distinct = index.len();
        self.rel.indices.insert(
            (rel_type.to_string(), property.to_string()),
            LayeredIndex::from(index),
        );
        distinct
    }

    /// Declare a relationship index: build it and record the declaration for
    /// the write-ahead log. The rebuild paths (load, `reindex`) call
    /// [`Self::build_rel_property_index`] directly, so a rebuild logs nothing.
    pub(crate) fn declare_rel_property_index(&mut self, rel_type: &str, property: &str) -> usize {
        let distinct = self.build_rel_property_index(rel_type, property);
        self.note_index_declaration(
            rel_type,
            vec![property.to_string()],
            PropertyIndexKind::Relationship,
            true,
        );
        distinct
    }

    /// Withdraw a relationship index and record the withdrawal. Reports whether
    /// one went; records nothing when none did.
    pub(crate) fn withdraw_rel_property_index(&mut self, rel_type: &str, property: &str) -> bool {
        let dropped = self.drop_rel_property_index(rel_type, property);
        if dropped {
            self.note_index_declaration(
                rel_type,
                vec![property.to_string()],
                PropertyIndexKind::Relationship,
                false,
            );
        }
        dropped
    }

    /// Drop the index on `(rel_type, property)`. Reports whether one went.
    pub(crate) fn drop_rel_property_index(&mut self, rel_type: &str, property: &str) -> bool {
        self.materialize_indexes();
        self.rel
            .indices
            .remove(&(rel_type.to_string(), property.to_string()))
            .is_some()
    }

    /// Rebuild every relationship index from the live edges, keeping which
    /// ones exist. Called by `reindex()` and the load path.
    pub(crate) fn rebuild_rel_property_indexes(&mut self) {
        let keys: Vec<IndexKey> = self.rel.indices.keys().cloned().collect();
        for (rel_type, property) in keys {
            self.build_rel_property_index(&rel_type, &property);
        }
    }

    /// The indexed keys of `rel_type`.
    fn rel_index_keys_of(&self, rel_type: &str) -> Vec<IndexKey> {
        self.rel
            .indices
            .keys()
            .filter(|(indexed_type, _)| indexed_type == rel_type)
            .cloned()
            .collect()
    }

    /// The type name of `edge`, when it is live.
    fn rel_type_of_edge(&self, edge: EdgeIndex) -> Option<String> {
        let key = self.graph.edge_weight(edge)?.connection_type;
        self.interner.try_resolve(key).map(str::to_string)
    }

    pub(crate) fn read_edge_value(&self, edge: EdgeIndex, property: &str) -> Option<Value> {
        self.graph
            .get_edge_property(edge, InternedKey::from_str(property))
    }

    /// File `edge` under `value` in the index `key`, journalling the append
    /// first so a rolled-back statement reverses it.
    fn file_rel_value(&mut self, key: &IndexKey, value: &Value, edge: EdgeIndex) {
        if matches!(value, Value::Null) {
            return;
        }
        let slot = edge_slot(edge);
        let Some(index) = self.rel.indices.get(key) else {
            return;
        };
        let bucket_was_new = !index.contains_key(value);
        if let Some(journal) = self.graph.undo_journal_mut() {
            journal.note_bucket_appended(
                BucketId::RelPropertyValue {
                    key: key.clone(),
                    value: value.clone(),
                },
                slot,
                bucket_was_new,
            );
        }
        if let Some(index) = self.rel.indices.get_mut(key) {
            index.entry_or_default(value).push(slot);
        }
    }

    /// Remove `edge` from the `value` bucket of the index `key`, journalling
    /// its position first.
    fn evict_rel_value(&mut self, key: &IndexKey, value: &Value, edge: EdgeIndex) {
        let slot = edge_slot(edge);
        let Some(pos) = self
            .rel
            .indices
            .get(key)
            .and_then(|index| index.get(value))
            .and_then(|members| members.iter().position(|member| *member == slot))
        else {
            return;
        };
        if let Some(journal) = self.graph.undo_journal_mut() {
            journal.note_bucket_removed(
                BucketId::RelPropertyValue {
                    key: key.clone(),
                    value: value.clone(),
                },
                slot,
                pos,
            );
        }
        if let Some(index) = self.rel.indices.get_mut(key) {
            let emptied = match index.get_mut(value) {
                Some(members) => {
                    members.remove(pos);
                    members.is_empty()
                }
                None => false,
            };
            if emptied {
                index.remove(value);
            }
        }
    }

    /// File a newly created edge into every index of its type. Called after
    /// the edge is in the graph.
    #[inline]
    pub(crate) fn file_created_edge(&mut self, edge: EdgeIndex) {
        if self.rel.indices.is_empty() {
            return;
        }
        self.file_created_edge_slow(edge);
    }

    fn file_created_edge_slow(&mut self, edge: EdgeIndex) {
        let Some(rel_type) = self.rel_type_of_edge(edge) else {
            return;
        };
        for key in self.rel_index_keys_of(&rel_type) {
            if let Some(value) = self.read_edge_value(edge, &key.1) {
                self.file_rel_value(&key, &value, edge);
            }
        }
    }

    /// Evict an edge that is about to be removed from every index of its type.
    /// Called while the edge is still in the graph.
    #[inline]
    pub(crate) fn evict_removed_edge(&mut self, edge: EdgeIndex) {
        if self.rel.indices.is_empty() {
            return;
        }
        self.evict_removed_edge_slow(edge);
    }

    fn evict_removed_edge_slow(&mut self, edge: EdgeIndex) {
        let Some(rel_type) = self.rel_type_of_edge(edge) else {
            return;
        };
        for key in self.rel_index_keys_of(&rel_type) {
            if let Some(value) = self.read_edge_value(edge, &key.1) {
                self.evict_rel_value(&key, &value, edge);
            }
        }
    }

    /// Capture the indexed values of `edge` before a property write. `None`
    /// when the edge's type has no index, which is also the fast-out.
    #[inline]
    pub(crate) fn rel_index_capture(&self, edge: EdgeIndex) -> Option<RelIndexPreImage> {
        if self.rel.indices.is_empty() {
            return None;
        }
        let rel_type = self.rel_type_of_edge(edge)?;
        let keys = self.rel_index_keys_of(&rel_type);
        if keys.is_empty() {
            return None;
        }
        let values = keys
            .into_iter()
            .map(|key| {
                let value = self.read_edge_value(edge, &key.1);
                (key, value)
            })
            .collect();
        Some(RelIndexPreImage { edge, values })
    }

    /// Move `edge` between buckets for every indexed property the write
    /// changed. Called after the write, with the capture taken before it.
    pub(crate) fn refresh_rel_indexes(&mut self, pre: Option<RelIndexPreImage>) {
        let Some(pre) = pre else {
            return;
        };
        for (key, old) in pre.values {
            let new = self.read_edge_value(pre.edge, &key.1);
            if old == new {
                continue;
            }
            if let Some(old) = &old {
                self.evict_rel_value(&key, old, pre.edge);
            }
            if let Some(new) = &new {
                self.file_rel_value(&key, new, pre.edge);
            }
        }
    }

    /// The live edges of `rel_type` holding `value` under `property`, from the
    /// index, or `None` when the index cannot prove the complete answer (no
    /// index, or a value whose equality domain a structural index cannot
    /// cover). The relationship twin of `lookup_by_index`.
    pub(crate) fn lookup_rel_index(
        &self,
        rel_type: &str,
        property: &str,
        value: &Value,
    ) -> Option<Vec<EdgeIndex>> {
        let key = (rel_type.to_string(), property.to_string());
        let index = self.rel.indices.get(&key)?;
        let keys = super::indexes::predicate_queries::equality_keys(value)?;
        if super::indexes::predicate_queries::temporal_probe_needs_scan(value, || {
            index.iter().map(|(key, _)| key)
        }) {
            return None;
        }
        let mut hits: Vec<NodeIndex> = Vec::new();
        for key in keys {
            if let Some(bucket) = index.get(&key) {
                hits.extend_from_slice(bucket);
            }
        }
        super::indexes::predicate_queries::dedup_hits(&mut hits);
        Some(hits.into_iter().map(slot_edge).collect())
    }

    /// Declared-but-unbuilt relationship indexes of a deferred load, listed
    /// beside the built ones.
    pub(crate) fn list_rel_property_indexes_with_state(
        &self,
    ) -> Vec<(String, String, super::indexes::IndexState)> {
        use super::indexes::IndexState;
        let mut listed: Vec<(String, String, IndexState)> = if self.indexes_deferred {
            self.rel
                .index_keys
                .iter()
                .cloned()
                .chain(self.deferred_owned_rel_slots())
                .map(|(t, p)| (t, p, IndexState::Deferred))
                .collect()
        } else {
            self.rel
                .indices
                .keys()
                .map(|(t, p)| (t.clone(), p.clone(), IndexState::Ready))
                .collect()
        };
        listed.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        listed
    }
}

#[cfg(test)]
#[path = "rel_index_tests.rs"]
pub(crate) mod tests;

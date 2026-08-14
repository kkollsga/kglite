//! The routed node-insert path for Cypher writes, split out of `mod.rs` to keep
//! it under the god-file ceiling.
//!
//! # This is NOT the graph's only create path
//!
//! `insert_node_routed` is the create path for **Cypher** `CREATE` and
//! `MERGE`-create, across all storage backends. Node creation has three funnels,
//! and anything that must hold for *every* new node has to be installed in all
//! three:
//!
//! 1. **`insert_node_routed`** (here) — Cypher `CREATE` / `MERGE`-create, via
//!    `languages/cypher/executor/write.rs::create_node`.
//! 2. **`mutation::batch::BatchProcessor::flush_chunk`** — every DataFrame-shaped
//!    ingest path: `add_nodes`, the blueprint builder, `from_records`, OKF, WAL
//!    replay, `extend_graph`, and edge stub vivification. It calls
//!    `GraphWrite::add_node` directly and never reaches this function.
//! 3. **Direct `GraphWrite::add_node`** in the RDF and N-Triples loaders
//!    (`io/rdf/loader.rs`, `io/ntriples/loader.rs`) and in `embedding_carry`.
//!
//! An earlier version of this comment claimed funnel 1 covered `add_nodes` too.
//! It does not, and believing it does is how a write-time guarantee ends up
//! enforced on Cypher only while the bulk path walks past it.
//!
//! Freshness provenance (`auto_timestamp`) is stamped here for funnel 1;
//! `mutation::maintain::add_nodes` stamps it independently for funnel 2.
//!
//! # One shape, every backend
//!
//! This function used to fork: disk pushed the node's id/title/properties
//! through the type's `ColumnStore`, memory and mapped built a row-shaped
//! node instead (the since-deleted `PropertyStorage::Compact`) and only became
//! columnar when `save()` rebuilt them. A graph therefore *changed write regime* the first
//! time it was saved, and every columnar defect was reachable only on a graph
//! that had been through that door.
//!
//! Every backend now takes the columnar branch: one row appended per node, id
//! and title in the store's reserved columns, and the inline `NodeData` fields
//! nulled to the same sentinel `enable_columnar` and the load path leave behind.
//! What `save()` does to a graph's shape is nothing.
//!
//! Funnels 1 and 2 both build that shape; funnel 3 is the residue. The
//! N-Triples loader consolidates once at the end of its build, so its output is
//! the same shape; the RDF loader and `embedding_carry` still add nodes holding
//! their properties inline, which the first consolidation pass converts. `Map`
//! staging like theirs is expected to survive — what the convergence removes is
//! the *row-shaped end state*, not every transient.

use std::collections::HashMap;
use std::sync::Arc;

use petgraph::graph::NodeIndex;

use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::schema::{ColumnarRow, InternedKey, NodeData, PropertyStorage};
use crate::graph::storage::undo::ColumnarAppendPreImage;
use crate::graph::storage::GraphWrite;

impl DirGraph {
    pub fn insert_node_routed(
        &mut self,
        id: Value,
        title: Value,
        node_type: &str,
        mut properties: HashMap<String, Value>,
    ) -> NodeIndex {
        // Freshness provenance: stamp `updated_at` (+ git_sha in phase 3) when
        // this type opted into `auto_timestamp`. Single chokepoint for every
        // create route — Cypher CREATE, `add_nodes`, and MERGE-create all land
        // here. A no-op for types that didn't opt in.
        self.inject_provenance(node_type, &mut properties);

        // Register property types in `node_type_metadata` from the values in
        // hand, because the store's column typing reads them and a column typed
        // `Mixed` cannot be spilled. Do NOT read the node back for this: on disk
        // the columnar store isn't synced to the read side until the end of the
        // clause, so a read-back would see no properties — and the
        // metadata-driven column persistence would then drop them on save.
        self.register_property_types(node_type, &properties);

        // Pre-intern property keys (and node type) before borrowing stores.
        let interned_props: Vec<(InternedKey, Value)> = properties
            .iter()
            .map(|(k, v)| (self.interner.get_or_intern(k), v.clone()))
            .collect();
        // Sort for a deterministic schema slot order — `properties` HashMap
        // iteration order is randomized per process, which would otherwise make
        // the saved column order (and the compressed `.kgl` bytes)
        // non-reproducible. `InternedKey`'s FNV hash is stable across
        // processes/versions.
        let mut keys: Vec<InternedKey> = interned_props.iter().map(|(k, _)| *k).collect();
        keys.sort_unstable_by_key(|k| k.as_u64());
        self.ensure_type_schema_keys(node_type, &keys);

        let row_id = self.push_node_row(node_type, &id, &title, &interned_props);

        let node_type_key = self.interner.get_or_intern(node_type);
        // id/title live in the store's reserved `__id__`/`__title__` columns
        // (pushed above), so the inline fields carry the `Null` sentinel every
        // other columnar producer leaves — `enable_columnar`, the bulk batch
        // funnel, and the `.kgl` load path. Keeping a second copy inline would
        // serialize every id and title twice (~27 B/node) and would make
        // `enable_columnar`'s drift check see a title override on every node.
        let node_data = NodeData {
            id: Value::Null,
            title: Value::Null,
            node_type: node_type_key,
            properties: PropertyStorage::Columnar(ColumnarRow::new(row_id)),
        };
        let idx = GraphWrite::add_node(&mut self.graph, node_data);
        // A no-op on the heap backends (the row id is already in the node's
        // `PropertyStorage`); on disk it re-stamps the slot, which is where
        // disk reads resolve the row from.
        GraphWrite::update_row_id(&mut self.graph, idx, row_id);
        idx
    }

    /// Append one node's row to its type's master store, journalling the
    /// pre-image a statement rollback needs.
    ///
    /// The store must be created (and, if it is still reading through an mmap
    /// base, materialized) *before* the pre-image is taken: the materialization
    /// re-derives the store's columns, so a pre-image captured across it would
    /// restore a schema that no longer describes the columns it names.
    fn push_node_row(
        &mut self,
        node_type: &str,
        id: &Value,
        title: &Value,
        interned_props: &[(InternedKey, Value)],
    ) -> u32 {
        let store_was_new = self.column_store(node_type).is_none();
        self.ensure_column_store_for_push(node_type);
        if self.graph.undo_journal_mut().is_some() {
            let type_key = InternedKey::from_str(node_type);
            let captured = self
                .column_store(node_type)
                .map(|store| ColumnarAppendPreImage::capture(store));
            if let (Some(captured), Some(journal)) = (captured, self.graph.undo_journal_mut()) {
                captured.record(journal, type_key, store_was_new);
            }
        }
        // `ensure_column_store_for_push` already ran; take the mutable handle
        // directly rather than paying its existence and mmap-base checks twice
        // on a path that runs once per node created.
        let store = Arc::make_mut(
            self.column_store_mut(node_type)
                .expect("ensure_column_store_for_push installed it"),
        );
        store.push_id(id);
        store.push_title(title);
        store.push_row(interned_props)
    }

    /// Record the property types this type has not registered yet.
    ///
    /// Declared metadata wins: an entry that already exists is left alone, so a
    /// `define_schema`'d `float64` survives the first row that happens to carry
    /// an integer — a column typed from the wrong value is a column the next
    /// write demotes to `Mixed`, and `Mixed` is the one shape that cannot be
    /// spilled. `Null` carries no type evidence and registers nothing, matching
    /// what the read-back-based `ensure_type_metadata` records (it enumerates
    /// through `NodeView`, which skips nulls).
    ///
    /// The common case — every key already registered — costs one hash lookup
    /// per property and allocates nothing.
    fn register_property_types(&mut self, node_type: &str, properties: &HashMap<String, Value>) {
        let known = self.node_type_metadata.get(node_type);
        let missing: Vec<(String, String)> = properties
            .iter()
            .filter(|(_, value)| !matches!(value, Value::Null))
            .filter(|(key, _)| known.is_none_or(|props| !props.contains_key(*key)))
            .map(|(key, value)| (key.clone(), value.type_name().to_string()))
            .collect();
        if missing.is_empty() {
            return;
        }
        self.upsert_node_type_metadata(node_type, missing.into_iter().collect());
    }
}

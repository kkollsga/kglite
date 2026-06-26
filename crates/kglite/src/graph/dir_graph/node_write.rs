//! The routed node-insert chokepoint, split out of `mod.rs` to keep it under
//! the god-file ceiling. `insert_node_routed` is the single create path for
//! Cypher CREATE / `add_nodes` / MERGE-create across all storage backends, so
//! it's also where freshness provenance (`auto_timestamp`) is stamped.

use std::collections::HashMap;
use std::sync::Arc;

use petgraph::graph::NodeIndex;

use super::DirGraph;
use crate::datatypes::values::Value;
use crate::graph::schema::{InternedKey, NodeData, PropertyStorage};
use crate::graph::storage::{GraphRead, GraphWrite};

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
        if self.graph.is_disk() {
            // Register property types in node_type_metadata from the values we
            // have in hand. Do NOT read the node back for this: on disk the
            // columnar store isn't synced to the read-side (`dg.column_stores`)
            // until the end of the clause, so a read-back would see no properties
            // — and the metadata-driven column persistence would then drop them on
            // save (properties survive in-memory but vanish after save/reload).
            // Merge-upsert, so it composes with any later `ensure_type_metadata`.
            //
            // Memory/mapped skip this: the caller (`create_node`) runs
            // `ensure_type_metadata` against the read-back node, which produces
            // identical metadata. Doing both was redundant per-node work — the
            // bulk-CREATE regression introduced in 0.10.17.
            let prop_types: HashMap<String, String> = properties
                .iter()
                .map(|(k, v)| (k.clone(), v.type_name().to_string()))
                .collect();
            self.upsert_node_type_metadata(node_type, prop_types);

            // Pre-intern property keys (and node type) before borrowing stores.
            let interned_props: Vec<(InternedKey, Value)> = properties
                .iter()
                .map(|(k, v)| (self.interner.get_or_intern(k), v.clone()))
                .collect();
            // Sort for a deterministic schema slot order (see the memory branch
            // below) — `properties` HashMap iteration order is randomized.
            let mut keys: Vec<InternedKey> = interned_props.iter().map(|(k, _)| *k).collect();
            keys.sort_unstable_by_key(|k| k.as_u64());
            self.ensure_type_schema_keys(node_type, &keys);

            let row_id = {
                let store = self.ensure_column_store_for_push(node_type);
                store.push_id(&id);
                store.push_title(&title);
                store.push_row(&interned_props)
            };
            // The Arc borrow above has ended; clone the (now-extended) store
            // handle for the node's Columnar pointer.
            let store_arc = Arc::clone(
                self.column_stores
                    .get(node_type)
                    .expect("ensure_column_store_for_push just inserted it"),
            );
            let node_type_key = self.interner.get_or_intern(node_type);
            // id/title live in the ColumnStore (pushed above); the disk
            // `add_node` drops NodeData.id/title anyway and reads row_id out of
            // the Columnar variant. update_row_id re-stamps it for parity with
            // the bulk path (harmless if already correct).
            let node_data = NodeData {
                id,
                title,
                node_type: node_type_key,
                properties: PropertyStorage::Columnar {
                    store: store_arc,
                    row_id,
                },
            };
            let idx = GraphWrite::add_node(&mut self.graph, node_data);
            GraphWrite::update_row_id(&mut self.graph, idx, row_id);
            idx
        } else {
            // Memory / mapped: Compact NodeData on the shared TypeSchema.
            // Sort keys for a deterministic schema slot order — `properties`
            // HashMap iteration is randomized per process, which would make the
            // saved column order (and compressed .kgl bytes) non-reproducible.
            // InternedKey's FNV hash is stable across processes/versions.
            let mut interned_keys: Vec<InternedKey> = properties
                .keys()
                .map(|k| self.interner.get_or_intern(k))
                .collect();
            interned_keys.sort_unstable_by_key(|k| k.as_u64());
            self.ensure_type_schema_keys(node_type, &interned_keys);
            let schema = Arc::clone(
                self.type_schemas
                    .get(node_type)
                    .expect("ensure_type_schema_keys just inserted it"),
            );
            let node_data = NodeData::new_compact(
                id,
                title,
                node_type.to_string(),
                properties,
                &mut self.interner,
                &schema,
            );
            GraphWrite::add_node(&mut self.graph, node_data)
        }
    }
}

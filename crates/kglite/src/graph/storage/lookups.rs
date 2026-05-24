// src/graph/storage/lookups.rs
use crate::datatypes::Value;
use crate::graph::schema::{GraphBackend, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeLookup {
    uid_to_index: HashMap<Value, NodeIndex>,
    title_to_index: HashMap<Value, NodeIndex>,
    node_type: String,
}

impl TypeLookup {
    pub fn new(graph: &GraphBackend, node_type: String) -> Result<Self, String> {
        if node_type.is_empty() {
            return Err("Node type cannot be empty".to_string());
        }

        let mut uid_to_index = HashMap::new();
        let mut title_to_index = HashMap::new();

        // Single pass through the graph
        for i in graph.node_indices() {
            if let Some(node_data) = graph.node_weight(i) {
                if node_data.node_type == InternedKey::from_str(&node_type) {
                    uid_to_index.insert(node_data.id().into_owned(), i);
                    title_to_index.insert(node_data.title().into_owned(), i);
                }
            }
        }

        Ok(TypeLookup {
            uid_to_index,
            title_to_index,
            node_type,
        })
    }

    /// Fast constructor using pre-built id_indices from DirGraph (avoids full-graph scan).
    /// Falls back to graph scan if id_indices has been invalidated (e.g., after node deletion).
    /// Does not build title_to_index since it's unused in the add_nodes hot path.
    pub fn from_id_indices(
        id_indices: &crate::graph::storage::disk::id_index::IdIndexStore,
        graph: &GraphBackend,
        node_type: String,
    ) -> Result<Self, String> {
        if node_type.is_empty() {
            return Err("Node type cannot be empty".to_string());
        }
        if let Some(type_index) = id_indices.get(&node_type) {
            let uid_map: HashMap<Value, NodeIndex> = type_index.iter().collect();
            Ok(TypeLookup {
                uid_to_index: uid_map,
                title_to_index: HashMap::new(),
                node_type,
            })
        } else {
            // id_indices invalidated — fall back to graph scan
            Self::new(graph, node_type)
        }
    }

    pub fn check_uid(&self, uid: &Value) -> Option<NodeIndex> {
        CombinedTypeLookup::lookup_with_type_fallback(&self.uid_to_index, uid)
    }

    pub fn check_title(&self, title: &Value) -> Option<NodeIndex> {
        self.title_to_index.get(title).copied()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedTypeLookup {
    source_uid_to_index: HashMap<Value, NodeIndex>,
    /// Only populated when source and target types differ (None when same_type is true)
    target_uid_to_index: Option<HashMap<Value, NodeIndex>>,
    source_type: String,
    target_type: String,
    same_type: bool,
}

impl CombinedTypeLookup {
    pub fn new(
        graph: &GraphBackend,
        source_type: String,
        target_type: String,
    ) -> Result<Self, String> {
        if source_type.is_empty() || target_type.is_empty() {
            return Err("Node types cannot be empty".to_string());
        }

        let same_type = source_type == target_type;
        let mut source_uid_to_index = HashMap::new();
        let mut target_uid_to_index_map: Option<HashMap<Value, NodeIndex>> = if same_type {
            None // Don't allocate separate map when types are the same
        } else {
            Some(HashMap::new())
        };

        // Single pass through graph - collect both source and target if different types
        for idx in graph.node_indices() {
            if let Some(node_data) = graph.node_weight(idx) {
                if node_data.node_type == InternedKey::from_str(&source_type) {
                    source_uid_to_index.insert(node_data.id().into_owned(), idx);
                }
                // Also collect target type in same pass (if different from source)
                if let Some(ref mut target_map) = target_uid_to_index_map {
                    if node_data.node_type == InternedKey::from_str(&target_type) {
                        target_map.insert(node_data.id().into_owned(), idx);
                    }
                }
            }
        }

        Ok(CombinedTypeLookup {
            source_uid_to_index,
            target_uid_to_index: target_uid_to_index_map,
            source_type,
            target_type,
            same_type,
        })
    }

    /// Fast constructor using pre-built id_indices from DirGraph (avoids full-graph scan).
    /// Falls back to graph scan if id_indices has been invalidated for either type.
    pub fn from_id_indices(
        id_indices: &crate::graph::storage::disk::id_index::IdIndexStore,
        graph: &GraphBackend,
        source_type: String,
        target_type: String,
    ) -> Result<Self, String> {
        if source_type.is_empty() || target_type.is_empty() {
            return Err("Node types cannot be empty".to_string());
        }
        let same_type = source_type == target_type;
        let has_source = id_indices.contains_key(&source_type);
        let has_target = same_type || id_indices.contains_key(&target_type);

        if has_source && has_target {
            let source_uid = id_indices
                .get(&source_type)
                .map(|idx| idx.iter().collect())
                .unwrap_or_default();
            let target_uid = if same_type {
                None
            } else {
                Some(
                    id_indices
                        .get(&target_type)
                        .map(|idx| idx.iter().collect())
                        .unwrap_or_default(),
                )
            };
            Ok(CombinedTypeLookup {
                source_uid_to_index: source_uid,
                target_uid_to_index: target_uid,
                source_type,
                target_type,
                same_type,
            })
        } else {
            // id_indices invalidated — fall back to graph scan
            Self::new(graph, source_type, target_type)
        }
    }

    pub fn check_source(&self, uid: &Value) -> Option<NodeIndex> {
        Self::lookup_with_type_fallback(&self.source_uid_to_index, uid)
    }

    pub fn check_target(&self, uid: &Value) -> Option<NodeIndex> {
        // Reuse source map when types are the same (avoids clone)
        let map = self
            .target_uid_to_index
            .as_ref()
            .unwrap_or(&self.source_uid_to_index);
        Self::lookup_with_type_fallback(map, uid)
    }

    /// Helper function to handle Int64/UniqueId/Float64 type mismatches during lookup.
    ///
    /// IDs in CSV sources sometimes arrive as floats (e.g. 260.0 instead of 260)
    /// due to pandas nullable-int promotion.  This method tries all plausible
    /// numeric representations so that a Float64(260.0) matches an Int64(260) node.
    fn lookup_with_type_fallback(
        map: &HashMap<Value, NodeIndex>,
        uid: &Value,
    ) -> Option<NodeIndex> {
        // First try direct lookup
        if let Some(idx) = map.get(uid).copied() {
            return Some(idx);
        }

        match uid {
            Value::Float64(f) => {
                // Float that is a whole number → try Int64 and UniqueId
                if f.is_finite() && f.fract() == 0.0 {
                    let i = *f as i64;
                    if let Some(idx) = map.get(&Value::Int64(i)).copied() {
                        return Some(idx);
                    }
                    if i >= 0 && i <= u32::MAX as i64 {
                        return map.get(&Value::UniqueId(i as u32)).copied();
                    }
                }
                None
            }
            Value::Int64(i) => {
                // Try UniqueId, then Float64
                if *i >= 0 && *i <= u32::MAX as i64 {
                    if let Some(idx) = map.get(&Value::UniqueId(*i as u32)).copied() {
                        return Some(idx);
                    }
                }
                map.get(&Value::Float64(*i as f64)).copied()
            }
            Value::UniqueId(u) => {
                // Try Int64, then Float64
                if let Some(idx) = map.get(&Value::Int64(*u as i64)).copied() {
                    return Some(idx);
                }
                map.get(&Value::Float64(*u as f64)).copied()
            }
            _ => None,
        }
    }

    pub fn get_source_type(&self) -> &str {
        &self.source_type
    }

    pub fn get_target_type(&self) -> &str {
        &self.target_type
    }
}

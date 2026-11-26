// src/graph/storage/lookups.rs
use crate::datatypes::Value;
use crate::graph::schema::{GraphBackend, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeLookup {
    /// FxHash, not the std SipHasher, for every id map in this module. These
    /// are the `add_nodes` conflict-check maps: one probe per incoming row
    /// against a map sized to the whole type, on the bulk-ingest hot path.
    /// `Value::hash` writes a discriminant then the payload, both of which Fx
    /// mixes as cheaply as any integer key.
    uid_to_index: FxHashMap<Value, NodeIndex>,
    title_to_index: FxHashMap<Value, NodeIndex>,
    node_type: String,
}

impl TypeLookup {
    pub fn new(graph: &GraphBackend, node_type: String) -> Result<Self, String> {
        if node_type.is_empty() {
            return Err("Node type cannot be empty".to_string());
        }

        let mut uid_to_index = FxHashMap::default();
        let mut title_to_index = FxHashMap::default();

        // Single pass through the graph
        for i in graph.node_indices() {
            if let Some(node_data) = graph.node_view(i) {
                if node_data.node_type() == InternedKey::from_str(&node_type) {
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

    pub fn check_uid(&self, uid: &Value) -> Option<NodeIndex> {
        CombinedTypeLookup::lookup_with_type_fallback(&self.uid_to_index, uid)
    }

    pub fn check_title(&self, title: &Value) -> Option<NodeIndex> {
        self.title_to_index.get(title).copied()
    }
}

// Test-only tally of the whole-graph scans that build an endpoint lookup, so a
// test can prove a resolution was answered from an id index instead.
#[cfg(test)]
thread_local! {
    static GRAPH_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn graph_scans() -> usize {
    GRAPH_SCANS.with(|scans| scans.get())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedTypeLookup {
    source_uid_to_index: FxHashMap<Value, NodeIndex>,
    /// Only populated when source and target types differ (None when same_type is true)
    target_uid_to_index: Option<FxHashMap<Value, NodeIndex>>,
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
        #[cfg(test)]
        GRAPH_SCANS.with(|scans| scans.set(scans.get() + 1));

        let same_type = source_type == target_type;
        let mut source_uid_to_index = FxHashMap::default();
        let mut target_uid_to_index_map: Option<FxHashMap<Value, NodeIndex>> = if same_type {
            None // Don't allocate separate map when types are the same
        } else {
            Some(FxHashMap::default())
        };

        // Single pass through graph - collect both source and target if different types
        for idx in graph.node_indices() {
            if let Some(node_data) = graph.node_view(idx) {
                if node_data.node_type() == InternedKey::from_str(&source_type) {
                    source_uid_to_index.insert(node_data.id().into_owned(), idx);
                }
                // Also collect target type in same pass (if different from source)
                if let Some(ref mut target_map) = target_uid_to_index_map {
                    if node_data.node_type() == InternedKey::from_str(&target_type) {
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

    fn lookup_with_type_fallback(
        map: &FxHashMap<Value, NodeIndex>,
        uid: &Value,
    ) -> Option<NodeIndex> {
        lookup_coerced(|v| map.get(v).copied(), uid)
    }
}

/// Resolve `uid` through `get`, then through the other keys a numeric id may
/// have been stored under.
///
/// IDs in CSV sources sometimes arrive as floats (e.g. 260.0 instead of 260)
/// due to pandas nullable-int promotion, so a `Float64(260.0)` must match an
/// `Int64(260)` node. The spellings come from `id_spellings`, which offers a
/// float only where it is exactly the integer: past 2^53 an `Int64` rounds
/// onto a neighbour's float, and a whole float past `i64` saturates onto
/// `i64::MAX`, so neither is ever probed as another id.
pub(crate) fn lookup_coerced(
    get: impl Fn(&Value) -> Option<NodeIndex>,
    uid: &Value,
) -> Option<NodeIndex> {
    get(uid).or_else(|| crate::graph::schema::id_spellings(uid).find_map(|key| get(&key)))
}

/// Resolves the endpoint ids of one `(source_type, target_type)` pair to nodes.
///
/// When both types have an id index — heap, mapped, or a delta over a mapped
/// entry — each id is one probe of that index, so the cost is the number of
/// ids resolved and never the size of either type: on a disk graph a type's
/// index stays in its mapping and is never copied onto the heap to answer a
/// batch of edges. When either index is missing, one scan of the graph builds
/// both maps.
pub enum EndpointResolver<'a> {
    Indexed {
        ids: &'a crate::graph::storage::disk::id_index::IdIndexStore,
        source_type: String,
        target_type: String,
    },
    Scanned(CombinedTypeLookup),
}

impl<'a> EndpointResolver<'a> {
    pub fn new(
        ids: &'a crate::graph::storage::disk::id_index::IdIndexStore,
        graph: &GraphBackend,
        source_type: String,
        target_type: String,
    ) -> Result<Self, String> {
        if source_type.is_empty() || target_type.is_empty() {
            return Err("Node types cannot be empty".to_string());
        }
        if ids.contains_key(&source_type) && ids.contains_key(&target_type) {
            return Ok(Self::Indexed {
                ids,
                source_type,
                target_type,
            });
        }
        CombinedTypeLookup::new(graph, source_type, target_type).map(Self::Scanned)
    }

    pub fn check_source(&self, uid: &Value) -> Option<NodeIndex> {
        match self {
            Self::Indexed {
                ids, source_type, ..
            } => ids.lookup(source_type, uid),
            Self::Scanned(lookup) => lookup.check_source(uid),
        }
    }

    pub fn check_target(&self, uid: &Value) -> Option<NodeIndex> {
        match self {
            Self::Indexed {
                ids, target_type, ..
            } => ids.lookup(target_type, uid),
            Self::Scanned(lookup) => lookup.check_target(uid),
        }
    }
}

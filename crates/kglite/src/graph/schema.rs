use crate::datatypes::values::{FilterCondition, Value};
pub use crate::graph::storage::interner::{InternedKey, StringInterner};
pub(crate) use crate::graph::storage::interner::{
    SerdeDeserializeGuard, SerdeSerializeGuard, StripPropertiesGuard,
};
use crate::graph::storage::GraphRead;
// `PropertyStorage` lives under `graph::storage` so a columnar node's store
// handle cannot be reached from outside the storage layer; re-exported here
// to preserve the `crate::graph::schema::PropertyStorage` import path.
pub(crate) use crate::graph::storage::property_storage::{ColumnarRow, PropertyStorage};

// Re-exported here to preserve the `crate::graph::schema::X` import paths.
pub use crate::graph::dir_graph::DirGraph;
use crate::graph::dir_graph::NodeRemap;
pub use crate::graph::storage::backend::GraphBackend;
// MemoryGraph re-export: required by `storage/recording.rs` tests.
// DO NOT REMOVE even if cargo fix suggests it.
#[allow(unused_imports)]
pub use crate::graph::storage::{MappedGraph, MemoryGraph};
use petgraph::graph::NodeIndex;
use rustc_hash::FxHashMap;

/// Engine-managed freshness-provenance keys. Stamped by the engine on writes to
/// `auto_timestamp`-opted types (never user-supplied); queryable directly
/// (`n.updated_at`, `r.updated_at`), but **hidden from property enumerations**
/// (`keys`/`properties`/`RETURN n`/`RETURN n.*`/`describe`) so they read as
/// metadata, not user data.
pub const RESERVED_PROVENANCE_KEYS: &[&str] = &["updated_at", "git_sha", "modified_by"];

#[inline]
pub fn is_reserved_provenance_key(key: &str) -> bool {
    RESERVED_PROVENANCE_KEYS.contains(&key)
}
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

/// Reserved property name marking an auto-vivified stub node — one
/// created to satisfy an edge whose endpoint had no row of its own.
/// Set to `true` at vivification; cleared (`Null`) when the real node
/// row later upserts the node ("promotion"). `purge_provisional()`
/// deletes whatever is still marked. Reserved: a user-supplied
/// property of this name is rejected at blueprint validation.
pub const PROVISIONAL_KEY: &str = "_provisional";

/// Shared schema for all nodes of one type — maps property keys to dense slot indices.
/// Held once per type by that type's `ColumnStore`; a node carries only its row
/// index (see `ColumnarRow`), never a schema handle of its own.
#[derive(Debug, Clone)]
pub struct TypeSchema {
    /// slot_index → interned key (for iteration / serialization)
    pub(crate) slots: Vec<InternedKey>,
    /// interned key → slot_index (for O(1) lookup). FxHash, not the std
    /// SipHasher: `InternedKey` is already a well-distributed FNV `u64`, so a
    /// cryptographic hash is pure overhead. `slot()` is the per-property,
    /// per-row, per-column lookup on the columnar read path —
    /// samply (2026-05-29) showed SipHash here at ~23% of in-memory query CPU.
    key_to_slot: FxHashMap<InternedKey, u16>,
}

impl TypeSchema {
    pub fn new() -> Self {
        TypeSchema {
            slots: Vec::new(),
            key_to_slot: FxHashMap::default(),
        }
    }

    pub fn from_keys(keys: impl IntoIterator<Item = InternedKey>) -> Self {
        let mut schema = TypeSchema::new();
        for key in keys {
            if !schema.key_to_slot.contains_key(&key) {
                let slot = schema.slots.len() as u16;
                schema.slots.push(key);
                schema.key_to_slot.insert(key, slot);
            }
        }
        schema
    }

    #[inline]
    pub fn slot(&self, key: InternedKey) -> Option<u16> {
        self.key_to_slot.get(&key).copied()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn merge(&self, other: &TypeSchema) -> TypeSchema {
        let mut merged = self.clone();
        for &key in &other.slots {
            merged.add_key(key);
        }
        merged
    }

    /// Returns the key's slot index, adding the key if it is not present.
    pub fn add_key(&mut self, key: InternedKey) -> u16 {
        if let Some(&slot) = self.key_to_slot.get(&key) {
            slot
        } else {
            let slot = self.slots.len() as u16;
            self.slots.push(key);
            self.key_to_slot.insert(key, slot);
            slot
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (u16, InternedKey)> + '_ {
        self.slots.iter().enumerate().map(|(i, &k)| (i as u16, k))
    }
}

/// Serialize a `HashMap` with key-sorted entries. `.kgl` bytes are contractually
/// reproducible (equivalent graphs → identical files; see the `test_phase4_parity`
/// golden-hash test and the byte-determinism regression in `io/file_tests.rs`),
/// but HashMap's per-process `RandomState` randomizes iteration order, so any
/// map serialized raw breaks that. Wire-compatible in both directions: the
/// payload is the same length-prefixed entry sequence, just ordered, and
/// `HashMap` deserialization accepts any order.
pub(crate) fn serialize_sorted_map<K, V, S>(
    map: &HashMap<K, V>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    K: Ord + Serialize + std::hash::Hash + Eq,
    V: Serialize,
    S: Serializer,
{
    let mut entries: Vec<(&K, &V)> = map.iter().collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
    serializer.collect_map(entries)
}

/// Spatial configuration for a node type. Declares which properties hold
/// spatial data (lat/lon pairs, WKT geometries) and enables auto-resolution
/// in Cypher `distance(a, b)` and fluent API methods.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SpatialConfig {
    /// Primary lat/lon location: (lat_field, lon_field). At most one per type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<(String, String)>,
    /// Primary WKT geometry field name. At most one per type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<String>,
    /// Named lat/lon points: name → (lat_field, lon_field). Zero or more.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub points: HashMap<String, (String, String)>,
    /// Named WKT shape fields: name → field_name. Zero or more.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub shapes: HashMap<String, String>,
}

/// Result of column-type parsing: optional config + cleaned (col_name, type_str) pairs.
pub type SpatialColumnParseResult = (Option<SpatialConfig>, Vec<(String, String)>);

/// Parse spatial column-type entries from the wheel's `column_types`
/// dict shape. Recognizes `location.lat`, `location.lon`, `geometry`,
/// `point.<name>.lat`, `point.<name>.lon`, `shape.<name>`. Returns
/// `(Some(config), cleaned_pairs)` if any spatial entries were found,
/// `(None, original_pairs)` otherwise. The cleaned pairs replace
/// recognized spatial type strings with their natural storage types
/// (`float` for lat/lon, `str` for WKT shapes) so downstream
/// dataframe loaders treat them correctly.
///
/// The wheel keeps only the `Bound<PyDict>` → `Vec<(String, String)>`
/// extraction wrapper.
pub fn parse_spatial_column_types_from_pairs(
    pairs: Vec<(String, String)>,
) -> Result<SpatialColumnParseResult, String> {
    let mut cleaned: Vec<(String, String)> = Vec::with_capacity(pairs.len());
    let mut config = SpatialConfig::default();
    let mut has_spatial = false;

    let mut location_lat: Option<String> = None;
    let mut location_lon: Option<String> = None;
    let mut point_lats: HashMap<String, String> = HashMap::new();
    let mut point_lons: HashMap<String, String> = HashMap::new();

    for (col_name, type_str) in pairs {
        let type_lower = type_str.to_lowercase();
        match type_lower.as_str() {
            "location.lat" => {
                location_lat = Some(col_name.clone());
                cleaned.push((col_name, "float".to_string()));
                has_spatial = true;
            }
            "location.lon" => {
                location_lon = Some(col_name.clone());
                cleaned.push((col_name, "float".to_string()));
                has_spatial = true;
            }
            "geometry" => {
                config.geometry = Some(col_name.clone());
                cleaned.push((col_name, "str".to_string()));
                has_spatial = true;
            }
            _ if type_lower.starts_with("point.") => {
                let parts: Vec<&str> = type_lower.splitn(3, '.').collect();
                if parts.len() == 3 {
                    let name = parts[1].to_string();
                    match parts[2] {
                        "lat" => {
                            point_lats.insert(name, col_name.clone());
                        }
                        "lon" => {
                            point_lons.insert(name, col_name.clone());
                        }
                        _ => {
                            return Err(format!(
                                "Invalid spatial type '{}' for column '{}'. \
                                 Expected 'point.<name>.lat' or 'point.<name>.lon'.",
                                type_str, col_name
                            ));
                        }
                    }
                    cleaned.push((col_name, "float".to_string()));
                    has_spatial = true;
                } else {
                    return Err(format!(
                        "Invalid spatial type '{}' for column '{}'. \
                         Expected 'point.<name>.lat' or 'point.<name>.lon'.",
                        type_str, col_name
                    ));
                }
            }
            _ if type_lower.starts_with("shape.") => {
                let parts: Vec<&str> = type_lower.splitn(2, '.').collect();
                if parts.len() == 2 {
                    let name = parts[1].to_string();
                    config.shapes.insert(name, col_name.clone());
                    cleaned.push((col_name, "str".to_string()));
                    has_spatial = true;
                } else {
                    return Err(format!(
                        "Invalid spatial type '{}' for column '{}'.",
                        type_str, col_name
                    ));
                }
            }
            _ => {
                cleaned.push((col_name, type_str));
            }
        }
    }

    if !has_spatial {
        return Ok((None, cleaned));
    }

    match (location_lat, location_lon) {
        (Some(lat), Some(lon)) => config.location = Some((lat, lon)),
        (Some(_), None) | (None, Some(_)) => {
            return Err(
                "Incomplete location: both 'location.lat' and 'location.lon' must be specified."
                    .to_string(),
            );
        }
        (None, None) => {}
    }

    let all_point_names: std::collections::HashSet<&String> =
        point_lats.keys().chain(point_lons.keys()).collect();
    for name in all_point_names {
        match (point_lats.get(name), point_lons.get(name)) {
            (Some(lat), Some(lon)) => {
                config
                    .points
                    .insert(name.clone(), (lat.clone(), lon.clone()));
            }
            _ => {
                return Err(format!(
                    "Incomplete point '{}': both 'point.{}.lat' and 'point.{}.lon' must be specified.",
                    name, name, name
                ));
            }
        }
    }

    Ok((Some(config), cleaned))
}

/// Temporal configuration for a node type or connection type.
/// Declares which properties hold validity-period dates (valid_from, valid_to).
/// When configured, temporal filtering is applied automatically in
/// `select()` (for nodes) and `traverse()` (for connections).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TemporalConfig {
    /// Property name holding the start date, e.g. "fldLicenseeFrom" or "date_from"
    pub valid_from: String,
    /// Property name holding the end date, e.g. "fldLicenseeTo" or "date_to"
    pub valid_to: String,
}

/// Result of temporal column-type parsing: optional config + cleaned pairs.
pub type TemporalColumnParseResult = (Option<TemporalConfig>, Vec<(String, String)>);

/// Parse temporal column-type entries from the wheel's `column_types`
/// dict shape. Recognizes `validFrom` / `validTo` (case-insensitive).
/// Returns `(Some(config), cleaned_pairs)` if BOTH validFrom and
/// validTo are found; `(None, original_pairs)` if neither is found.
/// Returns `Err` if exactly one is found (asymmetric config is a
/// data-shape mistake — better to fail loudly).
pub fn parse_temporal_column_types_from_pairs(
    pairs: Vec<(String, String)>,
) -> Result<TemporalColumnParseResult, String> {
    let mut cleaned: Vec<(String, String)> = Vec::with_capacity(pairs.len());
    let mut valid_from_col: Option<String> = None;
    let mut valid_to_col: Option<String> = None;

    for (col_name, type_str) in pairs {
        let type_lower = type_str.to_lowercase();
        match type_lower.as_str() {
            "validfrom" => {
                valid_from_col = Some(col_name.clone());
                cleaned.push((col_name, "datetime".to_string()));
            }
            "validto" => {
                valid_to_col = Some(col_name.clone());
                cleaned.push((col_name, "datetime".to_string()));
            }
            _ => {
                cleaned.push((col_name, type_str));
            }
        }
    }

    match (valid_from_col, valid_to_col) {
        (Some(from), Some(to)) => Ok((
            Some(TemporalConfig {
                valid_from: from,
                valid_to: to,
            }),
            cleaned,
        )),
        (Some(_), None) | (None, Some(_)) => Err(
            "Incomplete temporal config: both 'validFrom' and 'validTo' column types must be specified."
                .to_string(),
        ),
        (None, None) => Ok((None, cleaned)),
    }
}

/// Per-type ID index. Uses compact u32 keys when all IDs are UniqueId,
/// falling back to general Value keys otherwise.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TypeIdIndex {
    /// All IDs are UniqueId(u32) — compact, ~8 bytes per entry.
    Integer(FxHashMap<u32, NodeIndex>),
    /// Mixed ID types — general, ~60 bytes per entry.
    General(FxHashMap<Value, NodeIndex>),
}

impl TypeIdIndex {
    /// Look up a node by ID value, with type coercion.
    pub fn get(&self, id: &Value) -> Option<NodeIndex> {
        match self {
            TypeIdIndex::Integer(map) => match id {
                Value::UniqueId(u) => map.get(u).copied(),
                Value::Int64(i) => {
                    if *i >= 0 && *i <= u32::MAX as i64 {
                        map.get(&(*i as u32)).copied()
                    } else {
                        None
                    }
                }
                Value::Float64(f) => {
                    if f.fract() == 0.0 {
                        let i = *f as i64;
                        if i >= 0 && i <= u32::MAX as i64 {
                            map.get(&(i as u32)).copied()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                // NB: no string→u32 coercion. A `String` id queried against a
                // UniqueId-keyed index does NOT match (e.g. `{id:'a1'}` must
                // not resolve to `UniqueId(1)`). Datasets that expose a
                // string id form (e.g. Wikidata `Q76`) store it as a queryable
                // property — query `{nid:'Q76'}`, not `{id:'Q76'}`.
                _ => None,
            },
            TypeIdIndex::General(map) => {
                if let Some(&idx) = map.get(id) {
                    return Some(idx);
                }
                // Type coercion fallback. This must cover the same numeric
                // family as `values_equal` (core/filtering.rs), which is what
                // a type scan compares with: the id anchors in
                // `try_index_lookup` treat a miss here as an empty result and
                // never scan, so a coercion the index declines but the scan
                // would have accepted is a lost row, not a slow one.
                match id {
                    Value::Int64(i) => {
                        if let Some(&idx) = map.get(&Value::Float64(*i as f64)) {
                            return Some(idx);
                        }
                        if *i >= 0 && *i <= u32::MAX as i64 {
                            map.get(&Value::UniqueId(*i as u32)).copied()
                        } else {
                            None
                        }
                    }
                    Value::UniqueId(u) => {
                        if let Some(&idx) = map.get(&Value::Int64(*u as i64)) {
                            return Some(idx);
                        }
                        map.get(&Value::Float64(*u as f64)).copied()
                    }
                    Value::Float64(f) => {
                        if f.fract() == 0.0 {
                            let i = *f as i64;
                            if let Some(&idx) = map.get(&Value::Int64(i)) {
                                return Some(idx);
                            }
                            if i >= 0 && i <= u32::MAX as i64 {
                                return map.get(&Value::UniqueId(i as u32)).copied();
                            }
                        }
                        None
                    }
                    // String ids match only by exact value. See the Integer arm.
                    _ => None,
                }
            }
        }
    }

    pub fn insert(&mut self, id: Value, idx: NodeIndex) {
        match self {
            TypeIdIndex::Integer(map) => {
                if let Value::UniqueId(u) = id {
                    map.insert(u, idx);
                } else {
                    let mut general: FxHashMap<Value, NodeIndex> =
                        map.drain().map(|(k, v)| (Value::UniqueId(k), v)).collect();
                    general.insert(id, idx);
                    *self = TypeIdIndex::General(general);
                }
            }
            TypeIdIndex::General(map) => {
                map.insert(id, idx);
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TypeIdIndex::Integer(map) => map.len(),
            TypeIdIndex::General(map) => map.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop the entry for `id`, but only when it currently resolves to `idx`.
    ///
    /// The `idx` check is what makes this safe to call while deleting: if the
    /// id has since been re-pointed at a different node, the live mapping is
    /// left alone rather than silently unindexing the survivor. Resolution
    /// goes through [`get`](Self::get), so the same `Int64`/`UniqueId`/
    /// `Float64` coercions apply — the key is removed under whichever
    /// spelling it was actually stored, not under the caller's spelling.
    pub fn remove_matching(&mut self, id: &Value, idx: NodeIndex) -> bool {
        if self.get(id) != Some(idx) {
            return false;
        }
        match self {
            TypeIdIndex::Integer(map) => {
                let key = match id {
                    Value::UniqueId(u) => Some(*u),
                    Value::Int64(i) if *i >= 0 && *i <= u32::MAX as i64 => Some(*i as u32),
                    Value::Float64(f) if f.fract() == 0.0 => {
                        let i = *f as i64;
                        if i >= 0 && i <= u32::MAX as i64 {
                            Some(i as u32)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                key.is_some_and(|k| map.remove(&k).is_some())
            }
            TypeIdIndex::General(map) => {
                if map.remove(id).is_some() {
                    return true;
                }
                // `get` resolved it, so it is stored under a coerced spelling.
                let coerced = match id {
                    Value::Int64(i) if *i >= 0 && *i <= u32::MAX as i64 => {
                        Some(Value::UniqueId(*i as u32))
                    }
                    Value::UniqueId(u) => Some(Value::Int64(*u as i64)),
                    Value::Float64(f) if f.fract() == 0.0 => {
                        let i = *f as i64;
                        if map.contains_key(&Value::Int64(i)) {
                            Some(Value::Int64(i))
                        } else if i >= 0 && i <= u32::MAX as i64 {
                            Some(Value::UniqueId(i as u32))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                coerced.is_some_and(|k| map.remove(&k).is_some())
            }
        }
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = (Value, NodeIndex)> + '_> {
        match self {
            TypeIdIndex::Integer(map) => {
                Box::new(map.iter().map(|(&k, &v)| (Value::UniqueId(k), v)))
            }
            TypeIdIndex::General(map) => Box::new(map.iter().map(|(k, &v)| (k.clone(), v))),
        }
    }
}

impl Default for TypeIdIndex {
    fn default() -> Self {
        TypeIdIndex::General(FxHashMap::default())
    }
}

/// Lightweight snapshot of a node, returned by node queries and traversals.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub id: Value,
    pub title: Value,
    pub node_type: String,
    pub properties: HashMap<String, Value>,
}

/// A filter, sort or traversal applied to a selection; shown by `explain()`.
#[derive(Clone, Debug)]
pub enum SelectionOperation {
    Filter(HashMap<String, FilterCondition>),
    Sort(Vec<(String, bool)>), // (field_name, ascending)
    Traverse {
        connection_type: String,
        direction: Option<String>,
        max_nodes: Option<usize>,
    },
    Custom(String),
}

/// A single level in the selection hierarchy — holds node sets grouped
/// by their parent (for traversals) and tracks applied operations.
#[derive(Clone, Debug)]
pub struct SelectionLevel {
    pub selections: HashMap<Option<NodeIndex>, Vec<NodeIndex>>, // parent_idx -> selected_children
    pub operations: Vec<SelectionOperation>,
}

impl SelectionLevel {
    // `new()` is the whole constructor surface; `Default` would add unused public API.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        SelectionLevel {
            selections: HashMap::new(),
            operations: Vec::new(),
        }
    }

    pub fn add_selection(&mut self, parent: Option<NodeIndex>, children: Vec<NodeIndex>) {
        self.selections.insert(parent, children);
    }

    pub fn get_all_nodes(&self) -> Vec<NodeIndex> {
        self.selections
            .values()
            .flat_map(|children| children.iter().copied())
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.selections.is_empty()
    }

    pub fn iter_groups(&self) -> impl Iterator<Item = (&Option<NodeIndex>, &Vec<NodeIndex>)> {
        self.selections.iter()
    }

    /// Non-allocating alternative to `get_all_nodes()` for iterating or counting.
    pub fn iter_node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.selections
            .values()
            .flat_map(|children| children.iter().copied())
    }

    pub fn node_count(&self) -> usize {
        self.selections.values().map(|v| v.len()).sum()
    }

    /// Rewrite this level's indices through a vacuum's `old → new` mapping.
    ///
    /// Semantics (see [`CurrentSelection::remap_indices`] for why remapping
    /// happens at all):
    ///
    /// - A selected node that did not survive the vacuum is **dropped** from
    ///   its group. A group that loses all of its children stays, empty — it
    ///   still records that this parent was traversed and matched nothing,
    ///   which is what a grouped `collect()` reports.
    /// - A group whose **parent** did not survive is dropped whole, children
    ///   included. The children are in this level *because of* that parent, so
    ///   keeping them under a different key (or under `None`) would invent a
    ///   traversal that never happened. Conservative by decision: losing rows
    ///   is recoverable by re-querying, inventing them is not.
    pub fn remap_indices(&mut self, remap: &NodeRemap) {
        let mut remapped: HashMap<Option<NodeIndex>, Vec<NodeIndex>> =
            HashMap::with_capacity(self.selections.len());
        for (parent, children) in self.selections.drain() {
            let new_parent = match parent {
                None => None,
                Some(p) => match remap.get(p) {
                    Some(new_p) => Some(new_p),
                    None => continue,
                },
            };
            remapped.insert(
                new_parent,
                children.into_iter().filter_map(|c| remap.get(c)).collect(),
            );
        }
        self.selections = remapped;
    }
}

#[derive(Clone, Debug)]
pub struct PlanStep {
    pub operation: String,
    pub node_type: Option<String>,
    pub estimated_rows: usize,
    pub actual_rows: Option<usize>,
}

impl PlanStep {
    pub fn new(operation: &str, node_type: Option<&str>, estimated_rows: usize) -> Self {
        PlanStep {
            operation: operation.to_string(),
            node_type: node_type.map(|s| s.to_string()),
            estimated_rows,
            actual_rows: None,
        }
    }

    pub fn with_actual_rows(mut self, actual: usize) -> Self {
        self.actual_rows = Some(actual);
        self
    }
}

/// Tracks the current selection state across a chain of query operations
/// (type_filter → filter → traverse → ...). Supports nested levels for
/// parent-child traversals and records execution plan steps for `explain()`.
#[derive(Clone, Default)]
pub struct CurrentSelection {
    levels: Vec<SelectionLevel>,
    current_level: usize,
    execution_plan: Vec<PlanStep>,
}

impl CurrentSelection {
    pub fn new() -> Self {
        let mut selection = CurrentSelection {
            levels: Vec::new(),
            current_level: 0,
            execution_plan: Vec::new(),
        };
        selection.add_level();
        selection
    }

    pub fn add_level(&mut self) {
        self.levels.push(SelectionLevel::new());
        self.current_level = self.levels.len() - 1;
    }

    pub fn clear(&mut self) {
        self.levels.clear();
        self.current_level = 0;
        self.execution_plan.clear();
        self.add_level(); // Ensure we always have at least one level after clearing
    }

    pub fn add_plan_step(&mut self, step: PlanStep) {
        self.execution_plan.push(step);
    }

    pub fn get_execution_plan(&self) -> &[PlanStep] {
        &self.execution_plan
    }

    pub fn clear_execution_plan(&mut self) {
        self.execution_plan.clear();
    }

    pub fn get_level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn get_level(&self, index: usize) -> Option<&SelectionLevel> {
        self.levels.get(index)
    }

    pub fn get_level_mut(&mut self, index: usize) -> Option<&mut SelectionLevel> {
        self.levels.get_mut(index)
    }

    pub fn current_node_count(&self) -> usize {
        self.levels.last().map(|l| l.node_count()).unwrap_or(0)
    }

    /// Whether this selection was *never narrowed*: the current level holds no
    /// nodes **and** no query operation was ever recorded.
    ///
    /// This is the discriminator between the two ways a selection can hold zero
    /// nodes, and they mean opposite things:
    ///
    /// - never selected (`true`) — the caller has not asked for anything yet,
    ///   so "everything" is the honest answer. `get_nodes()` returns the whole
    ///   graph on it, and so do `vector_search` / `search_text` / `embeddings`.
    /// - a query that matched nothing (`false`) — an empty result the caller
    ///   asked for, which must stay empty.
    ///
    /// Node-count alone cannot tell them apart, and reading a virgin selection
    /// as "matched nothing" is what made vector search answer `[]` to a
    /// question it had never been asked.
    pub fn never_selected(&self) -> bool {
        let level_is_empty = self
            .levels
            .last()
            .map(|level| level.node_count() == 0)
            .unwrap_or(true);
        level_is_empty && self.execution_plan.is_empty()
    }

    /// Returns true if any filtering/selection operation has been applied to the current level.
    pub fn has_active_selection(&self) -> bool {
        self.levels
            .last()
            .map(|l| !l.operations.is_empty())
            .unwrap_or(false)
    }

    pub fn current_node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.levels
            .last()
            .into_iter()
            .flat_map(|l| l.iter_node_indices())
    }

    /// Rewrite every held node index through a vacuum's `old → new` mapping,
    /// at every level.
    ///
    /// A selection held across a vacuum used to be *reset*, and a reset
    /// `CurrentSelection` reads as "no filter has been applied" — so the same
    /// held handle answered `ids()` with `[]` (silently empty) and `len()`
    /// with the whole graph (silently widened), two contradictory answers
    /// about the same set. Remapping keeps them describing the selection the
    /// caller actually made, minus whatever the deletes took out of it.
    ///
    /// A mapping that describes no rebuild — [`NodeRemap::describes_rebuild`]
    /// is false, which is what the disk backend and a columnar-only reclaim
    /// return — leaves the selection untouched, because no index moved. That
    /// is why an auto-vacuum on a disk graph, which reclaims nothing, no
    /// longer costs the caller their selection.
    ///
    /// Per-level rules are in [`SelectionLevel::remap_indices`]. The execution
    /// plan is deliberately untouched: it records what was *run*, not what is
    /// currently selected.
    pub fn remap_indices(&mut self, remap: &NodeRemap) {
        if !remap.describes_rebuild() {
            return;
        }
        for level in &mut self.levels {
            level.remap_indices(remap);
        }
    }

    /// Node type of the first selected node, for spatial `SpatialConfig` lookup.
    pub fn first_node_type(&self, graph: &DirGraph) -> Option<String> {
        // Arena guard: node_weight materializes on the disk backend
        // (protocol in disk/graph.rs); no-op on memory/mapped.
        let _arena_guard = graph.graph.begin_query();
        self.current_node_indices()
            .next()
            .and_then(|idx| graph.graph.node_view(idx))
            .map(|node| node.node_type_str(&graph.interner).to_string())
    }
}

/// Copy-on-write wrapper for `CurrentSelection` — avoids cloning it on every
/// method call that does not modify it.
#[derive(Clone, Default)]
pub struct CowSelection {
    inner: Arc<CurrentSelection>,
}

impl CowSelection {
    pub fn new() -> Self {
        CowSelection {
            inner: Arc::new(CurrentSelection::new()),
        }
    }
}

impl std::ops::Deref for CowSelection {
    type Target = CurrentSelection;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for CowSelection {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.inner)
    }
}

/// Key for single-property indexes: (node_type, property_name)
pub type IndexKey = (String, String);

/// Key for composite indexes: (node_type, property_names)
pub type CompositeIndexKey = (String, Vec<String>);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompositeValue(pub Vec<Value>);

/// Current `.kgl` on-disk snapshot format version stamped into new saves.
///
/// The single source of truth for [`SaveMetadata::current`]; also surfaced to
/// bindings through `kglite::api::io` so an embedder can report the persisted
/// format it writes, distinct from the engine SemVer.
pub const KGL_FORMAT_VERSION: u32 = 2;

/// Version info stamped into saved files, reported through `graph_info()`.
/// Nothing gates a load on it — the loader enforces format through the `.kgl`
/// container magic (`io/magic.rs`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SaveMetadata {
    /// [`KGL_FORMAT_VERSION`] on a fresh save; `0` for a file written before
    /// this field existed (via serde default). Not itself persisted — a
    /// portable load stamps its own constant (`io/file.rs`).
    pub format_version: u32,
    /// Library version at save time, e.g. "0.4.7".
    pub library_version: String,
}

impl SaveMetadata {
    pub fn current() -> Self {
        SaveMetadata {
            format_version: KGL_FORMAT_VERSION,
            library_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Type connectivity triple: one row of the type-level graph.
/// (source_type) -[connection_type]-> (target_type) with edge count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityTriple {
    pub src: String,
    pub conn: String,
    pub tgt: String,
    pub count: usize,
}

/// Metadata about a connection type: which node types it connects and what properties it carries.
#[derive(Debug, Clone, Default)]
pub struct ConnectionTypeInfo {
    pub source_types: HashSet<String>,
    pub target_types: HashSet<String>,
    /// property_name → type_string (e.g. "weight" → "Float64")
    pub property_types: HashMap<String, String>,
}

/// Custom serializer emits sorted keys for the two HashSet<String> and the
/// HashMap<String, String> so that `.kgl` saves stay byte-deterministic
/// regardless of per-run HashMap seed. The `test_phase4_parity` golden-hash test
/// pins the current digest, and its fixtures deliberately carry multiple
/// source/target types and property keys so an unsorted collection cannot slip
/// through.
impl Serialize for ConnectionTypeInfo {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut sorted_sources: Vec<&String> = self.source_types.iter().collect();
        sorted_sources.sort();
        let mut sorted_targets: Vec<&String> = self.target_types.iter().collect();
        sorted_targets.sort();
        let mut sorted_props: Vec<(&String, &String)> = self.property_types.iter().collect();
        sorted_props.sort_by(|a, b| a.0.cmp(b.0));
        let property_types: std::collections::BTreeMap<&String, &String> =
            sorted_props.into_iter().collect();

        let mut state = serializer.serialize_struct("ConnectionTypeInfo", 3)?;
        state.serialize_field("source_types", &sorted_sources)?;
        state.serialize_field("target_types", &sorted_targets)?;
        state.serialize_field("property_types", &property_types)?;
        state.end()
    }
}

/// Custom deserializer to handle both old format (source_type/target_type as single strings)
/// and new format (source_types/target_types as sets).
impl<'de> Deserialize<'de> for ConnectionTypeInfo {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Legacy {
            source_type: Option<String>,
            target_type: Option<String>,
            #[serde(default)]
            source_types: Option<HashSet<String>>,
            #[serde(default)]
            target_types: Option<HashSet<String>>,
            #[serde(default)]
            property_types: HashMap<String, String>,
        }

        let legacy = Legacy::deserialize(deserializer)?;
        let source_types = legacy.source_types.unwrap_or_else(|| {
            legacy
                .source_type
                .map(|s| HashSet::from([s]))
                .unwrap_or_default()
        });
        let target_types = legacy.target_types.unwrap_or_else(|| {
            legacy
                .target_type
                .map(|s| HashSet::from([s]))
                .unwrap_or_default()
        });
        Ok(ConnectionTypeInfo {
            source_types,
            target_types,
            property_types: legacy.property_types,
        })
    }
}

/// Contiguous columnar storage for f32 embeddings associated with a (node_type, property_name).
/// All vectors in one store share the same dimensionality.
/// The flat Vec<f32> layout enables SIMD-friendly linear scans during vector search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingStore {
    pub dimension: usize,
    /// Contiguous f32 buffer: embedding i occupies data[i*dimension..(i+1)*dimension]
    pub data: Vec<f32>,
    /// Maps NodeIndex.index() -> slot position in the contiguous buffer.
    /// Key-sorted on write so equivalent stores serialize byte-identically.
    #[serde(serialize_with = "serialize_sorted_map")]
    pub node_to_slot: HashMap<usize, usize>,
    /// Reverse map: slot -> NodeIndex.index(), needed for returning results
    pub slot_to_node: Vec<usize>,
    /// Default distance metric for this embedding store (e.g. "cosine", "poincare").
    /// Used when no explicit metric is provided at query time.
    #[serde(default)]
    pub metric: Option<String>,
    /// Identity of the model that produced these vectors (e.g. "BAAI/bge-m3").
    /// Set from the embedder's `model_id()` during `embed_texts`; `None` for
    /// vectors supplied directly via `add_embeddings`/`set_embeddings` or by a
    /// backend that doesn't name its model. Surfaced via `embedding_info()`.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Per-node content hash of the embedded text (`NodeIndex.index()` → hash),
    /// stamped during `embed_texts`. Lets `embed_texts(mode='changed')`
    /// re-embed exactly the nodes whose text changed since the last pass,
    /// instead of every node or only the missing ones. Empty for stores built
    /// from raw vectors (no source text to hash).
    /// Key-sorted on write so equivalent stores serialize byte-identically.
    #[serde(default, serialize_with = "serialize_sorted_map")]
    pub text_hashes: HashMap<usize, u64>,
    /// Cached L2 norm (‖v‖) per slot, parallel to `slot_to_node` — `norms[slot]`
    /// is the Euclidean norm of the vector at `data[slot*dimension..]`. A derived
    /// acceleration structure: cosine search reads it instead of recomputing the
    /// stored vector's norm on every query (collapsing cosine's inner loop to a
    /// single dot product). Deliberately NOT serialized (`#[serde(skip)]`) — it
    /// is fully determined by `data`, so persisting it would bloat the `.kgl`
    /// file and risk drift. Rebuilt from `data` after load via `rebuild_norms`.
    #[serde(skip)]
    pub norms: Vec<f32>,
    /// Optional HNSW approximate-nearest-neighbour index over this store's
    /// vectors, built on demand via `build_vector_index`. When present (and the
    /// query isn't `exact`), top-k vector search dispatches through it instead of
    /// a full linear scan. Indexes nodes by *slot*, so any change to the slot
    /// layout (in-place vector replacement, `add_embeddings`, compaction, the
    /// pruning a node deletion does via `remove_embedding`)
    /// invalidates it — callers drop it via `invalidate_index`. NOT serialized
    /// here (`#[serde(skip)]`): it rides in a dedicated versioned, skippable
    /// sub-section of the `.kgl` so the on-disk format can evolve independently;
    /// absent that section it's simply rebuilt.
    #[serde(skip)]
    pub index: Option<crate::graph::algorithms::hnsw::HnswIndex>,
}

/// One vector lifted out of an [`EmbeddingStore`] by
/// [`EmbeddingStore::remove_embedding`] — everything
/// [`EmbeddingStore::restore_embedding`] needs to put it back on the slot it
/// vacated. Carried by the undo journal so a rolled-back `DELETE` leaves
/// vector search returning exactly what it returned before.
#[derive(Debug, Clone, PartialEq)]
pub struct RemovedEmbedding {
    /// The slot the vector occupied. The restore reverses the tail swap that
    /// freed it, so slot identity survives a rollback.
    pub slot: usize,
    pub vector: Vec<f32>,
    /// The node's `text_hashes` entry, if it had one (`embed_texts` stamps
    /// it; raw `add_embeddings` does not). Restoring it keeps
    /// `embed_texts(mode='changed')` from re-embedding a node whose delete
    /// was rolled back.
    pub text_hash: Option<u64>,
}

/// Squared L2 norm of a vector, computed with the same 4-accumulator
/// 8-lane chunk pattern the cosine kernel uses for the stored vector, so a
/// cached norm matches the value cosine would have computed inline (within fp
/// rounding). Auto-vectorizes (SSE2/AVX2/NEON) on the hot rebuild path.
#[inline]
fn l2_norm_sq(v: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (chunks, rem) = v.as_chunks::<8>();
    for c in chunks {
        s0 += c[0] * c[0];
        s1 += c[1] * c[1];
        s2 += c[2] * c[2];
        s3 += c[3] * c[3];
        s0 += c[4] * c[4];
        s1 += c[5] * c[5];
        s2 += c[6] * c[6];
        s3 += c[7] * c[7];
    }
    for &x in rem {
        s0 += x * x;
    }
    (s0 + s1) + (s2 + s3)
}

impl EmbeddingStore {
    pub fn new(dimension: usize) -> Self {
        EmbeddingStore {
            dimension,
            data: Vec::new(),
            node_to_slot: HashMap::new(),
            slot_to_node: Vec::new(),
            metric: None,
            model_id: None,
            text_hashes: HashMap::new(),
            norms: Vec::new(),
            index: None,
        }
    }

    pub fn with_metric(dimension: usize, metric: &str) -> Self {
        EmbeddingStore {
            dimension,
            data: Vec::new(),
            node_to_slot: HashMap::new(),
            slot_to_node: Vec::new(),
            metric: Some(metric.to_string()),
            model_id: None,
            text_hashes: HashMap::new(),
            norms: Vec::new(),
            index: None,
        }
    }

    /// FNV-1a 64-bit hash of an embedded text — the content fingerprint
    /// stored in `text_hashes` for change detection. Deterministic across
    /// processes (same rationale as `InternedKey::from_str`), so a hash
    /// persisted in one process matches a re-hash in another after reload.
    pub fn text_hash(text: &str) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut h = FNV_OFFSET;
        for &byte in text.as_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    }

    /// Record the source-text hash for a node (by `NodeIndex.index()`).
    pub fn set_text_hash(&mut self, node_index: usize, hash: u64) {
        self.text_hashes.insert(node_index, hash);
    }

    /// `true` if the node has no embedding yet, or its stored text hash
    /// differs from `current_hash` (the text changed since it was embedded),
    /// or it was embedded without a recorded hash. Drives `mode='changed'`.
    pub fn is_stale(&self, node_index: usize, current_hash: u64) -> bool {
        if !self.node_to_slot.contains_key(&node_index) {
            return true;
        }
        match self.text_hashes.get(&node_index) {
            Some(&stored) => stored != current_hash,
            None => true,
        }
    }

    /// Add or replace an embedding for a node. Returns the slot index.
    /// Keeps the cached L2 norm (`norms[slot]`) in sync with `data`, and drops
    /// any HNSW index — mutating the vectors/slot layout would invalidate its
    /// topology (rebuild via `build_index` after a batch of edits).
    pub fn set_embedding(&mut self, node_index: usize, embedding: &[f32]) -> usize {
        self.index = None;
        let norm = l2_norm_sq(embedding).sqrt();
        if let Some(&slot) = self.node_to_slot.get(&node_index) {
            let start = slot * self.dimension;
            self.data[start..start + self.dimension].copy_from_slice(embedding);
            self.norms[slot] = norm;
            slot
        } else {
            let slot = self.slot_to_node.len();
            self.node_to_slot.insert(node_index, slot);
            self.slot_to_node.push(node_index);
            self.data.extend_from_slice(embedding);
            self.norms.push(norm);
            slot
        }
    }

    /// Drop a node's embedding, returning what it held so a statement
    /// rollback can put it back exactly (see [`RemovedEmbedding`]).
    /// `None` when the node has no vector in this store.
    ///
    /// **Deleting a node must call this.** `StableDiGraph` reuses a freed
    /// `NodeIndex`, and the store is keyed by that index, so a vector left
    /// behind is inherited by the next node to land on the slot — a
    /// full-similarity hit for a node that was never embedded, of any type.
    ///
    /// Compacts by swapping the tail slot into the vacated one, so the store
    /// keeps the dense, hole-free layout every reader (and the `.kgl`
    /// serializer) already assumes; the slot layout changes, so the HNSW
    /// index is dropped.
    pub fn remove_embedding(&mut self, node_index: usize) -> Option<RemovedEmbedding> {
        let slot = self.node_to_slot.remove(&node_index)?;
        self.index = None;
        let text_hash = self.text_hashes.remove(&node_index);
        let dim = self.dimension;
        let vector = self.data[slot * dim..(slot + 1) * dim].to_vec();
        let last = self.slot_to_node.len() - 1;
        if slot != last {
            let moved = self.slot_to_node[last];
            self.slot_to_node[slot] = moved;
            self.node_to_slot.insert(moved, slot);
            self.data
                .copy_within(last * dim..(last + 1) * dim, slot * dim);
            self.norms[slot] = self.norms[last];
        }
        self.slot_to_node.pop();
        self.data.truncate(last * dim);
        self.norms.pop();
        Some(RemovedEmbedding {
            slot,
            vector,
            text_hash,
        })
    }

    /// Exact inverse of [`Self::remove_embedding`] — the undo journal's
    /// restore half.
    ///
    /// Reverses the tail swap as well as the removal: whatever now occupies
    /// `removed.slot` is pushed back to the end and the restored vector takes
    /// its original slot. Replayed in reverse capture order (as the journal
    /// always replays), a statement's removals therefore rebuild the exact
    /// pre-statement slot layout, not merely the same set of vectors — which
    /// matters because slot order is scan order, and scan order decides
    /// score ties.
    pub fn restore_embedding(&mut self, node_index: usize, removed: &RemovedEmbedding) {
        self.index = None;
        let dim = self.dimension;
        let slot = removed.slot;
        let len = self.slot_to_node.len();
        debug_assert!(slot <= len, "a restore may only refill the slot it vacated");
        let norm = l2_norm_sq(&removed.vector).sqrt();
        if slot < len {
            // Evict the current occupant to the tail the removal vacated.
            let moved = self.slot_to_node[slot];
            let moved_vector: Vec<f32> = self.data[slot * dim..(slot + 1) * dim].to_vec();
            self.slot_to_node.push(moved);
            self.node_to_slot.insert(moved, len);
            self.data.extend_from_slice(&moved_vector);
            self.norms.push(self.norms[slot]);
            self.data[slot * dim..(slot + 1) * dim].copy_from_slice(&removed.vector);
            self.norms[slot] = norm;
            self.slot_to_node[slot] = node_index;
        } else {
            self.slot_to_node.push(node_index);
            self.data.extend_from_slice(&removed.vector);
            self.norms.push(norm);
        }
        self.node_to_slot.insert(node_index, slot);
        if let Some(hash) = removed.text_hash {
            self.text_hashes.insert(node_index, hash);
        }
    }

    /// Get the embedding slice for a node (by NodeIndex.index()).
    #[inline]
    pub fn get_embedding(&self, node_index: usize) -> Option<&[f32]> {
        self.node_to_slot.get(&node_index).map(|&slot| {
            let start = slot * self.dimension;
            &self.data[start..start + self.dimension]
        })
    }

    /// Get the embedding slice and its cached L2 norm for a node, in one lookup.
    /// The norm lets cosine scoring skip recomputing the stored vector's norm.
    #[inline]
    pub fn get_embedding_with_norm(&self, node_index: usize) -> Option<(&[f32], f32)> {
        self.node_to_slot.get(&node_index).map(|&slot| {
            let start = slot * self.dimension;
            (&self.data[start..start + self.dimension], self.norms[slot])
        })
    }

    /// Drop any HNSW index (it must be rebuilt after the vectors change).
    #[inline]
    pub fn invalidate_index(&mut self) {
        self.index = None;
    }

    #[inline]
    pub fn has_index(&self) -> bool {
        self.index.is_some()
    }

    /// Build (or rebuild) an HNSW index over this store's vectors for `metric`.
    /// Errors for Poincaré (unsupported by HNSW — stays brute-force). Ensures
    /// the norm cache is populated first (cosine navigation needs it).
    pub fn build_index(
        &mut self,
        metric: crate::graph::algorithms::vector::DistanceMetric,
        params: crate::graph::algorithms::hnsw::HnswParams,
        seed: u64,
    ) -> Result<(), String> {
        let hm = crate::graph::algorithms::hnsw::HnswMetric::from_distance(metric).ok_or_else(
            || "HNSW does not support the Poincaré metric; it stays on the exact (brute-force) path.".to_string(),
        )?;
        if self.dimension == 0 {
            return Err(
                "HNSW index construction requires a non-zero embedding dimension".to_string(),
            );
        }
        params.validate().map_err(str::to_string)?;
        self.validate_shape().map_err(str::to_string)?;
        if self.norms.len() != self.slot_to_node.len() {
            self.rebuild_norms();
        }
        self.index = Some(crate::graph::algorithms::hnsw::HnswIndex::build(
            &self.data,
            &self.norms,
            self.dimension,
            hm,
            params,
            seed,
        ));
        Ok(())
    }

    /// Validate the serialized, parallel embedding-store columns before any
    /// derived cache indexes into them. Persistence callers must run this
    /// before [`Self::rebuild_norms`] so malformed cardinalities become a
    /// load error rather than a slice panic.
    pub(crate) fn validate_shape(&self) -> Result<(), &'static str> {
        let expected_data_len = self
            .slot_to_node
            .len()
            .checked_mul(self.dimension)
            .ok_or("embedding data cardinality overflows usize")?;
        if self.data.len() != expected_data_len {
            return Err("embedding data cardinality does not match its slot count");
        }
        if self.node_to_slot.len() != self.slot_to_node.len() {
            return Err("embedding node/slot maps have different cardinalities");
        }
        for (slot, &node) in self.slot_to_node.iter().enumerate() {
            if self.node_to_slot.get(&node) != Some(&slot) {
                return Err("embedding node/slot maps are not a bijection");
            }
        }
        Ok(())
    }

    /// Recompute every cached norm from `data`. Call after any wholesale
    /// replacement of `data` that bypasses `set_embedding` — deserialization
    /// (`norms` is `#[serde(skip)]`, so it loads empty) and slot remapping
    /// during compaction. Idempotent.
    pub fn rebuild_norms(&mut self) {
        let n = self.slot_to_node.len();
        self.norms.clear();
        self.norms.reserve(n);
        for slot in 0..n {
            let start = slot * self.dimension;
            self.norms
                .push(l2_norm_sq(&self.data[start..start + self.dimension]).sqrt());
        }
    }

    #[inline]
    // An `is_empty()` companion would add public API the curated surface does not want.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.slot_to_node.len()
    }
}
/// Structural conveniences KGLite exposes on a node (`n.label`, `n.type`,
/// `n.node_type`, `n.name`) that a user may also legitimately store as a real
/// property. For these the stored property WINS during resolution (KG-1,
/// 2026-05-30); the structural value is only the fallback used when no such
/// property exists. `id`/`title` are deliberately NOT soft aliases — they are
/// the node's identity fields and are always returned canonically (no stored
/// property can shadow them).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SoftAliasFallback {
    /// `name` falls back to the node title.
    Title,
    /// `type` / `node_type` / `label` fall back to the node type string.
    TypeString,
}

/// Every name [`soft_alias_fallback`] answers `Some` for — the enumerated form
/// of the match below, for callers that need to go the other way and ask which
/// property names *could* resolve structurally (projection completion asks
/// exactly that, per type rather than per node). Kept adjacent to the match it
/// mirrors, and pinned in lockstep by `soft_alias_names_match_the_classifier`.
pub const SOFT_ALIAS_NAMES: [&str; 4] = ["name", "type", "node_type", "label"];

/// Classify a (post-alias) property name as a soft structural alias, or
/// `None` for a normal property / identity field. Single source of truth so
/// every resolution path (RETURN, WHERE, inline-map, EXISTS, disk fast path)
/// agrees on which names are property-first.
#[inline]
pub fn soft_alias_fallback(resolved: &str) -> Option<SoftAliasFallback> {
    match resolved {
        "name" => Some(SoftAliasFallback::Title),
        "type" | "node_type" | "label" => Some(SoftAliasFallback::TypeString),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeData {
    /// **Not public, deliberately.** Holds the `Value::Null` *sentinel* on the
    /// memory and mapped backends; [`NodeData::id`] carries the full contract
    /// and the resolving reads to use instead. Same for [`NodeData::title`].
    pub(crate) id: Value,
    pub(crate) title: Value,
    pub node_type: InternedKey,
    pub(crate) properties: PropertyStorage,
}

impl NodeData {
    /// Create a new NodeData, interning all property keys and the node type.
    ///
    /// Builds the **staging** `PropertyStorage::Map`: the node's properties
    /// are held inline until a consolidation pass (`DirGraph::enable_columnar`,
    /// or the loader's own) pushes them into the type's `ColumnStore`. The
    /// converged construction funnels build `PropertyStorage::Columnar`
    /// directly and never call this; see `dir_graph/node_write.rs`.
    pub fn new(
        id: Value,
        title: Value,
        node_type: String,
        properties: HashMap<String, Value>,
        interner: &mut StringInterner,
    ) -> Self {
        let type_key = interner.get_or_intern(&node_type);
        let interned_props = properties
            .into_iter()
            .map(|(k, v)| {
                let key = interner.get_or_intern(&k);
                (key, v)
            })
            .collect();
        NodeData {
            id,
            title,
            node_type: type_key,
            properties: PropertyStorage::Map(interned_props),
        }
    }

    pub fn new_preinterned(
        id: Value,
        title: Value,
        node_type: InternedKey,
        properties: Vec<(InternedKey, Value)>,
    ) -> Self {
        let map: HashMap<InternedKey, Value> = properties.into_iter().collect();
        NodeData {
            id,
            title,
            node_type,
            properties: PropertyStorage::Map(map),
        }
    }

    /// The node's **stored** `id` field — not a resolved read.
    ///
    /// This returns `self.id` verbatim and consults no `ColumnStore`. Since
    /// 0.16.0 every ingest path is columnar from the first node, so on the
    /// memory and mapped backends the inline field is the `Value::Null`
    /// sentinel and the node's canonical id lives in its type's store. The
    /// disk backend materialises real values into its arena copy, so the same
    /// call answers differently there — see [`crate::graph::dir_graph::DirGraph::get_node`].
    ///
    /// **Read an id through [`GraphRead::get_node_id`] or `NodeView::id`**,
    /// which resolve the store; never off a bare `NodeData`.
    ///
    /// [`GraphRead::get_node_id`]: crate::graph::storage::GraphRead::get_node_id
    #[inline]
    pub fn id(&self) -> Cow<'_, Value> {
        Cow::Borrowed(&self.id)
    }

    /// The node's **stored** `title` field — not a resolved read.
    ///
    /// Same contract as [`NodeData::id`]: the inline field verbatim, which is
    /// the `Value::Null` sentinel on the memory and mapped backends. Read a
    /// title through `NodeView::title`.
    #[inline]
    pub fn title(&self) -> Cow<'_, Value> {
        Cow::Borrowed(&self.title)
    }

    #[inline]
    pub fn node_type_str<'a>(&self, interner: &'a StringInterner) -> &'a str {
        interner.resolve(self.node_type)
    }
}

pub struct EdgeData {
    pub connection_type: InternedKey,
    pub properties: Vec<(InternedKey, Value)>,
}

// Stable Serde struct shape: connection_type as an InternedKey (auto-resolves
// to a string), properties as a map (backward-compatible with the old format).
impl Serialize for EdgeData {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("EdgeData", 2)?;
        s.serialize_field("connection_type", &self.connection_type)?;
        // Serde maps have the same backward-compatible wire shape regardless
        // of their Rust container. Sort by the persisted InternedKey value so
        // equivalent property vectors always produce identical `.kgl` bytes.
        let props_map: BTreeMap<&InternedKey, &Value> =
            self.properties.iter().map(|(k, v)| (k, v)).collect();
        s.serialize_field("properties", &props_map)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for EdgeData {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct EdgeDataHelper {
            connection_type: InternedKey,
            #[serde(default)]
            properties: HashMap<InternedKey, Value>,
        }
        let helper = EdgeDataHelper::deserialize(deserializer)?;
        Ok(EdgeData {
            connection_type: helper.connection_type,
            properties: helper.properties.into_iter().collect(),
        })
    }
}

impl std::fmt::Debug for EdgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeData")
            .field("connection_type", &self.connection_type)
            .field("properties", &self.properties)
            .finish()
    }
}

impl Clone for EdgeData {
    fn clone(&self) -> Self {
        EdgeData {
            connection_type: self.connection_type,
            properties: self.properties.clone(),
        }
    }
}

impl EdgeData {
    /// Create a new EdgeData, interning connection_type and all property keys.
    pub fn new(
        connection_type: String,
        properties: HashMap<String, Value>,
        interner: &mut StringInterner,
    ) -> Self {
        let ct_key = interner.get_or_intern(&connection_type);
        let interned_props: Vec<(InternedKey, Value)> = properties
            .into_iter()
            .map(|(k, v)| {
                let key = interner.get_or_intern(&k);
                (key, v)
            })
            .collect();
        EdgeData {
            connection_type: ct_key,
            properties: interned_props,
        }
    }

    pub fn new_interned(
        connection_type: InternedKey,
        properties: Vec<(InternedKey, Value)>,
    ) -> Self {
        EdgeData {
            connection_type,
            properties,
        }
    }

    #[inline]
    pub fn connection_type_str<'a>(&self, interner: &'a StringInterner) -> &'a str {
        interner.resolve(self.connection_type)
    }

    /// Uses hash-based lookup — no interner needed.
    #[inline]
    pub fn get_property(&self, key: &str) -> Option<&Value> {
        let ik = InternedKey::from_str(key);
        self.properties
            .iter()
            .find(|(k, _)| *k == ik)
            .map(|(_, v)| v)
    }

    #[inline]
    pub fn property_keys<'a>(
        &'a self,
        interner: &'a StringInterner,
    ) -> impl Iterator<Item = &'a str> {
        self.properties
            .iter()
            .map(move |(k, _)| interner.resolve(*k))
    }

    #[inline]
    pub fn property_iter<'a>(
        &'a self,
        interner: &'a StringInterner,
    ) -> impl Iterator<Item = (&'a str, &'a Value)> {
        self.properties
            .iter()
            .map(move |(k, v)| (interner.resolve(*k), v))
    }

    #[inline]
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    #[inline]
    pub fn properties_cloned(&self, interner: &StringInterner) -> HashMap<String, Value> {
        self.properties
            .iter()
            .map(|(k, v)| (interner.resolve(*k).to_string(), v.clone()))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeSchemaDefinition {
    pub required_fields: Vec<String>,
    /// Fields that may be present (for documentation purposes)
    pub optional_fields: Vec<String>,
    /// Expected types for fields: "string", "integer", "float", "boolean", "datetime"
    pub field_types: HashMap<String, String>,
    /// Declared PRIMARY KEY property for this node type (opt-in). When set, the
    /// write path enforces that the key is **unique and present** on every node
    /// of the type — the NODE KEY semantics — so a CREATE that would duplicate
    /// or omit it is rejected (use MERGE to upsert). `None` = no constraint
    /// (the permissive default).
    ///
    /// The key may be any property. Two enforcement routes, chosen by whether
    /// the key is the type's identity field:
    ///
    /// - `Some("id")` — enforced through the per-type id-index, an O(1)
    ///   amortised probe across every backend. `id` is a `NodeData` field, not
    ///   an entry in the property map, so it needs no secondary index.
    /// - `Some(other)` — enforced through a unique secondary index, installed
    ///   automatically by [`crate::graph::DirGraph::set_schema`] and rebuilt on
    ///   load like every other index.
    ///
    /// Serialized additively in the JSON metadata, so older `.kgl` files load
    /// with `None`.
    #[serde(default)]
    pub primary_key: Option<String>,
    /// Declared UNIQUE constraints beyond the primary key, as property tuples:
    /// `[["email"], ["first", "last"]]` declares `email` unique on its own and
    /// `(first, last)` unique as a pair. A tuple only constrains nodes that
    /// carry *every* property in it — matching Neo4j, where uniqueness does not
    /// apply to nodes missing the property. Enforced on every write path,
    /// including the bulk loader.
    ///
    /// `None` = no constraints. Additive serde field, so older `.kgl` files load
    /// with `None` and stay permissive.
    #[serde(default)]
    pub unique: Option<Vec<Vec<String>>>,
    /// Ownership layer for the two-writer contract: `"managed"` (rebuilt from
    /// source by a batch writer) or `"runtime"` (owned/mutated live by another
    /// writer, e.g. an agent). A **managed reload** (`add_nodes(...,
    /// managed_reload=True)`) skips a `runtime` type instead of writing it.
    /// That is a lane the batch writer opts into, not an enforced perimeter:
    /// an `add_nodes` call without the flag writes a `runtime` type normally,
    /// nothing gates a runtime writer out of `managed` types, and
    /// `add_connections` is not covered. `None` = unlayered (no restriction).
    /// Additive serde field.
    #[serde(default)]
    pub layer: Option<String>,
    /// Opt-in freshness provenance: when `Some(true)`, every write to a node of
    /// this type auto-stamps an `updated_at` timestamp (and the caller-supplied
    /// `git_sha`/`modified_by`, when provided) — so "this node describes the
    /// world as of X" is checkable and drift becomes a query, not a silent lie.
    /// `None`/`Some(false)` = off (the default), which keeps writes
    /// deterministic. Independent of `layer` and of `schema_locked`. Additive
    /// serde field — older `.kgl` files load with `None`.
    #[serde(default)]
    pub auto_timestamp: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSchemaDefinition {
    pub source_type: String,
    pub target_type: String,
    /// Optional cardinality constraint: "one-to-one", "one-to-many", "many-to-one", "many-to-many"
    pub cardinality: Option<String>,
    pub required_properties: Vec<String>,
    pub property_types: HashMap<String, String>,
    /// Opt-in freshness provenance (mirrors [`NodeSchemaDefinition::auto_timestamp`]):
    /// `Some(true)` stamps a reserved `updated_at` on every write to an edge of
    /// this type. `None`/`Some(false)` = off (default). Additive serde field.
    #[serde(default)]
    pub auto_timestamp: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchemaDefinition {
    pub node_schemas: HashMap<String, NodeSchemaDefinition>,
    pub connection_schemas: HashMap<String, ConnectionSchemaDefinition>,
}

/// How an incoming [`SchemaDefinition`] combines with the one already installed.
///
/// The distinction exists because a declaration is also a *withdrawal*: whatever
/// the outgoing schema declared and the incoming one does not is no longer
/// enforced. Under [`SchemaInstall::Replace`] that reaches every type in the
/// graph, so a call naming one type silently stops enforcing every other type's
/// constraints — the failure mode is duplicates entering the data unnoticed.
/// Under [`SchemaInstall::Merge`] the withdrawal is scoped to the types the call
/// actually names, so declaring per module is safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchemaInstall {
    /// Types the incoming schema names replace their previous entry; every other
    /// type keeps the declaration it had. The default, because it is the mode
    /// whose mistake is a rejected write rather than an admitted duplicate.
    #[default]
    Merge,
    /// The incoming schema becomes the whole schema — every type it does not
    /// name loses its declarations, and with them their enforcement.
    Replace,
}

impl SchemaDefinition {
    pub fn new() -> Self {
        SchemaDefinition {
            node_schemas: HashMap::new(),
            connection_schemas: HashMap::new(),
        }
    }

    /// This schema with `incoming`'s entries layered over it: a named type takes
    /// the incoming declaration wholesale, an unnamed one keeps its own.
    ///
    /// Per *type* rather than per *field* on purpose — a type's entry is one
    /// declaration ("here is what a `Task` is"), so merging field-by-field would
    /// make a re-declaration unable to withdraw anything and leave no way to
    /// narrow a type short of replacing the whole schema.
    pub fn merged_with(mut self, incoming: SchemaDefinition) -> Self {
        self.node_schemas.extend(incoming.node_schemas);
        self.connection_schemas.extend(incoming.connection_schemas);
        self
    }

    pub fn add_node_schema(&mut self, node_type: String, schema: NodeSchemaDefinition) {
        self.node_schemas.insert(node_type, schema);
    }

    pub fn add_connection_schema(
        &mut self,
        connection_type: String,
        schema: ConnectionSchemaDefinition,
    ) {
        self.connection_schemas.insert(connection_type, schema);
    }
}

#[derive(Debug, Clone)]
pub enum ValidationError {
    MissingRequiredField {
        node_type: String,
        node_title: String,
        field: String,
    },
    TypeMismatch {
        node_type: String,
        node_title: String,
        field: String,
        expected_type: String,
        actual_type: String,
    },
    InvalidConnectionEndpoint {
        connection_type: String,
        expected_source: String,
        expected_target: String,
        actual_source: String,
        actual_target: String,
    },
    MissingConnectionProperty {
        connection_type: String,
        source_title: String,
        target_title: String,
        property: String,
    },
    UndefinedNodeType {
        node_type: String,
        count: usize,
    },
    UndefinedConnectionType {
        connection_type: String,
        count: usize,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingRequiredField {
                node_type,
                node_title,
                field,
            } => {
                write!(
                    f,
                    "Missing required field '{}' on {} node '{}'",
                    field, node_type, node_title
                )
            }
            ValidationError::TypeMismatch {
                node_type,
                node_title,
                field,
                expected_type,
                actual_type,
            } => {
                write!(
                    f,
                    "Type mismatch on {} node '{}': field '{}' expected {}, got {}",
                    node_type, node_title, field, expected_type, actual_type
                )
            }
            ValidationError::InvalidConnectionEndpoint {
                connection_type,
                expected_source,
                expected_target,
                actual_source,
                actual_target,
            } => {
                write!(
                    f,
                    "Invalid connection '{}': expected {}->{}  but found {}->{}",
                    connection_type, expected_source, expected_target, actual_source, actual_target
                )
            }
            ValidationError::MissingConnectionProperty {
                connection_type,
                source_title,
                target_title,
                property,
            } => {
                write!(
                    f,
                    "Missing required property '{}' on {} connection from '{}' to '{}'",
                    property, connection_type, source_title, target_title
                )
            }
            ValidationError::UndefinedNodeType { node_type, count } => {
                write!(
                    f,
                    "Node type '{}' ({} nodes) exists in graph but not defined in schema",
                    node_type, count
                )
            }
            ValidationError::UndefinedConnectionType {
                connection_type,
                count,
            } => {
                write!(f, "Connection type '{}' ({} connections) exists in graph but not defined in schema", connection_type, count)
            }
        }
    }
}

#[cfg(test)]
#[path = "schema_tests.rs"]
mod schema_tests;

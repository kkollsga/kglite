use crate::datatypes::values::{FilterCondition, Value};
pub(crate) use crate::graph::embedding_store::VectorIndexState;
pub use crate::graph::embedding_store::{EmbeddingStore, HnswRead, RemovedEmbedding};
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

/// Refuse a declared constraint on a reserved provenance key, naming the key
/// and the `subject` (e.g. "node type 'Task'").
///
/// The engine writes these keys on every write to an `auto_timestamp` type,
/// and the write paths stamp them after (bulk loads) or before (Cypher CREATE)
/// the constraint gate, and never gate a SET's stamp, so a constraint on one
/// is not enforceable. Refused whether or not the type has opted in yet,
/// since the opt-in may follow the declaration. Every declaration surface
/// calls this: both `CREATE CONSTRAINT` arms, the `define_schema` parser, and
/// `DirGraph::set_schema` as the backstop for a hand-built definition.
pub fn reject_reserved_provenance_constraint<'a>(
    properties: impl IntoIterator<Item = &'a str>,
    subject: &str,
) -> Result<(), String> {
    match properties
        .into_iter()
        .find(|p| is_reserved_provenance_key(p))
    {
        None => Ok(()),
        Some(key) => Err(format!(
            "cannot constrain '{key}' on {subject}: updated_at, git_sha and modified_by are \
             provenance keys the engine owns and stamps on writes to auto_timestamp types, so \
             they cannot carry a constraint. Constrain a property of your own instead."
        )),
    }
}

/// Node-type names the engine owns — graph-carried skills and recipes. A
/// system-labelled node is ordinary data to Cypher (`MATCH (s:KgliteSkill)`
/// returns it, `MATCH (n) RETURN count(n)` counts it, exports and digests
/// carry it), but is **hidden from every
/// surface that enumerates node types** — `node_types()`, `db.labels()`,
/// `schema()`, `describe()` — so the label reads as engine machinery rather
/// than as part of the user's model. Same posture as
/// [`RESERVED_PROVENANCE_KEYS`] one level up, and applied the same way: per
/// enumerating site, not at a chokepoint, because `DirGraph::type_indices` is
/// public and every surface walks it directly.
///
/// Conventional, not enforced: `CREATE (:KgliteSkill {...})` is accepted like
/// any other label. Counts that accompany a listing filter with it, or a
/// rendered document contradicts itself.
pub const SYSTEM_LABELS: &[&str] = &["KgliteSkill", "KgliteRecipe"];

#[inline]
pub fn is_system_label(label: &str) -> bool {
    SYSTEM_LABELS.contains(&label)
}
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use thin_vec::ThinVec;

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
    /// Property name holding the start date, e.g. "prjHolderFrom" or "date_from"
    pub valid_from: String,
    /// Property name holding the end date, e.g. "prjHolderTo" or "date_to"
    pub valid_to: String,
    /// Whether the `valid_to` day belongs to the interval. Omitted from the
    /// serialized form when closed, so a closed config writes the bytes a
    /// config without the field always wrote.
    #[serde(
        default,
        skip_serializing_if = "crate::graph::features::temporal::IntervalConvention::is_closed"
    )]
    pub convention: crate::graph::features::temporal::IntervalConvention,
    /// The source node type a relationship config applies to; `None` covers
    /// every source. Always `None` for a node config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_type: Option<String>,
    /// An interval the `closed` convention accepts as empty (valid on no
    /// day) instead of refusing. Omitted from the serialized form when
    /// unset, so a file without it is byte-identical to one written before
    /// the field existed and an older build, which ignores unknown keys,
    /// answers every query the same.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub empty_when: Option<crate::graph::features::temporal::EmptyWhen>,
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
                ..Default::default()
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

/// 2^63: the first float past `i64`, where `as i64` saturates.
const I64_FLOAT_END: f64 = 9_223_372_036_854_775_808.0;

/// The integer a numeric id denotes — the one coercion every id-index
/// lookup applies: a `UniqueId`, an `Int64` and an integral `Float64` of the
/// same integer are one id. `None` for any other value, which is only ever
/// its own id — including a whole float outside `i64`, which would otherwise
/// saturate onto `i64::MAX` (every whole float inside it converts exactly).
pub(crate) fn id_integer(id: &Value) -> Option<i64> {
    match id {
        Value::UniqueId(u) => Some(i64::from(*u)),
        Value::Int64(i) => Some(*i),
        Value::Float64(f) if f.fract() == 0.0 && (-I64_FLOAT_END..I64_FLOAT_END).contains(f) => {
            Some(*f as i64)
        }
        _ => None,
    }
}

/// `n` as a float, when the float is exactly `n`: past 2^53 an integer
/// rounds onto a neighbour's float, which is another value (another id, and a
/// value a `Float64` column cannot hold).
pub(crate) fn exact_float(n: i64) -> Option<f64> {
    let f = n as f64;
    (f < I64_FLOAT_END && f as i64 == n).then_some(f)
}

/// `id` as the compact `Integer` index's key, when it has one.
pub(crate) fn id_u32(id: &Value) -> Option<u32> {
    id_integer(id).and_then(|n| u32::try_from(n).ok())
}

/// The other keys a `General` id index may hold `id` under, in the order a
/// lookup probes them after missing `id` itself.
pub(crate) fn id_spellings(id: &Value) -> impl Iterator<Item = Value> {
    let n = id_integer(id);
    let own = std::mem::discriminant(id);
    [
        n.map(Value::Int64),
        n.and_then(exact_float).map(Value::Float64),
        id_u32(id).map(Value::UniqueId),
    ]
    .into_iter()
    .flatten()
    .filter(move |key| std::mem::discriminant(key) != own)
}

/// [`TypeIdIndex::insert`] into a `General` map.
pub(crate) fn insert_general(map: &mut FxHashMap<Value, NodeIndex>, id: Value, idx: NodeIndex) {
    for key in id_spellings(&id) {
        map.remove(&key);
    }
    map.insert(id, idx);
}

/// A `General` index's map from `(id, node)` pairs in the type's node
/// order: one spelling per id, the last node's. Ids of a single numeric kind
/// cannot spell one id twice, so only a mix pays the dedup.
pub(crate) fn general_id_map(entries: Vec<(Value, NodeIndex)>) -> FxHashMap<Value, NodeIndex> {
    if !mixed_numeric_kinds(entries.iter().map(|(id, _)| id)) {
        return entries.into_iter().collect();
    }
    let mut map = FxHashMap::with_capacity_and_hasher(entries.len(), Default::default());
    for (id, idx) in entries {
        insert_general(&mut map, id, idx);
    }
    map
}

/// Whether `ids` hold more than one numeric kind — the only way two keys can
/// spell one id.
pub(crate) fn mixed_numeric_kinds<'a>(ids: impl Iterator<Item = &'a Value>) -> bool {
    let kinds = ids.fold(0u8, |kinds, id| {
        kinds
            | match id {
                Value::UniqueId(_) => 1,
                Value::Int64(_) => 2,
                Value::Float64(_) => 4,
                _ => 0,
            }
    });
    kinds.count_ones() > 1
}

/// `map` with one spelling per id, keeping the entry of the node last in node
/// order. A `General` index persisted through 0.18.1 kept a key per spelling
/// (`UniqueId(1)` and `Int64(1)` pointing at different nodes); a persisted
/// type's node order is ascending `NodeIndex` (the type-index writer sorts
/// it), so the highest index is the one a fresh build would keep.
pub(crate) fn heal_general_spellings(
    map: FxHashMap<Value, NodeIndex>,
) -> FxHashMap<Value, NodeIndex> {
    if !mixed_numeric_kinds(map.keys()) {
        return map;
    }
    let mut entries: Vec<(Value, NodeIndex)> = map.into_iter().collect();
    entries.sort_unstable_by_key(|(_, idx)| *idx);
    general_id_map(entries)
}

/// `id` in the spelling all its [`id_spellings`] share, so two ids the index
/// treats as one compare equal here.
pub(crate) fn canonical_id(id: &Value) -> Cow<'_, Value> {
    match id_integer(id) {
        Some(n) if !matches!(id, Value::Int64(_)) => Cow::Owned(Value::Int64(n)),
        _ => Cow::Borrowed(id),
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
    /// Look up only the stored `Value` spelling, without numeric normalization.
    pub(crate) fn get_exact(&self, id: &Value) -> Option<NodeIndex> {
        match (self, id) {
            (TypeIdIndex::Integer(map), Value::UniqueId(value)) => map.get(value).copied(),
            (TypeIdIndex::Integer(_), _) => None,
            (TypeIdIndex::General(map), value) => map.get(value).copied(),
        }
    }

    /// Look up a node by ID value, with type coercion: `id` itself, then
    /// each of its [`id_spellings`]. The coercion covers the same numeric
    /// family as `values_equal` (core/filtering.rs), which is what a type
    /// scan compares with: the id anchors in `try_index_lookup` treat a miss
    /// here as an empty result and never scan, so a coercion the index
    /// declines but the scan would have accepted is a lost row, not a slow
    /// one. A string id matches only by exact value: `{id:'a1'}` must not
    /// resolve to `UniqueId(1)` (datasets with a string id form such as
    /// Wikidata `Q76` store it as another property — query `{nid:'Q76'}`).
    pub fn get(&self, id: &Value) -> Option<NodeIndex> {
        match self {
            TypeIdIndex::Integer(map) => map.get(&id_u32(id)?).copied(),
            TypeIdIndex::General(map) => map
                .get(id)
                .or_else(|| id_spellings(id).find_map(|key| map.get(&key)))
                .copied(),
        }
    }

    /// Point `id` at `idx`. A `General` index holds one spelling per id, so
    /// the id's other spellings are dropped: the node inserted last answers
    /// every spelling, as one stored in a single kind does.
    pub fn insert(&mut self, id: Value, idx: NodeIndex) {
        match self {
            TypeIdIndex::Integer(map) => {
                // An integer id that fits the compact key is the same id
                // whatever its spelling (`get` coerces the lookup the same
                // way), so it stays in the compact map. Demoting for it
                // rebuilt the whole map as `Value` keys: a `CREATE` with an int
                // id on a type whose ids a DataFrame load stored as `UniqueId`
                // cost O(type) time and about 100 bytes per id of heap.
                if let Some(key) = id_u32(&id) {
                    map.insert(key, idx);
                } else {
                    let mut general: FxHashMap<Value, NodeIndex> =
                        map.drain().map(|(k, v)| (Value::UniqueId(k), v)).collect();
                    insert_general(&mut general, id, idx);
                    *self = TypeIdIndex::General(general);
                }
            }
            TypeIdIndex::General(map) => insert_general(map, id, idx),
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
            TypeIdIndex::Integer(map) => id_u32(id).is_some_and(|k| map.remove(&k).is_some()),
            TypeIdIndex::General(map) => {
                // The spelling `get` resolved: the first one present.
                let key = std::iter::once(id.clone())
                    .chain(id_spellings(id))
                    .find(|key| map.contains_key(key));
                key.is_some_and(|k| map.remove(&k).is_some())
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

    /// The groups in a fixed order: the parentless group first, then parents by
    /// ascending node index (creation order). `selections` is a hash map, so
    /// its own iteration order changes between processes; every accessor that
    /// lists groups or nodes goes through here, and the children of each group
    /// are already in ascending node index order from the traversal that built
    /// them.
    fn ordered_groups(&self) -> Vec<(&Option<NodeIndex>, &Vec<NodeIndex>)> {
        let mut groups: Vec<_> = self.selections.iter().collect();
        if groups.len() > 1 {
            groups.sort_unstable_by_key(|(parent, _)| **parent);
        }
        groups
    }

    pub fn get_all_nodes(&self) -> Vec<NodeIndex> {
        self.ordered_groups()
            .into_iter()
            .flat_map(|(_, children)| children.iter().copied())
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.selections.is_empty()
    }

    pub fn iter_groups(&self) -> impl Iterator<Item = (&Option<NodeIndex>, &Vec<NodeIndex>)> {
        self.ordered_groups().into_iter()
    }

    /// Non-allocating alternative to `get_all_nodes()` for iterating or counting.
    pub fn iter_node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.ordered_groups()
            .into_iter()
            .flat_map(|(_, children)| children.iter().copied())
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

/// Key for composite indexes: (node_type, property_names **sorted**). Sorting
/// is what makes the key independent of declaration order — see
/// `DirGraph::create_composite_index`.
pub type CompositeIndexKey = (String, Vec<String>);

/// One tuple of a composite index or unique constraint, its values in the same
/// order as the names of the key it is filed under.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompositeValue(pub Vec<Value>);

/// Current `.kgl` on-disk snapshot format version stamped into new saves.
///
/// The single source of truth for [`SaveMetadata::current`]; also surfaced to
/// bindings through `kglite::api::io` so an embedder can report the persisted
/// format it writes, distinct from the engine SemVer.
///
/// **Derived from the container magic, never written as a literal.** Two
/// hand-maintained copies of this number drifted for four container bumps: the
/// save side froze at `2` (the v2 era) and the portable load side at `3`, so
/// the documented promise — "changes when the engine's storage format changes"
/// (`docs/python/guides/schema-migrations.md`) — was false in both directions.
/// Deriving it means a new `Vn_MAGIC` moves it by construction.
pub const KGL_FORMAT_VERSION: u32 = crate::graph::io::magic::CURRENT_CONTAINER_VERSION as u32;

/// Version info stamped into saved files, reported through `graph_info()`.
/// Nothing gates a load on it — the loader enforces format through the `.kgl`
/// container magic (`io/magic.rs`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SaveMetadata {
    /// The `.kgl` container version of the file this graph was last saved to
    /// or loaded from (the header byte: 8 for a compact save, 9 when the save
    /// wrote a data region). A graph never saved or loaded, and a disk
    /// directory, report [`KGL_FORMAT_VERSION`]; `0` for a file written before
    /// this field existed (via serde default). Not itself persisted — load and
    /// save stamp it from the header (`io/file.rs`).
    pub format_version: u32,
    /// Library version at save time, e.g. "0.4.7".
    pub library_version: String,
    /// The oldest library version that has written this graph's data: carried
    /// through every load and re-save, so a file an old build wrote and a new
    /// build re-saved still names the old writer. Empty on a graph built in
    /// this process (its oldest writer is the running build). Persisted only
    /// when it differs from the saving build's own version.
    pub(crate) oldest_writer: String,
}

impl SaveMetadata {
    pub fn current() -> Self {
        SaveMetadata {
            format_version: KGL_FORMAT_VERSION,
            library_version: env!("CARGO_PKG_VERSION").to_string(),
            oldest_writer: String::new(),
        }
    }

    /// The oldest writer a save from this build would record: the carried one
    /// when it parses as older than the running version, else the running
    /// version.
    pub(crate) fn oldest_writer_for_save(&self) -> String {
        let current = env!("CARGO_PKG_VERSION");
        match (
            crate::graph::advisories::parse_version(&self.oldest_writer),
            crate::graph::advisories::parse_version(current),
        ) {
            (Some(carried), Some(now)) if carried < now => self.oldest_writer.clone(),
            _ => current.to_string(),
        }
    }
}

/// Type connectivity triple: one row of the type-level graph.
/// (source_type) -\[connection_type\]-> (target_type) with edge count.
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

impl ConnectionTypeInfo {
    /// Whether recording `observed` for `property` would change nothing.
    pub fn records(&self, property: &str, observed: &str) -> bool {
        self.property_types
            .get(property)
            .is_some_and(|prior| merged_property_type(prior, observed).is_none())
    }

    /// Record one write's observed type for `property`.
    ///
    /// The recorded type covers every write, not the last one: a type that
    /// disagrees with a concrete recorded type records `"mixed"` (the spelling
    /// `update_node_properties` and WAL replay use), so one outlier `CREATE`
    /// after a string load does not re-type the whole column in `describe()`.
    /// `"Unknown"`/`"Null"` carry no evidence and never displace a concrete type.
    pub fn record_property_type(&mut self, property: String, observed: String) {
        match self.property_types.get(&property) {
            Some(prior) => {
                if let Some(merged) = merged_property_type(prior, &observed) {
                    self.property_types.insert(property, merged);
                }
            }
            None => {
                self.property_types.insert(property, observed);
            }
        }
    }
}

/// The record a numeric write leaves when it disagrees with the recorded
/// kind: `mixed` for an `Int64` against `Float64` (either order) or against an
/// already-`mixed` record. A float column no longer converts an integer, so the
/// column holds both kinds and a bare `Int64` or `Float64` record would
/// contradict a value read back. `None` leaves the incoming kind as it is.
pub(crate) fn numeric_record_after(recorded: &str, incoming: &str) -> Option<&'static str> {
    use crate::graph::storage::column_store::TypedColumn;
    let numeric = |kind| matches!(kind, Some("int64" | "float64"));
    let recorded_mixed = recorded.eq_ignore_ascii_case("mixed");
    let (recorded, incoming) = (
        TypedColumn::canonical_type_str(recorded),
        TypedColumn::canonical_type_str(incoming),
    );
    let diverges = (recorded == Some("float64") && incoming == Some("int64"))
        || (recorded == Some("int64") && incoming == Some("float64"))
        || (recorded_mixed && numeric(incoming));
    diverges.then_some("mixed")
}

/// What a recorded property type becomes when a write observes `observed` —
/// `None` when `prior` already says it.
pub(crate) fn merged_property_type(prior: &str, observed: &str) -> Option<String> {
    let no_evidence = |t: &str| t == "Unknown" || t == "Null";
    if no_evidence(observed) || prior.eq_ignore_ascii_case(observed) || prior == "mixed" {
        None
    } else if no_evidence(prior) {
        Some(observed.to_string())
    } else {
        Some("mixed".to_string())
    }
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

/// A node's identity, type and staged properties — 32 bytes, so a petgraph
/// node entry is 40 B.
///
/// `id` and `title` are not fields: on every steady-state node they are the
/// `Value::Null` sentinel (the values live in the type's `ColumnStore`), so
/// they sit behind `ident`, which is `None` for the sentinel pair. Only the
/// staging paths (RDF / N-Triples loaders, the `set_node_title` fallback,
/// pre-column `.kgl` files, vacuum and fork placeholders) hold a `Some`.
///
/// `repr(C)` pins the field order (`node_type` at 0, `ident` at 8,
/// `properties` at 16) so a frozen `.kgl` topology section can be read as a
/// `[NodeData]` in place; `storage::frozen::node_record` asserts the layout.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct NodeData {
    pub node_type: InternedKey,
    /// `(id, title)`; never a `(Null, Null)` pair — [`NodeData::set_ident`]
    /// collapses that back to `None`.
    pub(crate) ident: Option<Box<(Value, Value)>>,
    pub(crate) properties: PropertyStorage,
}

static NULL_VALUE: Value = Value::Null;

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
            ident: Self::pack_ident(id, title),
            node_type: type_key,
            properties: PropertyStorage::Map(Box::new(interned_props)),
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
            ident: Self::pack_ident(id, title),
            node_type,
            properties: PropertyStorage::Map(Box::new(map)),
        }
    }

    /// The `ident` encoding of an `(id, title)` pair: `None` for the sentinel
    /// pair, one boxed allocation otherwise.
    #[inline]
    pub(crate) fn pack_ident(id: Value, title: Value) -> Option<Box<(Value, Value)>> {
        if matches!(id, Value::Null) && matches!(title, Value::Null) {
            None
        } else {
            Some(Box::new((id, title)))
        }
    }

    /// The inline id, borrowed; the `Null` sentinel when none is held.
    #[inline]
    pub(crate) fn id_ref(&self) -> &Value {
        self.ident.as_deref().map_or(&NULL_VALUE, |p| &p.0)
    }

    /// The inline title, borrowed; the `Null` sentinel when none is held.
    #[inline]
    pub(crate) fn title_ref(&self) -> &Value {
        self.ident.as_deref().map_or(&NULL_VALUE, |p| &p.1)
    }

    /// Replace the inline `(id, title)` pair; a `(Null, Null)` pair frees the box.
    #[inline]
    pub(crate) fn set_ident(&mut self, id: Value, title: Value) {
        self.ident = Self::pack_ident(id, title);
    }

    /// Replace the inline title, leaving the id as it is.
    pub(crate) fn set_title(&mut self, title: Value) {
        let id = self.take_id();
        self.set_ident(id, title);
    }

    /// Move the inline id out, leaving the sentinel in its place.
    pub(crate) fn take_id(&mut self) -> Value {
        match self.ident.as_deref_mut() {
            Some(p) => std::mem::replace(&mut p.0, Value::Null),
            None => Value::Null,
        }
    }

    /// Reset both inline fields to the sentinel, freeing the box.
    #[inline]
    pub(crate) fn clear_ident(&mut self) {
        self.ident = None;
    }

    /// The node's **stored** `id` field — not a resolved read.
    ///
    /// This returns the inline id verbatim and consults no `ColumnStore`. Since
    /// 0.16.0 every ingest path is columnar from the first node, so on the
    /// memory and mapped backends the inline field is the `Value::Null`
    /// sentinel and the node's canonical id lives in its type's store. The
    /// disk backend materialises real values into the record it hands out, so the same
    /// call answers differently there — see [`crate::graph::dir_graph::DirGraph::get_node`].
    ///
    /// **Read an id through [`GraphRead::get_node_id`] or `NodeView::id`**,
    /// which resolve the store; never off a bare `NodeData`.
    ///
    /// [`GraphRead::get_node_id`]: crate::graph::storage::GraphRead::get_node_id
    #[inline]
    pub fn id(&self) -> Cow<'_, Value> {
        Cow::Borrowed(self.id_ref())
    }

    /// The node's **stored** `title` field — not a resolved read.
    ///
    /// Same contract as [`NodeData::id`]: the inline field verbatim, which is
    /// the `Value::Null` sentinel on the memory and mapped backends. Read a
    /// title through `NodeView::title`.
    #[inline]
    pub fn title(&self) -> Cow<'_, Value> {
        Cow::Borrowed(self.title_ref())
    }

    #[inline]
    pub fn node_type_str<'a>(&self, interner: &'a StringInterner) -> &'a str {
        interner.resolve(self.node_type)
    }
}

pub struct EdgeData {
    pub connection_type: InternedKey,
    /// A `ThinVec` is one pointer wide (and null-niched), which keeps
    /// `Option<EdgeData>` at 16 B so a petgraph edge entry is 32 B.
    pub properties: ThinVec<(InternedKey, Value)>,
}

impl EdgeData {
    /// Create a new EdgeData, interning connection_type and all property keys.
    pub fn new(
        connection_type: String,
        properties: HashMap<String, Value>,
        interner: &mut StringInterner,
    ) -> Self {
        let ct_key = interner.get_or_intern(&connection_type);
        let interned_props: ThinVec<(InternedKey, Value)> = properties
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
        properties: impl Into<ThinVec<(InternedKey, Value)>>,
    ) -> Self {
        EdgeData {
            connection_type,
            properties: properties.into(),
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

    /// Refuse a schema whose constraint-bearing fields name a reserved
    /// provenance key — the store-level backstop behind the parser's earlier
    /// per-type refusal, so a hand-built definition passed to `set_schema` is
    /// refused the same way. Types are walked in name order so the reported
    /// type is deterministic.
    pub fn reject_reserved_provenance_constraints(&self) -> Result<(), String> {
        let mut node_types: Vec<&String> = self.node_schemas.keys().collect();
        node_types.sort();
        for node_type in node_types {
            reject_reserved_provenance_constraint(
                self.node_schemas[node_type].constrained_properties(),
                &format!("node type '{node_type}'"),
            )?;
        }
        let mut conn_types: Vec<&String> = self.connection_schemas.keys().collect();
        conn_types.sort();
        for conn_type in conn_types {
            reject_reserved_provenance_constraint(
                self.connection_schemas[conn_type].constrained_properties(),
                &format!("connection type '{conn_type}'"),
            )?;
        }
        Ok(())
    }
}

impl NodeSchemaDefinition {
    /// Every property a constraint-bearing field names: `required`, `types`,
    /// `primary_key` and each `unique` tuple. `optional` constrains nothing.
    pub fn constrained_properties(&self) -> impl Iterator<Item = &str> {
        self.required_fields
            .iter()
            .chain(self.field_types.keys())
            .chain(self.primary_key.iter())
            .chain(self.unique.iter().flatten().flatten())
            .map(String::as_str)
    }
}

impl ConnectionSchemaDefinition {
    /// Every property a constraint-bearing field names: `required_properties`
    /// and `property_types`.
    pub fn constrained_properties(&self) -> impl Iterator<Item = &str> {
        self.required_properties
            .iter()
            .chain(self.property_types.keys())
            .map(String::as_str)
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

#[path = "schema_record_impls.rs"]
mod record_impls;

#[cfg(test)]
#[path = "schema_tests.rs"]
mod schema_tests;

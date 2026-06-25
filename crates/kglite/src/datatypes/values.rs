// src/datatypes/values.rs
use chrono::{NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub enum FilterCondition {
    Equals(Value),
    NotEquals(Value),
    GreaterThan(Value),
    GreaterThanEquals(Value),
    LessThan(Value),
    LessThanEquals(Value),
    In(Vec<Value>),
    Between(Value, Value), // Inclusive range [min, max]
    IsNull,
    IsNotNull,
    Contains(Value),
    StartsWith(Value),
    EndsWith(Value),
    Regex(String),
    Not(Box<FilterCondition>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Value {
    UniqueId(u32),
    Int64(i64),
    Float64(f64),
    String(String),
    Boolean(bool),
    DateTime(NaiveDate),
    Point {
        lat: f64,
        lon: f64,
    },
    Null,
    /// Internal: petgraph NodeIndex reference, used to preserve node identity
    /// through collect() → index → WITH → property access pipelines.
    /// Never persisted — only exists during Cypher execution.
    NodeRef(u32),
    /// Calendar duration: months + days + seconds (Neo4j shape).
    /// 0.9.0 Cluster 2 — replaces the soft-duration Int64-as-days
    /// hack from §3 v1. Calendar units (months, years) and clock
    /// units (days, hours, minutes, seconds) are kept separate so
    /// `duration({months: 1, days: 5}).months` returns 1, not 35.
    /// Sub-day precision (hours/minutes/seconds) is wired in seconds
    /// — Value::DateTime is still NaiveDate (Cluster 1, deferred),
    /// so DateTime + Duration discards the seconds component for
    /// now.
    ///
    /// Field widths (months/days as i32) sized to keep the enum
    /// payload at 16 bytes (matches Point's 2×f64). months/days are
    /// bounded around ±2e9 — 178 M years / 5.8 M years respectively
    /// — far past anything the user can reasonably need.
    ///
    /// **Layout note**: Duration was the LAST variant in `.kgl` v3.
    /// Phase A.1 (0.10.0) appends Node/Relationship/Path/List/Map
    /// after it and bumps the `.kgl` format to v4 — a hard break;
    /// v3 files do not load with v4 binaries. Discriminants 0..=8
    /// (Null .. NodeRef .. Duration) stay stable; 9..=13 are the
    /// new collection / graph-entity variants.
    Duration {
        months: i32,
        days: i32,
        seconds: i64,
    },
    /// A materialised graph node — the projection result for `RETURN n`.
    /// Boxed because [`NodeValue`] is large (id + labels + props map)
    /// and Node values are rarer than scalars; the indirection cost is
    /// amortised over typical query workloads.
    ///
    /// `Value::NodeRef(u32)` (variant 8) stays as the *transient*
    /// internal handle used during WITH/UNWIND chains. NodeRef is
    /// never user-visible — it gets materialised to `Node` at
    /// projection time and never persisted.
    Node(Box<NodeValue>),
    /// A materialised graph relationship — the projection result for
    /// `RETURN r` where `r` is a relationship variable.
    Relationship(Box<RelValue>),
    /// A materialised path — the projection result for variable-length
    /// path patterns and `shortestPath(...)` results.
    Path(Box<PathValue>),
    /// An ordered, heterogeneous list of values.
    ///
    /// `[]` in Cypher syntax; `labels(n)`, `nodes(p)`, `collect(...)`,
    /// `range(...)` all produce this. Kept inline (not Boxed) because
    /// list iteration is a hot path; the +24 bytes vs the prior
    /// largest variant (Point at 16) is the deliberate cost of
    /// having native collections.
    List(Vec<Value>),
    /// A string-keyed map of values.
    ///
    /// `{key: val, ...}` in Cypher syntax; `properties(n)`,
    /// `RETURN n.*` produce this. `BTreeMap` chosen over `HashMap`
    /// so equality / hashing / serialisation are deterministic by
    /// key order (Cypher consumers expect stable iteration order).
    Map(BTreeMap<String, Value>),
    /// A date *and* time-of-day, second precision (`NaiveDateTime`).
    ///
    /// Complements [`Value::DateTime`] (date-only `NaiveDate`): use
    /// `Timestamp` when the wall-clock time matters (event logs,
    /// `created_at`, scheduling). Produced by the `datetime()` /
    /// `localdatetime()` Cypher constructors and by passing a Python
    /// `datetime.datetime` with a non-midnight time component.
    ///
    /// **Layout note**: appended LAST (serde discriminant 15) so
    /// existing `.kgl` files — which never contain this variant —
    /// still deserialize unchanged; no format bump. Discriminants
    /// 0..=14 are untouched. Timestamp properties ride the generic
    /// `Value` serialization path (no dedicated typed column), so the
    /// hot date-only columnar path is unaffected.
    Timestamp(NaiveDateTime),
}

/// Owned, serialisable shape for a node value at the consumer
/// boundary. Distinct from [`crate::graph::schema::NodeData`], which
/// is interner-bound (carries `InternedKey` fields tied to the
/// graph's StringInterner) and therefore not portable across the
/// projection boundary.
///
/// Built at projection time (`Expression::Variable` → `Value::Node`)
/// by resolving the NodeData's interned fields against the active
/// graph's interner.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeValue {
    /// Stable integer id; mirrors what Bolt encodes as the Node struct's
    /// `identity` field. Sourced from `NodeData.id` if numeric, else a
    /// fallback derived from the petgraph NodeIndex.
    pub id: u32,
    /// Type labels. KGLite is single-label (one entry today), but the
    /// list shape matches Neo4j/Bolt's `labels` field and forward-
    /// compatible-with-multi-label work (ROADMAP §5).
    pub labels: Vec<String>,
    /// Properties as a string-keyed map. Key order is stable
    /// (BTreeMap), so equality/hash/serialisation are deterministic.
    pub properties: BTreeMap<String, Value>,
}

/// Owned, serialisable shape for a relationship value. See
/// [`NodeValue`] for the rationale (interner-decoupled, projection-
/// boundary-friendly).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelValue {
    pub id: u32,
    pub start_id: u32,
    pub end_id: u32,
    pub rel_type: String,
    pub properties: BTreeMap<String, Value>,
}

/// Owned, serialisable shape for a path value (sequence of nodes +
/// relationships from a variable-length pattern).
///
/// Stored as parallel vectors rather than alternating segments so
/// the common iteration patterns (just the nodes, just the rels)
/// are cheap. For a path of length k there are k+1 nodes and k
/// rels; consumers that need alternation can `zip` them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PathValue {
    pub nodes: Vec<NodeValue>,
    pub rels: Vec<RelValue>,
}

/// Zero-copy view of a [`Value`] for hot read paths that don't need
/// owned heap data. Strings borrow from the source buffer (e.g. an
/// mmap region) instead of cloning into a `String`.
///
/// Used by `save_subset_streaming_disk` to avoid the
/// `Value::String(s.to_string())` clone per property × per row, which
/// dominated the v3 node walk wall time on Wikidata (298 s out of
/// 446 s — heap pressure from ~510 M `String` allocations).
///
/// `to_value()` materializes an owned `Value` when one is needed
/// (e.g. for the heterogeneous Mixed column path).
#[derive(Clone, Copy, Debug)]
pub enum BorrowedValue<'a> {
    Null,
    Boolean(bool),
    Int64(i64),
    Float64(f64),
    UniqueId(u32),
    String(&'a str),
    DateTime(NaiveDate),
    /// A borrowed list of owned values. Unlike the scalar variants this
    /// borrows the `Vec<Value>` slice from the source rather than copying;
    /// it lets native list properties survive the streaming-disk save path
    /// (which otherwise can only carry scalars).
    List(&'a [Value]),
}

impl<'a> BorrowedValue<'a> {
    /// Materialize into an owned [`Value`]. Allocates for `String`.
    /// Takes `self` by value since `BorrowedValue` is `Copy`.
    pub fn to_value(self) -> Value {
        match self {
            BorrowedValue::Null => Value::Null,
            BorrowedValue::Boolean(b) => Value::Boolean(b),
            BorrowedValue::Int64(v) => Value::Int64(v),
            BorrowedValue::Float64(v) => Value::Float64(v),
            BorrowedValue::UniqueId(v) => Value::UniqueId(v),
            BorrowedValue::String(s) => Value::String(s.to_string()),
            BorrowedValue::DateTime(d) => Value::DateTime(d),
            BorrowedValue::List(items) => Value::List(items.to_vec()),
        }
    }
}

// Implement Eq for Value
impl Eq for Value {
    // We need this empty impl because we already have PartialEq
    // and all variants can be exactly equal except Float64,
    // which we handle specially in PartialEq
}

// Manual PartialOrd + Ord for Value.
// NaN sorts after all other floats; cross-variant ordering uses discriminant index.
impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        // Helper to get discriminant order. Independent of the enum's
        // serde discriminant — this is the ordering used for
        // cross-variant comparisons in Ord, where Null sorts first
        // and Duration last (mirrors Neo4j-ish "values < types <
        // structured types"). The serde discriminant is positional
        // (see enum doc).
        fn disc(v: &Value) -> u8 {
            match v {
                Value::Null => 0,
                Value::Boolean(_) => 1,
                Value::UniqueId(_) => 2,
                Value::Int64(_) => 3,
                Value::Float64(_) => 4,
                Value::String(_) => 5,
                Value::DateTime(_) => 6,
                Value::Duration { .. } => 7,
                Value::Point { .. } => 8,
                Value::NodeRef(_) => 9,
                // Phase A.1 — collection / graph-entity variants sort
                // after the scalars. Mirrors openCypher's general
                // "scalars < lists < maps < entities" ordering loosely;
                // exact ordering within is by id / structural compare.
                Value::List(_) => 10,
                Value::Map(_) => 11,
                Value::Node(_) => 12,
                Value::Relationship(_) => 13,
                Value::Path(_) => 14,
                // Sorts after the date-only DateTime in mixed compares;
                // same-variant timestamps order chronologically below.
                Value::Timestamp(_) => 15,
            }
        }
        match (self, other) {
            // Same variant comparisons
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
            (Value::UniqueId(a), Value::UniqueId(b)) => a.cmp(b),
            (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
            (Value::Float64(a), Value::Float64(b)) => {
                a.partial_cmp(b).unwrap_or_else(|| {
                    // NaN handling: NaN sorts last
                    match (a.is_nan(), b.is_nan()) {
                        (true, true) => Ordering::Equal,
                        (true, false) => Ordering::Greater,
                        (false, true) => Ordering::Less,
                        _ => unreachable!(),
                    }
                })
            }
            (Value::String(a), Value::String(b)) => a.cmp(b),
            (Value::DateTime(a), Value::DateTime(b)) => a.cmp(b),
            (
                Value::Point {
                    lat: a_lat,
                    lon: a_lon,
                },
                Value::Point {
                    lat: b_lat,
                    lon: b_lon,
                },
            ) => a_lat
                .partial_cmp(b_lat)
                .unwrap_or(Ordering::Equal)
                .then(a_lon.partial_cmp(b_lon).unwrap_or(Ordering::Equal)),
            (Value::NodeRef(a), Value::NodeRef(b)) => a.cmp(b),
            (
                Value::Duration {
                    months: am,
                    days: ad,
                    seconds: as_,
                },
                Value::Duration {
                    months: bm,
                    days: bd,
                    seconds: bs,
                },
            ) => am.cmp(bm).then(ad.cmp(bd)).then(as_.cmp(bs)),
            // Phase A.1 same-variant arms — defer to the derived Ord
            // on the contained payload types (NodeValue, RelValue,
            // PathValue all `#[derive(Ord)]`; Vec/BTreeMap do too).
            (Value::List(a), Value::List(b)) => a.cmp(b),
            (Value::Map(a), Value::Map(b)) => a.cmp(b),
            (Value::Node(a), Value::Node(b)) => a.cmp(b),
            (Value::Relationship(a), Value::Relationship(b)) => a.cmp(b),
            (Value::Path(a), Value::Path(b)) => a.cmp(b),
            (Value::Timestamp(a), Value::Timestamp(b)) => a.cmp(b),
            // Cross-variant: order by discriminant
            _ => disc(self).cmp(&disc(other)),
        }
    }
}

// Implement Hash for Value
impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // First hash discriminant to differentiate variants
        std::mem::discriminant(self).hash(state);

        // Then hash the contained value
        match self {
            Value::UniqueId(v) => v.hash(state),
            Value::Int64(v) => v.hash(state),
            Value::Float64(v) => {
                // Special handling for NaN and -0.0
                if v.is_nan() {
                    // Hash all NaN values the same
                    f64::NAN.to_bits().hash(state)
                } else {
                    // Handle -0.0 == 0.0
                    if *v == 0.0 {
                        0.0f64.to_bits().hash(state)
                    } else {
                        v.to_bits().hash(state)
                    }
                }
            }
            Value::String(v) => v.hash(state),
            Value::Boolean(v) => v.hash(state),
            Value::DateTime(v) => v.hash(state),
            Value::Point { lat, lon } => {
                lat.to_bits().hash(state);
                lon.to_bits().hash(state);
            }
            Value::Duration {
                months,
                days,
                seconds,
            } => {
                months.hash(state);
                days.hash(state);
                seconds.hash(state);
            }
            Value::Null => 0.hash(state),
            Value::NodeRef(v) => v.hash(state),
            // Phase A.1 — defer to derived Hash on payload types.
            // BTreeMap<String, Value> implements Hash (iterates in
            // key order); Vec<Value> implements Hash; NodeValue/
            // RelValue/PathValue all derive Hash.
            Value::List(v) => v.hash(state),
            Value::Map(v) => {
                // BTreeMap doesn't implement Hash in std. Hash the
                // length, then each (key, value) pair in iteration
                // order (BTreeMap iterates in sorted key order, so
                // this is deterministic).
                v.len().hash(state);
                for (k, val) in v {
                    k.hash(state);
                    val.hash(state);
                }
            }
            Value::Node(v) => v.hash(state),
            Value::Relationship(v) => v.hash(state),
            Value::Path(v) => v.hash(state),
            Value::Timestamp(v) => v.hash(state),
        }
    }
}

impl Value {
    pub fn as_string(&self) -> Option<String> {
        match self {
            Value::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Canonical PascalCase variant name. Phase A.1 / C7a — added so
    /// the (formerly duplicated) `value_type_name` / `value_kind`
    /// helpers across executor/write.rs and mutation/subgraph_streaming_
    /// writer.rs share one source of truth. Other classifiers
    /// (introspection/schema_overview.rs `str`/`int`/..., validation.rs
    /// `string`/`integer`/..., export.rs blueprint shape) use
    /// consumer-specific conventions and keep their own tables.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "Null",
            Value::Boolean(_) => "Boolean",
            Value::Int64(_) => "Int64",
            Value::Float64(_) => "Float64",
            Value::UniqueId(_) => "UniqueId",
            Value::String(_) => "String",
            Value::DateTime(_) => "DateTime",
            Value::Point { .. } => "Point",
            Value::NodeRef(_) => "NodeRef",
            Value::Duration { .. } => "Duration",
            Value::List(_) => "List",
            Value::Map(_) => "Map",
            Value::Node(_) => "Node",
            Value::Relationship(_) => "Relationship",
            Value::Path(_) => "Path",
            Value::Timestamp(_) => "Timestamp",
        }
    }
}

/// Phase A.1 / C7a — Display impl delegating to the existing
/// `format_value` free function. Lets `format!("{}", value)` and
/// `to_string()` work directly on `Value`, replacing the need to
/// import + call `format_value` everywhere. The free function stays
/// for now (some callers explicitly import it); a follow-up pass can
/// retire it once every site converts.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_value(self))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    UniqueId,
    Int64,
    Float64,
    String,
    Boolean,
    DateTime,
    /// A list-valued column — each cell is a `Value::List`. Heterogeneous
    /// inner values (matches `Value::List(Vec<Value>)`), so no inner type tag.
    List,
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_str = match self {
            ColumnType::UniqueId => "UniqueId",
            ColumnType::Int64 => "Int64",
            ColumnType::Float64 => "Float64",
            ColumnType::String => "String",
            ColumnType::Boolean => "Boolean",
            ColumnType::DateTime => "DateTime",
            ColumnType::List => "List",
        };
        write!(f, "{}", type_str)
    }
}

#[derive(Debug)]
pub struct Column {
    pub(crate) name: String,
    pub(crate) col_type: ColumnType,
    pub(crate) data: ColumnData,
}

#[derive(Debug)]
pub enum ColumnData {
    UniqueId(Vec<Option<u32>>),
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
    Boolean(Vec<Option<bool>>),
    DateTime(Vec<Option<NaiveDate>>),
    /// One `Value::List` payload per cell (None = null). The inner `Vec<Value>`
    /// is the list; values are heterogeneous, mirroring `Value::List`.
    List(Vec<Option<Vec<Value>>>),
}

#[derive(Debug)]
pub struct DataFrame {
    columns: Vec<Column>,
    column_indices: HashMap<String, usize>,
}

impl Column {
    fn get_value(&self, row_idx: usize) -> Option<Value> {
        match &self.data {
            ColumnData::UniqueId(vec) => vec.get(row_idx)?.map(Value::UniqueId),
            ColumnData::Int64(vec) => vec.get(row_idx)?.map(Value::Int64),
            ColumnData::Float64(vec) => vec.get(row_idx)?.map(Value::Float64),
            ColumnData::String(vec) => vec.get(row_idx)?.as_ref().map(|s| Value::String(s.clone())),
            ColumnData::Boolean(vec) => vec.get(row_idx)?.map(Value::Boolean),
            ColumnData::DateTime(vec) => vec.get(row_idx)?.map(Value::DateTime),
            ColumnData::List(vec) => vec.get(row_idx)?.as_ref().map(|v| Value::List(v.clone())),
        }
    }

    fn len(&self) -> usize {
        match &self.data {
            ColumnData::UniqueId(vec) => vec.len(),
            ColumnData::Int64(vec) => vec.len(),
            ColumnData::Float64(vec) => vec.len(),
            ColumnData::String(vec) => vec.len(),
            ColumnData::Boolean(vec) => vec.len(),
            ColumnData::DateTime(vec) => vec.len(),
            ColumnData::List(vec) => vec.len(),
        }
    }
}

impl DataFrame {
    pub fn new(columns: Vec<(String, ColumnType)>) -> Self {
        let mut column_indices = HashMap::with_capacity(columns.len());
        let columns: Vec<Column> = columns
            .into_iter()
            .enumerate()
            .map(|(idx, (name, col_type))| {
                let data = match col_type {
                    ColumnType::UniqueId => ColumnData::UniqueId(Vec::new()),
                    ColumnType::Int64 => ColumnData::Int64(Vec::new()),
                    ColumnType::Float64 => ColumnData::Float64(Vec::new()),
                    ColumnType::String => ColumnData::String(Vec::new()),
                    ColumnType::Boolean => ColumnData::Boolean(Vec::new()),
                    ColumnType::DateTime => ColumnData::DateTime(Vec::new()),
                    ColumnType::List => ColumnData::List(Vec::new()),
                };
                column_indices.insert(name.clone(), idx);
                Column {
                    name,
                    col_type,
                    data,
                }
            })
            .collect();

        DataFrame {
            columns,
            column_indices,
        }
    }

    pub fn get_value(&self, row: usize, column: &str) -> Option<Value> {
        self.column_indices
            .get(column)
            .and_then(|&idx| self.columns.get(idx))
            .and_then(|col| col.get_value(row))
    }

    pub fn get_value_by_index(&self, row_idx: usize, col_idx: usize) -> Option<Value> {
        self.columns
            .get(col_idx)
            .and_then(|col| col.get_value(row_idx))
    }

    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.column_indices.get(name).copied()
    }

    pub fn verify_column(&self, name: &str) -> bool {
        self.column_indices.contains_key(name)
    }

    pub fn row_count(&self) -> usize {
        self.columns.first().map_or(0, |col| col.len())
    }

    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn get_column_names(&self) -> Vec<String> {
        self.columns.iter().map(|col| col.name.clone()).collect()
    }

    pub fn get_column_type(&self, col_name: &str) -> ColumnType {
        self.column_indices
            .get(col_name)
            .and_then(|&idx| self.columns.get(idx))
            .map(|col| col.col_type.clone())
            .unwrap_or_else(|| panic!("Column {} not found", col_name))
    }

    pub fn add_column(
        &mut self,
        name: String,
        col_type: ColumnType,
        data: ColumnData,
    ) -> Result<(), String> {
        if self.column_indices.contains_key(&name) {
            return Err(format!("Column {} already exists", name));
        }

        // Validate that the provided data matches the column type
        match (&col_type, &data) {
            (ColumnType::UniqueId, ColumnData::UniqueId(_))
            | (ColumnType::Int64, ColumnData::Int64(_))
            | (ColumnType::Float64, ColumnData::Float64(_))
            | (ColumnType::String, ColumnData::String(_))
            | (ColumnType::Boolean, ColumnData::Boolean(_))
            | (ColumnType::DateTime, ColumnData::DateTime(_))
            | (ColumnType::List, ColumnData::List(_)) => (),
            _ => return Err(format!("Data type mismatch for column {}", name)),
        }

        let idx = self.columns.len();
        self.column_indices.insert(name.clone(), idx);
        self.columns.push(Column {
            name,
            col_type,
            data,
        });

        Ok(())
    }

    /// Create a DataFrame from Cypher query result rows.
    ///
    /// Converts row-oriented `Vec<Vec<Value>>` (from CypherResult) into the
    /// columnar DataFrame format used by `add_connections` and other fluent APIs.
    ///
    /// Type inference: scans each column for the first non-Null value to determine
    /// ColumnType. All-null columns default to Int64.
    pub fn from_cypher_rows(columns: Vec<String>, rows: Vec<Vec<Value>>) -> Result<Self, String> {
        let num_cols = columns.len();
        let num_rows = rows.len();

        if num_rows == 0 {
            // Empty result: create DataFrame with Int64 columns (no rows)
            let col_specs: Vec<(String, ColumnType)> = columns
                .into_iter()
                .map(|name| (name, ColumnType::Int64))
                .collect();
            return Ok(DataFrame::new(col_specs));
        }

        // Validate row width
        for (i, row) in rows.iter().enumerate() {
            if row.len() != num_cols {
                return Err(format!(
                    "Row {} has {} values but expected {} columns",
                    i,
                    row.len(),
                    num_cols
                ));
            }
        }

        // Infer column types from first non-null value in each column
        let mut col_types = vec![None; num_cols];
        for row in &rows {
            for (col_idx, val) in row.iter().enumerate() {
                if col_types[col_idx].is_some() {
                    continue;
                }
                col_types[col_idx] = match val {
                    Value::UniqueId(_) => Some(ColumnType::UniqueId),
                    Value::Int64(_) => Some(ColumnType::Int64),
                    Value::Float64(_) => Some(ColumnType::Float64),
                    Value::String(_) => Some(ColumnType::String),
                    Value::Boolean(_) => Some(ColumnType::Boolean),
                    Value::DateTime(_) => Some(ColumnType::DateTime),
                    // Timestamp has no dedicated DataFrame column; serialize
                    // via the String column as ISO 8601 (round-trips as text).
                    Value::Timestamp(_) => Some(ColumnType::String),
                    Value::Point { .. } => Some(ColumnType::String), // Serialize as WKT
                    // Durations are query-time-only — never persisted as
                    // a column (Cluster 2). Serialize via the String column.
                    Value::Duration { .. } => Some(ColumnType::String),
                    // Lists get a dedicated columnar shape so they round-trip
                    // structurally (UNWIND/IN), not as stringified JSON.
                    Value::List(_) => Some(ColumnType::List),
                    // The remaining collection / graph-entity variants don't fit
                    // columnar; serialise via the String column (JSON-ish).
                    Value::Map(_) | Value::Node(_) | Value::Relationship(_) | Value::Path(_) => {
                        Some(ColumnType::String)
                    }
                    Value::Null | Value::NodeRef(_) => None,
                };
            }
            if col_types.iter().all(|t| t.is_some()) {
                break;
            }
        }

        // Default all-null columns to Int64
        let col_types: Vec<ColumnType> = col_types
            .into_iter()
            .map(|t| t.unwrap_or(ColumnType::Int64))
            .collect();

        // Build columnar data by transposing rows
        let mut col_data: Vec<ColumnData> = col_types
            .iter()
            .map(|ct| match ct {
                ColumnType::UniqueId => ColumnData::UniqueId(Vec::with_capacity(num_rows)),
                ColumnType::Int64 => ColumnData::Int64(Vec::with_capacity(num_rows)),
                ColumnType::Float64 => ColumnData::Float64(Vec::with_capacity(num_rows)),
                ColumnType::String => ColumnData::String(Vec::with_capacity(num_rows)),
                ColumnType::Boolean => ColumnData::Boolean(Vec::with_capacity(num_rows)),
                ColumnType::DateTime => ColumnData::DateTime(Vec::with_capacity(num_rows)),
                ColumnType::List => ColumnData::List(Vec::with_capacity(num_rows)),
            })
            .collect();

        for row in rows {
            for (col_idx, val) in row.into_iter().enumerate() {
                match &mut col_data[col_idx] {
                    ColumnData::UniqueId(vec) => match val {
                        Value::UniqueId(v) => vec.push(Some(v)),
                        Value::Int64(v) => vec.push(Some(v as u32)),
                        Value::Null => vec.push(None),
                        _ => vec.push(None),
                    },
                    ColumnData::Int64(vec) => match val {
                        Value::Int64(v) => vec.push(Some(v)),
                        Value::UniqueId(v) => vec.push(Some(v as i64)),
                        Value::Float64(v) => vec.push(Some(v as i64)),
                        Value::Null => vec.push(None),
                        _ => vec.push(None),
                    },
                    ColumnData::Float64(vec) => match val {
                        Value::Float64(v) => vec.push(Some(v)),
                        Value::Int64(v) => vec.push(Some(v as f64)),
                        Value::UniqueId(v) => vec.push(Some(v as f64)),
                        Value::Null => vec.push(None),
                        _ => vec.push(None),
                    },
                    ColumnData::String(vec) => match val {
                        Value::String(v) => vec.push(Some(v)),
                        Value::Point { lat, lon } => {
                            vec.push(Some(format!("POINT({} {})", lon, lat)))
                        }
                        Value::Int64(v) => vec.push(Some(v.to_string())),
                        Value::Float64(v) => vec.push(Some(v.to_string())),
                        Value::UniqueId(v) => vec.push(Some(v.to_string())),
                        Value::Boolean(v) => vec.push(Some(v.to_string())),
                        Value::DateTime(v) => vec.push(Some(v.to_string())),
                        Value::Null => vec.push(None),
                        _ => vec.push(None),
                    },
                    ColumnData::Boolean(vec) => match val {
                        Value::Boolean(v) => vec.push(Some(v)),
                        Value::Null => vec.push(None),
                        _ => vec.push(None),
                    },
                    ColumnData::DateTime(vec) => match val {
                        Value::DateTime(v) => vec.push(Some(v)),
                        Value::Null => vec.push(None),
                        _ => vec.push(None),
                    },
                    ColumnData::List(vec) => match val {
                        Value::List(v) => vec.push(Some(v)),
                        Value::Null => vec.push(None),
                        // A non-list value in an inferred-list column is a
                        // heterogeneous mix; store it as a 1-element list so it
                        // isn't silently dropped.
                        other => vec.push(Some(vec![other])),
                    },
                }
            }
        }

        // Assemble DataFrame
        let mut column_indices = HashMap::with_capacity(num_cols);
        let built_columns: Vec<Column> = columns
            .into_iter()
            .zip(col_types)
            .zip(col_data)
            .enumerate()
            .map(|(idx, ((name, col_type), data))| {
                column_indices.insert(name.clone(), idx);
                Column {
                    name,
                    col_type,
                    data,
                }
            })
            .collect();

        Ok(DataFrame {
            columns: built_columns,
            column_indices,
        })
    }

    /// Add a constant-value column (every row gets the same value).
    ///
    /// Used by `add_connections(extra_properties=...)` to stamp static
    /// properties onto edges derived from a Cypher query.
    pub fn add_constant_column(&mut self, name: String, value: Value) -> Result<(), String> {
        let num_rows = self.row_count();
        let (col_type, data) = match value {
            Value::UniqueId(v) => (
                ColumnType::UniqueId,
                ColumnData::UniqueId(vec![Some(v); num_rows]),
            ),
            Value::Int64(v) => (
                ColumnType::Int64,
                ColumnData::Int64(vec![Some(v); num_rows]),
            ),
            Value::Float64(v) => (
                ColumnType::Float64,
                ColumnData::Float64(vec![Some(v); num_rows]),
            ),
            Value::String(v) => (
                ColumnType::String,
                ColumnData::String(vec![Some(v); num_rows]),
            ),
            Value::Boolean(v) => (
                ColumnType::Boolean,
                ColumnData::Boolean(vec![Some(v); num_rows]),
            ),
            Value::DateTime(v) => (
                ColumnType::DateTime,
                ColumnData::DateTime(vec![Some(v); num_rows]),
            ),
            Value::Timestamp(v) => (
                ColumnType::String,
                ColumnData::String(vec![
                    Some(v.format("%Y-%m-%dT%H:%M:%S").to_string());
                    num_rows
                ]),
            ),
            Value::Null => return Err("Cannot add a constant column with Null value".to_string()),
            Value::Point { lat, lon } => (
                ColumnType::String,
                ColumnData::String(vec![Some(format!("POINT({} {})", lon, lat)); num_rows]),
            ),
            Value::NodeRef(_) => {
                return Err("Cannot add a constant column with NodeRef value".to_string())
            }
            Value::Duration { .. } => {
                return Err(
                    "Cannot add a constant column with Duration value — durations are \
                     query-time-only (0.9.0 Cluster 2)"
                        .to_string(),
                )
            }
            // Phase A.1 — collection / graph-entity variants don't
            // fit columnar storage. Same rationale as Duration: these
            // are query-result-time values, not column types.
            Value::List(_)
            | Value::Map(_)
            | Value::Node(_)
            | Value::Relationship(_)
            | Value::Path(_) => {
                return Err(
                    "Cannot add a constant column with List/Map/Node/Relationship/Path value \
                     — collection and graph-entity variants are query-result-time values, \
                     not column types"
                        .to_string(),
                )
            }
        };
        self.add_column(name, col_type, data)
    }
}

impl std::fmt::Display for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let row_limit = 10.min(self.row_count());
        let columns = self.get_column_names();

        // Determine max width for each column
        let mut col_widths: Vec<usize> = columns.iter().map(|col| col.len()).collect();

        // Adjust widths based on values and column types
        for (col_idx, col) in self.columns.iter().enumerate() {
            // Include column type width
            let type_width = format_col_type(&col.col_type).len();
            col_widths[col_idx] = col_widths[col_idx].max(type_width);

            // Include value widths
            for row_idx in 0..row_limit {
                if let Some(value) = col.get_value(row_idx) {
                    col_widths[col_idx] = col_widths[col_idx].max(format_value(&value).len());
                }
            }
        }

        // Format helper
        let format_row = |values: Vec<String>| -> String {
            values
                .into_iter()
                .enumerate()
                .map(|(i, val)| format!(" {:^width$} ", val, width = col_widths[i]))
                .collect::<Vec<_>>()
                .join("|")
        };

        // Print headers
        writeln!(f, "\n| #  |{}|", format_row(columns))?;

        // Print column types
        let type_row: Vec<String> = self
            .columns
            .iter()
            .map(|col| format_col_type(&col.col_type))
            .collect();
        writeln!(f, "|    |{}|", format_row(type_row))?;

        // Print separator
        let separator = col_widths
            .iter()
            .map(|w| format!("{:-^width$}", "-", width = w + 2))
            .collect::<Vec<_>>()
            .join("|");
        writeln!(f, "|----|{}|", separator)?;

        // Print data rows
        for row_idx in 0..row_limit {
            let row_data: Vec<String> = (0..self.column_count())
                .map(|col_idx| {
                    format_value(
                        &self
                            .get_value_by_index(row_idx, col_idx)
                            .unwrap_or(Value::Null),
                    )
                })
                .collect();
            writeln!(f, "| {:^2} |{}|", row_idx, format_row(row_data))?;
        }

        // Show if there are more rows
        if self.row_count() > row_limit {
            let more_row = format_row(col_widths.iter().map(|_| "...".to_string()).collect());
            writeln!(f, "| .. |{}|", more_row)?;
        }

        Ok(())
    }
}

/// Render a `Value` as a plain unquoted string — the form used for CSV
/// cells, XML escaping, agent-facing human display, etc. Distinct from
/// [`format_value`] which produces a Cypher-literal-style rendering
/// (quoted strings, `NULL` for null, `%.2f` for floats).
///
/// `Null` → empty string. The Phase A.1 collection / graph-entity
/// variants delegate to [`format_value`] (their multi-line shapes are
/// the same in both contexts).
///
/// Consolidated 0.9.53 from three nearly-identical copies in
/// `graph/mod.rs`, `graph/explore.rs`, `graph/io/export.rs`.
pub fn raw_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Int64(n) => n.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::DateTime(dt) => dt.to_string(),
        Value::Timestamp(dt) => dt.to_string(),
        Value::UniqueId(id) => id.to_string(),
        Value::Point { lat, lon } => format!("point({}, {})", lat, lon),
        Value::Duration {
            months,
            days,
            seconds,
        } => format!("duration(M={}, D={}, S={})", months, days, seconds),
        Value::Null => String::new(),
        Value::NodeRef(idx) => format!("node#{}", idx),
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => format_value(value),
    }
}

pub fn format_value(value: &Value) -> String {
    match value {
        Value::UniqueId(v) => format!("{}", v),
        Value::Int64(v) => format!("{}", v),
        Value::Float64(v) => {
            if v.is_nan() {
                "NULL".to_string()
            } else {
                format!("{:.2}", v)
            }
        }
        Value::String(v) => format!("\"{}\"", v),
        Value::Boolean(v) => format!("{}", v),
        Value::DateTime(v) => format!("\"{}\"", v.format("%Y-%m-%d")),
        Value::Timestamp(v) => format!("\"{}\"", v.format("%Y-%m-%dT%H:%M:%S")),
        Value::Point { lat, lon } => format!("point({}, {})", lat, lon),
        Value::Null => "NULL".to_string(),
        Value::NodeRef(idx) => format!("node#{}", idx),
        Value::Duration {
            months,
            days,
            seconds,
        } => format!(
            "duration(months={}, days={}, seconds={})",
            months, days, seconds
        ),
        // Phase A.1 — Cypher-ish surface syntax for the collection /
        // graph-entity variants. Not round-trip-parseable; this fn is
        // for display / debug, not serialisation.
        Value::List(items) => {
            let inner: Vec<String> = items.iter().map(format_value).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Map(entries) => {
            let inner: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", k, format_value(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Node(n) => {
            format!("(:{} {{id: {}}})", n.labels.join(":"), n.id)
        }
        Value::Relationship(r) => {
            format!(
                "[:{} {{id: {}, start: {}, end: {}}}]",
                r.rel_type, r.id, r.start_id, r.end_id
            )
        }
        Value::Path(p) => {
            format!("path(nodes={}, rels={})", p.nodes.len(), p.rels.len())
        }
    }
}

fn format_col_type(col_type: &ColumnType) -> String {
    match col_type {
        ColumnType::UniqueId => "uID",
        ColumnType::Int64 => "i64",
        ColumnType::Float64 => "f64",
        ColumnType::String => "str",
        ColumnType::Boolean => "bool",
        ColumnType::DateTime => "datetime",
        ColumnType::List => "list",
    }
    .to_string()
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    // ========================================================================
    // Value::as_string
    // ========================================================================

    #[test]
    fn test_as_string_with_string_value() {
        let v = Value::String("hello".to_string());
        assert_eq!(v.as_string(), Some("hello".to_string()));
    }

    #[test]
    fn test_timestamp_roundtrip_order_and_meta() {
        use chrono::{NaiveDate, NaiveDateTime};
        let dt: NaiveDateTime = NaiveDate::from_ymd_opt(2024, 3, 15)
            .unwrap()
            .and_hms_opt(10, 30, 45)
            .unwrap();
        let v = Value::Timestamp(dt);

        // type_name + display carry the time component.
        assert_eq!(v.type_name(), "Timestamp");
        assert_eq!(format_value(&v), "\"2024-03-15T10:30:45\"");

        // serde round-trip (the .kgl path for Mixed columns).
        let bytes = bincode::serialize(&v).unwrap();
        assert_eq!(bincode::deserialize::<Value>(&bytes).unwrap(), v);

        // Appended last → discriminant 15 unchanged-prefix property:
        // a date-only value still orders before any timestamp.
        let date = Value::DateTime(NaiveDate::from_ymd_opt(2024, 3, 15).unwrap());
        assert!(date < v);

        // Same-variant ordering is chronological.
        let later = Value::Timestamp(
            NaiveDate::from_ymd_opt(2024, 3, 15)
                .unwrap()
                .and_hms_opt(10, 30, 46)
                .unwrap(),
        );
        assert!(v < later);
    }

    #[test]
    fn test_as_string_with_non_string_values() {
        assert_eq!(Value::Int64(42).as_string(), None);
        assert_eq!(Value::Float64(3.14).as_string(), None);
        assert_eq!(Value::Boolean(true).as_string(), None);
        assert_eq!(Value::Null.as_string(), None);
        assert_eq!(Value::UniqueId(1).as_string(), None);
    }

    // ========================================================================
    // Value equality and hash
    // ========================================================================

    #[test]
    fn test_value_equality_same_types() {
        assert_eq!(Value::Int64(42), Value::Int64(42));
        assert_eq!(Value::Float64(3.14), Value::Float64(3.14));
        assert_eq!(
            Value::String("a".to_string()),
            Value::String("a".to_string())
        );
        assert_eq!(Value::Boolean(true), Value::Boolean(true));
        assert_eq!(Value::Null, Value::Null);
        assert_eq!(Value::UniqueId(5), Value::UniqueId(5));
    }

    #[test]
    fn test_value_inequality() {
        assert_ne!(Value::Int64(1), Value::Int64(2));
        assert_ne!(
            Value::String("a".to_string()),
            Value::String("b".to_string())
        );
        assert_ne!(Value::Boolean(true), Value::Boolean(false));
    }

    #[test]
    fn test_value_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Value::Int64(42));
        set.insert(Value::Int64(42)); // duplicate
        assert_eq!(set.len(), 1);

        set.insert(Value::String("test".to_string()));
        assert_eq!(set.len(), 2);

        set.insert(Value::Null);
        set.insert(Value::Null); // duplicate
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_float_hash_negative_zero() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Value::Float64(0.0));
        set.insert(Value::Float64(-0.0));
        // 0.0 and -0.0 should hash the same
        assert_eq!(set.len(), 1);
    }

    // ========================================================================
    // format_value
    // ========================================================================

    #[test]
    fn test_format_value_types() {
        assert_eq!(format_value(&Value::UniqueId(42)), "42");
        assert_eq!(format_value(&Value::Int64(-5)), "-5");
        assert_eq!(format_value(&Value::Float64(3.14)), "3.14");
        assert_eq!(format_value(&Value::String("hi".to_string())), "\"hi\"");
        assert_eq!(format_value(&Value::Boolean(true)), "true");
        assert_eq!(format_value(&Value::Null), "NULL");
    }

    #[test]
    fn test_format_value_nan_is_null() {
        assert_eq!(format_value(&Value::Float64(f64::NAN)), "NULL");
    }

    // ========================================================================
    // ColumnType Display
    // ========================================================================

    #[test]
    fn test_column_type_display() {
        assert_eq!(format!("{}", ColumnType::UniqueId), "UniqueId");
        assert_eq!(format!("{}", ColumnType::Int64), "Int64");
        assert_eq!(format!("{}", ColumnType::Float64), "Float64");
        assert_eq!(format!("{}", ColumnType::String), "String");
        assert_eq!(format!("{}", ColumnType::Boolean), "Boolean");
        assert_eq!(format!("{}", ColumnType::DateTime), "DateTime");
    }

    // ========================================================================
    // DataFrame
    // ========================================================================

    #[test]
    fn test_dataframe_new_empty() {
        let df = DataFrame::new(vec![
            ("id".to_string(), ColumnType::Int64),
            ("name".to_string(), ColumnType::String),
        ]);
        assert_eq!(df.row_count(), 0);
        assert_eq!(df.column_count(), 2);
        assert!(df.verify_column("id"));
        assert!(df.verify_column("name"));
        assert!(!df.verify_column("missing"));
    }

    #[test]
    fn test_dataframe_column_names() {
        let df = DataFrame::new(vec![
            ("a".to_string(), ColumnType::Int64),
            ("b".to_string(), ColumnType::String),
        ]);
        let names = df.get_column_names();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_dataframe_column_type() {
        let df = DataFrame::new(vec![
            ("id".to_string(), ColumnType::Int64),
            ("name".to_string(), ColumnType::String),
        ]);
        assert_eq!(df.get_column_type("id"), ColumnType::Int64);
        assert_eq!(df.get_column_type("name"), ColumnType::String);
    }

    #[test]
    fn test_dataframe_add_column() {
        let mut df = DataFrame::new(vec![("id".to_string(), ColumnType::Int64)]);
        let result = df.add_column(
            "name".to_string(),
            ColumnType::String,
            ColumnData::String(vec![]),
        );
        assert!(result.is_ok());
        assert_eq!(df.column_count(), 2);
    }

    #[test]
    fn test_dataframe_add_duplicate_column() {
        let mut df = DataFrame::new(vec![("id".to_string(), ColumnType::Int64)]);
        let result = df.add_column(
            "id".to_string(),
            ColumnType::Int64,
            ColumnData::Int64(vec![]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_add_column_type_mismatch() {
        let mut df = DataFrame::new(vec![]);
        let result = df.add_column(
            "x".to_string(),
            ColumnType::Int64,
            ColumnData::String(vec![]),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_get_column_index() {
        let df = DataFrame::new(vec![
            ("a".to_string(), ColumnType::Int64),
            ("b".to_string(), ColumnType::String),
        ]);
        assert_eq!(df.get_column_index("a"), Some(0));
        assert_eq!(df.get_column_index("b"), Some(1));
        assert_eq!(df.get_column_index("c"), None);
    }
}

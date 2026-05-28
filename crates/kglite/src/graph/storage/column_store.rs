// src/graph/column_store.rs
//
// Per-type columnar property storage. Each node type gets a ColumnStore
// containing one TypedColumn per property key. Rows map 1:1 to nodes
// via a u32 row_id stored in PropertyStorage::Columnar.
//
// TypedColumn uses MmapOrVec<T> for fixed-size types (i64, f64, u32, bool, i32)
// and MmapBytes for string data. Mixed columns stay heap-only (Vec<Value>).

use crate::datatypes::values::Value;
use crate::graph::schema::{InternedKey, StringInterner, TypeSchema};
use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
use chrono::NaiveDate;
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

// ─── TypedColumn ─────────────────────────────────────────────────────────────

/// A single column of homogeneously-typed property values.
/// Column type is determined from `node_type_metadata` at construction time.
/// Falls back to `Mixed` for heterogeneous or unknown types.
///
/// Fixed-size columns use `MmapOrVec<T>` which can be heap- or file-backed.
/// String columns use `MmapOrVec<u64>` for offsets and `MmapBytes` for UTF-8 data.
/// Mixed columns use plain `Vec<Value>` (not mmap-eligible).
#[derive(Debug, Clone)]
pub enum TypedColumn {
    Int64 {
        data: MmapOrVec<i64>,
        nulls: MmapOrVec<u8>, // 0 = non-null, 1 = null
    },
    Float64 {
        data: MmapOrVec<f64>,
        nulls: MmapOrVec<u8>,
    },
    UniqueId {
        data: MmapOrVec<u32>,
        nulls: MmapOrVec<u8>,
    },
    Bool {
        data: MmapOrVec<u8>, // 0 = false, 1 = true
        nulls: MmapOrVec<u8>,
    },
    /// Days since Unix epoch (1970-01-01)
    Date {
        data: MmapOrVec<i32>,
        nulls: MmapOrVec<u8>,
    },
    /// Offset-based string storage: `offsets[i]..offsets[i+1]` is the byte range in `data`.
    /// Updates land in `relocated` instead of mutating `offsets`/`data` — rewriting
    /// `offsets[i+1]` corrupts the start of row `i+1`. `write_to` folds the overlay
    /// back into the canonical (offsets, data) layout on save.
    Str {
        offsets: MmapOrVec<u64>,
        data: MmapBytes,
        nulls: MmapOrVec<u8>,
        relocated: HashMap<u32, String>,
    },
    /// Fallback for heterogeneous columns — stores boxed Values directly.
    /// Cannot be mmap'd, but preserves correctness.
    Mixed { data: Vec<Value> },
}

/// Number of days from the Unix epoch to chrono's internal epoch.
/// Column data smaller than this threshold is loaded into heap Vec instead of
/// being written to a temp file and mmap'd. Avoids file I/O overhead for small columns.
const MMAP_THRESHOLD: usize = 262_144; // 256 KB

const UNIX_EPOCH_DATE: NaiveDate = match NaiveDate::from_ymd_opt(1970, 1, 1) {
    Some(d) => d,
    None => unreachable!(),
};

impl TypedColumn {
    /// Create an empty column of the given type based on metadata type string.
    /// Matching is case-insensitive (metadata stores "Int64", "String", etc.).
    pub fn from_type_str(type_str: &str) -> Self {
        match type_str.to_ascii_lowercase().as_str() {
            "int64" => TypedColumn::Int64 {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "float64" => TypedColumn::Float64 {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "uniqueid" => TypedColumn::UniqueId {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "bool" | "boolean" => TypedColumn::Bool {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "date" | "datetime" => TypedColumn::Date {
                data: MmapOrVec::new(),
                nulls: MmapOrVec::new(),
            },
            "string" => TypedColumn::Str {
                offsets: {
                    let mut o = MmapOrVec::new();
                    o.push(0u64);
                    o
                },
                data: MmapBytes::new(),
                nulls: MmapOrVec::new(),
                relocated: HashMap::new(),
            },
            _ => TypedColumn::Mixed { data: Vec::new() },
        }
    }

    /// Number of rows in this column.
    pub fn len(&self) -> usize {
        match self {
            TypedColumn::Int64 { nulls, .. }
            | TypedColumn::Float64 { nulls, .. }
            | TypedColumn::UniqueId { nulls, .. }
            | TypedColumn::Bool { nulls, .. }
            | TypedColumn::Date { nulls, .. }
            | TypedColumn::Str { nulls, .. } => nulls.len(),
            TypedColumn::Mixed { data } => data.len(),
        }
    }

    /// Push a value onto this column. Returns Ok(()) on success,
    /// Err(value) if the value type doesn't match (caller should demote to Mixed).
    pub fn push(&mut self, value: &Value) -> Result<(), ()> {
        match (self, value) {
            (TypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                data.push(*v);
                nulls.push(0);
            }
            (TypedColumn::Int64 { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(1);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                data.push(*v);
                nulls.push(0);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Int64(v)) => {
                // Allow int→float promotion (common from pandas)
                data.push(*v as f64);
                nulls.push(0);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Null) => {
                data.push(0.0);
                nulls.push(1);
            }
            (TypedColumn::UniqueId { data, nulls }, Value::UniqueId(v)) => {
                data.push(*v);
                nulls.push(0);
            }
            (TypedColumn::UniqueId { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(1);
            }
            (TypedColumn::Bool { data, nulls }, Value::Boolean(v)) => {
                data.push(*v as u8);
                nulls.push(0);
            }
            (TypedColumn::Bool { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(1);
            }
            (TypedColumn::Date { data, nulls }, Value::DateTime(d)) => {
                let days = (*d - UNIX_EPOCH_DATE).num_days() as i32;
                data.push(days);
                nulls.push(0);
            }
            (TypedColumn::Date { data, nulls }, Value::Null) => {
                data.push(0);
                nulls.push(1);
            }
            (
                TypedColumn::Str {
                    offsets,
                    data,
                    nulls,
                    ..
                },
                Value::String(s),
            ) => {
                data.extend(s.as_bytes());
                offsets.push(data.len() as u64);
                nulls.push(0);
            }
            (TypedColumn::Str { offsets, nulls, .. }, Value::Null) => {
                // Null string: push same offset (zero-length range)
                let last = if !offsets.is_empty() {
                    offsets.get(offsets.len() - 1)
                } else {
                    0
                };
                offsets.push(last);
                nulls.push(1);
            }
            (TypedColumn::Mixed { data }, value) => {
                data.push(value.clone());
            }
            _ => return Err(()),
        }
        Ok(())
    }

    /// Read the value at the given row index.
    pub fn get(&self, row: u32) -> Option<Value> {
        let idx = row as usize;
        match self {
            TypedColumn::Int64 { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                Some(Value::Int64(data.get(idx)))
            }
            TypedColumn::Float64 { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                Some(Value::Float64(data.get(idx)))
            }
            TypedColumn::UniqueId { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                Some(Value::UniqueId(data.get(idx)))
            }
            TypedColumn::Bool { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                Some(Value::Boolean(data.get(idx) != 0))
            }
            TypedColumn::Date { data, nulls } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                let date = UNIX_EPOCH_DATE + chrono::Duration::days(data.get(idx) as i64);
                Some(Value::DateTime(date))
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => {
                if idx >= nulls.len() {
                    return None;
                }
                if nulls.get(idx) != 0 {
                    return None;
                }
                if let Some(s) = relocated.get(&row) {
                    return Some(Value::String(s.clone()));
                }
                let start = offsets.get(idx) as usize;
                let end = offsets.get(idx + 1) as usize;
                let bytes = data.slice(start, end);
                // SAFETY: We only write valid UTF-8 via Value::String, so skip validation.
                let s = unsafe { String::from_utf8_unchecked(bytes.to_vec()) };
                Some(Value::String(s))
            }
            TypedColumn::Mixed { data } => {
                let val = data.get(idx)?;
                if matches!(val, Value::Null) {
                    return None;
                }
                Some(val.clone())
            }
        }
    }

    /// Get a string column value as a borrowed &str, avoiding heap allocation.
    /// Returns None if the column is not a Str variant, row is out of bounds, or null.
    #[inline]
    pub fn get_str(&self, row: u32) -> Option<&str> {
        let idx = row as usize;
        match self {
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => {
                if idx >= nulls.len() || nulls.get(idx) != 0 {
                    return None;
                }
                if let Some(s) = relocated.get(&row) {
                    return Some(s.as_str());
                }
                let start = offsets.get(idx) as usize;
                let end = offsets.get(idx + 1) as usize;
                let bytes = data.slice(start, end);
                // SAFETY: We only write valid UTF-8 via Value::String.
                Some(unsafe { std::str::from_utf8_unchecked(bytes) })
            }
            _ => None,
        }
    }

    /// Update the value at the given row index.
    /// Returns Ok(()) on success, Err(()) on type mismatch.
    pub fn set(&mut self, row: u32, value: &Value) -> Result<(), ()> {
        let idx = row as usize;
        match (self, value) {
            (TypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v);
                nulls.set(idx, 0);
            }
            (TypedColumn::Int64 { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0);
                nulls.set(idx, 1);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v);
                nulls.set(idx, 0);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Int64(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v as f64);
                nulls.set(idx, 0);
            }
            (TypedColumn::Float64 { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0.0);
                nulls.set(idx, 1);
            }
            (TypedColumn::UniqueId { data, nulls }, Value::UniqueId(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v);
                nulls.set(idx, 0);
            }
            (TypedColumn::UniqueId { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0);
                nulls.set(idx, 1);
            }
            (TypedColumn::Bool { data, nulls }, Value::Boolean(v)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, *v as u8);
                nulls.set(idx, 0);
            }
            (TypedColumn::Bool { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0);
                nulls.set(idx, 1);
            }
            (TypedColumn::Date { data, nulls }, Value::DateTime(d)) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, (*d - UNIX_EPOCH_DATE).num_days() as i32);
                nulls.set(idx, 0);
            }
            (TypedColumn::Date { data, nulls }, Value::Null) => {
                if idx >= data.len() {
                    return Err(());
                }
                data.set(idx, 0);
                nulls.set(idx, 1);
            }
            (
                TypedColumn::Str {
                    nulls, relocated, ..
                },
                Value::String(s),
            ) => {
                if idx >= nulls.len() {
                    return Err(());
                }
                // Park the new value in the relocated overlay. Mutating
                // `offsets[idx+1]` in place corrupts row idx+1's start —
                // see write_to for the on-save compaction.
                relocated.insert(row, s.clone());
                nulls.set(idx, 0);
            }
            (
                TypedColumn::Str {
                    nulls, relocated, ..
                },
                Value::Null,
            ) => {
                if idx >= nulls.len() {
                    return Err(());
                }
                relocated.remove(&row);
                nulls.set(idx, 1);
            }
            (TypedColumn::Mixed { data }, value) => {
                if idx >= data.len() {
                    return Err(());
                }
                data[idx] = value.clone();
            }
            _ => return Err(()),
        }
        Ok(())
    }

    /// Push a null value for this column type.
    pub fn push_null(&mut self) {
        let _ = self.push(&Value::Null);
    }

    /// Whether this column's data is currently file-backed.
    pub fn is_mapped(&self) -> bool {
        match self {
            TypedColumn::Int64 { data, .. } => data.is_mapped(),
            TypedColumn::Float64 { data, .. } => data.is_mapped(),
            TypedColumn::UniqueId { data, .. } => data.is_mapped(),
            TypedColumn::Bool { data, .. } => data.is_mapped(),
            TypedColumn::Date { data, .. } => data.is_mapped(),
            TypedColumn::Str { data, .. } => data.is_mapped(),
            TypedColumn::Mixed { .. } => false,
        }
    }

    /// Heap-resident bytes across all sub-buffers (0 if fully mmap'd).
    pub fn heap_bytes(&self) -> usize {
        match self {
            TypedColumn::Int64 { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Float64 { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::UniqueId { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Bool { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Date { data, nulls } => data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => {
                let relocated_bytes: usize = relocated.values().map(|s| s.capacity()).sum();
                offsets.heap_bytes() + data.heap_bytes() + nulls.heap_bytes() + relocated_bytes
            }
            TypedColumn::Mixed { data } => data.len() * std::mem::size_of::<Value>(),
        }
    }

    /// Materialize this column's data to file-backed mmap.
    /// `base_path` is the directory; files are named `{col_name}.{ext}`.
    pub fn materialize_to_file(&mut self, base_dir: &Path, col_name: &str) -> io::Result<()> {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.i64")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Float64 { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.f64")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::UniqueId { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.u32")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Bool { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.bool")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Date { data, nulls } => {
                data.materialize_to_file(&base_dir.join(format!("{col_name}.i32")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                ..
            } => {
                offsets.materialize_to_file(&base_dir.join(format!("{col_name}.off")))?;
                data.materialize_to_file(&base_dir.join(format!("{col_name}.str")))?;
                nulls.materialize_to_file(&base_dir.join(format!("{col_name}.null")))?;
            }
            TypedColumn::Mixed { .. } => {
                // Mixed columns cannot be mmap'd — no-op
            }
        }
        Ok(())
    }

    /// Flush dirty mmap pages to disk (msync) and advise the kernel to
    /// drop them from page cache. Heap-backed columns are no-ops. See
    /// `MmapOrVec::flush_and_release_pages` for the contract.
    #[allow(dead_code)]
    pub fn flush_and_release_pages(&self) -> io::Result<()> {
        let mut first: Option<io::Error> = None;
        let mut record = |r: io::Result<()>| {
            if let Err(e) = r {
                first.get_or_insert(e);
            }
        };
        match self {
            TypedColumn::Int64 { data, nulls } => {
                record(data.flush_and_release_pages());
                record(nulls.flush_and_release_pages());
            }
            TypedColumn::Float64 { data, nulls } => {
                record(data.flush_and_release_pages());
                record(nulls.flush_and_release_pages());
            }
            TypedColumn::UniqueId { data, nulls } => {
                record(data.flush_and_release_pages());
                record(nulls.flush_and_release_pages());
            }
            TypedColumn::Bool { data, nulls } => {
                record(data.flush_and_release_pages());
                record(nulls.flush_and_release_pages());
            }
            TypedColumn::Date { data, nulls } => {
                record(data.flush_and_release_pages());
                record(nulls.flush_and_release_pages());
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                ..
            } => {
                record(offsets.flush_and_release_pages());
                record(data.flush_and_release_pages());
                record(nulls.flush_and_release_pages());
            }
            TypedColumn::Mixed { .. } => {} // heap only — no mmap to flush
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Convert this column back to heap-backed storage.
    #[allow(dead_code)] // Test-only chain (ColumnStore::materialize_to_heap).
    pub fn materialize_to_heap(&mut self) {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Float64 { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::UniqueId { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Bool { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Date { data, nulls } => {
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                ..
            } => {
                offsets.materialize_to_heap();
                data.materialize_to_heap();
                nulls.materialize_to_heap();
            }
            TypedColumn::Mixed { .. } => {} // already heap
        }
    }

    /// Write column data to a writer (for v3 packed format).
    /// Writes data bytes, then null bytes. For strings: offsets + data + nulls.
    /// For mixed: bincode-serialized Vec<Value>.
    pub fn write_to(&self, writer: &mut impl io::Write) -> io::Result<()> {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::Float64 { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::UniqueId { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::Bool { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::Date { data, nulls } => {
                data.write_to(writer)?;
                nulls.write_to(writer)?;
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => {
                if relocated.is_empty() {
                    // Fast path: no overlay, write raw buffers.
                    offsets.write_to(writer)?;
                    data.write_to(writer)?;
                    nulls.write_to(writer)?;
                } else {
                    // Fold the relocated overlay back into a fresh
                    // offsets+data layout. The on-disk format expects
                    // N+1 offsets + concatenated data + N null bytes.
                    let n = nulls.len();
                    let mut new_offsets: Vec<u64> = Vec::with_capacity(n + 1);
                    let mut new_data: Vec<u8> = Vec::new();
                    new_offsets.push(0);
                    for row in 0..n {
                        if nulls.get(row) == 0 {
                            let bytes: Vec<u8> = if let Some(s) = relocated.get(&(row as u32)) {
                                s.as_bytes().to_vec()
                            } else {
                                let start = offsets.get(row) as usize;
                                let end = offsets.get(row + 1) as usize;
                                data.slice(start, end).to_vec()
                            };
                            new_data.extend_from_slice(&bytes);
                        }
                        new_offsets.push(new_data.len() as u64);
                    }
                    for off in &new_offsets {
                        writer.write_all(&off.to_le_bytes())?;
                    }
                    writer.write_all(&new_data)?;
                    nulls.write_to(writer)?;
                }
            }
            TypedColumn::Mixed { data } => {
                let encoded = bincode::serialize(data)
                    .map_err(|e| io::Error::other(format!("bincode error: {}", e)))?;
                writer.write_all(&encoded)?;
            }
        }
        Ok(())
    }

    /// Return the type tag string for serialization.
    pub fn type_tag(&self) -> &'static str {
        match self {
            TypedColumn::Int64 { .. } => "int64",
            TypedColumn::Float64 { .. } => "float64",
            TypedColumn::UniqueId { .. } => "uniqueid",
            TypedColumn::Bool { .. } => "bool",
            TypedColumn::Date { .. } => "date",
            TypedColumn::Str { .. } => "string",
            TypedColumn::Mixed { .. } => "mixed",
        }
    }
}

// ─── ColumnStore ─────────────────────────────────────────────────────────────

/// Per-node-type columnar store. Holds one TypedColumn per property key.
/// All columns have the same number of rows.
#[derive(Debug)]
pub struct ColumnStore {
    /// Schema mapping property keys to slot indices (shared with Compact storage)
    schema: Arc<TypeSchema>,
    /// One column per property key, indexed by slot index from schema
    columns: Vec<TypedColumn>,
    /// Number of rows (nodes of this type)
    row_count: u32,
    /// Tombstone bitmap: true = row deleted
    tombstones: Vec<bool>,
    /// Node ID column (mapped mode only). When present, NodeData.id is Value::Null sentinel.
    id_column: Option<TypedColumn>,
    /// Node title column (mapped mode only). When present, NodeData.title is Value::Null sentinel.
    title_column: Option<TypedColumn>,
    /// Overflow bag for sparse properties: offset array + data blob.
    overflow_offsets: Option<MmapOrVec<u64>>,
    overflow_data: Option<MmapBytes>,
    /// Optional mmap-backed store for disk mode. When present, get/get_id/get_title
    /// delegate to this instead of the TypedColumn arrays above.
    mmap_store: Option<Arc<crate::graph::storage::mapped::column_store::MmapColumnStore>>,
}

impl Clone for ColumnStore {
    fn clone(&self) -> Self {
        ColumnStore {
            schema: self.schema.clone(),
            columns: self.columns.clone(),
            row_count: self.row_count,
            tombstones: self.tombstones.clone(),
            mmap_store: self.mmap_store.clone(),
            id_column: self.id_column.clone(),
            title_column: self.title_column.clone(),
            overflow_offsets: self.overflow_offsets.clone(),
            overflow_data: self.overflow_data.clone(),
        }
    }
}

impl ColumnStore {
    /// Create a new ColumnStore from a TypeSchema and type metadata.
    /// `type_meta` maps property name → type string (e.g., "int64", "string").
    pub fn new(
        schema: Arc<TypeSchema>,
        type_meta: &HashMap<String, String>,
        interner: &StringInterner,
    ) -> Self {
        let mut columns = Vec::with_capacity(schema.len());
        for (_slot, ik) in schema.iter() {
            let prop_name = interner.resolve(ik);
            let type_str = type_meta
                .get(prop_name)
                .map(|s| s.as_str())
                .unwrap_or("mixed");
            columns.push(TypedColumn::from_type_str(type_str));
        }
        ColumnStore {
            schema,
            columns,
            row_count: 0,
            tombstones: Vec::new(),
            id_column: None,
            title_column: None,
            overflow_offsets: None,
            overflow_data: None,
            mmap_store: None,
        }
    }

    /// Create a ColumnStore from an existing schema with all Mixed columns (for unknown types).
    #[allow(dead_code)] // Test-only.
    pub fn new_mixed(schema: Arc<TypeSchema>) -> Self {
        let columns = (0..schema.len())
            .map(|_| TypedColumn::Mixed { data: Vec::new() })
            .collect();
        ColumnStore {
            schema,
            columns,
            row_count: 0,
            tombstones: Vec::new(),
            id_column: None,
            title_column: None,
            overflow_offsets: None,
            overflow_data: None,
            mmap_store: None,
        }
    }

    /// Create a ColumnStore backed by a shared mmap (disk mode).
    /// All get/get_id/get_title calls delegate to the MmapColumnStore.
    pub fn from_mmap_store(
        mmap_store: Arc<crate::graph::storage::mapped::column_store::MmapColumnStore>,
    ) -> Self {
        let rc = mmap_store.row_count();
        ColumnStore {
            schema: Arc::new(TypeSchema::new()),
            columns: Vec::new(),
            row_count: rc,
            tombstones: Vec::new(),
            id_column: None,
            title_column: None,
            overflow_offsets: None,
            overflow_data: None,
            mmap_store: Some(mmap_store),
        }
    }

    /// Look up a property in the overflow bag for a given row.
    /// Scans the bag entries for the matching key.
    pub fn get_overflow_property(&self, row_id: u32, key: InternedKey) -> Option<Value> {
        let offsets = self.overflow_offsets.as_ref()?;
        let data = self.overflow_data.as_ref()?;
        let idx = row_id as usize;
        if idx + 1 >= offsets.len() {
            return None;
        }
        let start = offsets.get(idx) as usize;
        let end = offsets.get(idx + 1) as usize;
        if start >= end || end > data.len() {
            return None;
        }
        let blob = data.slice(start, end);
        Self::scan_overflow_blob(blob, key)
    }

    /// Decode all properties from an overflow blob for a given row.
    fn overflow_row_properties(&self, row_id: u32) -> Vec<(InternedKey, Value)> {
        let offsets = match self.overflow_offsets.as_ref() {
            Some(o) => o,
            None => return Vec::new(),
        };
        let data = match self.overflow_data.as_ref() {
            Some(d) => d,
            None => return Vec::new(),
        };
        let idx = row_id as usize;
        if idx + 1 >= offsets.len() {
            return Vec::new();
        }
        let start = offsets.get(idx) as usize;
        let end = offsets.get(idx + 1) as usize;
        if start >= end || end > data.len() {
            return Vec::new();
        }
        let blob = data.slice(start, end);
        Self::decode_overflow_blob(blob)
    }

    /// Scan an overflow blob for a specific key. Returns the value if found.
    fn scan_overflow_blob(blob: &[u8], key: InternedKey) -> Option<Value> {
        let target = key.as_u64();
        if blob.len() < 2 {
            return None;
        }
        let num_entries = u16::from_le_bytes([blob[0], blob[1]]) as usize;
        let mut pos = 2;
        for _ in 0..num_entries {
            if pos + 9 > blob.len() {
                break;
            }
            let entry_key = u64::from_le_bytes(blob[pos..pos + 8].try_into().ok()?);
            let type_tag = blob[pos + 8];
            pos += 9;
            if entry_key == target {
                return Self::read_overflow_value(blob, &mut pos, type_tag);
            }
            // Skip value
            Self::skip_overflow_value(blob, &mut pos, type_tag);
        }
        None
    }

    /// Decode all entries from an overflow blob.
    fn decode_overflow_blob(blob: &[u8]) -> Vec<(InternedKey, Value)> {
        if blob.len() < 2 {
            return Vec::new();
        }
        let num_entries = u16::from_le_bytes([blob[0], blob[1]]) as usize;
        let mut result = Vec::with_capacity(num_entries);
        let mut pos = 2;
        for _ in 0..num_entries {
            if pos + 9 > blob.len() {
                break;
            }
            let entry_key = u64::from_le_bytes(blob[pos..pos + 8].try_into().unwrap_or([0; 8]));
            let type_tag = blob[pos + 8];
            pos += 9;
            let key = InternedKey::from_u64(entry_key);
            if let Some(val) = Self::read_overflow_value(blob, &mut pos, type_tag) {
                result.push((key, val));
            }
        }
        result
    }

    /// Read a single value from the overflow blob at the given position.
    fn read_overflow_value(blob: &[u8], pos: &mut usize, type_tag: u8) -> Option<Value> {
        match type_tag {
            0 => Some(Value::Null),
            1 => {
                // Int64: 8 bytes
                if *pos + 8 > blob.len() {
                    return None;
                }
                let v = i64::from_le_bytes(blob[*pos..*pos + 8].try_into().ok()?);
                *pos += 8;
                Some(Value::Int64(v))
            }
            2 => {
                // Float64: 8 bytes
                if *pos + 8 > blob.len() {
                    return None;
                }
                let v = f64::from_le_bytes(blob[*pos..*pos + 8].try_into().ok()?);
                *pos += 8;
                Some(Value::Float64(v))
            }
            3 => {
                // UniqueId: 4 bytes
                if *pos + 4 > blob.len() {
                    return None;
                }
                let v = u32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?);
                *pos += 4;
                Some(Value::UniqueId(v))
            }
            4 => {
                // Bool: 1 byte
                if *pos + 1 > blob.len() {
                    return None;
                }
                let v = blob[*pos] != 0;
                *pos += 1;
                Some(Value::Boolean(v))
            }
            5 => {
                // Date: 4 bytes as i32 days since epoch
                if *pos + 4 > blob.len() {
                    return None;
                }
                let days = i32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?);
                *pos += 4;
                let date = UNIX_EPOCH_DATE + chrono::Duration::days(days as i64);
                Some(Value::DateTime(date))
            }
            6 => {
                // String: u32 length prefix + bytes
                if *pos + 4 > blob.len() {
                    return None;
                }
                let slen = u32::from_le_bytes(blob[*pos..*pos + 4].try_into().ok()?) as usize;
                *pos += 4;
                if *pos + slen > blob.len() {
                    return None;
                }
                let s = String::from_utf8_lossy(&blob[*pos..*pos + slen]).into_owned();
                *pos += slen;
                Some(Value::String(s))
            }
            _ => None,
        }
    }

    /// Skip over a value in the overflow blob without decoding it.
    fn skip_overflow_value(blob: &[u8], pos: &mut usize, type_tag: u8) {
        match type_tag {
            0 => {}             // Null: no data
            1 | 2 => *pos += 8, // Int64 or Float64
            3 | 5 => *pos += 4, // UniqueId or Date
            4 => *pos += 1,     // Bool
            6 if *pos + 4 <= blob.len() => {
                // String: u32 length prefix + bytes
                let slen =
                    u32::from_le_bytes(blob[*pos..*pos + 4].try_into().unwrap_or([0; 4])) as usize;
                *pos += 4 + slen;
            }
            _ => {}
        }
    }

    /// Serialize a Value into overflow bag format, appending to the buffer.
    pub fn serialize_overflow_value(buf: &mut Vec<u8>, key: InternedKey, value: &Value) {
        buf.extend_from_slice(&key.as_u64().to_le_bytes());
        match value {
            Value::Null => buf.push(0),
            Value::Int64(v) => {
                buf.push(1);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Value::Float64(v) => {
                buf.push(2);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Value::UniqueId(v) => {
                buf.push(3);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Value::Boolean(v) => {
                buf.push(4);
                buf.push(*v as u8);
            }
            Value::DateTime(d) => {
                buf.push(5);
                let days = (*d - UNIX_EPOCH_DATE).num_days() as i32;
                buf.extend_from_slice(&days.to_le_bytes());
            }
            Value::String(s) => {
                buf.push(6);
                buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            Value::Point { lat, lon } => {
                // Serialize Point as a string "lat,lon"
                let s = format!("{},{}", lat, lon);
                buf.push(6);
                buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
            Value::NodeRef(_) => {
                // NodeRef is transient — skip (write as null)
                buf.push(0);
            }
            Value::Duration { .. } => {
                // Durations are query-time-only — never persisted in
                // the overflow bag (Cluster 2). Skip as null.
                buf.push(0);
            }
            // Phase A.1 — collection / graph-entity variants are
            // query-result-time values; they don't belong in the
            // overflow property bag. Skip as null (matches the
            // Duration/NodeRef precedent). C5 (the .kgl v4 format
            // bump) revisits this if structural property storage
            // becomes a requirement.
            Value::List(_)
            | Value::Map(_)
            | Value::Node(_)
            | Value::Relationship(_)
            | Value::Path(_) => {
                buf.push(0);
            }
        }
    }

    // ─── Id/Title column methods (mapped mode only) ──────────────────────

    /// Push a node ID value into the id column. Creates a Mixed column if None.
    pub fn push_id(&mut self, value: &Value) {
        let col = self
            .id_column
            .get_or_insert_with(|| TypedColumn::Mixed { data: Vec::new() });
        if col.push(value).is_err() {
            // Type mismatch — demote to Mixed
            let mut mixed = Vec::with_capacity(col.len() + 1);
            for i in 0..col.len() {
                mixed.push(col.get(i as u32).unwrap_or(Value::Null));
            }
            mixed.push(value.clone());
            *col = TypedColumn::Mixed { data: mixed };
        }
    }

    /// Push a node title value into the title column. Creates a Str column if None.
    pub fn push_title(&mut self, value: &Value) {
        let col = self.title_column.get_or_insert_with(|| TypedColumn::Str {
            offsets: {
                let mut o = MmapOrVec::new();
                o.push(0u64);
                o
            },
            data: MmapBytes::new(),
            nulls: MmapOrVec::new(),
            relocated: HashMap::new(),
        });
        if col.push(value).is_err() {
            // Type mismatch — demote to Mixed
            let mut mixed = Vec::with_capacity(col.len() + 1);
            for i in 0..col.len() {
                mixed.push(col.get(i as u32).unwrap_or(Value::Null));
            }
            mixed.push(value.clone());
            *col = TypedColumn::Mixed { data: mixed };
        }
    }

    /// Overwrite the title value at `row_id`. Used by update-path mutations
    /// on mapped / disk graphs where properties live in the columnar store
    /// rather than in a per-node heap map. Returns `true` on success.
    pub fn set_title(&mut self, row_id: u32, value: &Value) -> bool {
        if (row_id as usize) >= self.row_count as usize {
            return false;
        }
        // Lazy promotion: if this store is mmap-backed, the local
        // `title_column` is None and `set_title` would silently drop the
        // write (pre-0.9.4 Bug C). Materialize a Mixed column from the
        // mmap-backed titles so subsequent reads via `get_title` see
        // both the override at `row_id` and the original titles for the
        // rest. The new column is dense (one entry per row); titles for
        // unmodified rows are read out of mmap once and rewritten as
        // owned Values, paying a one-time RAM cost on first SET-title.
        if self.title_column.is_none() {
            if let Some(ref ms) = self.mmap_store {
                let row_count = ms.row_count();
                let mut mixed: Vec<Value> = Vec::with_capacity(row_count as usize);
                for i in 0..row_count {
                    mixed.push(ms.get_title(i).unwrap_or(Value::Null));
                }
                self.title_column = Some(TypedColumn::Mixed { data: mixed });
            } else {
                return false;
            }
        }
        let col = self.title_column.as_mut().unwrap();
        if (row_id as usize) >= col.len() {
            return false;
        }
        if col.set(row_id, value).is_err() {
            let mut mixed: Vec<Value> = (0..col.len())
                .map(|i| col.get(i as u32).unwrap_or(Value::Null))
                .collect();
            mixed[row_id as usize] = value.clone();
            *col = TypedColumn::Mixed { data: mixed };
        }
        true
    }

    /// Get the node ID from the id column at the given row.
    #[inline]
    pub fn get_id(&self, row_id: u32) -> Option<Value> {
        if let Some(ref ms) = self.mmap_store {
            return ms.get_id(row_id);
        }
        self.id_column.as_ref()?.get(row_id)
    }

    /// Get the node title from the title column at the given row.
    #[inline]
    pub fn get_title(&self, row_id: u32) -> Option<Value> {
        // Same overlay rule as `get`: in-memory `title_column`
        // (populated lazily by `set_title` on first override) always
        // wins over the mmap-backed read.
        if let Some(ref col) = self.title_column {
            return col.get(row_id);
        }
        if let Some(ref ms) = self.mmap_store {
            return ms.get_title(row_id);
        }
        None
    }

    /// Whether this store has id/title columns (mapped mode).
    #[inline]
    pub fn has_id_title_columns(&self) -> bool {
        self.id_column.is_some() || self.title_column.is_some() || self.mmap_store.is_some()
    }

    /// Borrowed view of the id column. Delegates to the underlying
    /// `MmapColumnStore` when present (the disk-graph case used by
    /// `save_subset_streaming_disk`); returns `None` otherwise.
    #[inline]
    pub fn id_borrowed(&self, row_id: u32) -> Option<crate::datatypes::values::BorrowedValue<'_>> {
        self.mmap_store.as_ref()?.id_borrowed(row_id)
    }

    /// Borrowed view of the title column. See [`id_borrowed`].
    #[inline]
    pub fn title_borrowed(&self, row_id: u32) -> Option<&str> {
        self.mmap_store.as_ref()?.title_borrowed(row_id)
    }

    /// Allocation-free property visitor. Used by
    /// `save_subset_streaming_disk` to skip the per-row
    /// `Vec<(InternedKey, Value)>` and `Value::String` clones that
    /// dominated v3's node walk on Wikidata (~298 s of 446 s).
    /// Mmap-backed stores hit the fast path; heap-overlay stores
    /// fall back to allocating `row_properties` (the streaming
    /// pipeline only ever sees disk-mode sources today).
    pub fn try_for_each_property_borrowed<F, E>(&self, row_id: u32, mut f: F) -> Result<(), E>
    where
        F: FnMut(InternedKey, crate::datatypes::values::BorrowedValue<'_>) -> Result<(), E>,
    {
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return Ok(());
        }
        if self.columns.is_empty() {
            if let Some(ref ms) = self.mmap_store {
                return ms.try_for_each_property_borrowed(row_id, f);
            }
            return Ok(());
        }
        // Heap-only / overlay path: convert through `row_properties`.
        let owned = self.row_properties(row_id);
        for (key, val) in owned.iter() {
            let bv = match val {
                Value::Null => crate::datatypes::values::BorrowedValue::Null,
                Value::Boolean(b) => crate::datatypes::values::BorrowedValue::Boolean(*b),
                Value::Int64(v) => crate::datatypes::values::BorrowedValue::Int64(*v),
                Value::Float64(v) => crate::datatypes::values::BorrowedValue::Float64(*v),
                Value::UniqueId(v) => crate::datatypes::values::BorrowedValue::UniqueId(*v),
                Value::String(s) => crate::datatypes::values::BorrowedValue::String(s.as_str()),
                Value::DateTime(d) => crate::datatypes::values::BorrowedValue::DateTime(*d),
                _ => continue,
            };
            f(*key, bv)?;
        }
        Ok(())
    }

    /// Type tag of the id column if known: `"string"` or `"uniqueid"`
    /// for the typed cases, `"mixed"` for heterogeneous ids, or
    /// `None` if there is no id column at all. External writers
    /// (`save_subset_streaming_disk`'s TypeWriter) use this to open a
    /// matching column file format on the dest side.
    pub fn id_type_str(&self) -> Option<&'static str> {
        if let Some(ref ms) = self.mmap_store {
            return Some(if ms.id_is_string {
                "string"
            } else {
                "uniqueid"
            });
        }
        self.id_column.as_ref().map(|c| c.type_tag())
    }

    /// Type tag of the title column. `MmapColumnStore`'s title is
    /// always a string column (per its data model); otherwise we
    /// report the in-memory `title_column`'s tag, or `None`.
    pub fn title_type_str(&self) -> Option<&'static str> {
        if self.mmap_store.is_some() {
            return Some("string");
        }
        self.title_column.as_ref().map(|c| c.type_tag())
    }

    /// Number of rows (including tombstoned).
    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    /// Number of live (non-tombstoned) rows.
    #[allow(dead_code)] // Test-only.
    pub fn live_count(&self) -> u32 {
        self.row_count - self.tombstones.iter().filter(|&&t| t).count() as u32
    }

    /// Reference to the shared schema.
    pub fn schema(&self) -> &Arc<TypeSchema> {
        &self.schema
    }

    /// Append a row of property values. Returns the row_id for this row.
    /// `values` is a list of (InternedKey, Value) pairs.
    pub fn push_row(&mut self, values: &[(InternedKey, Value)]) -> u32 {
        let row_id = self.row_count;

        // Build slot→value lookup to push values directly (avoids null-then-overwrite).
        let mut slot_values: Vec<Option<&Value>> = vec![None; self.columns.len()];
        for (key, value) in values {
            if let Some(slot) = self.schema.slot(*key) {
                slot_values[slot as usize] = Some(value);
            }
        }

        for (slot, slot_val) in slot_values.iter().enumerate() {
            let col = &mut self.columns[slot];
            if let Some(value) = slot_val {
                if col.push(value).is_err() {
                    // Type mismatch — demote column to Mixed and retry
                    self.demote_to_mixed(slot);
                    let _ = self.columns[slot].push(value);
                }
            } else {
                col.push_null();
            }
        }

        // Keep id/title columns in sync (push null placeholders for property-only rows)
        if let Some(ref mut col) = self.id_column {
            if col.len() < self.row_count as usize + 1 {
                col.push_null();
            }
        }
        if let Some(ref mut col) = self.title_column {
            if col.len() < self.row_count as usize + 1 {
                col.push_null();
            }
        }

        self.row_count += 1;
        self.tombstones.push(false);
        row_id
    }

    /// Get a property value by (row_id, interned key).
    /// Falls back to the overflow bag when the key isn't in the schema or the
    /// dense column value is null.
    pub fn get(&self, row_id: u32, key: InternedKey) -> Option<Value> {
        if row_id >= self.row_count {
            return None;
        }
        if self
            .tombstones
            .get(row_id as usize)
            .copied()
            .unwrap_or(false)
        {
            return None;
        }
        // In-memory write overlay always wins over the mmap-backed read.
        // Pre-0.9.4 the mmap-backed branch short-circuited at the top of
        // this method, so any Cypher SET that landed in `self.columns`
        // via `set()` was invisible on read — `MATCH … SET p.x = 1` would
        // succeed (count=1 returned) but a subsequent `RETURN p.x` saw
        // `None`. Triggered by the `load_ntriples` build path that
        // constructs ColumnStores via `from_mmap_store`. Bug C in the
        // 0.9.3 disk-mode regression report.
        if let Some(slot) = self.schema.slot(key) {
            if let Some(val) = self.columns.get(slot as usize).and_then(|c| c.get(row_id)) {
                return Some(val);
            }
        }
        if let Some(ref ms) = self.mmap_store {
            return ms.get(row_id, key);
        }
        // Fall back to overflow bag
        self.get_overflow_property(row_id, key)
    }

    /// Zero-allocation string equality check for (row_id, key) against `target`.
    /// Returns `None` if the property is missing/null for this row, otherwise
    /// `Some(bool)`. Avoids the `String::from_utf8_unchecked(bytes.to_vec())`
    /// that a full `get()` would trigger for mmap-backed string columns —
    /// significant on mapped graphs where string property scans are the
    /// main perf gap vs in-memory mode.
    pub fn str_prop_eq(&self, row_id: u32, key: InternedKey, target: &str) -> Option<bool> {
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return None;
        }
        // In-memory overlay wins over mmap (mirrors `get` — Bug C fix).
        if let Some(slot) = self.schema.slot(key) {
            if let Some(col) = self.columns.get(slot as usize) {
                if let Some(s) = col.get_str(row_id) {
                    return Some(s == target);
                }
                if let Some(v) = col.get(row_id) {
                    return Some(matches!(v, Value::String(ref s) if s == target));
                }
            }
        }
        if let Some(ref ms) = self.mmap_store {
            return ms.str_prop_eq(row_id, key, target);
        }
        self.get_overflow_property(row_id, key)
            .map(|v| matches!(v, Value::String(ref s) if s == target))
    }

    /// Resolve a property name to a column slot index.
    #[inline]
    #[allow(dead_code)] // Test-only.
    pub fn slot(&self, key: InternedKey) -> Option<u16> {
        self.schema.slot(key)
    }

    /// Fast property access by pre-resolved slot index.
    /// Caller must ensure row_id is valid and not tombstoned.
    #[inline]
    #[allow(dead_code)] // Test-only.
    pub fn get_by_slot(&self, row_id: u32, slot: u16) -> Option<Value> {
        self.columns.get(slot as usize)?.get(row_id)
    }

    /// Fast string access by pre-resolved slot. Returns borrowed &str without allocation.
    #[inline]
    pub fn get_str_by_slot(&self, row_id: u32, slot: u16) -> Option<&str> {
        self.columns.get(slot as usize)?.get_str(row_id)
    }

    /// Fast string comparison by pre-resolved slot. No allocation.
    #[inline]
    #[allow(dead_code)] // Test-only.
    pub fn compare_str_by_slot(&self, row_id: u32, slot: u16, target: &str) -> bool {
        self.columns
            .get(slot as usize)
            .and_then(|c| c.get_str(row_id))
            .is_some_and(|s| s == target)
    }

    /// Set a property value for a given row.
    /// Extends the schema if the key is new.
    pub fn set(
        &mut self,
        row_id: u32,
        key: InternedKey,
        value: &Value,
        type_meta: Option<&str>,
    ) -> bool {
        if row_id >= self.row_count {
            return false;
        }
        let slot = match self.schema.slot(key) {
            Some(s) => s,
            None => {
                // New property — extend schema and add a column
                let s = Arc::make_mut(&mut self.schema).add_key(key);
                let type_str = type_meta.unwrap_or("mixed");
                let mut col = TypedColumn::from_type_str(type_str);
                // Backfill nulls for existing rows
                for _ in 0..self.row_count {
                    col.push_null();
                }
                self.columns.push(col);
                s
            }
        };
        let col = &mut self.columns[slot as usize];
        if col.set(row_id, value).is_err() {
            self.demote_to_mixed(slot as usize);
            let _ = self.columns[slot as usize].set(row_id, value);
        }
        true
    }

    /// Mark a row as deleted (tombstoned).
    pub fn tombstone(&mut self, row_id: u32) {
        if let Some(t) = self.tombstones.get_mut(row_id as usize) {
            *t = true;
        }
    }

    /// Check if a row has a property (non-null, non-tombstoned).
    #[allow(dead_code)] // Test-only.
    pub fn contains(&self, row_id: u32, key: InternedKey) -> bool {
        self.get(row_id, key).is_some()
    }

    /// Iterate over all non-null properties for a row.
    /// Returns (InternedKey, Value) pairs from both dense columns and overflow bag.
    pub fn row_properties(&self, row_id: u32) -> Vec<(InternedKey, Value)> {
        if row_id >= self.row_count
            || self
                .tombstones
                .get(row_id as usize)
                .copied()
                .unwrap_or(false)
        {
            return Vec::new();
        }
        // Build up the in-memory overlay first so `keys(node)` and
        // similar surface operators can see Cypher-SET-introduced
        // properties on mmap-backed stores. Then merge with the
        // mmap-backed row, with the in-memory overlay winning on
        // collisions. Pre-0.9.4 the mmap-backed branch short-
        // circuited and SET-introduced keys never appeared.
        let mut result = Vec::new();
        let mut seen: std::collections::HashSet<InternedKey> = std::collections::HashSet::new();
        for (slot, ik) in self.schema.iter() {
            if let Some(val) = self.columns.get(slot as usize).and_then(|c| c.get(row_id)) {
                seen.insert(ik);
                result.push((ik, val));
            }
        }
        if let Some(ref ms) = self.mmap_store {
            for (ik, val) in ms.row_properties(row_id) {
                if !seen.contains(&ik) {
                    result.push((ik, val));
                }
            }
            return result;
        }
        for (slot, ik) in self.schema.iter() {
            // re-iterate to keep overflow-bag fall-through unchanged for
            // the non-mmap path (the loop above already inserted dense
            // entries; below we only fill blanks).
            if seen.contains(&ik) {
                continue;
            }
            if let Some(val) = self.columns.get(slot as usize).and_then(|c| c.get(row_id)) {
                result.push((ik, val));
            }
        }
        // Append overflow bag properties
        let overflow = self.overflow_row_properties(row_id);
        result.extend(overflow);
        result
    }

    /// Reconstruct all properties for a row as a HashMap<String, Value>.
    #[allow(dead_code)] // Test-only.
    pub fn row_properties_map(
        &self,
        row_id: u32,
        interner: &StringInterner,
    ) -> HashMap<String, Value> {
        self.row_properties(row_id)
            .into_iter()
            .map(|(ik, v)| (interner.resolve(ik).to_string(), v))
            .collect()
    }

    /// Demote a column from typed to Mixed, preserving all existing data.
    fn demote_to_mixed(&mut self, slot: usize) {
        let old_col = &self.columns[slot];
        let mut mixed_data = Vec::with_capacity(old_col.len());
        for i in 0..old_col.len() {
            mixed_data.push(old_col.get(i as u32).unwrap_or(Value::Null));
        }
        self.columns[slot] = TypedColumn::Mixed { data: mixed_data };
    }

    /// Materialize all columns to file-backed mmap in the given directory.
    pub fn materialize_to_files(
        &mut self,
        dir: &Path,
        interner: &StringInterner,
    ) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        for (slot, ik) in self.schema.iter() {
            let col_name = interner.resolve(ik);
            if let Some(col) = self.columns.get_mut(slot as usize) {
                col.materialize_to_file(dir, col_name)?;
            }
        }
        // Spill id/title columns too
        if let Some(ref mut col) = self.id_column {
            col.materialize_to_file(dir, "__id__")?;
        }
        if let Some(ref mut col) = self.title_column {
            col.materialize_to_file(dir, "__title__")?;
        }
        Ok(())
    }

    /// Flush dirty pages of every mmap-backed underlying file to disk and
    /// advise the kernel to drop them from page cache. Used by streaming
    /// builders to keep peak RSS bounded during long push loops — without
    /// this, dirty mmap pages accumulate in RAM until the kernel evicts
    /// on its own schedule.
    ///
    /// Heap-backed columns are no-ops. Returns the first error from any
    /// underlying msync; subsequent columns are still attempted.
    ///
    /// As of v2 the streaming subgraph filter no longer calls this on
    /// the hot path — chunk-and-spill handles eviction by closing file
    /// handles between chunks. Retained as a Linux-friendly explicit-
    /// flush primitive for future callers.
    #[allow(dead_code)]
    pub fn flush_and_release_pages(&self) -> io::Result<()> {
        let mut first_err: Option<io::Error> = None;
        for col in &self.columns {
            if let Err(e) = col.flush_and_release_pages() {
                first_err.get_or_insert(e);
            }
        }
        if let Some(ref col) = self.id_column {
            if let Err(e) = col.flush_and_release_pages() {
                first_err.get_or_insert(e);
            }
        }
        if let Some(ref col) = self.title_column {
            if let Err(e) = col.flush_and_release_pages() {
                first_err.get_or_insert(e);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Convert all columns back to heap-backed storage.
    #[allow(dead_code)] // Test-only.
    pub fn materialize_to_heap(&mut self) {
        for col in &mut self.columns {
            col.materialize_to_heap();
        }
        if let Some(ref mut col) = self.id_column {
            col.materialize_to_heap();
        }
        if let Some(ref mut col) = self.title_column {
            col.materialize_to_heap();
        }
    }

    /// Whether any column is file-backed.
    pub fn is_mapped(&self) -> bool {
        self.columns.iter().any(|c| c.is_mapped())
    }

    /// Heap-resident bytes across all columns (0 if fully mmap'd).
    pub fn heap_bytes(&self) -> usize {
        let col_bytes: usize = self.columns.iter().map(|c| c.heap_bytes()).sum();
        let id_bytes = self.id_column.as_ref().map_or(0, |c| c.heap_bytes());
        let title_bytes = self.title_column.as_ref().map_or(0, |c| c.heap_bytes());
        let overflow_bytes = self.overflow_offsets.as_ref().map_or(0, |o| o.heap_bytes())
            + self.overflow_data.as_ref().map_or(0, |d| d.heap_bytes());
        col_bytes + id_bytes + title_bytes + overflow_bytes + self.tombstones.len()
    }

    /// Access columns for introspection (e.g., getting type tags).
    pub fn columns_ref(&self) -> &[TypedColumn] {
        &self.columns
    }

    /// Access the optional id sidecar column.
    pub fn id_column_ref(&self) -> Option<&TypedColumn> {
        self.id_column.as_ref()
    }

    /// Access the optional title sidecar column.
    pub fn title_column_ref(&self) -> Option<&TypedColumn> {
        self.title_column.as_ref()
    }

    /// Raw bytes of the overflow_offsets array (u64 values, native
    /// endian). Returns `None` when no overflow bag is installed.
    pub fn overflow_offsets_bytes(&self) -> Option<Vec<u8>> {
        self.overflow_offsets
            .as_ref()
            .map(|o| o.as_raw_bytes().to_vec())
    }

    /// Raw bytes of the overflow_data blob. Returns `None` when no
    /// overflow bag is installed.
    pub fn overflow_data_bytes(&self) -> Option<Vec<u8>> {
        self.overflow_data
            .as_ref()
            .map(|d| d.as_raw_bytes().to_vec())
    }

    // ── External-builder accessors ──────────────────────────────────
    //
    // The streaming subgraph filter (`save_subset`) builds a destination
    // ColumnStore in chunks, spilling each to disk and merging at the
    // end. Those steps need to inject finished `TypedColumn` values
    // (mmap-backed at the merged file paths) into a freshly-constructed
    // ColumnStore shell. Plain `ColumnStore::new` has no way to do this;
    // these accessors fill the gap.
    //
    // `dead_code` is allowed at the impl-block level here because the
    // first consumer ships in commit 2 of the v2 chunk-spill PR; commit
    // 1 lands these accessors alone so the API change passes parity
    // tests in isolation before any new behavior is introduced.

    /// Replace the schema-keyed property columns wholesale. The new
    /// `Vec<TypedColumn>` must have exactly `self.schema().len()` entries
    /// in slot order; the caller is responsible for the correspondence.
    #[allow(dead_code)]
    pub fn replace_columns(&mut self, columns: Vec<TypedColumn>) {
        self.columns = columns;
    }

    /// Replace the id sidecar column.
    #[allow(dead_code)]
    pub fn replace_id_column(&mut self, col: TypedColumn) {
        self.id_column = Some(col);
    }

    /// Replace the title sidecar column.
    #[allow(dead_code)]
    pub fn replace_title_column(&mut self, col: TypedColumn) {
        self.title_column = Some(col);
    }

    /// Replace the overflow bag (offsets + data blob).
    ///
    /// Used by the streaming subgraph carve to persist non-schema
    /// properties that the source had stored as per-row overflow
    /// blobs. The wire format matches what `write_packed` emits and
    /// `load_packed` reads back via the `__overflow_offsets__` /
    /// `__overflow_data__` pseudo-columns.
    pub fn replace_overflow_bag(&mut self, offsets: MmapOrVec<u64>, data: MmapBytes) {
        self.overflow_offsets = Some(offsets);
        self.overflow_data = Some(data);
    }

    /// Borrowed-value equivalent of [`Self::serialize_overflow_value`]
    /// for hot-path callers that already hold a `BorrowedValue` slice
    /// (avoids the clone into `Value`).
    pub fn serialize_overflow_value_borrowed(
        buf: &mut Vec<u8>,
        key: InternedKey,
        value: &crate::datatypes::values::BorrowedValue<'_>,
    ) {
        use crate::datatypes::values::BorrowedValue;
        buf.extend_from_slice(&key.as_u64().to_le_bytes());
        match value {
            BorrowedValue::Null => buf.push(0),
            BorrowedValue::Int64(v) => {
                buf.push(1);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            BorrowedValue::Float64(v) => {
                buf.push(2);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            BorrowedValue::UniqueId(v) => {
                buf.push(3);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            BorrowedValue::Boolean(v) => {
                buf.push(4);
                buf.push(*v as u8);
            }
            BorrowedValue::DateTime(d) => {
                buf.push(5);
                let days = (*d - UNIX_EPOCH_DATE).num_days() as i32;
                buf.extend_from_slice(&days.to_le_bytes());
            }
            BorrowedValue::String(s) => {
                buf.push(6);
                buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
        }
    }

    /// Set the row count after wiring up replaced columns. The store's
    /// authoritative row count is the merged total; without this the
    /// fresh shell reports 0 rows even though the columns hold data.
    #[allow(dead_code)]
    pub fn set_row_count(&mut self, n: u32) {
        self.row_count = n;
    }

    /// Type-tag string for the column at `slot`, e.g. `"int64"`,
    /// `"string"`, `"mixed"`. Delegates to [`TypedColumn::type_tag`].
    /// Used by the chunked-spill merge to dispatch to the right merge
    /// kernel per typed-column variant.
    #[allow(dead_code)]
    pub fn column_type_str(&self, slot: usize) -> Option<&'static str> {
        self.columns.get(slot).map(|c| c.type_tag())
    }

    /// Borrow the `Vec<Value>` inside a `TypedColumn::Mixed` at `slot`.
    /// Returns `None` for non-Mixed variants. Used by the chunked-spill
    /// builder to serialize Mixed columns to per-chunk bincode sidecars
    /// (since `materialize_to_files` skips Mixed).
    #[allow(dead_code)]
    pub fn column_values_mixed(&self, slot: usize) -> Option<&Vec<Value>> {
        match self.columns.get(slot)? {
            TypedColumn::Mixed { data } => Some(data),
            _ => None,
        }
    }

    /// Serialize all columns to a packed byte buffer for the v3 file format.
    ///
    /// Format per column:
    ///   [2B] col_name_len  [NB] col_name_utf8
    ///   [2B] type_tag_len  [NB] type_tag
    ///   [8B] data_len      [NB] data_bytes (+ null_bytes for typed columns)
    ///   For "string": data_bytes = offsets + str_data + null_bitmap
    ///   For "mixed": data_bytes = bincode(Vec<Value>)
    pub fn write_packed(&self, interner: &StringInterner) -> io::Result<Vec<u8>> {
        // If this ColumnStore is mmap-backed (from_mmap_store), materialize
        // rows from the mmap store so they can be serialized.
        if let Some(ref mmap_store) = self.mmap_store {
            return self.write_packed_from_mmap(mmap_store, interner);
        }

        let mut buf: Vec<u8> = Vec::new();

        // Write ALL schema columns (including empty ones) to preserve metadata round-trip.
        // Empty columns are cheap — just type tag + zero-length data blob.
        let extra = self.id_column.is_some() as u32
            + self.title_column.is_some() as u32
            + if self.overflow_offsets.is_some() {
                2
            } else {
                0
            };
        let num_cols = self.columns.len() as u32 + extra;
        buf.extend_from_slice(&num_cols.to_le_bytes());

        for (slot, ik) in self.schema.iter() {
            let col_name = interner.resolve(ik);
            let col = &self.columns[slot as usize];
            if col.len() == 0 && self.row_count > 0 {
                // Column is in schema but has no data. Write it as all-null
                // with the correct row_count so load_packed can deserialize it.
                let mut padded = TypedColumn::Mixed { data: Vec::new() };
                for _ in 0..self.row_count {
                    let _ = padded.push(&Value::Null);
                }
                Self::write_packed_column(&mut buf, col_name, &padded)?;
            } else {
                Self::write_packed_column(&mut buf, col_name, col)?;
            }
        }

        // Write id/title columns with reserved names
        if let Some(ref col) = self.id_column {
            Self::write_packed_column(&mut buf, "__id__", col)?;
        }
        if let Some(ref col) = self.title_column {
            Self::write_packed_column(&mut buf, "__title__", col)?;
        }

        // Write overflow bag as two pseudo-columns
        if let (Some(ref offsets), Some(ref data)) = (&self.overflow_offsets, &self.overflow_data) {
            // __overflow_offsets__: raw bytes of the u64 offset array
            {
                let name = b"__overflow_offsets__";
                buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
                buf.extend_from_slice(name);
                let tag = b"raw";
                buf.extend_from_slice(&(tag.len() as u16).to_le_bytes());
                buf.extend_from_slice(tag);
                let raw = offsets.as_raw_bytes();
                buf.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                buf.extend_from_slice(raw);
            }
            // __overflow_data__: raw bytes blob
            {
                let name = b"__overflow_data__";
                buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
                buf.extend_from_slice(name);
                let tag = b"raw";
                buf.extend_from_slice(&(tag.len() as u16).to_le_bytes());
                buf.extend_from_slice(tag);
                let raw = data.as_raw_bytes();
                buf.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                buf.extend_from_slice(raw);
            }
        }

        Ok(buf)
    }

    /// Write packed format from an mmap-backed ColumnStore.
    /// Materializes rows from the MmapColumnStore into Mixed TypedColumns, then serializes.
    /// This is used when a disk graph is loaded (creating mmap-backed stores) and then re-saved.
    fn write_packed_from_mmap(
        &self,
        mmap_store: &crate::graph::storage::mapped::column_store::MmapColumnStore,
        interner: &StringInterner,
    ) -> io::Result<Vec<u8>> {
        let rc = mmap_store.row_count();
        let mut buf: Vec<u8> = Vec::new();

        // Materialize ID column
        let id_col = TypedColumn::Mixed {
            data: (0..rc)
                .map(|r| mmap_store.get_id(r).unwrap_or(Value::Null))
                .collect(),
        };

        // Materialize title column
        let title_col = TypedColumn::Mixed {
            data: (0..rc)
                .map(|r| mmap_store.get_title(r).unwrap_or(Value::Null))
                .collect(),
        };

        // Materialize property columns from col_map
        let mut prop_columns: Vec<(String, TypedColumn)> = Vec::new();
        for &key in mmap_store.col_map.keys() {
            let col_name = interner.resolve(key).to_string();
            let col = TypedColumn::Mixed {
                data: (0..rc)
                    .map(|r| mmap_store.get(r, key).unwrap_or(Value::Null))
                    .collect(),
            };
            prop_columns.push((col_name, col));
        }

        // Count columns
        let has_overflow = mmap_store.has_overflow && mmap_store.overflow_offsets.len > 0;
        let mut num_cols = prop_columns.len() as u32 + 2; // +2 for id + title
        if has_overflow {
            num_cols += 2;
        }
        buf.extend_from_slice(&num_cols.to_le_bytes());

        // Write property columns
        for (name, col) in &prop_columns {
            Self::write_packed_column(&mut buf, name, col)?;
        }

        // Write id/title
        Self::write_packed_column(&mut buf, "__id__", &id_col)?;
        Self::write_packed_column(&mut buf, "__title__", &title_col)?;

        // Write overflow if present
        if has_overflow {
            let off_r = &mmap_store.overflow_offsets;
            let dat_r = &mmap_store.overflow_data;
            // __overflow_offsets__
            {
                let name = b"__overflow_offsets__";
                buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
                buf.extend_from_slice(name);
                let tag = b"raw";
                buf.extend_from_slice(&(tag.len() as u16).to_le_bytes());
                buf.extend_from_slice(tag);
                let raw = &mmap_store.mmap[off_r.offset..off_r.offset + off_r.len];
                buf.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                buf.extend_from_slice(raw);
            }
            // __overflow_data__
            {
                let name = b"__overflow_data__";
                buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
                buf.extend_from_slice(name);
                let tag = b"raw";
                buf.extend_from_slice(&(tag.len() as u16).to_le_bytes());
                buf.extend_from_slice(tag);
                let raw = &mmap_store.mmap[dat_r.offset..dat_r.offset + dat_r.len];
                buf.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                buf.extend_from_slice(raw);
            }
        }

        Ok(buf)
    }

    /// Write a single column entry to a packed buffer.
    fn write_packed_column(buf: &mut Vec<u8>, col_name: &str, col: &TypedColumn) -> io::Result<()> {
        let type_tag = col.type_tag();

        // Column name
        let name_bytes = col_name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        // Type tag
        let tag_bytes = type_tag.as_bytes();
        buf.extend_from_slice(&(tag_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(tag_bytes);

        // Column data — write length placeholder, then data directly, then patch length
        let len_offset = buf.len();
        buf.extend_from_slice(&0u64.to_le_bytes()); // placeholder
        col.write_to(buf)?;
        let data_len = (buf.len() - len_offset - 8) as u64;
        buf[len_offset..len_offset + 8].copy_from_slice(&data_len.to_le_bytes());
        Ok(())
    }

    /// Load columns from a packed byte buffer (v3 format).
    ///
    /// If `temp_dir` is `Some`, writes column data to temp files and mmaps them
    /// (for larger-than-RAM support). If `None`, loads into heap.
    pub fn load_packed(
        schema: Arc<TypeSchema>,
        type_meta: &HashMap<String, String>,
        interner: &StringInterner,
        packed: &[u8],
        row_count: u32,
        temp_dir: Option<&Path>,
    ) -> io::Result<Self> {
        use std::io::Read;

        let mut store = ColumnStore::new(Arc::clone(&schema), type_meta, interner);
        store.row_count = row_count;
        store.tombstones = vec![false; row_count as usize];

        let mut cursor = std::io::Cursor::new(packed);

        // Read number of columns
        let mut u32_buf = [0u8; 4];
        cursor.read_exact(&mut u32_buf)?;
        let num_cols = u32::from_le_bytes(u32_buf);

        for _ in 0..num_cols {
            // Column name
            let mut u16_buf = [0u8; 2];
            cursor.read_exact(&mut u16_buf)?;
            let name_len = u16::from_le_bytes(u16_buf) as usize;
            let mut name_bytes = vec![0u8; name_len];
            cursor.read_exact(&mut name_bytes)?;
            let col_name = String::from_utf8(name_bytes)
                .map_err(|e| io::Error::other(format!("invalid column name: {}", e)))?;

            // Type tag
            cursor.read_exact(&mut u16_buf)?;
            let tag_len = u16::from_le_bytes(u16_buf) as usize;
            let mut tag_bytes = vec![0u8; tag_len];
            cursor.read_exact(&mut tag_bytes)?;
            let type_tag = String::from_utf8(tag_bytes)
                .map_err(|e| io::Error::other(format!("invalid type tag: {}", e)))?;

            // Data blob
            let mut u64_buf = [0u8; 8];
            cursor.read_exact(&mut u64_buf)?;
            let data_len = u64::from_le_bytes(u64_buf) as usize;
            let mut data_blob = vec![0u8; data_len];
            cursor.read_exact(&mut data_blob)?;

            // Check for special id/title columns first
            if col_name == "__id__" {
                let col =
                    Self::unpack_column(&type_tag, &data_blob, row_count, temp_dir, &col_name)?;
                store.id_column = Some(col);
                continue;
            }
            if col_name == "__title__" {
                let col =
                    Self::unpack_column(&type_tag, &data_blob, row_count, temp_dir, &col_name)?;
                store.title_column = Some(col);
                continue;
            }

            // Check for overflow pseudo-columns
            if col_name == "__overflow_offsets__" {
                let num_offsets = data_blob.len() / std::mem::size_of::<u64>();
                let offsets = Self::load_typed_vec::<u64>(
                    &data_blob,
                    num_offsets,
                    temp_dir,
                    &col_name,
                    "off",
                )?;
                store.overflow_offsets = Some(offsets);
                continue;
            }
            if col_name == "__overflow_data__" {
                let data = Self::load_bytes(&data_blob, temp_dir, &col_name, "dat")?;
                store.overflow_data = Some(data);
                continue;
            }

            // Find the slot for this column
            let ik = InternedKey::from_str(&col_name);
            let slot = match schema.slot(ik) {
                Some(s) => s as usize,
                None => continue, // schema doesn't have this column, skip
            };

            // Build the TypedColumn from the data blob
            let col = Self::unpack_column(&type_tag, &data_blob, row_count, temp_dir, &col_name)?;

            if slot < store.columns.len() {
                store.columns[slot] = col;
            }
        }

        Ok(store)
    }

    /// Unpack a single column from its raw data blob.
    fn unpack_column(
        type_tag: &str,
        data_blob: &[u8],
        row_count: u32,
        temp_dir: Option<&Path>,
        col_name: &str,
    ) -> io::Result<TypedColumn> {
        let rc = row_count as usize;
        match type_tag {
            "int64" => {
                let data_size = rc * std::mem::size_of::<i64>();
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<i64>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "i64",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::Int64 { data, nulls })
            }
            "float64" => {
                let data_size = rc * std::mem::size_of::<f64>();
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<f64>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "f64",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::Float64 { data, nulls })
            }
            "uniqueid" => {
                let data_size = rc * std::mem::size_of::<u32>();
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<u32>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "u32",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::UniqueId { data, nulls })
            }
            "bool" | "boolean" => {
                let data_size = rc; // u8 per row
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<u8>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "bool",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::Bool { data, nulls })
            }
            "date" | "datetime" => {
                let data_size = rc * std::mem::size_of::<i32>();
                let null_size = rc;
                Self::check_blob_size(data_blob, data_size + null_size, type_tag, col_name)?;
                let data = Self::load_typed_vec::<i32>(
                    &data_blob[..data_size],
                    rc,
                    temp_dir,
                    col_name,
                    "i32",
                )?;
                let nulls = Self::load_typed_vec::<u8>(
                    &data_blob[data_size..],
                    rc,
                    temp_dir,
                    col_name,
                    "null",
                )?;
                Ok(TypedColumn::Date { data, nulls })
            }
            "string" => {
                // offsets: (rc+1) * u64, then str_data, then nulls: rc * u8
                let offsets_size = (rc + 1) * std::mem::size_of::<u64>();
                if data_blob.len() < offsets_size {
                    return Err(io::Error::other(format!(
                        "column '{}' (string): blob too small for offsets ({} < {})",
                        col_name,
                        data_blob.len(),
                        offsets_size
                    )));
                }
                let offsets_bytes = &data_blob[..offsets_size];
                let rest = &data_blob[offsets_size..];

                // Determine string data length from last offset
                let last_offset = u64::from_le_bytes(
                    offsets_bytes[offsets_size - 8..offsets_size]
                        .try_into()
                        .unwrap(),
                ) as usize;
                let null_size = rc;

                if rest.len() < last_offset + null_size {
                    return Err(io::Error::other(format!(
                        "column '{}' (string): blob too small for data+nulls",
                        col_name
                    )));
                }
                let str_bytes = &rest[..last_offset];
                let null_bytes = &rest[last_offset..last_offset + null_size];

                let offsets =
                    Self::load_typed_vec::<u64>(offsets_bytes, rc + 1, temp_dir, col_name, "off")?;
                let data = Self::load_bytes(str_bytes, temp_dir, col_name, "str")?;
                let nulls = Self::load_typed_vec::<u8>(null_bytes, rc, temp_dir, col_name, "null")?;
                Ok(TypedColumn::Str {
                    offsets,
                    data,
                    nulls,
                    relocated: HashMap::new(),
                })
            }
            _ => {
                // Mixed — bincode deserialize
                let data: Vec<Value> = bincode::deserialize(data_blob).map_err(|e| {
                    io::Error::other(format!("bincode error for '{}': {}", col_name, e))
                })?;
                Ok(TypedColumn::Mixed { data })
            }
        }
    }

    /// Load raw bytes into a MmapOrVec<T>, optionally via temp file + mmap.
    fn load_typed_vec<T: Copy + Default + 'static>(
        bytes: &[u8],
        len: usize,
        temp_dir: Option<&Path>,
        col_name: &str,
        ext: &str,
    ) -> io::Result<MmapOrVec<T>> {
        // Skip mmap for small columns — file I/O overhead exceeds memory savings
        if let Some(dir) = temp_dir.filter(|_| bytes.len() >= MMAP_THRESHOLD) {
            let path = dir.join(format!("{col_name}.{ext}"));
            std::fs::write(&path, bytes)?;
            MmapOrVec::load_mapped(&path, len)
        } else {
            // Load into heap
            let elem_size = std::mem::size_of::<T>();
            if elem_size == 0 {
                return Ok(MmapOrVec::new());
            }
            let mut data = Vec::with_capacity(len);
            for i in 0..len {
                let offset = i * elem_size;
                // SAFETY: `bytes.len() >= len * elem_size` (caller contract
                // in the .kgl v3 packed-column reader); `T: Copy+Default`
                // has no drop glue and no alignment requirement stronger
                // than the natural `elem_size` offset.
                let val = unsafe { std::ptr::read(bytes.as_ptr().add(offset) as *const T) };
                data.push(val);
            }
            Ok(MmapOrVec::Heap { data })
        }
    }

    /// Load raw bytes into a MmapBytes, optionally via temp file + mmap.
    fn load_bytes(
        bytes: &[u8],
        temp_dir: Option<&Path>,
        col_name: &str,
        ext: &str,
    ) -> io::Result<MmapBytes> {
        // Skip mmap for small data — file I/O overhead exceeds memory savings
        if let Some(dir) = temp_dir.filter(|_| bytes.len() >= MMAP_THRESHOLD) {
            let path = dir.join(format!("{col_name}.{ext}"));
            std::fs::write(&path, bytes)?;
            MmapBytes::load_mapped(&path, bytes.len())
        } else {
            Ok(MmapBytes::Heap {
                data: bytes.to_vec(),
            })
        }
    }

    fn check_blob_size(
        blob: &[u8],
        expected: usize,
        type_tag: &str,
        col_name: &str,
    ) -> io::Result<()> {
        if blob.len() < expected {
            Err(io::Error::other(format!(
                "column '{}' ({}): blob too small ({} < {})",
                col_name,
                type_tag,
                blob.len(),
                expected
            )))
        } else {
            Ok(())
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────
// Hosted in `column_store_tests.rs` to keep this file under the
// 2500-line god-file gate (see test_phase7_parity.py::test_god_file_gate).

#[cfg(test)]
#[path = "column_store_tests.rs"]
mod tests;

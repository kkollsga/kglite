//! `TypedColumn` — one column of homogeneously-typed property values.
//!
//! Split out of `column_store.rs` when that file passed its 2500-line ceiling.
//! This half is the *storage element*: how one property's values are laid out
//! (`MmapOrVec<T>` for fixed-size types, `MmapBytes` for strings, a heap
//! `Vec<Value>` for `Mixed`), pushed, read, spilled and materialised. The
//! `ColumnStore` that owns a vector of them — schema, rows, tombstones,
//! id/title columns, the packed `.kgl` codec — lives in the sibling `mod.rs`.

use crate::datatypes::values::Value;
use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec, MmapPod};
use crate::graph::storage::packed_codec::write_packed_values;
use crate::graph::storage::StrField;
use chrono::NaiveDate;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::io;
use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

// ─── TypedColumn ─────────────────────────────────────────────────────────────

/// A single column of homogeneously-typed property values.
/// Column type is determined from `node_type_metadata` at construction time.
/// Falls back to `Mixed` for heterogeneous or unknown types.
///
/// Fixed-size columns use `MmapOrVec<T>` which can be heap- or file-backed.
/// String columns use `MmapOrVec<u64>` for offsets and `MmapBytes` for UTF-8 data.
/// Mixed columns use plain `Vec<Value>` (not mmap-eligible).
#[derive(Debug)]
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
        /// FxHash, not the std SipHasher: the key is a bare `u32` row id and
        /// this map is probed on *every* row of every scan of the column once
        /// it is non-empty (`str_at`). One differing-length `SET` was measured
        /// at +35% on every later scan of that column; the hash was the part
        /// of that a swap can remove (compacting the overlay away is the
        /// filed follow-on).
        relocated: FxHashMap<u32, String>,
    },
    /// Fallback for heterogeneous columns — stores boxed Values directly.
    /// Cannot be mmap'd, but preserves correctness.
    Mixed { data: Vec<Value> },
}

#[cfg(test)]
thread_local! {
    /// `TypedColumn` deep copies performed since the last reset.
    ///
    /// The **unit oracle** of the copy-on-write family, and the one the
    /// store-level counter cannot express. `COLUMN_STORE_CLONES` counts
    /// `ColumnStore` copies, which was the right unit while a store copy meant
    /// copying every column it held; since the columns became individually
    /// shared it no longer is — a store copy is now a handful of refcount
    /// bumps, and the cost that matters is *which columns the copy actually
    /// deep-copied*. A shared store forked for a one-cell write must copy
    /// exactly one column; the store counter reads 1 whether it copied one
    /// column or twenty-five.
    ///
    /// Thread-local like its siblings: it sees copies performed on the calling
    /// thread only, which is where every statement-scoped write happens.
    static COLUMN_CLONES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_column_clones() {
    COLUMN_CLONES.set(0);
}

/// Individual columns deep-copied on this thread since the last reset.
#[cfg(test)]
pub(crate) fn column_clones() -> usize {
    COLUMN_CLONES.get()
}

/// Hand-written so the copy can be counted — see [`COLUMN_CLONES`]. The arms
/// are exactly what `#[derive(Clone)]` produced.
impl Clone for TypedColumn {
    fn clone(&self) -> Self {
        #[cfg(test)]
        COLUMN_CLONES.set(COLUMN_CLONES.get() + 1);
        match self {
            Self::Int64 { data, nulls } => Self::Int64 {
                data: data.clone(),
                nulls: nulls.clone(),
            },
            Self::Float64 { data, nulls } => Self::Float64 {
                data: data.clone(),
                nulls: nulls.clone(),
            },
            Self::UniqueId { data, nulls } => Self::UniqueId {
                data: data.clone(),
                nulls: nulls.clone(),
            },
            Self::Bool { data, nulls } => Self::Bool {
                data: data.clone(),
                nulls: nulls.clone(),
            },
            Self::Date { data, nulls } => Self::Date {
                data: data.clone(),
                nulls: nulls.clone(),
            },
            Self::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => Self::Str {
                offsets: offsets.clone(),
                data: data.clone(),
                nulls: nulls.clone(),
                relocated: relocated.clone(),
            },
            Self::Mixed { data } => Self::Mixed { data: data.clone() },
        }
    }
}

impl TypedColumn {
    /// Unique targets, including unique mappings, retain the existing push
    /// path. Shared mapped columns clone to heap just like ordinary Clone;
    /// declined reservations fall back to Arc::make_mut.
    pub(super) fn make_mut_for_append<'a>(
        handle: &'a mut Arc<Self>,
        value: &Value,
    ) -> &'a mut Self {
        if Arc::strong_count(handle) > 1 {
            // Weak-only sharing keeps Arc::make_mut's dissociation behavior.
            if let Some(copied) = handle.try_clone_for_append(value) {
                *handle = Arc::new(copied);
            }
        }
        Arc::make_mut(handle)
    }

    fn try_clone_for_append(&self, value: &Value) -> Option<Self> {
        macro_rules! scalar {
            ($variant:ident, $data:ident, $nulls:ident) => {
                Self::$variant {
                    data: $data.try_clone_for_append(1)?,
                    nulls: $nulls.try_clone_for_append(1)?,
                }
            };
        }
        let copied = match self {
            Self::Int64 { data, nulls } if matches!(value, Value::Int64(_) | Value::Null) => {
                scalar!(Int64, data, nulls)
            }
            Self::Float64 { data, nulls }
                if matches!(value, Value::Float64(_) | Value::Int64(_) | Value::Null) =>
            {
                scalar!(Float64, data, nulls)
            }
            Self::UniqueId { data, nulls } if matches!(value, Value::UniqueId(_) | Value::Null) => {
                scalar!(UniqueId, data, nulls)
            }
            Self::Bool { data, nulls } if matches!(value, Value::Boolean(_) | Value::Null) => {
                scalar!(Bool, data, nulls)
            }
            Self::Date { data, nulls } if matches!(value, Value::DateTime(_) | Value::Null) => {
                scalar!(Date, data, nulls)
            }
            Self::Str {
                offsets,
                data,
                nulls,
                relocated,
            } if matches!(value, Value::String(_) | Value::Null) => {
                let additional = if let Value::String(text) = value {
                    text.len()
                } else {
                    0
                };
                Self::Str {
                    offsets: offsets.try_clone_for_append(1)?,
                    data: data.try_clone_for_append(additional)?,
                    nulls: nulls.try_clone_for_append(1)?,
                    relocated: relocated.clone(),
                }
            }
            // Mixed values and typed demotion retain generic cloning and
            // conversion rather than predicting their replacement storage.
            _ => return None,
        };
        #[cfg(test)]
        COLUMN_CLONES.set(COLUMN_CLONES.get() + 1);
        Some(copied)
    }
}

#[cfg(test)]
mod push_failure_tests {
    use super::*;
    use std::fs::File;

    #[test]
    fn typed_push_distinguishes_mismatch_from_storage_and_rolls_back() {
        let mut mismatch = TypedColumn::from_type_str("int64");
        assert!(matches!(
            mismatch.push(&Value::String("wrong".to_string())),
            Err(ColumnPushError::TypeMismatch)
        ));

        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("data.bin");
        let nulls_path = dir.path().join("nulls.bin");
        let mut data = MmapOrVec::mapped(&data_path, 64).unwrap();
        let mut nulls = MmapOrVec::mapped(&nulls_path, 64).unwrap();
        for value in 0..64 {
            data.try_push(value).unwrap();
            nulls.try_push(0).unwrap();
        }
        if let MmapOrVec::Mapped { file, .. } = &mut nulls {
            *file = File::open(&nulls_path).unwrap();
        }
        let mut column = TypedColumn::Int64 { data, nulls };

        assert!(matches!(
            column.push(&Value::Int64(64)),
            Err(ColumnPushError::Storage(_))
        ));
        assert_eq!(column.len(), 64);
        assert_eq!(column.get(63), Some(Value::Int64(63)));
    }
}

#[derive(Debug)]
pub(crate) enum ColumnPushError {
    TypeMismatch,
    Storage(io::Error),
}

impl std::fmt::Display for ColumnPushError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatch => formatter.write_str("column value type mismatch"),
            Self::Storage(error) => write!(formatter, "column storage append failed: {error}"),
        }
    }
}

fn push_pair<T: MmapPod>(
    data: &mut MmapOrVec<T>,
    value: T,
    nulls: &mut MmapOrVec<u8>,
    null_flag: u8,
) -> Result<(), ColumnPushError> {
    let data_len = data.len();
    data.try_push(value).map_err(ColumnPushError::Storage)?;
    if let Err(error) = nulls.try_push(null_flag) {
        data.truncate(data_len);
        return Err(ColumnPushError::Storage(error));
    }
    Ok(())
}

/// One row of a `Str` column, borrowed. `None` for out-of-range or null.
///
/// The single reader for the layout — every string read (predicate, grouping,
/// projection, id/title) comes through here, so the per-element `Heap`/`Mapped`
/// dispatch is taken once per array rather than once per access, and the
/// relocation overlay (empty on every store that has not taken a string `SET`)
/// is not hashed into at all until it has an entry.
///
/// `inline(always)`: three callers, all per-row, and the `Option<&str>` return
/// plus the `Heap`/`Mapped` dispatch only fold away once it is inlined into
/// them — left to the optimiser's discretion it cost a measured 14% on an
/// equality scan.
#[inline(always)]
fn str_at<'c>(
    offsets: &'c MmapOrVec<u64>,
    data: &'c MmapBytes,
    nulls: &'c MmapOrVec<u8>,
    relocated: &'c FxHashMap<u32, String>,
    row: u32,
) -> Option<&'c str> {
    let idx = row as usize;
    if *nulls.as_slice().get(idx)? != 0 {
        return None;
    }
    if !relocated.is_empty() {
        if let Some(s) = relocated.get(&row) {
            return Some(s.as_str());
        }
    }
    let offsets = offsets.as_slice();
    let start = *offsets.get(idx)? as usize;
    let end = *offsets.get(idx + 1)? as usize;
    let bytes = data.as_raw_bytes().get(start..end)?;
    // SAFETY: `Str` column bytes are either written in-process from
    // `Value::String` (`String::as_bytes()` — valid UTF-8 by Rust's core
    // invariant) or come from a packed file via `unpack_column`, which
    // validates the whole blob as UTF-8 and checks offset monotonicity and
    // bounds at load time.
    Some(unsafe { std::str::from_utf8_unchecked(bytes) })
}

/// Number of days from the Unix epoch to chrono's internal epoch.
/// Column data smaller than this threshold is loaded into heap Vec instead of
/// being written to a temp file and mmap'd. Avoids file I/O overhead for small columns.
pub(super) const MMAP_THRESHOLD: usize = 262_144; // 256 KB
pub(super) static NEXT_TEMP_COLUMN_FILE: AtomicU64 = AtomicU64::new(0);

const UNIX_EPOCH_DATE: NaiveDate = match NaiveDate::from_ymd_opt(1970, 1, 1) {
    Some(d) => d,
    None => unreachable!(),
};

impl TypedColumn {
    /// The dense-column tag `type_str` names, or `None` when it names nothing
    /// this store can hold densely.
    ///
    /// `node_type_metadata` is not a controlled vocabulary — it carries
    /// whatever `Value::type_name`, a DataFrame dtype or a `define_schema`
    /// declaration wrote — so a caller that wants to *prefer* declared metadata
    /// over the value in hand needs to know when the metadata is unusable
    /// rather than silently getting `Mixed`, which is the one shape that
    /// cannot be spilled. Matching is case-insensitive (metadata stores
    /// "Int64", "String", etc.).
    pub fn canonical_type_str(type_str: &str) -> Option<&'static str> {
        Some(match type_str.to_ascii_lowercase().as_str() {
            "int64" => "int64",
            "float64" => "float64",
            "uniqueid" => "uniqueid",
            "bool" | "boolean" => "bool",
            "date" | "datetime" => "date",
            "string" => "string",
            _ => return None,
        })
    }

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
                offsets: MmapOrVec::from_vec(vec![0u64]),
                data: MmapBytes::new(),
                nulls: MmapOrVec::new(),
                relocated: FxHashMap::default(),
            },
            _ => TypedColumn::Mixed { data: Vec::new() },
        }
    }

    /// The [`from_type_str`](Self::from_type_str) tag a value would need.
    ///
    /// The column-typing rule for every write site that creates a column
    /// *without* declared metadata to consult — a `SET` for a property the
    /// type has never carried, a `push_row` whose key is new to the store,
    /// the id column. Those sites used to fall through to
    /// `TypedColumn::Mixed`, which is 24-32 B/row of `Value` enums, has no
    /// file representation at all (`materialize_to_file` is a no-op for it),
    /// and therefore cannot be spilled or mapped — so a `set_memory_limit`
    /// could be escaped by writing one new property. The same mapping is what
    /// the disk create path already computes into `node_type_metadata` via
    /// `Value::type_name`.
    ///
    /// Values with no dense representation (`List`, `Map`, `Point`, `Duration`,
    /// `Timestamp`, graph entities) and `Null` — which carries no type
    /// evidence — still answer `"mixed"`.
    pub fn type_str_for_value(value: &Value) -> &'static str {
        match value {
            Value::Int64(_) => "int64",
            Value::Float64(_) => "float64",
            Value::UniqueId(_) => "uniqueid",
            Value::Boolean(_) => "bool",
            Value::DateTime(_) => "date",
            Value::String(_) => "string",
            _ => "mixed",
        }
    }

    /// An empty column typed to hold `value`. See [`Self::type_str_for_value`].
    pub fn for_value(value: &Value) -> Self {
        Self::from_type_str(Self::type_str_for_value(value))
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
    pub(crate) fn push(&mut self, value: &Value) -> Result<(), ColumnPushError> {
        match (self, value) {
            (TypedColumn::Int64 { data, nulls }, Value::Int64(v)) => {
                push_pair(data, *v, nulls, 0)?;
            }
            (TypedColumn::Int64 { data, nulls }, Value::Null) => {
                push_pair(data, 0, nulls, 1)?;
            }
            (TypedColumn::Float64 { data, nulls }, Value::Float64(v)) => {
                push_pair(data, *v, nulls, 0)?;
            }
            (TypedColumn::Float64 { data, nulls }, Value::Int64(v)) => {
                // Allow int→float promotion (common from pandas)
                push_pair(data, *v as f64, nulls, 0)?;
            }
            (TypedColumn::Float64 { data, nulls }, Value::Null) => {
                push_pair(data, 0.0, nulls, 1)?;
            }
            (TypedColumn::UniqueId { data, nulls }, Value::UniqueId(v)) => {
                push_pair(data, *v, nulls, 0)?;
            }
            (TypedColumn::UniqueId { data, nulls }, Value::Null) => {
                push_pair(data, 0, nulls, 1)?;
            }
            (TypedColumn::Bool { data, nulls }, Value::Boolean(v)) => {
                push_pair(data, *v as u8, nulls, 0)?;
            }
            (TypedColumn::Bool { data, nulls }, Value::Null) => {
                push_pair(data, 0, nulls, 1)?;
            }
            (TypedColumn::Date { data, nulls }, Value::DateTime(d)) => {
                let days = (*d - UNIX_EPOCH_DATE).num_days() as i32;
                push_pair(data, days, nulls, 0)?;
            }
            (TypedColumn::Date { data, nulls }, Value::Null) => {
                push_pair(data, 0, nulls, 1)?;
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
                // On mmap growth failure, report a failed typed push. The
                // caller's existing demotion path preserves the logical row
                // in a heap-backed Mixed column instead of panicking.
                let (data_len, offsets_len, nulls_len) = (data.len(), offsets.len(), nulls.len());
                let result = (|| {
                    data.extend(s.as_bytes())
                        .map_err(ColumnPushError::Storage)?;
                    offsets
                        .try_push(data.len() as u64)
                        .map_err(ColumnPushError::Storage)?;
                    nulls.try_push(0).map_err(ColumnPushError::Storage)
                })();
                if result.is_err() {
                    data.truncate(data_len);
                    offsets.truncate(offsets_len);
                    nulls.truncate(nulls_len);
                }
                result?;
            }
            (TypedColumn::Str { offsets, nulls, .. }, Value::Null) => {
                // Null string: push same offset (zero-length range)
                let last = if !offsets.is_empty() {
                    offsets.get(offsets.len() - 1)
                } else {
                    0
                };
                let offsets_len = offsets.len();
                offsets.try_push(last).map_err(ColumnPushError::Storage)?;
                if let Err(error) = nulls.try_push(1) {
                    offsets.truncate(offsets_len);
                    return Err(ColumnPushError::Storage(error));
                }
            }
            (TypedColumn::Mixed { data }, value) => {
                data.push(value.clone());
            }
            _ => return Err(ColumnPushError::TypeMismatch),
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
            } => str_at(offsets, data, nulls, relocated, row).map(|s| Value::String(s.to_owned())),
            TypedColumn::Mixed { data } => {
                let val = data.get(idx)?;
                if matches!(val, Value::Null) {
                    return None;
                }
                Some(val.clone())
            }
        }
    }

    /// Borrow the value at `row` when the column stores it as a `Value`.
    ///
    /// Only `Mixed` can answer: every other variant stores a decoded
    /// representation (a packed int, a string arena) and has to *build* a
    /// `Value` in [`Self::get`], so there is nothing to lend. `None` therefore
    /// means "read it through `get`", not "absent" — callers pair the two.
    ///
    /// This exists for the values whose clone is unbounded: a `Mixed` column is
    /// where a list property lands, and `get` clones the whole list on every
    /// element access. Measured on a 200-node graph doing 16 subscripts per
    /// node, release build: 0.19 µs/access at a stored length of 16 and
    /// 3.95 µs/access at 1024 — the per-access cost was the *list's* length.
    #[inline]
    pub fn get_ref(&self, row: u32) -> Option<&Value> {
        match self {
            TypedColumn::Mixed { data } => match data.get(row as usize) {
                Some(Value::Null) | None => None,
                some => some,
            },
            _ => None,
        }
    }

    /// Get a string column value as a borrowed &str, avoiding heap allocation.
    /// Returns None if the column is not a Str variant, row is out of bounds, or null.
    #[inline]
    pub fn get_str(&self, row: u32) -> Option<&str> {
        match self {
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => str_at(offsets, data, nulls, relocated, row),
            _ => None,
        }
    }

    /// Whether this row holds a non-null value, without materialising it.
    #[inline]
    pub fn is_present(&self, row: u32) -> bool {
        let idx = row as usize;
        match self {
            TypedColumn::Int64 { nulls, .. }
            | TypedColumn::Float64 { nulls, .. }
            | TypedColumn::UniqueId { nulls, .. }
            | TypedColumn::Bool { nulls, .. }
            | TypedColumn::Date { nulls, .. }
            | TypedColumn::Str { nulls, .. } => nulls.as_slice().get(idx).copied() == Some(0),
            TypedColumn::Mixed { data } => data.get(idx).is_some_and(|v| !matches!(v, Value::Null)),
        }
    }

    /// Borrowed string read that also reports *why* it could not borrow.
    ///
    /// Unlike [`Self::get_str`] — a `Str`-only accessor whose `None` the disk
    /// property index reads as "try the next route" — this answers for every
    /// column shape, so a caller can tell an absent field from one holding a
    /// non-string value. `Mixed` columns hold `Value`s and can still lend a
    /// `&str` out of one.
    #[inline]
    pub fn str_field(&self, row: u32) -> StrField<'_> {
        let idx = row as usize;
        match self {
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => match str_at(offsets, data, nulls, relocated, row) {
                Some(s) => StrField::Str(Cow::Borrowed(s)),
                None => StrField::Absent,
            },
            TypedColumn::Mixed { data } => match data.get(idx) {
                None | Some(Value::Null) => StrField::Absent,
                Some(Value::String(s)) => StrField::Str(Cow::Borrowed(s.as_str())),
                Some(_) => StrField::NotString,
            },
            // Fixed-width columns never hold a string; `get` distinguishes
            // present from null for them without allocating.
            fixed => match fixed.get(row) {
                Some(_) => StrField::NotString,
                None => StrField::Absent,
            },
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
                    offsets,
                    data,
                    nulls,
                    relocated,
                },
                Value::String(s),
            ) => {
                if idx >= nulls.len() {
                    return Err(());
                }
                // A same-length replacement writes where the value already is.
                // The overlay exists because `offsets[idx+1]` may not move —
                // and a value of the same length does not move it. This is the
                // whole cost of a re-`add_nodes` upsert, which rewrites every
                // row's title: an overlay entry per row is a `String` clone
                // plus a hash insert, and it never gets reclaimed until save.
                let same_length = !relocated.contains_key(&row)
                    && nulls.as_slice().get(idx).copied() == Some(0)
                    && {
                        let offsets = offsets.as_slice();
                        match (offsets.get(idx), offsets.get(idx + 1)) {
                            (Some(&start), Some(&end)) => {
                                (end - start) as usize == s.len()
                                    && data.overwrite_heap(start as usize, s.as_bytes())
                            }
                            _ => false,
                        }
                    };
                if !same_length {
                    // Park the new value in the relocated overlay. Mutating
                    // `offsets[idx+1]` in place corrupts row idx+1's start —
                    // see write_to for the on-save compaction.
                    relocated.insert(row, s.clone());
                }
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
        if self.push(&Value::Null).is_err() {
            // Infallible ColumnStore mutation boundary: preserve all existing
            // values and the new NULL in an explicit heap-backed fallback.
            let mut mixed = Vec::with_capacity(self.len() + 1);
            for row in 0..self.len() {
                mixed.push(self.get(row as u32).unwrap_or(Value::Null));
            }
            mixed.push(Value::Null);
            *self = Self::Mixed { data: mixed };
        }
    }

    /// Drop every row past `len`.
    ///
    /// The inverse of a tail of [`Self::push`] calls, and only that: a column
    /// is a stack, so truncation is exact for rows the same statement appended
    /// and meaningless for anything else. `Str` needs three buffers cut in
    /// step — `offsets` keeps its `len + 1` fence post, `data` loses the bytes
    /// past the surviving fence, and the relocation overlay drops keys that no
    /// longer name a row — because a stale entry there would resurface under
    /// the next row pushed onto the same index.
    pub(crate) fn truncate_rows(&mut self, len: usize) {
        if len >= self.len() {
            return;
        }
        match self {
            TypedColumn::Int64 { data, nulls } => {
                data.truncate(len);
                nulls.truncate(len);
            }
            TypedColumn::Float64 { data, nulls } => {
                data.truncate(len);
                nulls.truncate(len);
            }
            TypedColumn::UniqueId { data, nulls } => {
                data.truncate(len);
                nulls.truncate(len);
            }
            TypedColumn::Bool { data, nulls } => {
                data.truncate(len);
                nulls.truncate(len);
            }
            TypedColumn::Date { data, nulls } => {
                data.truncate(len);
                nulls.truncate(len);
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => {
                let keep_bytes = if len < offsets.len() {
                    offsets.get(len) as usize
                } else {
                    data.len()
                };
                offsets.truncate(len + 1);
                data.truncate(keep_bytes);
                nulls.truncate(len);
                relocated.retain(|row, _| (*row as usize) < len);
            }
            TypedColumn::Mixed { data } => data.truncate(len),
        }
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

    /// The subset of [`Self::heap_bytes`] that [`Self::materialize_to_file`]
    /// can actually move off the heap.
    ///
    /// Two parts of `heap_bytes` are structurally unspillable and must not
    /// enter the spill *trigger*'s arithmetic (they stay in `heap_bytes`,
    /// which is what `graph_info()['columnar_heap_bytes']` reports):
    ///
    /// * `Mixed` has no file representation at all — `materialize_to_file` is
    ///   a documented no-op for it.
    /// * the `Str` `relocated` overlay is deliberately left behind: it is the
    ///   write overlay a mapping cannot hold, and folding it away is a
    ///   compaction, not a spill.
    ///
    /// Counting them made the trigger unable to converge. `StorageMode::Mapped`
    /// pins `memory_limit = Some(0)`, so any unspillable byte left the total
    /// permanently over the limit and `maybe_spill_columns` re-ran its entire
    /// per-type loop — Vec + sort + a `create_dir_all` syscall per type — after
    /// every mutating statement, spilling nothing each time.
    pub fn spillable_heap_bytes(&self) -> usize {
        match self {
            TypedColumn::Int64 { .. }
            | TypedColumn::Float64 { .. }
            | TypedColumn::UniqueId { .. }
            | TypedColumn::Bool { .. }
            | TypedColumn::Date { .. } => self.heap_bytes(),
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                ..
            } => offsets.heap_bytes() + data.heap_bytes() + nulls.heap_bytes(),
            TypedColumn::Mixed { .. } => 0,
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
    /// For mixed: codec-selected `Vec<Value>`.
    pub(super) fn write_to_with_codec(
        &self,
        writer: &mut impl io::Write,
        codec: crate::serde_codec::CodecVersion,
    ) -> io::Result<()> {
        match self {
            TypedColumn::Int64 { data, nulls } => {
                write_packed_values(data, writer)?;
                write_packed_values(nulls, writer)?;
            }
            TypedColumn::Float64 { data, nulls } => {
                write_packed_values(data, writer)?;
                write_packed_values(nulls, writer)?;
            }
            TypedColumn::UniqueId { data, nulls } => {
                write_packed_values(data, writer)?;
                write_packed_values(nulls, writer)?;
            }
            TypedColumn::Bool { data, nulls } => {
                write_packed_values(data, writer)?;
                write_packed_values(nulls, writer)?;
            }
            TypedColumn::Date { data, nulls } => {
                write_packed_values(data, writer)?;
                write_packed_values(nulls, writer)?;
            }
            TypedColumn::Str {
                offsets,
                data,
                nulls,
                relocated,
            } => {
                if relocated.is_empty() {
                    // Fast path: no overlay, write raw buffers.
                    write_packed_values(offsets, writer)?;
                    data.write_to(writer)?;
                    write_packed_values(nulls, writer)?;
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
                    write_packed_values(nulls, writer)?;
                }
            }
            TypedColumn::Mixed { data } => {
                let encoded = crate::serde_codec::encode_versioned(codec, data, u64::MAX)
                    .map_err(|e| io::Error::other(format!("column codec error: {e}")))?;
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

#[cfg(test)]
mod append_capacity_tests;

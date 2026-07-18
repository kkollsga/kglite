//! Streaming-write per-type column writer for save_subset v3.
//!
//! `TypeWriter` writes a single node type's column store *directly* to
//! its final on-disk files via per-file [`BufWriter`]s — no heap
//! accumulation of typed columns, no chunking, no merge step. Each
//! typed column corresponds to a small set of files (offsets / data /
//! nulls) opened on creation and appended to in lockstep with row
//! pushes. `finalize()` flushes the writers, mmaps the closed files
//! into [`TypedColumn`] variants, and returns an `Arc<ColumnStore>`
//! ready to install in `dest.column_stores`.
//!
//! This replaces v2's `ChunkedColumnBuilder` (which buffered N rows in
//! a heap-backed `ColumnStore`, spilled to mmap files, and merged at
//! finalize). v2 had two issues: (a) it solved the wrong problem
//! (bounded the *transient* heap inside a type but not the aggregate
//! live mmap-page footprint across types), and (b) the
//! `sanitize_type_name` collapse on Unicode-only types caused
//! finalize() races to corrupt each other's mmap-backed files. v3
//! sidesteps both by streaming directly to the dest's final files —
//! bounded heap = O(file-handle-buffer-size × open-typed-cols), and
//! each type's files have a unique hash-suffixed path.
//!
//! Per-typed-column on-disk layout (matches
//! `ColumnStore::materialize_to_files`):
//!
//! - Int64:    `<col>.i64`   + `<col>.null`
//! - Float64:  `<col>.f64`   + `<col>.null`
//! - UniqueId: `<col>.u32`   + `<col>.null`
//! - Bool:     `<col>.bool`  + `<col>.null`
//! - Date:     `<col>.i32`   + `<col>.null`
//! - Str:      `<col>.off`   + `<col>.str`  + `<col>.null`
//! - Mixed:    heap-buffered until finalize (no streaming format)

use crate::datatypes::values::{BorrowedValue, Value};
use crate::graph::schema::{InternedKey, StringInterner, TypeSchema};
use crate::graph::storage::column_store::{ColumnStore, TypedColumn};
use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Default BufWriter buffer size per file. 64 KB matches the default
/// Linux pipe buffer; trades a bit of heap for fewer syscalls during
/// the write loop. With ~5 files per typed column × ~5 columns × 4500
/// types = ~100 K open BufWriters, 64 KB × 100 K = 6.4 GB worst-case
/// — too high. Tune down for small types if needed; for now, the
/// kernel only allocates the buffer on first write so types with zero
/// rows pay no cost.
const BUF_SIZE: usize = 64 * 1024;

/// Per-type streaming writer.
pub(super) struct TypeWriter {
    schema: Arc<TypeSchema>,
    meta: HashMap<String, String>,
    out_dir: PathBuf,
    /// Per-schema-slot writer (matches `schema.iter()` slot order).
    columns: Vec<ColumnWriter>,
    /// `__id__` column. Pre-created as Str; non-string id values are
    /// rejected with an error from `push_row`.
    id_col: ColumnWriter,
    /// `__title__` column. Pre-created as Str.
    title_col: ColumnWriter,
    row_count: u32,
    /// Per-row offsets into [`Self::overflow_data`]. Length =
    /// `row_count + 1` once `finalize` is called (with the final entry
    /// being the past-the-end offset, matching the source's wire
    /// format — see [`crate::graph::storage::overflow`]).
    overflow_offsets: Vec<u64>,
    /// Concatenated per-row overflow blobs. Each row's blob is
    /// `[u16 LE: num_entries] + entries…` where each entry is
    /// `[u64 LE: key] + [u8: type_tag] + [type-specific value bytes]`.
    /// Format mirrors [`crate::graph::io::ntriples::column_builder`].
    overflow_data: Vec<u8>,
    /// True once any property has been routed to overflow on any row.
    /// Avoids emitting an overflow bag at all on types whose source
    /// stored everything in schema slots.
    overflow_used: bool,
}

/// Per-typed-column streaming writer. Each variant owns the open
/// `BufWriter`s and the running state (cursor / len).
enum ColumnWriter {
    Int64 {
        data: BufWriter<File>,
        nulls: BufWriter<File>,
        len: u32,
        data_path: PathBuf,
        nulls_path: PathBuf,
    },
    Float64 {
        data: BufWriter<File>,
        nulls: BufWriter<File>,
        len: u32,
        data_path: PathBuf,
        nulls_path: PathBuf,
    },
    UniqueId {
        data: BufWriter<File>,
        nulls: BufWriter<File>,
        len: u32,
        data_path: PathBuf,
        nulls_path: PathBuf,
    },
    Bool {
        data: BufWriter<File>,
        nulls: BufWriter<File>,
        len: u32,
        data_path: PathBuf,
        nulls_path: PathBuf,
    },
    Date {
        data: BufWriter<File>,
        nulls: BufWriter<File>,
        len: u32,
        data_path: PathBuf,
        nulls_path: PathBuf,
    },
    Str {
        offsets: BufWriter<File>,
        data: BufWriter<File>,
        nulls: BufWriter<File>,
        /// Cumulative byte cursor — updated on every push to track
        /// where the next string's bytes start in the data file.
        cursor: u64,
        len: u32,
        offsets_path: PathBuf,
        data_path: PathBuf,
        nulls_path: PathBuf,
    },
    /// Heterogeneous-typed column. Buffered in heap because there's
    /// no streaming-friendly serialization that round-trips Mixed.
    Mixed { values: Vec<Value> },
}

impl TypeWriter {
    /// Open all files for a type's columns under `out_dir`.
    ///
    /// `id_type` and `title_type` should be the source's id/title
    /// column type tags (e.g. `"string"` or `"uniqueid"`) — typically
    /// fetched via `ColumnStore::id_type_str()` /
    /// `title_type_str()`. Wikidata mixes string-Q-code ids with
    /// `UniqueId` ids across types, so a hard-coded "string" choice
    /// fails on real data with "expected String/Null, got UniqueId".
    /// Defaults to "string" when the caller has nothing better — the
    /// writer's push() will surface a typed error if reality
    /// disagrees.
    pub(super) fn new(
        schema: Arc<TypeSchema>,
        meta: HashMap<String, String>,
        out_dir: PathBuf,
        interner: &StringInterner,
        id_type: &str,
        title_type: &str,
    ) -> io::Result<Self> {
        std::fs::create_dir_all(&out_dir)?;

        let mut columns = Vec::with_capacity(schema.len());
        for (_slot, ik) in schema.iter() {
            let col_name = interner.resolve(ik);
            let type_str = meta.get(col_name).map(|s| s.as_str()).unwrap_or("mixed");
            columns.push(ColumnWriter::open(type_str, col_name, &out_dir)?);
        }

        let id_col = ColumnWriter::open(id_type, "__id__", &out_dir)?;
        let title_col = ColumnWriter::open(title_type, "__title__", &out_dir)?;

        // Overflow offsets always start with offset 0 for row 0; final
        // past-the-end offset gets pushed in `finalize()`.
        let overflow_offsets = vec![0u64];

        Ok(Self {
            schema,
            meta,
            out_dir,
            columns,
            id_col,
            title_col,
            row_count: 0,
            overflow_offsets,
            overflow_data: Vec::new(),
            overflow_used: false,
        })
    }

    /// Append one row. Pushes id/title first, then per-schema-slot
    /// property values in slot order, padding null for unset slots.
    /// Non-schema keys in `props` are silently dropped — matches v1
    /// `ColumnStore::push_row` behavior.
    #[cfg(test)]
    pub(super) fn push_row(
        &mut self,
        id: &Value,
        title: &Value,
        props: &[(InternedKey, Value)],
    ) -> io::Result<u32> {
        self.id_col.push(id)?;
        self.title_col.push(title)?;

        // Build slot lookup from props.
        let mut slot_values: Vec<Option<&Value>> = vec![None; self.columns.len()];
        for (key, value) in props {
            if let Some(slot) = self.schema.slot(*key) {
                slot_values[slot as usize] = Some(value);
            }
        }
        for (slot, slot_val) in slot_values.iter().enumerate() {
            let v: &Value = slot_val.unwrap_or(&Value::Null);
            self.columns[slot].push(v)?;
        }

        let row_id = self.row_count;
        self.row_count = self
            .row_count
            .checked_add(1)
            .ok_or_else(|| io::Error::other("TypeWriter: row count overflow > u32::MAX"))?;
        Ok(row_id)
    }

    /// Allocation-free row push. Caller invokes `f(visitor)` and
    /// for each property calls `visitor.push_property(key,
    /// borrowed_value)`. Strings stay borrowed from the source mmap;
    /// dest's BufWriters write `&[u8]` directly without going through
    /// `Value::String(s.to_string())`.
    ///
    /// On Wikidata Articles+P50+Authors this is the lever: the v3
    /// node walk's `read_source` was 381 s of 407 s wall time,
    /// dominated by ~510 M `Value::String` clones inside
    /// `MmapColumnStore::row_properties`. Borrowing eliminates those
    /// clones entirely; only Mixed-column writes still allocate.
    pub(super) fn push_row_borrowed<F>(
        &mut self,
        id: BorrowedValue<'_>,
        title: BorrowedValue<'_>,
        f: F,
    ) -> io::Result<u32>
    where
        F: FnOnce(&mut RowVisitor<'_>) -> io::Result<()>,
    {
        self.id_col.push_borrowed(id)?;
        self.title_col.push_borrowed(title)?;

        // Per-row overflow scratch buffer. Starts with a 2-byte
        // num_entries placeholder; finalised after the closure runs.
        let mut row_overflow: Vec<u8> = Vec::new();
        row_overflow.extend_from_slice(&0u16.to_le_bytes());
        let mut row_overflow_count: u16 = 0;

        let filled = {
            let mut visitor = RowVisitor {
                schema: &self.schema,
                columns: &mut self.columns,
                filled: SmallBitmap::new(self.schema.len()),
                overflow_buf: &mut row_overflow,
                overflow_count: &mut row_overflow_count,
            };
            f(&mut visitor)?;
            visitor.filled
        };
        for slot in 0..self.columns.len() {
            if !filled.get(slot) {
                self.columns[slot].push_borrowed(BorrowedValue::Null)?;
            }
        }

        // Finalise the row's overflow blob: backfill the entry count.
        // Append unconditionally — every row needs an entry in
        // overflow_data so the offsets array stays in step with
        // row_count, even when this row had no overflow keys.
        if row_overflow_count > 0 {
            self.overflow_used = true;
        }
        row_overflow[0..2].copy_from_slice(&row_overflow_count.to_le_bytes());
        self.overflow_data.extend_from_slice(&row_overflow);
        self.overflow_offsets.push(self.overflow_data.len() as u64);

        let row_id = self.row_count;
        self.row_count = self
            .row_count
            .checked_add(1)
            .ok_or_else(|| io::Error::other("TypeWriter: row count overflow > u32::MAX"))?;
        Ok(row_id)
    }

    /// Flush writers, mmap the closed files, build TypedColumns, and
    /// wrap in an Arc<ColumnStore>. The writer is consumed; its files
    /// outlive it via the mmaps held by the returned ColumnStore.
    pub(super) fn finalize(self, interner: &StringInterner) -> io::Result<Arc<ColumnStore>> {
        let row_count = self.row_count;

        let mut typed_columns: Vec<TypedColumn> = Vec::with_capacity(self.columns.len());
        for col in self.columns {
            typed_columns.push(col.finalize(row_count)?);
        }
        let id_typed = self.id_col.finalize(row_count)?;
        let title_typed = self.title_col.finalize(row_count)?;

        let mut store = ColumnStore::new(Arc::clone(&self.schema), &self.meta, interner);
        store.replace_columns(typed_columns);
        store.replace_id_column(id_typed);
        store.replace_title_column(title_typed);
        store.set_row_count(row_count);

        // Install the overflow bag if any row touched it. The offsets
        // array invariant is `len == row_count + 1` — the trailing
        // entry is the past-the-end byte offset, which the read path
        // uses to bound the last row's blob. We've been pushing
        // `overflow_data.len()` after each row, so the invariant
        // already holds.
        if self.overflow_used {
            use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
            let offsets_vec = MmapOrVec::from_vec(self.overflow_offsets);
            let mut data_bytes = MmapBytes::new();
            data_bytes.extend(&self.overflow_data)?;
            store.replace_overflow_bag(offsets_vec, data_bytes);
        }

        // Discard out_dir reference — the open file handles live inside
        // the mmaps held by `store`. They keep the files alive as long
        // as the Arc is alive.
        let _ = self.out_dir;
        Ok(Arc::new(store))
    }
}

/// Visitor handed to `push_row_borrowed`'s closure. Forwards each
/// property to its schema slot's column writer and tracks which
/// slots have been filled this row, so unfilled slots get nulls
/// after the closure returns. Properties whose key isn't in the
/// schema are routed to the per-row overflow buffer in the same
/// wire format the source's `MmapColumnStore` uses, so they
/// round-trip through reload.
pub(super) struct RowVisitor<'a> {
    schema: &'a Arc<TypeSchema>,
    columns: &'a mut Vec<ColumnWriter>,
    filled: SmallBitmap,
    /// Scratch buffer for this row's overflow blob. Pre-seeded with a
    /// 2-byte LE entry-count placeholder; `push_row_borrowed` writes
    /// the real count back after the closure returns.
    overflow_buf: &'a mut Vec<u8>,
    /// Number of entries appended to `overflow_buf` this row.
    overflow_count: &'a mut u16,
}

impl<'a> RowVisitor<'a> {
    /// Push a borrowed property into its schema slot, or into the
    /// per-row overflow buffer when the key is not in the schema.
    pub(super) fn push_property(
        &mut self,
        key: InternedKey,
        value: BorrowedValue<'_>,
    ) -> io::Result<()> {
        if let Some(slot) = self.schema.slot(key) {
            let s = slot as usize;
            if self.filled.get(s) {
                // Defensive: each row should hit each schema slot at
                // most once. A double-fill desyncs the column
                // writer's row count from `self.row_count`.
                return Err(io::Error::other(format!(
                    "TypeWriter::push_row_borrowed: slot {} filled twice in one row",
                    s
                )));
            }
            self.columns[s].push_borrowed(value)?;
            self.filled.set(s);
        } else {
            // Non-schema key: route to the per-row overflow buffer.
            // Skip Null entries — they round-trip as "absent" anyway
            // and just bloat the blob (matches the column_builder's
            // matches!(value, Value::Null) skip on the ntriples
            // ingestion path).
            if !matches!(value, BorrowedValue::Null) {
                crate::graph::storage::overflow::encode_value_borrowed(
                    self.overflow_buf,
                    key,
                    &value,
                );
                *self.overflow_count = self.overflow_count.saturating_add(1);
            }
        }
        Ok(())
    }
}

/// Per-row "which slots have been filled" bitmap. 64-bit inline;
/// falls back to a `Vec<u64>` only for schemas wider than 64.
struct SmallBitmap {
    inline: u64,
    overflow: Vec<u64>,
    len: usize,
}

impl SmallBitmap {
    fn new(len: usize) -> Self {
        let overflow = if len > 64 {
            vec![0u64; (len - 64).div_ceil(64)]
        } else {
            Vec::new()
        };
        Self {
            inline: 0,
            overflow,
            len,
        }
    }

    fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len);
        if idx < 64 {
            (self.inline >> idx) & 1 == 1
        } else {
            let i = idx - 64;
            (self.overflow[i / 64] >> (i % 64)) & 1 == 1
        }
    }

    fn set(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        if idx < 64 {
            self.inline |= 1u64 << idx;
        } else {
            let i = idx - 64;
            self.overflow[i / 64] |= 1u64 << (i % 64);
        }
    }
}

impl ColumnWriter {
    fn open(type_str: &str, col_name: &str, out_dir: &Path) -> io::Result<Self> {
        let lower = type_str.to_ascii_lowercase();
        match lower.as_str() {
            "int64" => Self::open_fixed(
                col_name,
                out_dir,
                "i64",
                |data, nulls, len, data_path, nulls_path| ColumnWriter::Int64 {
                    data,
                    nulls,
                    len,
                    data_path,
                    nulls_path,
                },
            ),
            "float64" => Self::open_fixed(
                col_name,
                out_dir,
                "f64",
                |data, nulls, len, data_path, nulls_path| ColumnWriter::Float64 {
                    data,
                    nulls,
                    len,
                    data_path,
                    nulls_path,
                },
            ),
            "uniqueid" => Self::open_fixed(
                col_name,
                out_dir,
                "u32",
                |data, nulls, len, data_path, nulls_path| ColumnWriter::UniqueId {
                    data,
                    nulls,
                    len,
                    data_path,
                    nulls_path,
                },
            ),
            "bool" | "boolean" => Self::open_fixed(
                col_name,
                out_dir,
                "bool",
                |data, nulls, len, data_path, nulls_path| ColumnWriter::Bool {
                    data,
                    nulls,
                    len,
                    data_path,
                    nulls_path,
                },
            ),
            "date" | "datetime" => Self::open_fixed(
                col_name,
                out_dir,
                "i32",
                |data, nulls, len, data_path, nulls_path| ColumnWriter::Date {
                    data,
                    nulls,
                    len,
                    data_path,
                    nulls_path,
                },
            ),
            "string" => {
                let offsets_path = out_dir.join(format!("{col_name}.off"));
                let data_path = out_dir.join(format!("{col_name}.str"));
                let nulls_path = out_dir.join(format!("{col_name}.null"));
                let mut offsets = open_buf_writer(&offsets_path)?;
                let data = open_buf_writer(&data_path)?;
                let nulls = open_buf_writer(&nulls_path)?;
                // Write the leading-0 entry of the offsets array.
                offsets.write_all(&0u64.to_le_bytes())?;
                Ok(ColumnWriter::Str {
                    offsets,
                    data,
                    nulls,
                    cursor: 0,
                    len: 0,
                    offsets_path,
                    data_path,
                    nulls_path,
                })
            }
            // "mixed" or unknown — buffer in heap.
            _ => Ok(ColumnWriter::Mixed { values: Vec::new() }),
        }
    }

    fn open_fixed<F>(col_name: &str, out_dir: &Path, data_ext: &str, make: F) -> io::Result<Self>
    where
        F: FnOnce(BufWriter<File>, BufWriter<File>, u32, PathBuf, PathBuf) -> Self,
    {
        let data_path = out_dir.join(format!("{col_name}.{data_ext}"));
        let nulls_path = out_dir.join(format!("{col_name}.null"));
        let data = open_buf_writer(&data_path)?;
        let nulls = open_buf_writer(&nulls_path)?;
        Ok(make(data, nulls, 0, data_path, nulls_path))
    }

    /// Allocation-free counterpart of [`push`]. Writes the borrowed
    /// `value` straight to the column's `BufWriter`s — `String` goes
    /// to `data` as `&[u8]` without ever materializing `Value::String`,
    /// fixed types are by-value and trivially cheap.
    fn push_borrowed(&mut self, value: BorrowedValue<'_>) -> io::Result<()> {
        match self {
            ColumnWriter::Int64 {
                data, nulls, len, ..
            } => {
                let (v, is_null): (i64, u8) = match value {
                    BorrowedValue::Int64(x) => (x, 0),
                    BorrowedValue::Null => (0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push_borrowed: expected Int64/Null, got {:?}",
                            borrowed_kind(&other)
                        )))
                    }
                };
                data.write_all(&v.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Float64 {
                data, nulls, len, ..
            } => {
                let (v, is_null): (f64, u8) = match value {
                    BorrowedValue::Float64(x) => (x, 0),
                    BorrowedValue::Int64(x) => (x as f64, 0),
                    BorrowedValue::Null => (0.0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push_borrowed: expected Float64/Int64/Null, got {:?}",
                            borrowed_kind(&other)
                        )))
                    }
                };
                data.write_all(&v.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::UniqueId {
                data, nulls, len, ..
            } => {
                let (v, is_null): (u32, u8) = match value {
                    BorrowedValue::UniqueId(x) => (x, 0),
                    BorrowedValue::Null => (0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push_borrowed: expected UniqueId/Null, got {:?}",
                            borrowed_kind(&other)
                        )))
                    }
                };
                data.write_all(&v.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Bool {
                data, nulls, len, ..
            } => {
                let (v, is_null): (u8, u8) = match value {
                    BorrowedValue::Boolean(b) => (b as u8, 0),
                    BorrowedValue::Null => (0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push_borrowed: expected Boolean/Null, got {:?}",
                            borrowed_kind(&other)
                        )))
                    }
                };
                data.write_all(&[v])?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Date {
                data, nulls, len, ..
            } => {
                let (days, is_null): (i32, u8) = match value {
                    BorrowedValue::DateTime(d) => {
                        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                            .expect("1970-01-01 is a valid date");
                        ((d - epoch).num_days() as i32, 0)
                    }
                    BorrowedValue::Null => (0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push_borrowed: expected DateTime/Null, got {:?}",
                            borrowed_kind(&other)
                        )))
                    }
                };
                data.write_all(&days.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Str {
                offsets,
                data,
                nulls,
                cursor,
                len,
                ..
            } => {
                let (bytes, is_null): (&[u8], u8) = match value {
                    BorrowedValue::String(s) => (s.as_bytes(), 0),
                    BorrowedValue::Null => (&[], 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push_borrowed: expected String/Null, got {:?}",
                            borrowed_kind(&other)
                        )))
                    }
                };
                if !bytes.is_empty() {
                    data.write_all(bytes)?;
                    *cursor = cursor
                        .checked_add(bytes.len() as u64)
                        .ok_or_else(|| io::Error::other("Str data cursor overflow"))?;
                }
                offsets.write_all(&cursor.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Mixed { values } => {
                // Mixed columns can't avoid heap (the binary Serde payload
                // at finalize captures owned `Value`s). Convert at
                // push time.
                values.push(value.to_value());
            }
        }
        Ok(())
    }

    /// Owned `&Value` push variant — kept for the test harness's
    /// `push_row` path. Production traffic goes through
    /// [`push_borrowed`].
    #[cfg(test)]
    fn push(&mut self, value: &Value) -> io::Result<()> {
        match self {
            ColumnWriter::Int64 {
                data, nulls, len, ..
            } => {
                let (v, is_null): (i64, u8) = match value {
                    Value::Int64(x) => (*x, 0),
                    Value::Null => (0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push: expected Int64/Null, got {:?}",
                            value_kind(other)
                        )))
                    }
                };
                data.write_all(&v.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Float64 {
                data, nulls, len, ..
            } => {
                let (v, is_null): (f64, u8) = match value {
                    Value::Float64(x) => (*x, 0),
                    // Permit Int64 → Float64 promotion to mirror
                    // TypedColumn::push's behavior on the heap path.
                    Value::Int64(x) => (*x as f64, 0),
                    Value::Null => (0.0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push: expected Float64/Int64/Null, got {:?}",
                            value_kind(other)
                        )))
                    }
                };
                data.write_all(&v.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::UniqueId {
                data, nulls, len, ..
            } => {
                let (v, is_null): (u32, u8) = match value {
                    Value::UniqueId(x) => (*x, 0),
                    Value::Null => (0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push: expected UniqueId/Null, got {:?}",
                            value_kind(other)
                        )))
                    }
                };
                data.write_all(&v.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Bool {
                data, nulls, len, ..
            } => {
                let (v, is_null): (u8, u8) = match value {
                    Value::Boolean(b) => (*b as u8, 0),
                    Value::Null => (0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push: expected Boolean/Null, got {:?}",
                            value_kind(other)
                        )))
                    }
                };
                data.write_all(&[v])?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Date {
                data, nulls, len, ..
            } => {
                let (days, is_null): (i32, u8) = match value {
                    Value::DateTime(d) => {
                        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                            .expect("1970-01-01 is a valid date");
                        ((*d - epoch).num_days() as i32, 0)
                    }
                    Value::Null => (0, 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push: expected DateTime/Null, got {:?}",
                            value_kind(other)
                        )))
                    }
                };
                data.write_all(&days.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Str {
                offsets,
                data,
                nulls,
                cursor,
                len,
                ..
            } => {
                let (bytes, is_null): (&[u8], u8) = match value {
                    Value::String(s) => (s.as_bytes(), 0),
                    Value::Null => (&[], 1),
                    other => {
                        return Err(io::Error::other(format!(
                            "TypeWriter::push: expected String/Null, got {:?}",
                            value_kind(other)
                        )))
                    }
                };
                if !bytes.is_empty() {
                    data.write_all(bytes)?;
                    *cursor = cursor
                        .checked_add(bytes.len() as u64)
                        .ok_or_else(|| io::Error::other("Str data cursor overflow"))?;
                }
                offsets.write_all(&cursor.to_le_bytes())?;
                nulls.write_all(&[is_null])?;
                *len = len.checked_add(1).ok_or_else(row_overflow)?;
            }
            ColumnWriter::Mixed { values } => {
                values.push(value.clone());
            }
        }
        Ok(())
    }

    fn finalize(self, row_count: u32) -> io::Result<TypedColumn> {
        match self {
            ColumnWriter::Int64 {
                data,
                nulls,
                len,
                data_path,
                nulls_path,
            } => {
                close_buf_writers([data, nulls])?;
                expect_len(len, row_count, &data_path)?;
                let data = MmapOrVec::<i64>::load_mapped(&data_path, row_count as usize)?;
                let nulls = MmapOrVec::<u8>::load_mapped(&nulls_path, row_count as usize)?;
                Ok(TypedColumn::Int64 { data, nulls })
            }
            ColumnWriter::Float64 {
                data,
                nulls,
                len,
                data_path,
                nulls_path,
            } => {
                close_buf_writers([data, nulls])?;
                expect_len(len, row_count, &data_path)?;
                let data = MmapOrVec::<f64>::load_mapped(&data_path, row_count as usize)?;
                let nulls = MmapOrVec::<u8>::load_mapped(&nulls_path, row_count as usize)?;
                Ok(TypedColumn::Float64 { data, nulls })
            }
            ColumnWriter::UniqueId {
                data,
                nulls,
                len,
                data_path,
                nulls_path,
            } => {
                close_buf_writers([data, nulls])?;
                expect_len(len, row_count, &data_path)?;
                let data = MmapOrVec::<u32>::load_mapped(&data_path, row_count as usize)?;
                let nulls = MmapOrVec::<u8>::load_mapped(&nulls_path, row_count as usize)?;
                Ok(TypedColumn::UniqueId { data, nulls })
            }
            ColumnWriter::Bool {
                data,
                nulls,
                len,
                data_path,
                nulls_path,
            } => {
                close_buf_writers([data, nulls])?;
                expect_len(len, row_count, &data_path)?;
                let data = MmapOrVec::<u8>::load_mapped(&data_path, row_count as usize)?;
                let nulls = MmapOrVec::<u8>::load_mapped(&nulls_path, row_count as usize)?;
                Ok(TypedColumn::Bool { data, nulls })
            }
            ColumnWriter::Date {
                data,
                nulls,
                len,
                data_path,
                nulls_path,
            } => {
                close_buf_writers([data, nulls])?;
                expect_len(len, row_count, &data_path)?;
                let data = MmapOrVec::<i32>::load_mapped(&data_path, row_count as usize)?;
                let nulls = MmapOrVec::<u8>::load_mapped(&nulls_path, row_count as usize)?;
                Ok(TypedColumn::Date { data, nulls })
            }
            ColumnWriter::Str {
                offsets,
                data,
                nulls,
                cursor,
                len,
                offsets_path,
                data_path,
                nulls_path,
            } => {
                close_buf_writers([offsets, data, nulls])?;
                expect_len(len, row_count, &offsets_path)?;
                let offsets =
                    MmapOrVec::<u64>::load_mapped(&offsets_path, (row_count as usize) + 1)?;
                let data = MmapBytes::load_mapped(&data_path, cursor as usize)?;
                let nulls = MmapOrVec::<u8>::load_mapped(&nulls_path, row_count as usize)?;
                Ok(TypedColumn::Str {
                    offsets,
                    data,
                    nulls,
                    relocated: std::collections::HashMap::new(),
                })
            }
            ColumnWriter::Mixed { values } => {
                if values.len() as u32 != row_count {
                    return Err(io::Error::other(format!(
                        "TypeWriter Mixed column finalize: expected {} values, got {}",
                        row_count,
                        values.len()
                    )));
                }
                Ok(TypedColumn::Mixed { data: values })
            }
        }
    }
}

fn open_buf_writer(path: &Path) -> io::Result<BufWriter<File>> {
    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    Ok(BufWriter::with_capacity(BUF_SIZE, f))
}

fn close_buf_writers<const N: usize>(writers: [BufWriter<File>; N]) -> io::Result<()> {
    for mut w in writers {
        w.flush()?;
        // Dropping `w` (and its underlying File) closes the fd. The
        // file is then re-openable for mmap by the caller. Explicit
        // sync is unnecessary — load_mapped reads via the kernel's
        // page cache which is already coherent with the just-flushed
        // bytes.
    }
    Ok(())
}

fn expect_len(actual: u32, expected: u32, path: &Path) -> io::Result<()> {
    if actual != expected {
        return Err(io::Error::other(format!(
            "TypeWriter::finalize: column {} pushed {} rows, expected {}",
            path.display(),
            actual,
            expected
        )));
    }
    Ok(())
}

fn row_overflow() -> io::Error {
    io::Error::other("TypeWriter column row count overflow > u32::MAX")
}

fn borrowed_kind(v: &BorrowedValue<'_>) -> &'static str {
    match v {
        BorrowedValue::Null => "Null",
        BorrowedValue::Boolean(_) => "Boolean",
        BorrowedValue::Int64(_) => "Int64",
        BorrowedValue::Float64(_) => "Float64",
        BorrowedValue::UniqueId(_) => "UniqueId",
        BorrowedValue::String(_) => "String",
        BorrowedValue::DateTime(_) => "DateTime",
        BorrowedValue::Timestamp(_) => "Timestamp",
        BorrowedValue::List(_) => "List",
        BorrowedValue::Map(_) => "Map",
    }
}

/// Short identifier for a Value's variant, used in type-mismatch
/// errors. We don't depend on `Value::Debug` because some variants
/// would print large payloads.
#[cfg(test)]
fn value_kind(v: &Value) -> &'static str {
    // Phase A.1 / C7a — delegates to Value::type_name (canonical
    // PascalCase form); was a duplicate per-variant match.
    v.type_name()
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::graph::schema::TypeSchema;
    use tempfile::TempDir;

    fn make_schema(interner: &mut StringInterner, props: &[&str]) -> Arc<TypeSchema> {
        let keys: Vec<InternedKey> = props.iter().map(|p| interner.get_or_intern(p)).collect();
        Arc::new(TypeSchema::from_keys(keys))
    }

    /// Build a TypeWriter with a mix of typed columns and verify the
    /// finalized ColumnStore reads back identical values per row.
    #[test]
    fn round_trip_mixed_types() {
        let mut interner = StringInterner::new();
        let schema = make_schema(&mut interner, &["age", "name", "active", "score"]);
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("age".to_string(), "int64".to_string());
        meta.insert("name".to_string(), "string".to_string());
        meta.insert("active".to_string(), "bool".to_string());
        meta.insert("score".to_string(), "float64".to_string());

        let dir = TempDir::new().unwrap();
        let mut writer = TypeWriter::new(
            Arc::clone(&schema),
            meta.clone(),
            dir.path().to_path_buf(),
            &interner,
            "string",
            "string",
        )
        .unwrap();

        let key_age = interner.get_or_intern("age");
        let key_name = interner.get_or_intern("name");
        let key_active = interner.get_or_intern("active");
        let key_score = interner.get_or_intern("score");

        let names = ["Alice", "Bob", "Carla-with-a-longer-name", "", "Dee"];
        for i in 0..5u32 {
            let id = Value::String(format!("id-{i}"));
            let title = Value::String(format!("Title {i}"));
            let props = vec![
                (key_age, Value::Int64(20 + i as i64)),
                (key_name, Value::String(names[i as usize].to_string())),
                (key_active, Value::Boolean(i % 2 == 0)),
                (key_score, Value::Float64(i as f64 * 1.5)),
            ];
            let row = writer.push_row(&id, &title, &props).unwrap();
            assert_eq!(row, i);
        }

        let store = writer.finalize(&interner).unwrap();
        assert_eq!(store.row_count(), 5);

        for i in 0..5u32 {
            assert_eq!(store.get_id(i).unwrap(), Value::String(format!("id-{i}")));
            assert_eq!(
                store.get_title(i).unwrap(),
                Value::String(format!("Title {i}"))
            );
            assert_eq!(store.get(i, key_age).unwrap(), Value::Int64(20 + i as i64));
            let expected_name = names[i as usize].to_string();
            if expected_name.is_empty() {
                // Empty string round-trips as Value::String("") (offset
                // delta = 0, null byte = 0). Verify we don't conflate
                // with Null.
                assert_eq!(
                    store.get(i, key_name).unwrap(),
                    Value::String(String::new())
                );
            } else {
                assert_eq!(
                    store.get(i, key_name).unwrap(),
                    Value::String(expected_name)
                );
            }
            assert_eq!(
                store.get(i, key_active).unwrap(),
                Value::Boolean(i % 2 == 0)
            );
            assert_eq!(
                store.get(i, key_score).unwrap(),
                Value::Float64(i as f64 * 1.5)
            );
        }
    }

    /// Null values for every typed column should round-trip as Null on
    /// read (not Default::default() for the type).
    #[test]
    fn null_values_round_trip() {
        let mut interner = StringInterner::new();
        let schema = make_schema(&mut interner, &["x", "s"]);
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("x".to_string(), "int64".to_string());
        meta.insert("s".to_string(), "string".to_string());
        let key_x = interner.get_or_intern("x");
        let key_s = interner.get_or_intern("s");

        let dir = TempDir::new().unwrap();
        let mut writer = TypeWriter::new(
            Arc::clone(&schema),
            meta,
            dir.path().to_path_buf(),
            &interner,
            "string",
            "string",
        )
        .unwrap();
        writer
            .push_row(
                &Value::Null,
                &Value::Null,
                &[(key_x, Value::Null), (key_s, Value::Null)],
            )
            .unwrap();
        writer
            .push_row(
                &Value::String("a".into()),
                &Value::String("T".into()),
                &[
                    (key_x, Value::Int64(7)),
                    (key_s, Value::String("hello".into())),
                ],
            )
            .unwrap();

        let store = writer.finalize(&interner).unwrap();
        assert_eq!(store.row_count(), 2);
        // Row 0 — all null
        assert!(store.get_id(0).is_none());
        assert!(store.get_title(0).is_none());
        assert!(store.get(0, key_x).is_none());
        assert!(store.get(0, key_s).is_none());
        // Row 1 — populated
        assert_eq!(store.get_id(1).unwrap(), Value::String("a".into()));
        assert_eq!(store.get(1, key_x).unwrap(), Value::Int64(7));
        assert_eq!(store.get(1, key_s).unwrap(), Value::String("hello".into()));
    }

    /// Mixed column declared via meta="mixed" should accept any value
    /// type and round-trip via the heap-buffered path.
    #[test]
    fn mixed_column_heterogeneous() {
        let mut interner = StringInterner::new();
        let schema = make_schema(&mut interner, &["payload"]);
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("payload".to_string(), "mixed".to_string());
        let key_p = interner.get_or_intern("payload");

        let dir = TempDir::new().unwrap();
        let mut writer = TypeWriter::new(
            Arc::clone(&schema),
            meta,
            dir.path().to_path_buf(),
            &interner,
            "string",
            "string",
        )
        .unwrap();
        let values = [
            Value::Int64(42),
            Value::String("hello".into()),
            Value::Float64(3.14),
            Value::Null,
            Value::Boolean(true),
        ];
        for (i, v) in values.iter().enumerate() {
            writer
                .push_row(
                    &Value::String(format!("id{i}")),
                    &Value::Null,
                    &[(key_p, v.clone())],
                )
                .unwrap();
        }
        let store = writer.finalize(&interner).unwrap();
        for (i, v) in values.iter().enumerate() {
            match v {
                Value::Null => assert!(store.get(i as u32, key_p).is_none()),
                _ => assert_eq!(store.get(i as u32, key_p).unwrap(), v.clone()),
            }
        }
    }

    #[test]
    fn borrowed_rows_preserve_map_columns_and_overflow_properties() {
        let mut interner = StringInterner::new();
        let schema = make_schema(&mut interner, &["payload"]);
        let mut meta = HashMap::new();
        meta.insert("payload".to_string(), "mixed".to_string());
        let key_payload = interner.get_or_intern("payload");
        let key_extra = interner.get_or_intern("extra");
        let payload = std::collections::BTreeMap::from([
            ("kind".into(), Value::String("mixed".into())),
            ("values".into(), Value::List(vec![Value::Int64(1)])),
        ]);
        let extra = std::collections::BTreeMap::from([(
            "nested".into(),
            Value::Map(std::collections::BTreeMap::from([(
                "ok".into(),
                Value::Boolean(true),
            )])),
        )]);

        let dir = TempDir::new().unwrap();
        let mut writer = TypeWriter::new(
            Arc::clone(&schema),
            meta,
            dir.path().to_path_buf(),
            &interner,
            "string",
            "string",
        )
        .unwrap();
        writer
            .push_row_borrowed(
                BorrowedValue::String("id"),
                BorrowedValue::String("title"),
                |visitor| {
                    visitor.push_property(key_payload, BorrowedValue::Map(&payload))?;
                    visitor.push_property(key_extra, BorrowedValue::Map(&extra))
                },
            )
            .unwrap();

        let store = writer.finalize(&interner).unwrap();
        assert_eq!(store.get(0, key_payload), Some(Value::Map(payload.clone())));
        assert_eq!(store.get(0, key_extra), Some(Value::Map(extra.clone())));

        let mut seen = Vec::new();
        store
            .try_for_each_property_borrowed(0, |key, value| -> Result<(), ()> {
                seen.push((key, value.to_value()));
                Ok(())
            })
            .unwrap();
        assert!(seen.contains(&(key_payload, Value::Map(payload))));
        assert!(seen.contains(&(key_extra, Value::Map(extra))));
    }
}

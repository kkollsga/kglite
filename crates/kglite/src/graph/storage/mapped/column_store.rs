// src/graph/mmap_column_store.rs
//
// Disk-backed column store that reads directly from a shared mmap file.
// No heap allocation for column data — values are read from mmap at query time.
// The OS page cache manages which pages are resident.
//
// String offset convention: offsets[row] = cumulative end byte for that row.
// Row 0 starts at byte 0; row i>0 starts at offsets[i-1].
// (No leading-zero prefix — same as ntriples build_columns_direct.)
//
// Null bitmap convention: 0 = non-null, non-zero = null (same as ColumnStore).

use crate::datatypes::values::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::type_build_meta::ColType;
use chrono::NaiveDate;
use memmap2::MmapMut;
use std::collections::HashMap;
use std::sync::Arc;

const UNIX_EPOCH_DATE: NaiveDate = match NaiveDate::from_ymd_opt(1970, 1, 1) {
    Some(d) => d,
    None => unreachable!(),
};

// ─── Region ──────────────────────────────────────────────────────────────────

/// Byte region in the shared mmap file.
/// A region with `len == 0` means "not present".
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub offset: usize,
    pub len: usize, // in bytes
}

impl Region {
    /// A zero-length sentinel meaning "not present".
    pub const EMPTY: Region = Region { offset: 0, len: 0 };
}

// ─── Column metadata ─────────────────────────────────────────────────────────

/// Metadata for a fixed-width column in the mmap.
#[derive(Clone, Debug)]
pub struct FixedColumnMeta {
    pub col_type: ColType,
    pub data: Region,  // raw typed data
    pub nulls: Region, // null bitmap (1 byte per row)
}

/// Metadata for a string column in the mmap.
#[derive(Clone, Debug)]
pub struct StrColumnMeta {
    pub data: Region,    // string bytes
    pub offsets: Region, // u64 offset array (one per row, cumulative end)
    pub nulls: Region,   // null bitmap
}

/// Reference to a column in the MmapColumnStore.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ColRef {
    Fixed(usize), // index into fixed_cols
    Str(usize),   // index into str_cols
}

// ─── MmapColumnStore ─────────────────────────────────────────────────────────

/// Per-type column store backed by a shared mmap.
///
/// Clone is cheap: it clones the `Arc` and small metadata vecs, not the mmap data.
#[derive(Clone, Debug)]
pub struct MmapColumnStore {
    /// Shared reference to the mmap file containing all column data.
    pub(crate) mmap: Arc<MmapMut>,
    /// Number of rows in this type.
    pub(crate) row_count: u32,
    /// Whether the id column is stored as string (true) or UniqueId (false).
    pub(crate) id_is_string: bool,
    /// Id column if stored as UniqueId (fixed-width).
    pub(crate) id_fixed: Option<FixedColumnMeta>,
    /// Id column if stored as string.
    pub(crate) id_str: Option<StrColumnMeta>,
    /// Title column (always string).
    pub(crate) title: StrColumnMeta,
    /// Property key → column reference (fixed or string).
    pub(crate) col_map: HashMap<InternedKey, ColRef>,
    /// Dense fixed-width columns.
    pub(crate) fixed_cols: Vec<FixedColumnMeta>,
    /// Dense string columns.
    pub(crate) str_cols: Vec<StrColumnMeta>,
    /// Overflow bag offset array region: u64 array, row_count+1 entries.
    pub(crate) overflow_offsets: Region,
    /// Overflow bag serialized data region.
    pub(crate) overflow_data: Region,
    /// Whether this type has overflow data.
    pub(crate) has_overflow: bool,
}

// ─── Constructor ─────────────────────────────────────────────────────────────

impl MmapColumnStore {
    /// Number of rows in this type.
    #[inline]
    pub fn row_count(&self) -> u32 {
        self.row_count
    }
}

// ─── Low-level mmap readers ──────────────────────────────────────────────────

impl MmapColumnStore {
    /// Read the null bitmap at `row`. Returns `true` if the value IS null.
    #[inline]
    fn read_null(&self, region: &Region, row: usize) -> bool {
        if region.len == 0 {
            return true;
        }
        self.mmap[region.offset + row] != 0
    }

    #[inline]
    fn read_i64(&self, region: &Region, row: usize) -> i64 {
        let off = region.offset + row * 8;
        i64::from_ne_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn read_f64(&self, region: &Region, row: usize) -> f64 {
        let off = region.offset + row * 8;
        f64::from_ne_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn read_u64(&self, region: &Region, row: usize) -> u64 {
        let off = region.offset + row * 8;
        u64::from_ne_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    #[inline]
    fn read_u32(&self, region: &Region, row: usize) -> u32 {
        let off = region.offset + row * 4;
        u32::from_ne_bytes(self.mmap[off..off + 4].try_into().unwrap())
    }

    #[inline]
    fn read_i32(&self, region: &Region, row: usize) -> i32 {
        let off = region.offset + row * 4;
        i32::from_ne_bytes(self.mmap[off..off + 4].try_into().unwrap())
    }

    #[inline]
    fn read_u8(&self, region: &Region, row: usize) -> u8 {
        self.mmap[region.offset + row]
    }

    /// Read a string from the mmap for the given row.
    ///
    /// Offset convention: `offsets[row]` is the cumulative end byte.
    /// Row 0 starts at byte 0; row i>0 starts at `offsets[i-1]`.
    ///
    /// SAFETY: the `from_utf8_unchecked` below is sound because every
    /// string column reachable here was validated **once, at load
    /// time**: the disk-graph loader (`io/file.rs::load_disk_dir`)
    /// calls [`Self::validate_utf8`] on each store mapped from an
    /// existing `columns.bin` (whole-blob UTF-8 check + offset
    /// monotonicity/bounds + char-boundary check per row), and the
    /// only other constructor path is the same-process direct-write
    /// builder (`ntriples/column_builder.rs`), which writes the bytes
    /// from `String::as_bytes()` itself. Skipping the per-access
    /// validator matters on the Wikidata streaming workload: at 30
    /// strings × 17 M rows × ~50 bytes it was processing ~25 GB of
    /// data per save, far more than the actual work.
    #[inline]
    fn read_str(&self, data_region: &Region, offsets_region: &Region, row: usize) -> &str {
        let end = self.read_u64(offsets_region, row) as usize;
        let start = if row > 0 {
            self.read_u64(offsets_region, row - 1) as usize
        } else {
            0
        };
        let bytes = &self.mmap[data_region.offset + start..data_region.offset + end];
        // SAFETY: see method-level note — validated at load time (or
        // written in-process from `String::as_bytes()`).
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }
}

// ─── Load-time validation ────────────────────────────────────────────────────

impl MmapColumnStore {
    /// Validate every string column of this store — bytes come from
    /// disk and are untrusted until proven UTF-8. Runs **once at load
    /// time** (called by `load_disk_dir` for stores mapped from an
    /// existing `columns.bin`), never on the per-access read path; the
    /// hot readers keep their `from_utf8_unchecked` with this pass as
    /// the soundness citation.
    ///
    /// Checks, per string column (id / title / dense str columns):
    /// - the data/offsets/nulls regions lie within the mmap;
    /// - the whole data blob is valid UTF-8;
    /// - offsets are monotonically non-decreasing, within the blob,
    ///   and land on char boundaries (a corrupt offset that slices a
    ///   multi-byte code point would make a *slice* of a valid blob
    ///   invalid).
    pub fn validate_utf8(&self, type_name: &str) -> std::io::Result<()> {
        if let Some(sc) = &self.id_str {
            self.validate_str_column(sc, type_name, "__id__")?;
        }
        self.validate_str_column(&self.title, type_name, "__title__")?;
        for (i, sc) in self.str_cols.iter().enumerate() {
            self.validate_str_column(sc, type_name, &format!("str_col[{i}]"))?;
        }
        Ok(())
    }

    fn validate_str_column(
        &self,
        sc: &StrColumnMeta,
        type_name: &str,
        col: &str,
    ) -> std::io::Result<()> {
        let corrupt = |what: &str| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "columns.bin string column '{type_name}.{col}' is corrupt: {what} \
                     (refusing to load — rebuild the graph or restore the file from backup)"
                ),
            )
        };
        // Fully-absent column (e.g. a store with no titles).
        if sc.data.len == 0 && sc.offsets.len == 0 && sc.nulls.len == 0 {
            return Ok(());
        }
        for (region, what) in [
            (&sc.data, "data"),
            (&sc.offsets, "offsets"),
            (&sc.nulls, "nulls"),
        ] {
            let end = region
                .offset
                .checked_add(region.len)
                .ok_or_else(|| corrupt(&format!("{what} region overflows usize")))?;
            if end > self.mmap.len() {
                return Err(corrupt(&format!("{what} region extends past the file")));
            }
        }
        let rows = self.row_count as usize;
        let need_offsets = rows
            .checked_mul(8)
            .ok_or_else(|| corrupt("offset count overflows usize"))?;
        if sc.offsets.len < need_offsets {
            return Err(corrupt("offsets region too small for the row count"));
        }
        let blob = &self.mmap[sc.data.offset..sc.data.offset + sc.data.len];
        let s = std::str::from_utf8(blob).map_err(|e| corrupt(&format!("invalid UTF-8: {e}")))?;
        let mut prev = 0u64;
        for row in 0..rows {
            let end = self.read_u64(&sc.offsets, row);
            if end < prev || end as usize > blob.len() {
                return Err(corrupt(&format!(
                    "non-monotonic or out-of-range offset at row {row}"
                )));
            }
            if !s.is_char_boundary(end as usize) {
                return Err(corrupt(&format!(
                    "offset at row {row} splits a UTF-8 code point"
                )));
            }
            prev = end;
        }
        Ok(())
    }
}

// ─── Public accessors ────────────────────────────────────────────────────────

impl MmapColumnStore {
    /// Read the node ID for a given row.
    pub fn get_id(&self, row_id: u32) -> Option<Value> {
        if row_id >= self.row_count {
            return None;
        }
        let row = row_id as usize;
        if self.id_is_string {
            let sc = self.id_str.as_ref()?;
            if self.read_null(&sc.nulls, row) {
                return None;
            }
            Some(Value::String(
                self.read_str(&sc.data, &sc.offsets, row).to_string(),
            ))
        } else {
            let fc = self.id_fixed.as_ref()?;
            if self.read_null(&fc.nulls, row) {
                return None;
            }
            Some(Value::UniqueId(self.read_u32(&fc.data, row)))
        }
    }

    /// Read the node title for a given row.
    pub fn get_title(&self, row_id: u32) -> Option<Value> {
        if row_id >= self.row_count {
            return None;
        }
        let row = row_id as usize;
        if self.title.nulls.len == 0 && self.title.data.len == 0 {
            return None;
        }
        if self.read_null(&self.title.nulls, row) {
            return None;
        }
        Some(Value::String(
            self.read_str(&self.title.data, &self.title.offsets, row)
                .to_string(),
        ))
    }

    /// Zero-allocation equality check for a string property.
    ///
    /// Returns:
    /// - `None` — property is missing / null for this row (matcher should fail)
    /// - `Some(true)` — stored value matches `target` byte-for-byte
    /// - `Some(false)` — stored value is present but differs
    ///
    /// Fast path: for a dense string column we compare mmap bytes directly,
    /// never allocating a `String`. Falls back to the normal `get()` for
    /// overflow-bag or non-string columns (numeric values are cheap to
    /// allocate, so it's fine to reuse the general path).
    pub fn str_prop_eq(&self, row_id: u32, key: InternedKey, target: &str) -> Option<bool> {
        if row_id >= self.row_count {
            return None;
        }
        let row = row_id as usize;
        if let Some(col_ref) = self.col_map.get(&key) {
            match col_ref {
                ColRef::Str(idx) => {
                    let sc = &self.str_cols[*idx];
                    if self.read_null(&sc.nulls, row) {
                        return None;
                    }
                    return Some(self.read_str(&sc.data, &sc.offsets, row) == target);
                }
                ColRef::Fixed(_) => {
                    // Fixed-width column — definitely not a string match
                    return Some(false);
                }
            }
        }
        // Overflow bag or unknown key: fall back to the allocating path.
        self.get_overflow_property(row_id, key)
            .map(|v| matches!(v, Value::String(ref s) if s == target))
    }

    /// Read a property value by (row_id, interned key).
    /// Checks dense columns first, falls back to the overflow bag.
    pub fn get(&self, row_id: u32, key: InternedKey) -> Option<Value> {
        if row_id >= self.row_count {
            return None;
        }
        let row = row_id as usize;

        if let Some(col_ref) = self.col_map.get(&key) {
            match col_ref {
                ColRef::Fixed(idx) => {
                    let fc = &self.fixed_cols[*idx];
                    if !self.read_null(&fc.nulls, row) {
                        return Some(self.read_fixed_value(fc, row));
                    }
                }
                ColRef::Str(idx) => {
                    let sc = &self.str_cols[*idx];
                    if !self.read_null(&sc.nulls, row) {
                        let s = self.read_str(&sc.data, &sc.offsets, row);
                        return Some(Value::String(s.to_string()));
                    }
                }
            }
        }

        // Fall back to overflow bag
        self.get_overflow_property(row_id, key)
    }

    /// Read a fixed-width value from the mmap, dispatching on ColType.
    #[inline]
    fn read_fixed_value(&self, fc: &FixedColumnMeta, row: usize) -> Value {
        match fc.col_type {
            ColType::Int64 => Value::Int64(self.read_i64(&fc.data, row)),
            ColType::Float64 => Value::Float64(self.read_f64(&fc.data, row)),
            ColType::UniqueId => Value::UniqueId(self.read_u32(&fc.data, row)),
            ColType::Bool => Value::Boolean(self.read_u8(&fc.data, row) != 0),
            ColType::Date => {
                let days = self.read_i32(&fc.data, row);
                let date = UNIX_EPOCH_DATE + chrono::Duration::days(days as i64);
                Value::DateTime(date)
            }
            ColType::Str => unreachable!("string columns use ColRef::Str"),
        }
    }

    /// Borrowed view of the id column for a given row, without
    /// allocating an owned `Value::String`. The returned `&str`
    /// (when `Some(BorrowedValue::String)`) is tied to `self`'s mmap.
    ///
    /// Streaming-write callers use this to push id bytes directly into
    /// dest column files without the per-row `to_string()` clone that
    /// `get_id` does. On Wikidata that clone was ~5 µs/row × 17 M rows
    /// = ~85 s of wall time.
    pub fn id_borrowed(&self, row_id: u32) -> Option<crate::datatypes::values::BorrowedValue<'_>> {
        use crate::datatypes::values::BorrowedValue;
        if row_id >= self.row_count {
            return None;
        }
        let row = row_id as usize;
        if self.id_is_string {
            let sc = self.id_str.as_ref()?;
            if self.read_null(&sc.nulls, row) {
                return None;
            }
            Some(BorrowedValue::String(self.read_str(
                &sc.data,
                &sc.offsets,
                row,
            )))
        } else {
            let fc = self.id_fixed.as_ref()?;
            if self.read_null(&fc.nulls, row) {
                return None;
            }
            Some(BorrowedValue::UniqueId(self.read_u32(&fc.data, row)))
        }
    }

    /// Borrowed view of the title column for a given row. Title is
    /// always a string column in `MmapColumnStore`, so this returns
    /// `Option<&str>` rather than `BorrowedValue<'_>`.
    pub fn title_borrowed(&self, row_id: u32) -> Option<&str> {
        if row_id >= self.row_count {
            return None;
        }
        let row = row_id as usize;
        if self.title.nulls.len == 0 && self.title.data.len == 0 {
            return None;
        }
        if self.read_null(&self.title.nulls, row) {
            return None;
        }
        Some(self.read_str(&self.title.data, &self.title.offsets, row))
    }

    /// Allocation-free counterpart of [`row_properties`]: visits each
    /// non-null `(InternedKey, BorrowedValue<'_>)` pair without
    /// building a `Vec<(InternedKey, Value)>` and without the
    /// `String::to_string()` heap clone for every string property.
    /// Stops at the first `Err` returned by `f`.
    ///
    /// On Wikidata Articles+P50+Authors, the allocating `row_properties`
    /// path was 298 s out of 446 s of node walk (67%) — almost
    /// entirely heap pressure from ~510 M `Value::String` clones. The
    /// borrowed visitor pushes `&str` straight to the dest column
    /// writer and drops back to allocation only when the writer
    /// genuinely needs a `Value` (Mixed columns / NodeData).
    pub fn try_for_each_property_borrowed<F, E>(&self, row_id: u32, mut f: F) -> Result<(), E>
    where
        F: FnMut(InternedKey, crate::datatypes::values::BorrowedValue<'_>) -> Result<(), E>,
    {
        use crate::datatypes::values::BorrowedValue;
        if row_id >= self.row_count {
            return Ok(());
        }
        let row = row_id as usize;

        for (&key, col_ref) in &self.col_map {
            match col_ref {
                ColRef::Fixed(idx) => {
                    let fc = &self.fixed_cols[*idx];
                    if !self.read_null(&fc.nulls, row) {
                        let bv = match fc.col_type {
                            ColType::Int64 => BorrowedValue::Int64(self.read_i64(&fc.data, row)),
                            ColType::Float64 => {
                                BorrowedValue::Float64(self.read_f64(&fc.data, row))
                            }
                            ColType::UniqueId => {
                                BorrowedValue::UniqueId(self.read_u32(&fc.data, row))
                            }
                            ColType::Bool => {
                                BorrowedValue::Boolean(self.read_u8(&fc.data, row) != 0)
                            }
                            ColType::Date => {
                                let days = self.read_i32(&fc.data, row);
                                BorrowedValue::DateTime(
                                    UNIX_EPOCH_DATE + chrono::Duration::days(days as i64),
                                )
                            }
                            ColType::Str => unreachable!("string columns use ColRef::Str"),
                        };
                        f(key, bv)?;
                    }
                }
                ColRef::Str(idx) => {
                    let sc = &self.str_cols[*idx];
                    if !self.read_null(&sc.nulls, row) {
                        let s = self.read_str(&sc.data, &sc.offsets, row);
                        f(key, BorrowedValue::String(s))?;
                    }
                }
            }
        }
        // Overflow bag — decode entries IN PLACE via the shared codec
        // ([`crate::graph::storage::overflow`]) and yield `BorrowedValue`
        // slices into the mmap blob. The naïve path would call
        // `overflow_row_properties` which allocates a
        // `Vec<(InternedKey, Value)>` + a `String` per entry; on
        // Wikidata articles ~20+ properties per row land in overflow,
        // so that's ~250 M `String` allocations across the node walk.
        // The shared borrowed decoder skips them: strings borrow the
        // raw mmap bytes (UTF-8 *checked* — disk bytes are untrusted),
        // unknown future tags are skipped without dropping the rest of
        // the row, and Timestamp (tag 7) round-trips (both were bugs of
        // this file's previous hand-rolled copy of the decoder).
        if let Some(blob) = self.overflow_blob(row_id) {
            crate::graph::storage::overflow::try_for_each_borrowed(blob, f)?;
        }
        Ok(())
    }

    /// Iterate over all non-null properties for a row.
    /// Returns (InternedKey, Value) pairs from both dense columns and overflow bag.
    pub fn row_properties(&self, row_id: u32) -> Vec<(InternedKey, Value)> {
        if row_id >= self.row_count {
            return Vec::new();
        }
        let row = row_id as usize;
        let mut result = Vec::new();

        for (&key, col_ref) in &self.col_map {
            match col_ref {
                ColRef::Fixed(idx) => {
                    let fc = &self.fixed_cols[*idx];
                    if !self.read_null(&fc.nulls, row) {
                        result.push((key, self.read_fixed_value(fc, row)));
                    }
                }
                ColRef::Str(idx) => {
                    let sc = &self.str_cols[*idx];
                    if !self.read_null(&sc.nulls, row) {
                        let s = self.read_str(&sc.data, &sc.offsets, row);
                        result.push((key, Value::String(s.to_string())));
                    }
                }
            }
        }

        // Append overflow bag properties
        result.extend(self.overflow_row_properties(row_id));
        result
    }
}

// ─── Overflow bag ────────────────────────────────────────────────────────────
//
// Wire format + tag table live in `crate::graph::storage::overflow` —
// the single shared codec for both this mmap-backed store and the heap
// `ColumnStore`.

impl MmapColumnStore {
    /// Look up a single property in the overflow bag for a given row.
    pub fn get_overflow_property(&self, row_id: u32, key: InternedKey) -> Option<Value> {
        let blob = self.overflow_blob(row_id)?;
        crate::graph::storage::overflow::scan_blob(blob, key)
    }

    /// Decode all properties from the overflow bag for a given row.
    fn overflow_row_properties(&self, row_id: u32) -> Vec<(InternedKey, Value)> {
        match self.overflow_blob(row_id) {
            Some(blob) => crate::graph::storage::overflow::decode_blob(blob),
            None => Vec::new(),
        }
    }

    /// Extract the overflow blob slice for a given row, or None if not present.
    fn overflow_blob(&self, row_id: u32) -> Option<&[u8]> {
        if !self.has_overflow {
            return None;
        }
        let idx = row_id as usize;
        // overflow_offsets has row_count + 1 entries (u64 each)
        let expected_len = (self.row_count as usize + 1) * 8;
        if self.overflow_offsets.len < expected_len {
            return None;
        }
        let start = self.read_u64(&self.overflow_offsets, idx) as usize;
        let end = self.read_u64(&self.overflow_offsets, idx + 1) as usize;
        if start >= end || end > self.overflow_data.len {
            return None;
        }
        Some(&self.mmap[self.overflow_data.offset + start..self.overflow_data.offset + end])
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::MmapMut;

    /// Build a minimal store over an anonymous mmap with one string
    /// title column: `titles` are laid out back-to-back, followed by a
    /// u64 cumulative-end offset per row, followed by a null byte per
    /// row. Returns the store plus the byte range of the title data
    /// (so tests can corrupt it).
    fn store_with_titles(titles: &[&str]) -> (MmapColumnStore, usize) {
        let data_bytes: Vec<u8> = titles.iter().flat_map(|t| t.bytes()).collect();
        let mut offsets: Vec<u8> = Vec::new();
        let mut end = 0u64;
        for t in titles {
            end += t.len() as u64;
            offsets.extend_from_slice(&end.to_le_bytes());
        }
        let nulls = vec![0u8; titles.len()];

        let total = data_bytes.len() + offsets.len() + nulls.len();
        let mut mmap = MmapMut::map_anon(total.max(1)).unwrap();
        mmap[..data_bytes.len()].copy_from_slice(&data_bytes);
        mmap[data_bytes.len()..data_bytes.len() + offsets.len()].copy_from_slice(&offsets);
        mmap[data_bytes.len() + offsets.len()..total].copy_from_slice(&nulls);

        let store = MmapColumnStore {
            mmap: Arc::new(mmap),
            row_count: titles.len() as u32,
            id_is_string: false,
            id_fixed: None,
            id_str: None,
            title: StrColumnMeta {
                data: Region {
                    offset: 0,
                    len: data_bytes.len(),
                },
                offsets: Region {
                    offset: data_bytes.len(),
                    len: offsets.len(),
                },
                nulls: Region {
                    offset: data_bytes.len() + offsets.len(),
                    len: titles.len(),
                },
            },
            col_map: HashMap::new(),
            fixed_cols: Vec::new(),
            str_cols: Vec::new(),
            overflow_offsets: Region::EMPTY,
            overflow_data: Region::EMPTY,
            has_overflow: false,
        };
        (store, data_bytes.len())
    }

    #[test]
    fn validate_utf8_accepts_valid_store() {
        let (store, _) = store_with_titles(&["Zebra", "blåbær", "日本語"]);
        store.validate_utf8("T").unwrap();
        assert_eq!(store.title_borrowed(1), Some("blåbær"));
    }

    #[test]
    fn validate_utf8_rejects_invalid_bytes() {
        let (mut store, _) = store_with_titles(&["Zebra", "Fjord"]);
        {
            // Corrupt a title byte: 0xFF is never valid UTF-8.
            let m = Arc::get_mut(&mut store.mmap).unwrap();
            m[2] = 0xFF;
        }
        let err = store.validate_utf8("T").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("invalid UTF-8"), "{err}");
    }

    /// A corrupt OFFSET that slices a multi-byte code point must be
    /// rejected even though the whole blob is valid UTF-8 — the
    /// per-row slice would be invalid, breaking the
    /// `from_utf8_unchecked` readers' invariant.
    #[test]
    fn validate_utf8_rejects_offset_splitting_code_point() {
        let (mut store, data_len) = store_with_titles(&["blåbær", "xyz"]);
        {
            let m = Arc::get_mut(&mut store.mmap).unwrap();
            // Row 0's end offset is 8 ("blåbær" = 8 bytes: å/æ are 2 each).
            // Move it to 3, landing inside the 2-byte 'å'.
            m[data_len..data_len + 8].copy_from_slice(&3u64.to_le_bytes());
        }
        let err = store.validate_utf8("T").unwrap_err();
        assert!(
            err.to_string().contains("splits a UTF-8 code point"),
            "{err}"
        );
    }

    #[test]
    fn validate_utf8_rejects_out_of_range_offset() {
        let (mut store, data_len) = store_with_titles(&["abc", "def"]);
        {
            let m = Arc::get_mut(&mut store.mmap).unwrap();
            // Row 1's end offset claims bytes past the data region.
            m[data_len + 8..data_len + 16].copy_from_slice(&999u64.to_le_bytes());
        }
        let err = store.validate_utf8("T").unwrap_err();
        assert!(err.to_string().contains("out-of-range"), "{err}");
    }

    #[test]
    fn validate_utf8_rejects_region_past_file() {
        let (mut store, _) = store_with_titles(&["abc"]);
        store.title.data.len = 1 << 20; // region claims more than the mmap holds
        let err = store.validate_utf8("T").unwrap_err();
        assert!(err.to_string().contains("extends past the file"), "{err}");
    }
}

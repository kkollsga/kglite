//! Unified mega-file writer for `ColumnStore`s.
//!
//! Produces `seg_000/columns.bin` + `seg_000/columns_meta.json` matching
//! the layout the ntriples builder emits and the loader's mmap fast
//! path expects (see [`crate::graph::io::ntriples::ColumnTypeMeta`]).
//!
//! Used by [`crate::graph::dir_graph::DirGraph::save_disk`] when no
//! pre-existing `columns.bin` exists, so saved DirGraphs (carves,
//! `save_subset`, mutation persists from a fresh in-memory build) load
//! with mmap-fast-path semantics rather than per-type-sidecar
//! decompression.
//!
//! Layout strategy:
//! 1. Plan: walk every (type, column, sub-array) once to compute
//!    region offsets in the mega-file.
//! 2. Allocate `seg_000/columns.bin` with the total size.
//! 3. Write each sub-array's raw bytes (via [`MmapOrVec::as_raw_bytes`]
//!    / [`MmapBytes::as_raw_bytes`]) at its planned offset.
//! 4. Emit `seg_000/columns_meta.json` with the per-type
//!    [`ColumnTypeMeta`].
//!
//! Types whose `ColumnStore` contains a `TypedColumn::Mixed` cannot be
//! represented in the mmap layout and are returned in
//! `unhandled_types` so the caller falls back to the legacy zstd
//! sidecar for those.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

use memmap2::MmapMut;
use serde_json;

use crate::graph::io::ntriples::{
    ColMapEntry, ColumnTypeMeta, FixedColMeta, RegionMeta, StrColMeta,
};
use crate::graph::schema::StringInterner;
use crate::graph::storage::column_store::{ColumnStore, TypedColumn};
use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
use rustc_hash::FxHashMap;

/// Result of a unified-columns write.
#[allow(dead_code)] // fields are part of the public API; consumed by save_disk in the future
pub struct WriteResult {
    /// Types successfully encoded into `seg_000/columns.bin`. The
    /// caller should skip sidecar emission for these.
    pub written: HashSet<String>,
    /// Types containing `TypedColumn::Mixed` columns (or otherwise
    /// unrepresentable in the mmap layout). Caller falls back to the
    /// legacy zstd sidecar path for these.
    pub unhandled: HashSet<String>,
}

/// Write all column stores for the given dir, producing the mmap-
/// friendly `seg_000/columns.bin` + `seg_000/columns_meta.json`.
///
/// Returns the set of types that landed in the mega-file (caller skips
/// them during sidecar emission) plus the set that needs sidecar
/// fallback (typed-incompatible).
pub fn write_unified_columns(
    dir: &Path,
    column_stores: &HashMap<String, Arc<ColumnStore>>,
    _interner: &StringInterner,
) -> io::Result<WriteResult> {
    let seg0 = dir.join("seg_000");
    fs::create_dir_all(&seg0)?;
    let bin_path = seg0.join("columns.bin");
    let json_path = seg0.join("columns_meta.json");

    // ── Pass 1: plan the layout ─────────────────────────────────────
    //
    // For every type whose ColumnStore is fully typed (no Mixed), walk
    // every sub-array (id, title, per-property, overflow) and assign
    // it a contiguous region in the mega-file. Skip types that contain
    // any Mixed column — those need the sidecar fallback.

    struct PlannedType {
        type_name: String,
        meta: ColumnTypeMeta,
        // Source bytes per region, in the order they will be written.
        // Each entry is (planned_offset_in_megafile, &[u8]).
        sources: Vec<(usize, Vec<u8>)>,
    }

    let mut planned: Vec<PlannedType> = Vec::with_capacity(column_stores.len());
    let mut unhandled: HashSet<String> = HashSet::new();
    let mut cursor: usize = 0;

    // Stable iteration order for deterministic mega-file layout.
    let mut type_names: Vec<&String> = column_stores.keys().collect();
    type_names.sort();

    for type_name in type_names {
        let store = &column_stores[type_name];

        // Mixed-column check — abort planning for this type if any
        // schema-slot column is Mixed. Id/title columns are also
        // checked (they should be Str / UniqueId, but defensively).
        let has_mixed = store_has_mixed(store);
        if has_mixed {
            unhandled.insert(type_name.clone());
            continue;
        }

        let row_count = store.row_count();

        // ── id column ─────────────────────────────────────────────
        let (id_is_string, id_data_bytes, id_nulls_bytes, id_str_data_bytes, id_str_offsets_bytes) =
            extract_id_column(store);
        let mut sources: Vec<(usize, Vec<u8>)> = Vec::new();

        let mut id_data = RegionMeta { offset: 0, len: 0 };
        let mut id_nulls = RegionMeta { offset: 0, len: 0 };
        let mut id_str_data = RegionMeta { offset: 0, len: 0 };
        let mut id_str_offsets = RegionMeta { offset: 0, len: 0 };

        if id_is_string {
            (id_str_data, cursor) = plan_region(cursor, &id_str_data_bytes);
            sources.push((id_str_data.offset, id_str_data_bytes));
            (id_str_offsets, cursor) = plan_region(cursor, &id_str_offsets_bytes);
            sources.push((id_str_offsets.offset, id_str_offsets_bytes));
            (id_nulls, cursor) = plan_region(cursor, &id_nulls_bytes);
            sources.push((id_nulls.offset, id_nulls_bytes));
        } else if !id_data_bytes.is_empty() {
            (id_data, cursor) = plan_region(cursor, &id_data_bytes);
            sources.push((id_data.offset, id_data_bytes));
            (id_nulls, cursor) = plan_region(cursor, &id_nulls_bytes);
            sources.push((id_nulls.offset, id_nulls_bytes));
        }

        // ── title column ──────────────────────────────────────────
        let (title_data_bytes, title_offsets_bytes, title_nulls_bytes) =
            extract_title_column(store);

        let (title_data, c) = plan_region(cursor, &title_data_bytes);
        cursor = c;
        sources.push((title_data.offset, title_data_bytes));
        let (title_offsets, c) = plan_region(cursor, &title_offsets_bytes);
        cursor = c;
        sources.push((title_offsets.offset, title_offsets_bytes));
        let (title_nulls, c) = plan_region(cursor, &title_nulls_bytes);
        cursor = c;
        sources.push((title_nulls.offset, title_nulls_bytes));

        // ── per-schema-slot property columns ──────────────────────
        let mut col_map: Vec<ColMapEntry> = Vec::new();
        let mut fixed_cols: Vec<FixedColMeta> = Vec::new();
        let mut str_cols: Vec<StrColMeta> = Vec::new();

        for (slot, ik) in store.schema().iter() {
            let s = slot as usize;
            let col = match store.column(s) {
                Some(c) => c,
                None => continue,
            };
            match col {
                TypedColumn::Mixed { .. } => {
                    // Defensive — should have been caught by store_has_mixed.
                    unreachable!("Mixed column slipped past store_has_mixed");
                }
                TypedColumn::Int64 { data, nulls } => {
                    let (data_r, c) = plan_region(cursor, data.as_raw_bytes());
                    cursor = c;
                    sources.push((data_r.offset, data.as_raw_bytes().to_vec()));
                    let (nulls_r, c) = plan_region(cursor, nulls.as_raw_bytes());
                    cursor = c;
                    sources.push((nulls_r.offset, nulls.as_raw_bytes().to_vec()));
                    let idx = fixed_cols.len();
                    fixed_cols.push(FixedColMeta {
                        col_type_str: "int64".into(),
                        data: data_r,
                        nulls: nulls_r,
                    });
                    col_map.push(ColMapEntry {
                        key_u64: ik.as_u64(),
                        col_type_str: "int64".into(),
                        idx,
                    });
                }
                TypedColumn::Float64 { data, nulls } => {
                    let (data_r, c) = plan_region(cursor, data.as_raw_bytes());
                    cursor = c;
                    sources.push((data_r.offset, data.as_raw_bytes().to_vec()));
                    let (nulls_r, c) = plan_region(cursor, nulls.as_raw_bytes());
                    cursor = c;
                    sources.push((nulls_r.offset, nulls.as_raw_bytes().to_vec()));
                    let idx = fixed_cols.len();
                    fixed_cols.push(FixedColMeta {
                        col_type_str: "float64".into(),
                        data: data_r,
                        nulls: nulls_r,
                    });
                    col_map.push(ColMapEntry {
                        key_u64: ik.as_u64(),
                        col_type_str: "float64".into(),
                        idx,
                    });
                }
                TypedColumn::UniqueId { data, nulls } => {
                    let (data_r, c) = plan_region(cursor, data.as_raw_bytes());
                    cursor = c;
                    sources.push((data_r.offset, data.as_raw_bytes().to_vec()));
                    let (nulls_r, c) = plan_region(cursor, nulls.as_raw_bytes());
                    cursor = c;
                    sources.push((nulls_r.offset, nulls.as_raw_bytes().to_vec()));
                    let idx = fixed_cols.len();
                    fixed_cols.push(FixedColMeta {
                        col_type_str: "uniqueid".into(),
                        data: data_r,
                        nulls: nulls_r,
                    });
                    col_map.push(ColMapEntry {
                        key_u64: ik.as_u64(),
                        col_type_str: "uniqueid".into(),
                        idx,
                    });
                }
                TypedColumn::Bool { data, nulls } => {
                    let (data_r, c) = plan_region(cursor, data.as_raw_bytes());
                    cursor = c;
                    sources.push((data_r.offset, data.as_raw_bytes().to_vec()));
                    let (nulls_r, c) = plan_region(cursor, nulls.as_raw_bytes());
                    cursor = c;
                    sources.push((nulls_r.offset, nulls.as_raw_bytes().to_vec()));
                    let idx = fixed_cols.len();
                    fixed_cols.push(FixedColMeta {
                        col_type_str: "bool".into(),
                        data: data_r,
                        nulls: nulls_r,
                    });
                    col_map.push(ColMapEntry {
                        key_u64: ik.as_u64(),
                        col_type_str: "bool".into(),
                        idx,
                    });
                }
                TypedColumn::Date { data, nulls } => {
                    let (data_r, c) = plan_region(cursor, data.as_raw_bytes());
                    cursor = c;
                    sources.push((data_r.offset, data.as_raw_bytes().to_vec()));
                    let (nulls_r, c) = plan_region(cursor, nulls.as_raw_bytes());
                    cursor = c;
                    sources.push((nulls_r.offset, nulls.as_raw_bytes().to_vec()));
                    let idx = fixed_cols.len();
                    fixed_cols.push(FixedColMeta {
                        col_type_str: "date".into(),
                        data: data_r,
                        nulls: nulls_r,
                    });
                    col_map.push(ColMapEntry {
                        key_u64: ik.as_u64(),
                        col_type_str: "date".into(),
                        idx,
                    });
                }
                TypedColumn::Str {
                    offsets,
                    data,
                    nulls,
                    relocated,
                } => {
                    let (data_bytes, offsets_bytes, nulls_bytes) =
                        pack_str_column(offsets, data, nulls, relocated);

                    let (data_r, c) = plan_region(cursor, &data_bytes);
                    cursor = c;
                    sources.push((data_r.offset, data_bytes));
                    let (offsets_r, c) = plan_region(cursor, &offsets_bytes);
                    cursor = c;
                    sources.push((offsets_r.offset, offsets_bytes));
                    let (nulls_r, c) = plan_region(cursor, &nulls_bytes);
                    cursor = c;
                    sources.push((nulls_r.offset, nulls_bytes));
                    let idx = str_cols.len();
                    str_cols.push(StrColMeta {
                        data: data_r,
                        offsets: offsets_r,
                        nulls: nulls_r,
                    });
                    col_map.push(ColMapEntry {
                        key_u64: ik.as_u64(),
                        col_type_str: "string".into(),
                        idx,
                    });
                }
            }
        }

        // ── overflow bag ─────────────────────────────────────────
        let (overflow_offsets, overflow_data, has_overflow) =
            if let (Some(off_bytes), Some(data_bytes)) =
                (store.overflow_offsets_bytes(), store.overflow_data_bytes())
            {
                let (off_r, c) = plan_region(cursor, &off_bytes);
                cursor = c;
                sources.push((off_r.offset, off_bytes));
                let (data_r, c) = plan_region(cursor, &data_bytes);
                cursor = c;
                sources.push((data_r.offset, data_bytes));
                (off_r, data_r, true)
            } else {
                (
                    RegionMeta { offset: 0, len: 0 },
                    RegionMeta { offset: 0, len: 0 },
                    false,
                )
            };

        let meta = ColumnTypeMeta {
            type_name: type_name.clone(),
            row_count,
            id_is_string,
            id_data,
            id_nulls,
            id_str_data,
            id_str_offsets,
            title_data,
            title_offsets,
            title_nulls,
            col_map,
            fixed_cols,
            str_cols,
            overflow_offsets,
            overflow_data,
            has_overflow,
        };
        planned.push(PlannedType {
            type_name: type_name.clone(),
            meta,
            sources,
        });
    }

    // ── Pass 2: allocate + write ──────────────────────────────────
    let total_bytes = cursor;
    if total_bytes == 0 {
        // Nothing to write; clean up any stale mega-file artifacts.
        // Note: the previous gate also required `unhandled.is_empty()`,
        // but unhandled types only need sidecar fallback (not anything
        // here in the mega-file), so the right gate is "no bytes
        // planned". Skipping with non-empty unhandled used to fall
        // through to `mmap::map_mut` of a 0-byte file, which returns
        // EINVAL on every Unix and broke disk-graph save_disk in 0.9.15
        // (every test in test_disk_property_index was a failure mode).
        let _ = fs::remove_file(&bin_path);
        let _ = fs::remove_file(&json_path);
        return Ok(WriteResult {
            written: HashSet::new(),
            unhandled,
        });
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&bin_path)?;
    file.set_len(total_bytes as u64)?;
    // SAFETY: `bin_path` was just create-truncated and `file` is the
    // sole writer in this process for the duration of this function.
    // memmap2::MmapMut requires the file not be modified externally
    // while the map is alive; this writer holds the only handle and
    // the temp path is unique-per-run, so the invariant holds.
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };

    for pt in &planned {
        for (off, bytes) in &pt.sources {
            let dst = &mut mmap[*off..*off + bytes.len()];
            dst.copy_from_slice(bytes);
        }
    }
    mmap.flush()?;

    // ── Pass 3: emit metadata ─────────────────────────────────────
    let metas: Vec<ColumnTypeMeta> = planned.iter().map(|pt| pt.meta.clone()).collect();
    let json = serde_json::to_string_pretty(&metas).map_err(io::Error::other)?;
    let mut f = File::create(&json_path)?;
    f.write_all(json.as_bytes())?;
    f.sync_all()?;

    let written: HashSet<String> = planned.into_iter().map(|pt| pt.type_name).collect();
    Ok(WriteResult { written, unhandled })
}

#[inline]
fn plan_region(cursor: usize, bytes: &[u8]) -> (RegionMeta, usize) {
    let region = RegionMeta {
        offset: cursor,
        len: bytes.len(),
    };
    (region, cursor + bytes.len())
}

/// Pack a `Str` column into the mega-file's `(data, offsets, nulls)` byte
/// triple.
///
/// Two conventions are reconciled here.
///
/// *Offsets.* The mega-file stores `row_count` cumulative **end** offsets — row
/// 0 starts at byte 0, row `i` starts at `offsets[i - 1]` (see
/// [`crate::graph::storage::mapped::column_store`]). An in-memory
/// `TypedColumn::Str` instead carries `row_count + 1` offsets with a leading
/// zero, which `str_at` reads as `offsets[i]..offsets[i + 1]`, while the
/// streaming carve's `TypeWriter` already emits the mega-file form. Both are
/// accepted; the leading zero is stripped.
///
/// *The write overlay.* `TypedColumn::set` cannot shift `offsets` for a
/// replacement of a different length — that would move row `i + 1`'s start —
/// so it parks the new string in `relocated` and leaves `offsets`/`data`
/// holding the pre-`SET` bytes. The raw buffers are therefore **stale** for
/// every overlaid row. Reading them straight through, as this writer did,
/// silently dropped every differing-length string `SET` on a disk-mode graph:
/// the value read back after a reload as its pre-`SET` string, or as `""` when
/// the `SET` itself was what created the column.
/// [`TypedColumn::write_to`](crate::graph::storage::column_store::TypedColumn)
/// folds the overlay back for the packed sidecars; this is that fold in the
/// mega-file's layout.
fn pack_str_column(
    offsets: &MmapOrVec<u64>,
    data: &MmapBytes,
    nulls: &MmapOrVec<u8>,
    relocated: &FxHashMap<u32, String>,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let row_count = nulls.len();
    let nulls_bytes = nulls.as_raw_bytes().to_vec();
    // `row_count + 1` offsets means the leading-zero form.
    let leading_zero = offsets.len() == row_count + 1;

    if relocated.is_empty() {
        let off_bytes = offsets.as_raw_bytes();
        let off_slice = if leading_zero {
            &off_bytes[8..]
        } else {
            off_bytes
        };
        return (
            data.as_raw_bytes().to_vec(),
            off_slice.to_vec(),
            nulls_bytes,
        );
    }

    let offsets = offsets.as_slice();
    let source = data.as_raw_bytes();
    let mut new_data: Vec<u8> = Vec::with_capacity(source.len());
    let mut new_offsets: Vec<u8> = Vec::with_capacity(row_count * 8);
    for row in 0..row_count {
        // A null row contributes no bytes but still advances an offset, so a
        // reader's `offsets[i - 1] == offsets[i]` empty range lines up with the
        // null flag. Same rule as `TypedColumn::write_to`.
        if nulls.get(row) == 0 {
            match relocated.get(&(row as u32)) {
                Some(s) => new_data.extend_from_slice(s.as_bytes()),
                None => {
                    // Bounds-checked throughout, like `str_at`: a malformed
                    // offsets array yields an empty row rather than a panic.
                    let range = if leading_zero {
                        offsets.get(row).copied().zip(offsets.get(row + 1).copied())
                    } else if row == 0 {
                        offsets.first().copied().map(|end| (0, end))
                    } else {
                        offsets.get(row - 1).copied().zip(offsets.get(row).copied())
                    };
                    if let Some(bytes) =
                        range.and_then(|(start, end)| source.get(start as usize..end as usize))
                    {
                        new_data.extend_from_slice(bytes);
                    }
                }
            }
        }
        new_offsets.extend_from_slice(&(new_data.len() as u64).to_le_bytes());
    }
    (new_data, new_offsets, nulls_bytes)
}

fn store_has_mixed(store: &ColumnStore) -> bool {
    if store
        .columns_ref()
        .any(|c| matches!(c, TypedColumn::Mixed { .. }))
    {
        return true;
    }
    if let Some(c) = store.id_column_ref() {
        if matches!(c, TypedColumn::Mixed { .. }) {
            return true;
        }
    }
    if let Some(c) = store.title_column_ref() {
        if matches!(c, TypedColumn::Mixed { .. }) {
            return true;
        }
    }
    false
}

/// Extract the id column's raw bytes per the layout expected by the
/// loader. Returns `(id_is_string, fixed_data_bytes, nulls_bytes,
/// str_data_bytes, str_offsets_bytes)`. Empty slices are used for the
/// unused branch (fixed vs string).
fn extract_id_column(store: &ColumnStore) -> (bool, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    match store.id_column_ref() {
        Some(TypedColumn::Str {
            offsets,
            data,
            nulls,
            relocated,
        }) => {
            let (data_bytes, offsets_bytes, nulls_bytes) =
                pack_str_column(offsets, data, nulls, relocated);
            (true, Vec::new(), nulls_bytes, data_bytes, offsets_bytes)
        }
        Some(TypedColumn::UniqueId { data, nulls }) => (
            false,
            data.as_raw_bytes().to_vec(),
            nulls.as_raw_bytes().to_vec(),
            Vec::new(),
            Vec::new(),
        ),
        Some(TypedColumn::Int64 { data, nulls }) => (
            false,
            data.as_raw_bytes().to_vec(),
            nulls.as_raw_bytes().to_vec(),
            Vec::new(),
            Vec::new(),
        ),
        _ => (false, Vec::new(), Vec::new(), Vec::new(), Vec::new()),
    }
}

/// Extract the title column's raw bytes (always Str). Returns
/// `(data_bytes, offsets_bytes, nulls_bytes)`. Empty if no title.
fn extract_title_column(store: &ColumnStore) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    match store.title_column_ref() {
        Some(TypedColumn::Str {
            offsets,
            data,
            nulls,
            relocated,
        }) => pack_str_column(offsets, data, nulls, relocated),
        _ => (Vec::new(), Vec::new(), Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::values::Value;

    /// Decode a packed `Str` column the way the mega-file's reader does:
    /// `offsets[row]` is the cumulative *end*, row 0 starts at 0, and row `i`
    /// starts at `offsets[i - 1]`.
    fn decode(data: &[u8], offsets: &[u8], nulls: &[u8]) -> Vec<Option<String>> {
        let ends: Vec<u64> = offsets
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(ends.len(), nulls.len(), "one end offset per row");
        let mut out = Vec::with_capacity(nulls.len());
        for (row, &is_null) in nulls.iter().enumerate() {
            if is_null != 0 {
                out.push(None);
                continue;
            }
            let start = if row == 0 { 0 } else { ends[row - 1] } as usize;
            let end = ends[row] as usize;
            out.push(Some(String::from_utf8(data[start..end].to_vec()).unwrap()));
        }
        out
    }

    /// Build the in-memory `Str` shape: `row_count + 1` offsets, leading zero.
    fn build(values: &[Option<&str>]) -> TypedColumn {
        let mut col = TypedColumn::from_type_str("string");
        for v in values {
            match v {
                Some(s) => col.push(&Value::String((*s).to_string())).unwrap(),
                None => col.push_null(),
            }
        }
        col
    }

    fn pack(col: &TypedColumn) -> Vec<Option<String>> {
        let TypedColumn::Str {
            offsets,
            data,
            nulls,
            relocated,
        } = col
        else {
            panic!("expected a Str column");
        };
        let (data_bytes, offsets_bytes, nulls_bytes) =
            pack_str_column(offsets, data, nulls, relocated);
        decode(&data_bytes, &offsets_bytes, &nulls_bytes)
    }

    #[test]
    fn packs_a_column_with_no_overlay() {
        let col = build(&[Some("a"), None, Some("ccc")]);
        assert_eq!(
            pack(&col),
            vec![Some("a".into()), None, Some("ccc".into())],
            "the leading zero must be stripped, not emitted as row 0's end"
        );
    }

    #[test]
    fn folds_a_differing_length_overwrite_back_into_the_layout() {
        // The regression this file's `pack_str_column` exists for: `set` parks
        // a differing-length value in `relocated` and leaves `offsets`/`data`
        // holding the pre-`SET` bytes, so a writer reading the raw buffers
        // emitted the stale string and lost the write.
        let mut col = build(&[Some("aa"), Some("bb"), Some("cc")]);
        col.set(1, &Value::String("a-much-longer-value".into()))
            .unwrap();
        assert_eq!(
            pack(&col),
            vec![
                Some("aa".into()),
                Some("a-much-longer-value".into()),
                Some("cc".into()),
            ],
            "the overlaid row must carry the new value and its neighbours the old ones"
        );
    }

    #[test]
    fn folds_a_shorter_overwrite_and_renumbers_the_tail() {
        let mut col = build(&[Some("aaaa"), Some("bbbb"), Some("cccc")]);
        col.set(0, &Value::String("z".into())).unwrap();
        assert_eq!(
            pack(&col),
            vec![Some("z".into()), Some("bbbb".into()), Some("cccc".into())],
            "shrinking row 0 must shift every later row's offsets down"
        );
    }

    #[test]
    fn folds_an_overlay_onto_a_column_whose_rows_are_all_null() {
        // The shape a brand-new key's column has: `ColumnStore::set` appends a
        // column of nulls and then writes one row. A dropped overlay left the
        // whole column empty, which read back as "" rather than null.
        let mut col = build(&[None, None, None]);
        col.set(2, &Value::String("xyz".into())).unwrap();
        assert_eq!(pack(&col), vec![None, None, Some("xyz".into())]);
    }

    #[test]
    fn a_null_row_advances_an_offset_without_contributing_bytes() {
        let mut col = build(&[Some("aa"), None, Some("cc")]);
        col.set(0, &Value::String("longer".into())).unwrap();
        assert_eq!(
            pack(&col),
            vec![Some("longer".into()), None, Some("cc".into())]
        );
    }

    #[test]
    fn folds_an_overlay_onto_the_cumulative_ends_offset_form() {
        // The streaming carve's `TypeWriter` emits `row_count` cumulative ends
        // with no leading zero. Both forms reach this writer, so the fold has
        // to read either one.
        let mut col = TypedColumn::Str {
            offsets: MmapOrVec::from_vec(vec![2u64, 4, 6]),
            data: {
                let mut d = MmapBytes::new();
                d.extend(b"aabbcc").unwrap();
                d
            },
            nulls: MmapOrVec::from_vec(vec![0u8, 0, 0]),
            relocated: FxHashMap::default(),
        };
        if let TypedColumn::Str { relocated, .. } = &mut col {
            relocated.insert(1, "BB-longer".to_string());
        }
        assert_eq!(
            pack(&col),
            vec![
                Some("aa".into()),
                Some("BB-longer".into()),
                Some("cc".into()),
            ]
        );
    }
}

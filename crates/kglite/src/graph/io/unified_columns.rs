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
            let col = match store.columns_ref().get(s) {
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
                    ..
                } => {
                    // The mega-file layout convention: offsets has
                    // `row_count` u64 entries (cumulative ends).
                    // TypedColumn::Str built via in-memory push has a
                    // leading 0 → `row_count + 1` entries; the
                    // streaming carve's TypeWriter writes only
                    // cumulative ends so it already matches. Detect
                    // and strip the leading zero on the in-memory
                    // build path.
                    let off_bytes = offsets.as_raw_bytes();
                    let off_slice = if offsets.len() == row_count as usize + 1 {
                        &off_bytes[8..]
                    } else {
                        off_bytes
                    };

                    let (data_r, c) = plan_region(cursor, data.as_raw_bytes());
                    cursor = c;
                    sources.push((data_r.offset, data.as_raw_bytes().to_vec()));
                    let (offsets_r, c) = plan_region(cursor, off_slice);
                    cursor = c;
                    sources.push((offsets_r.offset, off_slice.to_vec()));
                    let (nulls_r, c) = plan_region(cursor, nulls.as_raw_bytes());
                    cursor = c;
                    sources.push((nulls_r.offset, nulls.as_raw_bytes().to_vec()));
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

fn store_has_mixed(store: &ColumnStore) -> bool {
    if store
        .columns_ref()
        .iter()
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
            ..
        }) => {
            let row_count = nulls.len();
            let off_bytes = offsets.as_raw_bytes();
            let off_slice = if offsets.len() == row_count + 1 {
                &off_bytes[8..]
            } else {
                off_bytes
            };
            (
                true,
                Vec::new(),
                nulls.as_raw_bytes().to_vec(),
                data.as_raw_bytes().to_vec(),
                off_slice.to_vec(),
            )
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
            ..
        }) => {
            let row_count = nulls.len();
            let off_bytes = offsets.as_raw_bytes();
            let off_slice = if offsets.len() == row_count + 1 {
                &off_bytes[8..]
            } else {
                off_bytes
            };
            (
                data.as_raw_bytes().to_vec(),
                off_slice.to_vec(),
                nulls.as_raw_bytes().to_vec(),
            )
        }
        _ => (Vec::new(), Vec::new(), Vec::new()),
    }
}

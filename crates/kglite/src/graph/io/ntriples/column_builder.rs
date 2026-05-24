//! Direct-write columnar build pipeline for the N-Triples loader.
//!
//! `build_columns_direct` replays the property log (collected during
//! Phase 1) into a single mmap'd `columns.bin` file with pre-allocated
//! exact-sized regions. Bookkeeping types (`RegionMeta`,
//! `FixedColMeta`, `StrColMeta`, `ColMapEntry`, `ColumnTypeMeta`)
//! describe the resulting layout for post-Phase-3 mmap reload.
//!
//! Split out of `loader.rs` to keep the loader entry point under the
//! 2,500-line cap.

use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, InternedKey, PropertyStorage, TypeSchema};
use crate::graph::storage::type_build_meta::{ColType, TypeBuildMeta};
use crate::graph::storage::{GraphRead, GraphWrite};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use super::loader::format_count;
use super::{ProgressEvent, ProgressSink};

macro_rules! eplog {
    ($($arg:tt)*) => {
        eprintln!("[{}] {}", chrono::Local::now().format("%H:%M:%S"), format_args!($($arg)*))
    };
}

// ─── Column metadata for mmap reload ────────────────────────────────────────

/// Serializable metadata for a single Region in the mmap file.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy)]
pub struct RegionMeta {
    pub offset: usize,
    pub len: usize,
}

/// Serializable metadata for a fixed-width column.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct FixedColMeta {
    pub col_type_str: String,
    pub data: RegionMeta,
    pub nulls: RegionMeta,
}

/// Serializable metadata for a string column.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct StrColMeta {
    pub data: RegionMeta,
    pub offsets: RegionMeta,
    pub nulls: RegionMeta,
}

/// Serializable metadata for a column mapping entry.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct ColMapEntry {
    pub key_u64: u64,
    pub col_type_str: String,
    pub idx: usize,
}

/// Parse a ColType from its tag string.
fn col_type_from_str(s: &str) -> ColType {
    match s {
        "int64" => ColType::Int64,
        "float64" => ColType::Float64,
        "uniqueid" => ColType::UniqueId,
        "bool" => ColType::Bool,
        "date" => ColType::Date,
        "string" => ColType::Str,
        _ => ColType::Str,
    }
}

/// Per-type metadata saved to columns_meta.json for post-Phase-3 mmap reload.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct ColumnTypeMeta {
    pub type_name: String,
    pub row_count: u32,
    pub id_is_string: bool,
    pub id_data: RegionMeta,
    pub id_nulls: RegionMeta,
    pub id_str_data: RegionMeta,
    pub id_str_offsets: RegionMeta,
    pub title_data: RegionMeta,
    pub title_offsets: RegionMeta,
    pub title_nulls: RegionMeta,
    pub col_map: Vec<ColMapEntry>,
    pub fixed_cols: Vec<FixedColMeta>,
    pub str_cols: Vec<StrColMeta>,
    pub overflow_offsets: RegionMeta,
    pub overflow_data: RegionMeta,
    pub has_overflow: bool,
}

impl RegionMeta {
    fn from_region(r: &crate::graph::storage::mapped::column_store::Region) -> Self {
        RegionMeta {
            offset: r.offset,
            len: r.len,
        }
    }

    fn to_region(self) -> crate::graph::storage::mapped::column_store::Region {
        crate::graph::storage::mapped::column_store::Region {
            offset: self.offset,
            len: self.len,
        }
    }
}

impl ColumnTypeMeta {
    /// Rebuild an MmapColumnStore from saved metadata + a shared mmap.
    pub fn to_mmap_store(
        &self,
        mmap: Arc<memmap2::MmapMut>,
    ) -> crate::graph::storage::mapped::column_store::MmapColumnStore {
        use crate::graph::storage::mapped::column_store::{
            ColRef, FixedColumnMeta, MmapColumnStore, StrColumnMeta,
        };

        MmapColumnStore {
            mmap,
            row_count: self.row_count,
            id_is_string: self.id_is_string,
            id_fixed: if !self.id_is_string {
                Some(FixedColumnMeta {
                    col_type: ColType::UniqueId,
                    data: self.id_data.to_region(),
                    nulls: self.id_nulls.to_region(),
                })
            } else {
                None
            },
            id_str: if self.id_is_string {
                Some(StrColumnMeta {
                    data: self.id_str_data.to_region(),
                    offsets: self.id_str_offsets.to_region(),
                    nulls: self.id_nulls.to_region(),
                })
            } else {
                None
            },
            title: StrColumnMeta {
                data: self.title_data.to_region(),
                offsets: self.title_offsets.to_region(),
                nulls: self.title_nulls.to_region(),
            },
            col_map: self
                .col_map
                .iter()
                .map(|e| {
                    let key = InternedKey::from_u64(e.key_u64);
                    let ct = col_type_from_str(&e.col_type_str);
                    let cr = if matches!(ct, ColType::Str) {
                        ColRef::Str(e.idx)
                    } else {
                        ColRef::Fixed(e.idx)
                    };
                    (key, cr)
                })
                .collect(),
            fixed_cols: self
                .fixed_cols
                .iter()
                .map(|fc| FixedColumnMeta {
                    col_type: col_type_from_str(&fc.col_type_str),
                    data: fc.data.to_region(),
                    nulls: fc.nulls.to_region(),
                })
                .collect(),
            str_cols: self
                .str_cols
                .iter()
                .map(|sc| StrColumnMeta {
                    data: sc.data.to_region(),
                    offsets: sc.offsets.to_region(),
                    nulls: sc.nulls.to_region(),
                })
                .collect(),
            overflow_offsets: self.overflow_offsets.to_region(),
            overflow_data: self.overflow_data.to_region(),
            has_overflow: self.has_overflow,
        }
    }
}

/// Build ColumnStores by replaying the property log with pre-allocated direct-write.
///
/// Uses TypeBuildMeta (collected during Phase 1) to pre-allocate exact-sized arrays,
/// then writes values directly to their final positions in a single pass.
/// No BlockPool, no eviction, no intermediate compression.
/// Entities between Phase 1b progress callbacks. Tuned so the bar
/// moves smoothly on Wikidata-scale (~120M entities) without firing
/// a callback on every line.
const PHASE1B_TICK: u64 = 250_000;

pub(super) fn build_columns_direct(
    graph: &mut DirGraph,
    log_path: &std::path::Path,
    type_meta: &HashMap<String, TypeBuildMeta>,
    type_rename_map: &HashMap<String, String>,
    verbose: bool,
    progress: Option<&dyn ProgressSink>,
) -> std::io::Result<()> {
    use crate::graph::storage::column_store::ColumnStore;
    use crate::graph::storage::mapped::column_store::{
        ColRef, FixedColumnMeta, MmapColumnStore, Region, StrColumnMeta,
    };
    use crate::graph::storage::memory::property_log::PropertyLogReader;
    use memmap2::MmapMut;

    let alloc_start = Instant::now();

    // Get data_dir for placing the final columns.bin. Disk graphs have a
    // persistent user-provided data_dir; mapped graphs reuse the same
    // per-process spill-dir scheme as the property log and edge buffer.
    // The resulting mmap'd `columns.bin` stays alive until graph drop.
    let data_dir = if let crate::graph::schema::GraphBackend::Disk(ref dg) = graph.graph {
        dg.data_dir.clone()
    } else {
        graph.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("kglite_build_{}", std::process::id()))
        })
    };
    let _ = std::fs::create_dir_all(&data_dir);

    // ── Step 1: Layout computation — single mmap file ────────────────────
    //
    // ONE file (`columns.bin`) in data_dir holds ALL column data for ALL types.
    //
    // Each region is 8-byte aligned so typed reads are safe.

    struct TypeWriter {
        row_cursor: u32,
        id_is_string: bool,
        id_data: Region,
        id_nulls: Region,
        id_str_data: Region,
        id_str_offsets: Region,
        title_data: Region,
        title_offsets: Region,
        title_nulls: Region,
        title_cursor: u64,
        col_map: HashMap<InternedKey, (ColType, usize)>,
        fixed_cols: Vec<FixedColLayout>,
        str_cols: Vec<StrColLayout>,
        /// Keys with fill rate < threshold go to the overflow bag instead of dense columns.
        overflow_keys: HashSet<InternedKey>,
        /// Grows during write pass — serialized overflow entries for all rows.
        overflow_bag_data: Vec<u8>,
        /// One offset per row (+ final sentinel), recording the start of each row's blob.
        overflow_offsets: Vec<u64>,
        /// Region in the mmap for overflow offsets (set after appending to file).
        overflow_offsets_region: Region,
        /// Region in the mmap for overflow bag data (set after appending to file).
        overflow_data_region: Region,
    }

    struct FixedColLayout {
        col_type: ColType,
        data: Region,
        nulls: Region,
    }

    struct StrColLayout {
        data: Region,
        offsets: Region,
        nulls: Region,
        cursor: u64,
    }

    // Helpers to write typed values into the mmap
    #[inline]
    fn write_i64(mmap: &mut MmapMut, region: &Region, row: usize, val: i64) {
        let off = region.offset + row * 8;
        mmap[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn write_f64(mmap: &mut MmapMut, region: &Region, row: usize, val: f64) {
        let off = region.offset + row * 8;
        mmap[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn write_u32(mmap: &mut MmapMut, region: &Region, row: usize, val: u32) {
        let off = region.offset + row * 4;
        mmap[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn write_i32(mmap: &mut MmapMut, region: &Region, row: usize, val: i32) {
        let off = region.offset + row * 4;
        mmap[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn write_u8(mmap: &mut MmapMut, region: &Region, row: usize, val: u8) {
        mmap[region.offset + row] = val;
    }
    #[inline]
    fn write_u64(mmap: &mut MmapMut, region: &Region, row: usize, val: u64) {
        let off = region.offset + row * 8;
        mmap[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    }
    #[inline]
    fn read_u64(mmap: &MmapMut, region: &Region, row: usize) -> u64 {
        let off = region.offset + row * 8;
        u64::from_ne_bytes(mmap[off..off + 8].try_into().unwrap())
    }

    /// Align a byte offset up to 8-byte boundary.
    #[inline]
    fn align8(v: usize) -> usize {
        (v + 7) & !7
    }

    let col_dir = data_dir.clone();

    let mut writers: HashMap<String, TypeWriter> = HashMap::new();

    // Accumulate regions that need null-fill (0xFF) so we can batch-fill after mmap creation.
    let mut null_regions: Vec<Region> = Vec::new();
    let mut cursor: usize = 0;

    // Keys with fill rate < threshold are routed to the overflow bag instead
    // of a dense column. Hoisted here so the Phase 1b summary below can cite
    // the same threshold the per-type loop applies.
    const FILL_RATE_THRESHOLD: f64 = 0.05;

    // Aggregate per-type column-layout stats instead of printing one line per
    // type. On Wikidata this replaced 88 k spam lines with a single summary.
    let mut types_with_overflow: u32 = 0;
    let mut total_dense_cols: u64 = 0;
    let mut total_overflow_cols: u64 = 0;

    for (type_name, meta) in type_meta {
        let rc = meta.row_count as usize;
        let id_is_string = meta.id_is_string;

        // Id column regions
        let id_data = if !id_is_string {
            cursor = align8(cursor);
            let r = Region {
                offset: cursor,
                len: rc * 4,
            };
            cursor += r.len;
            r
        } else {
            Region { offset: 0, len: 0 }
        };

        cursor = align8(cursor);
        let id_nulls = Region {
            offset: cursor,
            len: rc,
        };
        null_regions.push(id_nulls);
        cursor += id_nulls.len;

        let id_str_data = if id_is_string {
            cursor = align8(cursor);
            let r = Region {
                offset: cursor,
                len: meta.id_string_bytes as usize,
            };
            cursor += r.len;
            r
        } else {
            Region { offset: 0, len: 0 }
        };

        let id_str_offsets = if id_is_string {
            cursor = align8(cursor);
            let r = Region {
                offset: cursor,
                len: rc * 8,
            };
            cursor += r.len;
            r
        } else {
            Region { offset: 0, len: 0 }
        };

        // Title column regions
        cursor = align8(cursor);
        let title_data = Region {
            offset: cursor,
            len: meta.title_string_bytes as usize,
        };
        cursor += title_data.len;

        cursor = align8(cursor);
        let title_offsets = Region {
            offset: cursor,
            len: rc * 8,
        };
        cursor += title_offsets.len;

        cursor = align8(cursor);
        let title_nulls = Region {
            offset: cursor,
            len: rc,
        };
        null_regions.push(title_nulls);
        cursor += title_nulls.len;

        // Property columns — split into dense (>= 5% fill) and overflow (< 5% fill).
        // Threshold hoisted above the per-type loop so the Phase 1b summary can reuse it.
        let mut col_map: HashMap<InternedKey, (ColType, usize)> = HashMap::new();
        let mut fixed_cols: Vec<FixedColLayout> = Vec::new();
        let mut str_cols: Vec<StrColLayout> = Vec::new();
        let mut overflow_keys: HashSet<InternedKey> = HashSet::new();
        let mut dense_count = 0u32;
        let mut overflow_count = 0u32;

        for (key, col_meta) in &meta.columns {
            let fill_rate = meta.fill_rate(col_meta);
            if fill_rate < FILL_RATE_THRESHOLD {
                overflow_keys.insert(*key);
                overflow_count += 1;
                continue;
            }

            dense_count += 1;
            if let Some(val_size) = col_meta.col_type.value_size() {
                let idx = fixed_cols.len();
                col_map.insert(*key, (col_meta.col_type, idx));

                cursor = align8(cursor);
                let data = Region {
                    offset: cursor,
                    len: rc * val_size,
                };
                cursor += data.len;

                cursor = align8(cursor);
                let nulls = Region {
                    offset: cursor,
                    len: rc,
                };
                null_regions.push(nulls);
                cursor += nulls.len;

                fixed_cols.push(FixedColLayout {
                    col_type: col_meta.col_type,
                    data,
                    nulls,
                });
            } else {
                // String column
                let idx = str_cols.len();
                col_map.insert(*key, (col_meta.col_type, idx));

                cursor = align8(cursor);
                let data = Region {
                    offset: cursor,
                    len: col_meta.string_bytes as usize,
                };
                cursor += data.len;

                cursor = align8(cursor);
                let offsets = Region {
                    offset: cursor,
                    len: rc * 8,
                };
                cursor += offsets.len;

                cursor = align8(cursor);
                let nulls = Region {
                    offset: cursor,
                    len: rc,
                };
                null_regions.push(nulls);
                cursor += nulls.len;

                str_cols.push(StrColLayout {
                    data,
                    offsets,
                    nulls,
                    cursor: 0,
                });
            }
        }

        total_dense_cols += dense_count as u64;
        total_overflow_cols += overflow_count as u64;
        if overflow_count > 0 {
            types_with_overflow += 1;
        }

        writers.insert(
            type_name.clone(),
            TypeWriter {
                row_cursor: 0,
                id_is_string,
                id_data,
                id_nulls,
                id_str_data,
                id_str_offsets,
                title_data,
                title_offsets,
                title_nulls,
                title_cursor: 0,
                col_map,
                fixed_cols,
                str_cols,
                overflow_keys,
                overflow_bag_data: Vec::new(),
                overflow_offsets: Vec::with_capacity(rc + 1),
                overflow_offsets_region: Region::EMPTY,
                overflow_data_region: Region::EMPTY,
            },
        );
    }

    let total_bytes = align8(cursor).max(8); // at least 8 bytes for valid mmap

    // ── Step 1b: Create single file and mmap ──────────────────────────────

    if verbose {
        eplog!(
            "  Phase 1b: layout computed — {:.1} GB for {} types ({:.1}s)",
            total_bytes as f64 / (1u64 << 30) as f64,
            writers.len(),
            alloc_start.elapsed().as_secs_f64(),
        );
        eplog!(
            "  Phase 1b: columns — {} dense, {} overflow across {} types with sparse (< {:.0}%) cols",
            total_dense_cols,
            total_overflow_cols,
            types_with_overflow,
            FILL_RATE_THRESHOLD * 100.0,
        );
        // Show top 10 types by space consumption
        let mut type_sizes: Vec<(&str, u64, u32)> = Vec::new();
        for (tn, meta) in type_meta {
            let mut sz = meta.title_string_bytes + (meta.row_count as u64) * 9; // title offsets+nulls + id
            if meta.id_is_string {
                sz += meta.id_string_bytes + (meta.row_count as u64) * 8;
            } else {
                sz += (meta.row_count as u64) * 4;
            }
            for cm in meta.columns.values() {
                sz += if let Some(vs) = cm.col_type.value_size() {
                    (meta.row_count as u64) * (vs as u64 + 1)
                } else {
                    cm.string_bytes + (meta.row_count as u64) * 9
                };
            }
            type_sizes.push((tn.as_str(), sz, meta.row_count));
        }
        type_sizes.sort_by_key(|t| std::cmp::Reverse(t.1));
        for (tn, sz, rc) in type_sizes.iter().take(10) {
            eplog!(
                "    {:>8.1} GB  {:>10} rows  {}",
                *sz as f64 / (1u64 << 30) as f64,
                format_count(*rc as u64),
                if tn.len() > 50 { &tn[..50] } else { tn },
            );
        }
    }

    if total_bytes == 0 && verbose {
        eplog!(
            "  Phase 1b: no types to pre-allocate ({:.1}s)",
            alloc_start.elapsed().as_secs_f64(),
        );
    }

    // Create the single mmap file (only if there are types to store)
    let mmap_path = col_dir.join("columns.bin");
    let mmap_opt: Option<MmapMut> = if total_bytes > 0 {
        if verbose {
            eplog!(
                "  Phase 1b: creating {:.1} GB mmap file...",
                total_bytes as f64 / (1u64 << 30) as f64,
            );
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&mmap_path)?;
        file.set_len(total_bytes as u64)?;
        // SAFETY: file was just created+truncated to total_bytes; mmap
        // region matches the on-disk length.
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        if verbose {
            eplog!("  Phase 1b: filling null bitmaps...");
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
        // Fill all null bitmap regions with 0xFF (all-null)
        for region in &null_regions {
            mmap[region.offset..region.offset + region.len].fill(0xFF);
        }
        Some(mmap)
    } else {
        None
    };
    drop(null_regions);

    if verbose {
        eplog!(
            "  Phase 1b: mmap ready — {:.1} GB for {} types ({:.1}s)",
            total_bytes as f64 / (1u64 << 30) as f64,
            writers.len(),
            alloc_start.elapsed().as_secs_f64(),
        );
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }

    // Unwrap mmap for the write/read phases (guaranteed to exist if writers is non-empty)
    let mut mmap = mmap_opt.unwrap_or_else(|| {
        // Create a dummy 1-byte mmap if no types
        let p = col_dir.join("columns.bin");
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&p)
            .unwrap();
        f.set_len(1).unwrap();
        // SAFETY: placeholder file just created+truncated to 1 byte; the
        // caller replaces this with a real-sized mmap once sizes are known.
        unsafe { MmapMut::map_mut(&f).unwrap() }
    });

    // ── Step 2: Buffered type-batched write pass ──────────────────────────
    //
    // Instead of writing each entity to the mmap immediately (random I/O across
    // 35 GB when entities are interleaved by type), we buffer entries grouped by
    // type and flush the biggest type when the buffer exceeds a threshold.
    // This makes writes sequential within each type's mmap region.

    struct BufferedEntry {
        node_idx: petgraph::graph::NodeIndex,
        id: Value,
        title: Value,
        properties: Vec<(InternedKey, Value)>,
    }

    /// Estimate the in-memory size of a log entry for buffer accounting.
    fn estimate_entry_bytes(
        id: &Value,
        title: &Value,
        properties: &[(InternedKey, Value)],
    ) -> usize {
        let mut sz = 100; // base overhead (Vec headers, NodeIndex, etc.)
        if let Value::String(s) = id {
            sz += s.len();
        }
        if let Value::String(s) = title {
            sz += s.len();
        }
        for (_, v) in properties {
            sz += match v {
                Value::String(s) => 16 + s.len(),
                _ => 16,
            };
        }
        sz
    }

    /// Flush all buffered entries for a single type to the mmap.
    /// Returns the approximate byte count that was flushed.
    #[allow(clippy::too_many_arguments)]
    fn flush_type_entries(
        type_name: &str,
        entries: &[BufferedEntry],
        writers: &mut HashMap<String, TypeWriter>,
        mmap: &mut MmapMut,
        row_ids: &mut Vec<(petgraph::graph::NodeIndex, u32)>,
    ) -> usize {
        let writer = match writers.get_mut(type_name) {
            Some(w) => w,
            None => return 0,
        };

        let mut flushed_bytes = 0;

        for entry in entries {
            let row = writer.row_cursor as usize;
            writer.row_cursor += 1;
            row_ids.push((entry.node_idx, row as u32));

            // Write id
            if writer.id_is_string {
                if let Value::String(s) = &entry.id {
                    let bytes = s.as_bytes();
                    let c = if row > 0 {
                        read_u64(mmap, &writer.id_str_offsets, row - 1) as usize
                    } else {
                        0
                    };
                    let end = c + bytes.len();
                    if c + bytes.len() <= writer.id_str_data.len {
                        let off = writer.id_str_data.offset + c;
                        mmap[off..off + bytes.len()].copy_from_slice(bytes);
                    }
                    write_u64(mmap, &writer.id_str_offsets, row, end as u64);
                    write_u8(mmap, &writer.id_nulls, row, 0);
                }
            } else if let Value::UniqueId(n) = &entry.id {
                write_u32(mmap, &writer.id_data, row, *n);
                write_u8(mmap, &writer.id_nulls, row, 0);
            }

            // Write title
            if let Value::String(s) = &entry.title {
                let bytes = s.as_bytes();
                let c = writer.title_cursor as usize;
                if c + bytes.len() <= writer.title_data.len {
                    let off = writer.title_data.offset + c;
                    mmap[off..off + bytes.len()].copy_from_slice(bytes);
                }
                writer.title_cursor += bytes.len() as u64;
                write_u64(mmap, &writer.title_offsets, row, writer.title_cursor);
                write_u8(mmap, &writer.title_nulls, row, 0);
            }

            // Write properties — dense columns and overflow bag
            let has_overflow = !writer.overflow_keys.is_empty();
            let overflow_start = writer.overflow_bag_data.len() as u64;
            if has_overflow {
                writer.overflow_offsets.push(overflow_start);
            }

            let mut overflow_entry_buf: Vec<u8> = Vec::new();
            let mut overflow_entry_count: u16 = 0;
            if has_overflow {
                overflow_entry_buf.extend_from_slice(&0u16.to_le_bytes());
            }

            for (key, value) in &entry.properties {
                if matches!(value, Value::Null) {
                    continue;
                }

                if writer.overflow_keys.contains(key) {
                    ColumnStore::serialize_overflow_value(&mut overflow_entry_buf, *key, value);
                    overflow_entry_count += 1;
                    continue;
                }

                let (col_type, idx) = match writer.col_map.get(key) {
                    Some(v) => *v,
                    None => continue,
                };

                match col_type {
                    ColType::Str => {
                        if let Value::String(s) = value {
                            let sc = &mut writer.str_cols[idx];
                            let bytes = s.as_bytes();
                            let c = sc.cursor as usize;
                            if c + bytes.len() <= sc.data.len {
                                let off = sc.data.offset + c;
                                mmap[off..off + bytes.len()].copy_from_slice(bytes);
                            }
                            sc.cursor += bytes.len() as u64;
                            write_u64(mmap, &sc.offsets, row, sc.cursor);
                            write_u8(mmap, &sc.nulls, row, 0);
                        }
                    }
                    _ => {
                        let fc = &writer.fixed_cols[idx];
                        let written = match (fc.col_type, value) {
                            (ColType::Int64, Value::Int64(v)) => {
                                write_i64(mmap, &fc.data, row, *v);
                                true
                            }
                            (ColType::Float64, Value::Float64(v)) => {
                                write_f64(mmap, &fc.data, row, *v);
                                true
                            }
                            (ColType::Float64, Value::Int64(v)) => {
                                write_f64(mmap, &fc.data, row, *v as f64);
                                true
                            }
                            (ColType::UniqueId, Value::UniqueId(v)) => {
                                write_u32(mmap, &fc.data, row, *v);
                                true
                            }
                            (ColType::Bool, Value::Boolean(v)) => {
                                write_u8(mmap, &fc.data, row, *v as u8);
                                true
                            }
                            (ColType::Date, Value::DateTime(dt)) => {
                                let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                                write_i32(mmap, &fc.data, row, (*dt - epoch).num_days() as i32);
                                true
                            }
                            _ => false, // type mismatch — leave as null
                        };
                        if written {
                            write_u8(mmap, &fc.nulls, row, 0);
                        }
                    }
                }
            }

            // Finalize this row's overflow blob
            if has_overflow {
                if overflow_entry_count > 0 {
                    overflow_entry_buf[0..2].copy_from_slice(&overflow_entry_count.to_le_bytes());
                    writer
                        .overflow_bag_data
                        .extend_from_slice(&overflow_entry_buf);
                } else {
                    writer
                        .overflow_bag_data
                        .extend_from_slice(&0u16.to_le_bytes());
                }
            }

            flushed_bytes += estimate_entry_bytes(&entry.id, &entry.title, &entry.properties);
        }

        flushed_bytes
    }

    let write_start = Instant::now();
    let reader = PropertyLogReader::open(log_path)?;
    let mut row_ids: Vec<(petgraph::graph::NodeIndex, u32)> = Vec::new();
    let mut entity_count = 0u64;
    let total_entities = type_meta.values().map(|m| m.row_count as u64).sum::<u64>();

    // Buffer: InternedKey -> list of entries (avoids per-entity string allocation)
    let mut buffers: HashMap<InternedKey, Vec<BufferedEntry>> = HashMap::new();
    let mut total_buffer_bytes: usize = 0;

    // Build InternedKey lookup for writers (resolve strings once, not per entity)
    let writer_keys: HashSet<InternedKey> = writers
        .keys()
        .filter_map(|name| graph.interner.try_resolve_to_key(name))
        .collect();

    // Build InternedKey rename map: old Q-code keys → new label keys.
    // Property log entries carry old InternedKeys from Phase 1; the type merge
    // renamed types after Phase 1 but before the log is replayed here.
    let key_rename: HashMap<InternedKey, InternedKey> = type_rename_map
        .iter()
        .filter_map(|(old, new)| {
            let old_key = graph.interner.try_resolve_to_key(old)?;
            let new_key = graph.interner.try_resolve_to_key(new)?;
            Some((old_key, new_key))
        })
        .collect();

    // Flush threshold: configurable via KGLITE_FLUSH_MB env var, default 2048 MB (2 GB)
    let flush_threshold_bytes: usize = std::env::var("KGLITE_FLUSH_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048)
        * (1 << 20);

    for entry_result in reader {
        let entry = entry_result?;
        // Remap old Q-code type keys to merged label keys
        let type_key = key_rename
            .get(&entry.node_type)
            .copied()
            .unwrap_or(entry.node_type);

        // Skip types not in writers (O(1) InternedKey lookup, no string allocation)
        if !writer_keys.contains(&type_key) {
            continue;
        }

        let entry_bytes = estimate_entry_bytes(&entry.id, &entry.title, &entry.properties);

        buffers.entry(type_key).or_default().push(BufferedEntry {
            node_idx: entry.node_idx,
            id: entry.id,
            title: entry.title,
            properties: entry.properties,
        });
        total_buffer_bytes += entry_bytes;
        entity_count += 1;

        if verbose && entity_count.is_multiple_of(10_000_000) {
            eplog!(
                "  Phase 1b: {}/{} entities read ({:.1}s)",
                format_count(entity_count),
                format_count(total_entities),
                write_start.elapsed().as_secs_f64(),
            );
        }
        // Phase 1b internal progress: keep tqdm bar live + responsive.
        // Cancel-check is best-effort here — the function's signature is
        // io::Result, so we swallow Cancelled rather than propagate (a
        // future change could thread Cancelled through).
        if entity_count.is_multiple_of(PHASE1B_TICK) {
            if let Some(s) = progress {
                let _ = s.emit(ProgressEvent::Update {
                    phase: "phase1b",
                    current: entity_count,
                    fields: &[],
                });
            }
        }

        // Flush the biggest type when buffer exceeds threshold
        if total_buffer_bytes > flush_threshold_bytes {
            let biggest_key = buffers
                .iter()
                .max_by_key(|(_, entries)| entries.len())
                .map(|(key, _)| *key);
            if let Some(flush_key) = biggest_key {
                let entries = buffers.remove(&flush_key).unwrap();
                let n_entries = entries.len();
                // Resolve InternedKey → String only at flush time (once per flush)
                let flush_name = graph.interner.resolve(flush_key).to_string();
                let flushed = flush_type_entries(
                    &flush_name,
                    &entries,
                    &mut writers,
                    &mut mmap,
                    &mut row_ids,
                );
                total_buffer_bytes = total_buffer_bytes.saturating_sub(flushed);
                if verbose {
                    eplog!(
                        "  Phase 1b: flushed {} ({} entries, {:.1} MB)",
                        flush_name,
                        format_count(n_entries as u64),
                        flushed as f64 / (1 << 20) as f64,
                    );
                }
            }
        }
    }

    // Flush all remaining buffers
    let remaining_keys: Vec<InternedKey> = buffers.keys().copied().collect();
    for flush_key in remaining_keys {
        let entries = buffers.remove(&flush_key).unwrap();
        if entries.is_empty() {
            continue;
        }
        let n_entries = entries.len();
        let flush_type = graph.interner.resolve(flush_key).to_string();
        flush_type_entries(&flush_type, &entries, &mut writers, &mut mmap, &mut row_ids);
        if verbose {
            eplog!(
                "  Phase 1b: flushed {} ({} entries, final)",
                flush_type,
                format_count(n_entries as u64),
            );
        }
    }

    // Finalize overflow offsets: push sentinel end offset for each type writer
    for writer in writers.values_mut() {
        if !writer.overflow_keys.is_empty() {
            writer
                .overflow_offsets
                .push(writer.overflow_bag_data.len() as u64);
        }
    }

    if verbose {
        eplog!(
            "  Phase 1b: write pass done — {} entities ({:.1}s)",
            format_count(entity_count),
            write_start.elapsed().as_secs_f64(),
        );
    }

    // ── Step 3: Forward-fill string offsets for null rows ───────────────────

    for writer in writers.values() {
        // Title offsets
        let r = &writer.title_offsets;
        let rc = r.len / 8;
        for i in 1..rc {
            if read_u64(&mmap, r, i) == 0 {
                let prev = read_u64(&mmap, r, i - 1);
                write_u64(&mut mmap, r, i, prev);
            }
        }
        // Id string offsets (if string type)
        if writer.id_is_string {
            let r = &writer.id_str_offsets;
            let rc = r.len / 8;
            for i in 1..rc {
                if read_u64(&mmap, r, i) == 0 {
                    let prev = read_u64(&mmap, r, i - 1);
                    write_u64(&mut mmap, r, i, prev);
                }
            }
        }
        // Property string offsets
        for sc in &writer.str_cols {
            let r = &sc.offsets;
            let rc = r.len / 8;
            for i in 1..rc {
                if read_u64(&mmap, r, i) == 0 {
                    let prev = read_u64(&mmap, r, i - 1);
                    write_u64(&mut mmap, r, i, prev);
                }
            }
        }
    }

    // ── Step 4: Append overflow data to mmap file and create MmapColumnStores ──

    let assemble_start = Instant::now();

    // First pass: compute total overflow bytes to append
    let mut total_overflow_bytes: usize = 0;
    for writer in writers.values() {
        if !writer.overflow_keys.is_empty() && !writer.overflow_bag_data.is_empty() {
            total_overflow_bytes = align8(total_overflow_bytes);
            // overflow offsets: (row_count+1) * 8 bytes (but stored as raw Vec<u64>)
            total_overflow_bytes += writer.overflow_offsets.len() * 8;
            total_overflow_bytes = align8(total_overflow_bytes);
            total_overflow_bytes += writer.overflow_bag_data.len();
        }
    }

    // If there's overflow data, extend the mmap file and append it
    if total_overflow_bytes > 0 {
        // Flush and drop the current mmap so we can extend the file
        mmap.flush()?;
        let current_len = mmap.len();
        drop(mmap);

        // Extend the file
        let new_len = align8(current_len) + total_overflow_bytes;
        {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&mmap_path)?;
            file.set_len(new_len as u64)?;
        }

        // Re-mmap the extended file
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&mmap_path)?;
        // SAFETY: file was just ftruncated to include the overflow region;
        // re-mmapping the full length is safe.
        mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write overflow data and record regions. Accumulate summary stats
        // instead of printing one line per type (Wikidata triggered ~90 k
        // lines of mostly-noise here pre-0.8.15).
        let mut overflow_types: u32 = 0;
        let mut overflow_sparse_cols: u64 = 0;
        let mut overflow_bytes_total: u64 = 0;
        let mut cursor = align8(current_len);
        for writer in writers.values_mut() {
            if writer.overflow_keys.is_empty() || writer.overflow_bag_data.is_empty() {
                continue;
            }

            overflow_types += 1;
            overflow_sparse_cols += writer.overflow_keys.len() as u64;
            overflow_bytes_total += writer.overflow_bag_data.len() as u64;

            // Write overflow offsets
            cursor = align8(cursor);
            let offsets_region = Region {
                offset: cursor,
                len: writer.overflow_offsets.len() * 8,
            };
            for (i, &off) in writer.overflow_offsets.iter().enumerate() {
                write_u64(&mut mmap, &offsets_region, i, off);
            }
            cursor += offsets_region.len;

            // Write overflow bag data
            cursor = align8(cursor);
            let data_region = Region {
                offset: cursor,
                len: writer.overflow_bag_data.len(),
            };
            mmap[data_region.offset..data_region.offset + data_region.len]
                .copy_from_slice(&writer.overflow_bag_data);
            cursor += data_region.len;

            // Store regions back into writer for MmapColumnStore creation
            writer.overflow_offsets_region = offsets_region;
            writer.overflow_data_region = data_region;
        }

        if verbose && overflow_types > 0 {
            eplog!(
                "  Phase 1b: overflow bags — {:.1} MB across {} types, {} sparse cols total",
                overflow_bytes_total as f64 / (1024.0 * 1024.0),
                overflow_types,
                overflow_sparse_cols,
            );
        }
    }

    // Create shared mmap Arc for all MmapColumnStore instances
    let mmap_arc = Arc::new(mmap);

    // Metadata to serialize for post-Phase-3 reload
    let mut columns_meta: Vec<ColumnTypeMeta> = Vec::new();

    for (type_name, writer) in &writers {
        let meta = match type_meta.get(type_name) {
            Some(m) => m,
            None => continue,
        };

        let has_overflow = !writer.overflow_keys.is_empty() && !writer.overflow_bag_data.is_empty();

        let mmap_store = MmapColumnStore {
            mmap: Arc::clone(&mmap_arc),
            row_count: meta.row_count,
            id_is_string: writer.id_is_string,
            id_fixed: if !writer.id_is_string {
                Some(FixedColumnMeta {
                    col_type: ColType::UniqueId,
                    data: writer.id_data,
                    nulls: writer.id_nulls,
                })
            } else {
                None
            },
            id_str: if writer.id_is_string {
                Some(StrColumnMeta {
                    data: writer.id_str_data,
                    offsets: writer.id_str_offsets,
                    nulls: writer.id_nulls,
                })
            } else {
                None
            },
            title: StrColumnMeta {
                data: writer.title_data,
                offsets: writer.title_offsets,
                nulls: writer.title_nulls,
            },
            col_map: writer
                .col_map
                .iter()
                .map(|(k, (ct, idx))| {
                    let cr = if matches!(ct, ColType::Str) {
                        ColRef::Str(*idx)
                    } else {
                        ColRef::Fixed(*idx)
                    };
                    (*k, cr)
                })
                .collect(),
            fixed_cols: writer
                .fixed_cols
                .iter()
                .map(|fc| FixedColumnMeta {
                    col_type: fc.col_type,
                    data: fc.data,
                    nulls: fc.nulls,
                })
                .collect(),
            str_cols: writer
                .str_cols
                .iter()
                .map(|sc| StrColumnMeta {
                    data: sc.data,
                    offsets: sc.offsets,
                    nulls: sc.nulls,
                })
                .collect(),
            overflow_offsets: if has_overflow {
                writer.overflow_offsets_region
            } else {
                Region::EMPTY
            },
            overflow_data: if has_overflow {
                writer.overflow_data_region
            } else {
                Region::EMPTY
            },
            has_overflow,
        };

        // Collect metadata for serialization
        columns_meta.push(ColumnTypeMeta {
            type_name: type_name.clone(),
            row_count: meta.row_count,
            id_is_string: writer.id_is_string,
            id_data: RegionMeta::from_region(&writer.id_data),
            id_nulls: RegionMeta::from_region(&writer.id_nulls),
            id_str_data: RegionMeta::from_region(&writer.id_str_data),
            id_str_offsets: RegionMeta::from_region(&writer.id_str_offsets),
            title_data: RegionMeta::from_region(&writer.title_data),
            title_offsets: RegionMeta::from_region(&writer.title_offsets),
            title_nulls: RegionMeta::from_region(&writer.title_nulls),
            col_map: writer
                .col_map
                .iter()
                .map(|(k, (ct, idx))| ColMapEntry {
                    key_u64: k.as_u64(),
                    col_type_str: ct.type_tag().to_string(),
                    idx: *idx,
                })
                .collect(),
            fixed_cols: writer
                .fixed_cols
                .iter()
                .map(|fc| FixedColMeta {
                    col_type_str: fc.col_type.type_tag().to_string(),
                    data: RegionMeta::from_region(&fc.data),
                    nulls: RegionMeta::from_region(&fc.nulls),
                })
                .collect(),
            str_cols: writer
                .str_cols
                .iter()
                .map(|sc| StrColMeta {
                    data: RegionMeta::from_region(&sc.data),
                    offsets: RegionMeta::from_region(&sc.offsets),
                    nulls: RegionMeta::from_region(&sc.nulls),
                })
                .collect(),
            overflow_offsets: RegionMeta::from_region(&writer.overflow_offsets_region),
            overflow_data: RegionMeta::from_region(&writer.overflow_data_region),
            has_overflow: !writer.overflow_keys.is_empty() && !writer.overflow_bag_data.is_empty(),
        });

        let store = ColumnStore::from_mmap_store(Arc::new(mmap_store));

        // Register schema from dense columns
        if !graph.type_schemas.contains_key(type_name) {
            let schema = TypeSchema::from_keys(writer.col_map.keys().copied());
            graph
                .type_schemas
                .insert(type_name.clone(), Arc::new(schema));
        }
        graph
            .column_stores
            .insert(type_name.clone(), Arc::new(store));
    }

    if verbose {
        eplog!(
            "  Phase 1b: assembled {} column stores ({:.1}s)",
            graph.column_stores.len(),
            assemble_start.elapsed().as_secs_f64(),
        );
    }

    // ── Step 5: Link nodes to their column store row_ids ───────────────────
    //
    // Disk backend: `update_row_id` writes into the DiskNodeSlot array so
    // later reads resolve `(type, row_id)` → column data.
    //
    // Mapped backend: replace each node's `PropertyStorage::Map` with
    // `PropertyStorage::Columnar { store, row_id }`, mirroring the second
    // pass in `DirGraph::enable_columnar`. Without this linkage the nodes
    // in MappedGraph's petgraph would have empty properties and queries
    // like `MATCH (n {id: "Q42"}) RETURN n.description` would return
    // `None`.
    //
    // Memory backend: `update_row_id` is a default no-op and mapped
    // linkage is skipped — memory-mode N-Triples builds don't go through
    // this path (they keep `PropertyStorage::Map/Compact` unchanged).
    if GraphRead::is_disk(&graph.graph) {
        for &(node_idx, row_id) in &row_ids {
            GraphWrite::update_row_id(&mut graph.graph, node_idx, row_id);
        }
        if verbose {
            eplog!(
                "  Phase 1b: fixed up {} row_ids",
                format_count(row_ids.len() as u64),
            );
        }
    } else if GraphRead::is_mapped(&graph.graph) {
        // Snapshot the Arc<ColumnStore> per type once so the inner loop
        // only does a HashMap lookup; avoids a clone on every node.
        let type_name_by_key: HashMap<InternedKey, String> = graph
            .column_stores
            .keys()
            .filter_map(|t| graph.interner.try_resolve_to_key(t).map(|k| (k, t.clone())))
            .collect();
        let stores_snapshot: HashMap<
            String,
            Arc<crate::graph::storage::column_store::ColumnStore>,
        > = graph
            .column_stores
            .iter()
            .map(|(t, s)| (t.clone(), Arc::clone(s)))
            .collect();
        let mut linked = 0u64;
        for &(node_idx, row_id) in &row_ids {
            let type_key = match GraphRead::node_type_of(&graph.graph, node_idx) {
                Some(k) => k,
                None => continue,
            };
            let type_name = match type_name_by_key.get(&type_key) {
                Some(n) => n,
                None => continue,
            };
            let store = match stores_snapshot.get(type_name) {
                Some(s) => Arc::clone(s),
                None => continue,
            };
            if let Some(node) = GraphWrite::node_weight_mut(&mut graph.graph, node_idx) {
                node.properties = PropertyStorage::Columnar { store, row_id };
                linked += 1;
            }
        }
        if verbose {
            eplog!(
                "  Phase 1b: linked {} mapped nodes to column stores",
                format_count(linked),
            );
        }
    }

    // Save columns metadata for post-Phase-3 reload.
    // The mmap file stays on disk at data_dir/columns.bin — don't delete it.
    let meta_path = data_dir.join("columns_meta.json");
    if !columns_meta.is_empty() {
        if let Ok(json) = serde_json::to_string(&columns_meta) {
            let _ = std::fs::write(&meta_path, json);
        }
        // Also save as bincode+zstd for fast loading (~10x faster than JSON parse)
        if let Ok(bytes) = bincode::serialize(&columns_meta) {
            if let Ok(compressed) = zstd::encode_all(bytes.as_slice(), 3) {
                let _ = std::fs::write(data_dir.join("columns_meta.bin.zst"), compressed);
            }
        }
    }

    if verbose {
        eplog!(
            "  Phase 1b: saved columns metadata ({} types) to {}",
            columns_meta.len(),
            meta_path.display(),
        );
    }

    Ok(())
}

//! Column buffers in a v9 container's data region: which columns a save moves
//! there, and how a load places them.
//!
//! A claimed column keeps its entry in the type's compressed column section
//! (name and type tag with an `@map` suffix, empty payload) and its buffers
//! become sections named `col:<type>:<column>:data|nulls|offsets|bytes`
//! (owner `columns`). On load each column is placed as a whole, by
//! `LoadOptions::columns`: mapped from the file, or decoded from the container
//! bytes through the same heap/spill path a v8 column takes.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use super::embedding_sections::escape;
use super::load_options::MapPolicy;
use super::mappable::{
    resolve_mapping, MapContext, MapEntry, MappableSection, MappableTable, MappedFile, Placement,
    SaveLayout, SectionDtype, SharedSection, FLAG_UTF8_VALIDATED,
};
use super::*;
use crate::graph::storage::column_store::{
    buffer_leaves_heap, ColumnSectionSource, ColumnSplit, PackedDecode, PartData,
};

pub(super) const OWNER: &str = "columns";

/// Section class (`MapEntry::kind`) of every column buffer.
pub(super) const KIND_COLUMN: u8 = 3;

fn section_name(type_name: &str, column: &str, suffix: &str) -> String {
    format!("col:{}:{}:{suffix}", escape(type_name), escape(column))
}

/// What [`encode_column_sections`] produces: the per-type metadata, the
/// compressed column sections, and the data-region sections moved out of them.
pub(super) struct EncodedColumns<'g> {
    pub(super) meta: Vec<PortableColumnSection>,
    pub(super) compressed: Vec<Vec<u8>>,
    pub(super) mappable: Vec<MappableSection<'g>>,
    /// Large columns written without a data-region form, and why.
    pub(super) fallbacks: Vec<crate::graph::dir_graph::MappedFallback>,
}

/// Column sections, one per node type sorted by type name (the backend's map
/// is a HashMap whose per-instance RandomState would otherwise vary the
/// section order across processes, breaking the byte-level reproducibility the
/// `test_phase4_parity` golden-hash test relies on), plus the data-region
/// sections of the columns `layout` moved out of them. Sorting is free (type
/// count is small) and doesn't affect the format: each section is
/// self-describing and the decoder iterates `column_sections_meta` in order.
pub(super) fn encode_column_sections(
    graph: &DirGraph,
    codec: serde_codec::CodecVersion,
    layout: SaveLayout,
) -> io::Result<EncodedColumns<'_>> {
    let claims = layout != SaveLayout::Compact;
    let mut meta_out = Vec::new();
    let mut data_out = Vec::new();
    let mut sections = Vec::new();
    let mut fallbacks = Vec::new();
    let mut stores: Vec<(&str, &Arc<ColumnStore>)> = graph.column_stores_by_name();
    stores.sort_by(|a, b| a.0.cmp(b.0));
    for (type_name, store) in stores {
        // Mapped columns are about to be copied into a new file.
        store.verify_storage()?;
        let int_encoding = crate::graph::storage::packed_codec::IntColumnEncoding::Auto;
        // A writer overlay's store can carry a heap tail; the sections are
        // written from its columns, so they must hold every row.
        let (store, packed, split) = match store.with_heap_tail_folded() {
            Cow::Borrowed(store) => {
                let mut split = ColumnSplit::new(claims);
                // v6: integer columns pick their smaller encoding per column.
                let packed =
                    store.write_packed_split(&graph.interner, codec, int_encoding, &mut split)?;
                (Cow::Borrowed(store), packed, split)
            }
            Cow::Owned(store) => {
                let mut split = ColumnSplit::new(claims);
                let packed =
                    store.write_packed_split(&graph.interner, codec, int_encoding, &mut split)?;
                let split = split.into_owned();
                (Cow::Owned(store), packed, split)
            }
        };
        let compressed = zstd_compress(&packed)?;
        drop(packed); // free uncompressed before next type

        let mut cols = HashMap::new();
        for (slot, ik) in store.schema().iter() {
            let prop_name = graph.interner.resolve(ik);
            if let Some(col) = store.column(slot as usize) {
                // The *logical* column type. The per-column encoding actually
                // used lives in the section itself (a v6 `Int64` column may be
                // written delta-varint); the loader reads the section's tag and
                // uses these entries only for their key set.
                cols.insert(prop_name.to_string(), col.type_tag().to_string());
            }
        }

        meta_out.push(PortableColumnSection {
            type_name: type_name.to_string(),
            compressed_size: compressed.len() as u64,
            row_count: store.row_count(),
            columns: cols,
        });
        data_out.push(compressed);
        for (column, reason) in &split.declined {
            let column = match column.as_str() {
                "__id__" => "id",
                "__title__" => "title",
                other => other,
            };
            fallbacks.push(crate::graph::dir_graph::MappedFallback::new(
                format!("columns:{type_name}.{column}"),
                *reason,
            ));
        }
        for column in split.columns {
            for part in column.parts {
                sections.push(MappableSection {
                    name: section_name(type_name, &column.name, part.suffix),
                    owner: OWNER.to_string(),
                    kind: KIND_COLUMN,
                    dtype: part.dtype,
                    elem_count: part.elem_count,
                    flags: if part.utf8 { FLAG_UTF8_VALIDATED } else { 0 },
                    data: part.bytes,
                });
            }
        }
    }
    Ok(EncodedColumns {
        meta: meta_out,
        compressed: data_out,
        mappable: sections,
        fallbacks,
    })
}

/// Decides, per column, whether a load maps its sections or decodes them.
pub(super) struct ColumnPlacer {
    largest: HashMap<String, u64>,
}

fn column_stem(name: &str) -> &str {
    name.rsplit_once(':').map_or(name, |(stem, _)| stem)
}

impl ColumnPlacer {
    pub(super) fn new(entries: &[MapEntry]) -> Self {
        let mut largest: HashMap<String, u64> = HashMap::new();
        for entry in entries.iter().filter(|e| e.kind == KIND_COLUMN) {
            let biggest = largest
                .entry(column_stem(&entry.name).to_string())
                .or_default();
            *biggest = (*biggest).max(entry.len);
        }
        ColumnPlacer { largest }
    }

    /// One decision for all of a column's buffers, made from its largest
    /// buffer against the rule a spill uses ([`buffer_leaves_heap`]), so a
    /// column is never half mapped and `Auto` maps exactly what a v8 load
    /// would have spilled.
    pub(super) fn place(
        &self,
        entry: &MapEntry,
        file: Option<&Arc<MappedFile>>,
        policy: MapPolicy,
    ) -> Placement {
        let largest = self
            .largest
            .get(column_stem(&entry.name))
            .copied()
            .unwrap_or(entry.len);
        let ctx = MapContext::host(file.map(|f| &**f), largest);
        let auto_floor = if buffer_leaves_heap(largest as usize) {
            0
        } else {
            u64::MAX
        };
        if resolve_mapping(policy, auto_floor, &ctx) {
            Placement::Mapped
        } else {
            Placement::Deferred
        }
    }
}

/// The buffers of one type's `@map` columns, found in the opened table.
struct TypeSections<'t> {
    table: &'t MappableTable,
    container: &'t [u8],
    type_name: &'t str,
    claimed: &'t RefCell<HashSet<String>>,
}

impl ColumnSectionSource for TypeSections<'_> {
    fn part(
        &self,
        column: &str,
        suffix: &str,
        dtype: SectionDtype,
        expected: Option<u64>,
    ) -> io::Result<PartData<'_>> {
        let name = section_name(self.type_name, column, suffix);
        let wrong = |what: &str| {
            invalid_data(format!(
                "mappable section '{name}' is {what}; the column's buffers do not match its type"
            ))
        };
        let (found_dtype, count, owner, kind) =
            match (self.table.get(&name), self.table.get_deferred(&name)) {
                (Some(s), _) => (s.dtype, s.elem_count, &s.owner, s.kind),
                (None, Some(d)) => (d.dtype, d.elem_count, &d.owner, d.kind),
                (None, None) => {
                    return Err(invalid_data(format!(
                        "the column section of '{}' refers to the data-region section '{name}', \
                     which the file does not contain",
                        self.type_name
                    )))
                }
            };
        if owner != OWNER || kind != KIND_COLUMN || found_dtype != dtype {
            return Err(wrong("not a column section of the expected element type"));
        }
        if expected.is_some_and(|n| n != count) {
            return Err(wrong("not the length its column needs"));
        }
        self.claimed.borrow_mut().insert(name.clone());
        match self.table.get(&name) {
            Some(section) => Ok(match SharedSection::new(section.clone()) {
                Some(shared) => PartData::Shared(shared),
                None => PartData::Bytes(section.as_bytes()),
            }),
            None => {
                let deferred = self.table.get_deferred(&name).expect("checked above");
                Ok(PartData::Bytes(deferred.bytes(self.container)))
            }
        }
    }
}

/// Refuse a column section no column claimed.
fn check_all_claimed(table: &MappableTable, claimed: &HashSet<String>) -> io::Result<()> {
    let stray = table
        .iter()
        .map(|s| (&s.name, &s.owner))
        .chain(table.iter_deferred().map(|s| (&s.name, &s.owner)))
        .find(|(name, owner)| *owner == OWNER && !claimed.contains(*name));
    match stray {
        Some((name, _)) => Err(invalid_data(format!(
            "mappable section '{name}' names no column of '{OWNER}'"
        ))),
        None => Ok(()),
    }
}

/// Where a mapped column's buffers are read from, for the Windows
/// destination check and `verify_integrity`.
pub(crate) fn mapped_sources(graph: &DirGraph) -> Vec<std::path::PathBuf> {
    graph
        .column_stores_by_name()
        .into_iter()
        .flat_map(|(_, store)| store.shared_sections())
        .filter_map(|section| section.source_path().map(std::path::Path::to_path_buf))
        .collect()
}

/// Copy every mapped column buffer of `graph` to the heap.
pub(crate) fn promote_all(graph: &mut DirGraph) {
    let names: Vec<String> = graph
        .column_stores_by_name()
        .into_iter()
        .filter(|(_, store)| !store.shared_sections().is_empty())
        .map(|(name, _)| name.to_string())
        .collect();
    for name in names {
        let Some(store) = graph.column_store(&name) else {
            continue;
        };
        let mut promoted = (**store).clone();
        promoted.promote_shared();
        graph.install_column_store(&name, Arc::new(promoted));
    }
}

fn load_portable_column_section(
    codec: serde_codec::CodecVersion,
    dir_graph: &mut DirGraph,
    sections: &mut SectionCursor<'_>,
    section_meta: &PortableColumnSection,
    section_index: usize,
    temp_dir: &Path,
    source: &TypeSections<'_>,
) -> io::Result<()> {
    let compressed = sections.take(
        section_meta.compressed_size,
        &column_section_key(&section_meta.type_name),
    )?;
    let packed = zstd_decompress_within(compressed, sections.ceiling())?;
    let expected_rows = dir_graph
        .type_indices
        .get(&section_meta.type_name)
        .map_or(0, |nodes| nodes.len());
    if section_meta.row_count as usize != expected_rows {
        return Err(invalid_data(format!(
            "column section {section_index} for '{}' declares {} rows; topology has {expected_rows}",
            section_meta.type_name, section_meta.row_count
        )));
    }
    // Column slot order comes from the PAYLOAD, not from `section_meta.columns`.
    //
    // The packed block is self-describing and ordered — it carries
    // `(name, type_tag, data)` per column in the writing store's slot order —
    // whereas `section_meta.columns` is a `HashMap` that records only the key
    // set and type tags (see the writer's own note that the loader "uses these
    // entries only for their key set"). Building the schema from that map made
    // slot order a `RandomState` artefact that differed on every load, so
    // re-saving a file produced different bytes each run.
    //
    // Reading the recorded order makes a re-save byte-identical to the
    // original, not merely deterministic.
    let mut ordered_names = ColumnStore::packed_column_names(&packed)?;
    // A key the metadata declares but the payload does not carry has no
    // recorded position; append such keys by name so they still get a slot and
    // the result stays deterministic. Expected to be empty — writer and payload
    // are built from the same schema — but silently dropping a declared column
    // would be worse than an arbitrary-but-stable position.
    let in_payload: std::collections::HashSet<&str> =
        ordered_names.iter().map(String::as_str).collect();
    let mut orphans: Vec<&String> = section_meta
        .columns
        .keys()
        .filter(|name| !in_payload.contains(name.as_str()))
        .collect();
    orphans.sort();
    let orphans: Vec<String> = orphans.into_iter().cloned().collect();
    ordered_names.extend(orphans);

    let col_keys = ordered_names
        .iter()
        .map(|name| {
            dir_graph
                .interner
                .try_get_or_intern(name)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let column_schema = Arc::new(crate::graph::schema::TypeSchema::from_keys(col_keys));
    let type_meta = dir_graph
        .node_type_metadata
        .get(&section_meta.type_name)
        .cloned()
        .unwrap_or_default();
    // A *name* for this section's spill directory, not a directory: the column
    // reader creates it when it has a blob to write there, and most sections
    // never do. Creating it here made every load — including one of a 12 KB
    // file that cannot spill a single column — mkdir under a shared `$TMPDIR`,
    // and racing that mkdir against another load's cleanup is what returned
    // bare `EEXIST`/`EINVAL`/`ENOENT` from `load_file`.
    let type_temp_dir = temp_dir.join(format!("type_{section_index}"));
    let store = ColumnStore::load_packed_sectioned(
        column_schema,
        &type_meta,
        &dir_graph.interner,
        &packed,
        section_meta.row_count,
        &PackedDecode {
            temp_dir: Some(&type_temp_dir),
            codec,
            sections: Some(source),
        },
    )?;
    dir_graph.install_column_store(&section_meta.type_name, Arc::new(store));
    Ok(())
}

pub(super) fn load_portable_columns(
    codec: serde_codec::CodecVersion,
    dir_graph: &mut DirGraph,
    sections: &mut SectionCursor<'_>,
    columns: &[PortableColumnSection],
    table: &MappableTable,
    container: &[u8],
) -> io::Result<()> {
    let temp_dir = portable_temp_dir();
    if let Ok(mut dirs) = dir_graph.temp_dirs.lock() {
        dirs.push(temp_dir.clone());
    }
    let claimed = RefCell::new(HashSet::new());
    for (index, metadata) in columns.iter().enumerate() {
        let source = TypeSections {
            table,
            container,
            type_name: &metadata.type_name,
            claimed: &claimed,
        };
        load_portable_column_section(
            codec, dir_graph, sections, metadata, index, &temp_dir, &source,
        )?;
    }
    check_all_claimed(table, &claimed.into_inner())?;
    // A frozen topology's records already carry their rows (checked against
    // the type indexes when it was built) and no inline identity to clear.
    if !dir_graph.graph.is_frozen() {
        attach_portable_column_stores(dir_graph);
    }
    Ok(())
}

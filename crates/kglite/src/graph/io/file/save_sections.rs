//! The `.kgl` section encoders and stream writer behind [`write_kgl_to`]
//! and the stamped backup path, split from `file.rs` (source-quality
//! ceilings). One serializer body serves both: the stamp is a parameter.

use super::mappable::{
    collect_mappable_sections, plan_data_region, write_data_region, MapEntry, MappableSection,
    MappableSet, SaveLayout,
};
use super::*;

/// Compressed optional sections, each `None` when the graph has nothing for it.
struct OptionalSections {
    embeddings: Option<Vec<u8>>,
    edge_embeddings: Option<Vec<u8>>,
    timeseries: Option<Vec<u8>>,
    secondary_labels: Option<Vec<u8>>,
    vector_index: Option<Vec<u8>>,
    text_index: Option<Vec<u8>>,
    edge_vector_index: Option<Vec<u8>>,
    edge_text_index: Option<Vec<u8>>,
    core_version: u32,
}

impl OptionalSections {
    /// In file order: the relationship BM25 section is last so a reader that
    /// predates it stops before those bytes.
    fn in_file_order(&self) -> [Option<&[u8]>; 8] {
        [
            self.embeddings.as_deref(),
            self.edge_embeddings.as_deref(),
            self.timeseries.as_deref(),
            self.secondary_labels.as_deref(),
            self.vector_index.as_deref(),
            self.text_index.as_deref(),
            self.edge_vector_index.as_deref(),
            self.edge_text_index.as_deref(),
        ]
    }
}

fn compress_some(payload: Option<Vec<u8>>) -> io::Result<Option<Vec<u8>>> {
    payload.map(|p| zstd_compress(&p)).transpose()
}

/// The topology of a frozen graph, or of an overlay over one, is copied out
/// of sections that were not checksummed at load; refuse to write a damaged
/// one, or one whose endpoint records are inconsistent, into a new file. Each
/// check runs once per mapping (the result is cached), so a repeat call is free. The serializer repeats
/// the check but can only report a generic error.
fn verify_topology_sources(graph: &DirGraph) -> io::Result<()> {
    match graph.graph.frozen_base() {
        Some(frozen) => {
            frozen.verify_sections()?;
            // A save writes every edge; an endpoint the file got wrong would
            // otherwise be dropped from the copy without a word.
            frozen.validated_endpoints()
        }
        None => Ok(()),
    }
}

/// Topology with node properties stripped into the column sections.
fn encode_topology(graph: &DirGraph, codec: serde_codec::CodecVersion) -> io::Result<Vec<u8>> {
    verify_topology_sources(graph)?;
    let topology_raw = {
        let _strip = StripPropertiesGuard::new();
        let _guard = SerdeSerializeGuard::new(&graph.interner);
        codec_ser(codec, &graph.graph)?
    };
    zstd_compress(&topology_raw)
}

/// `graph.embeddings` and `graph.timeseries_store` are HashMaps whose
/// per-process RandomState would randomize entry order; serializing through
/// a BTreeMap view keeps the bytes reproducible (same wire shape).
fn encode_ordered_map<K: Ord + Serialize, V: Serialize>(
    map: impl Iterator<Item = (K, V)>,
    is_empty: bool,
    codec: serde_codec::CodecVersion,
) -> io::Result<Option<Vec<u8>>> {
    if is_empty {
        return Ok(None);
    }
    let ordered: BTreeMap<K, V> = map.collect();
    let raw = codec_ser(codec, &ordered)?;
    Ok(Some(zstd_compress(&raw)?))
}

fn encode_optional_sections(
    graph: &DirGraph,
    codec: serde_codec::CodecVersion,
    layout: SaveLayout,
) -> io::Result<OptionalSections> {
    let records = graph
        .embeddings
        .iter()
        .map(|(key, store)| Ok((key, NodeStoreRecord::new(store, layout)?)))
        .collect::<io::Result<Vec<_>>>()?;
    let embeddings = encode_ordered_map(records.into_iter(), graph.embeddings.is_empty(), codec)?;
    let (edge_embeddings, core_version) = encode_portable_edge_embeddings(graph, codec, layout)?;
    let timeseries = encode_ordered_map(
        graph.timeseries_store.iter(),
        graph.timeseries_store.is_empty(),
        codec,
    )?;
    // Hand-rolled binary format — InternedKey doesn't derive serde, and the
    // same layout is reused by the disk sidecar. Single-label graphs skip it.
    let secondary_labels = compress_some(encode_secondary_label_index(graph))?;
    let vector_index = compress_some(encode_vector_indexes(graph)?)?;
    let text_index = compress_some(encode_text_indexes(graph)?)?;
    Ok(OptionalSections {
        embeddings,
        edge_embeddings,
        timeseries,
        secondary_labels,
        vector_index,
        text_index,
        edge_vector_index: encode_edge_vector_index_section(graph)?,
        edge_text_index: encode_edge_text_index_section(graph)?,
        core_version,
    })
}

/// What the metadata records about the data region: its section table and the
/// parts the save wrote without a mapped form.
struct DataRegionRecord {
    entries: Vec<MapEntry>,
    fallbacks: Vec<crate::graph::dir_graph::MappedFallback>,
}

/// Canonical metadata JSON: round-trips through `serde_json::Value`, whose
/// object map is a BTreeMap, so every nested `HashMap<String, T>` emits with
/// sorted keys. Prevents per-process HashMap randomization from producing
/// different save bytes for the same graph — the byte-level tripwire in
/// `tests/test_phase4_parity.py` depends on this.
fn build_metadata_json(
    graph: &DirGraph,
    checkpoint_lsn: Option<u64>,
    topology: &[u8],
    column_meta: Vec<PortableColumnSection>,
    column_data: &[Vec<u8>],
    optional: &OptionalSections,
    region: DataRegionRecord,
) -> io::Result<Vec<u8>> {
    let keys = [
        EMBEDDINGS_SECTION,
        EDGE_EMBEDDINGS_SECTION,
        TIMESERIES_SECTION,
        SECONDARY_LABELS_SECTION,
        VECTOR_INDEX_SECTION,
        TEXT_INDEX_SECTION,
        EDGE_VECTOR_INDEX_SECTION,
        EDGE_TEXT_INDEX_SECTION,
    ];
    let present = optional.in_file_order();
    let digests = build_section_digests(
        topology,
        &column_meta,
        column_data,
        std::array::from_fn(|i| (keys[i], present[i])),
    );
    let mut metadata = FileMetadata::from_graph_version(graph, optional.core_version);
    if let Some(lsn) = checkpoint_lsn {
        metadata.checkpoint_lsn = lsn;
    }
    metadata.section_digests = digests;
    metadata.mappable = region.entries;
    metadata.mapped_fallbacks = region.fallbacks;
    metadata.topology_compressed_size = topology.len() as u64;
    metadata.column_sections = column_meta;
    metadata.embeddings_compressed_size = compressed_len(&optional.embeddings);
    metadata.edge_embeddings_compressed_size = compressed_len(&optional.edge_embeddings);
    metadata.timeseries_compressed_size = compressed_len(&optional.timeseries);
    metadata.secondary_labels_compressed_size = compressed_len(&optional.secondary_labels);
    metadata.vector_index_compressed_size = compressed_len(&optional.vector_index);
    metadata.text_index_compressed_size = compressed_len(&optional.text_index);
    metadata.edge_vector_index_compressed_size = compressed_len(&optional.edge_vector_index);
    metadata.edge_text_index_compressed_size = compressed_len(&optional.edge_text_index);
    let value = serde_json::to_value(&metadata).map_err(io::Error::other)?;
    serde_json::to_vec(&value).map_err(io::Error::other)
}

/// [`write_kgl_to`] with the metadata's `checkpoint_lsn` overridden (`None`
/// keeps the graph's own stamp). Reads the graph only.
pub fn write_kgl_to_stamped<W: Write>(
    graph: &DirGraph,
    writer: &mut W,
    checkpoint_lsn: Option<u64>,
) -> io::Result<()> {
    write_kgl_to_layout(graph, writer, checkpoint_lsn, SaveLayout::Auto)
}

/// Collect the mappable sections while the source-section checksum pass runs
/// on a second thread: the pass is page-fault bound and independent of the
/// collection. A thread rather than `rayon::join`, which would move the
/// collection off this thread. A graph with no frozen base has nothing to
/// check and spawns nothing. Nothing is written before both results are read,
/// and a collection error is reported ahead of a checksum error, as when the
/// two ran in sequence.
fn collect_beside_verify(
    graph: &DirGraph,
    layout: SaveLayout,
) -> (io::Result<MappableSet<'_>>, io::Result<()>) {
    if graph.graph.frozen_base().is_none() {
        return (collect_mappable_sections(graph, layout), Ok(()));
    }
    std::thread::scope(|scope| {
        let verifier = scope.spawn(|| verify_topology_sources(graph));
        let collected = collect_mappable_sections(graph, layout);
        let verified = verifier
            .join()
            .unwrap_or_else(|_| Err(io::Error::other("the section checksum pass panicked")));
        (collected, verified)
    })
}

/// [`write_kgl_to_stamped`] under an explicit [`SaveLayout`]: `Compact` writes
/// the v8 container every kglite release can read.
pub fn write_kgl_to_layout<W: Write>(
    graph: &DirGraph,
    writer: &mut W,
    checkpoint_lsn: Option<u64>,
    layout: SaveLayout,
) -> io::Result<()> {
    let (collected, verified) = collect_beside_verify(graph, layout);
    let set = collected?;
    verified?;
    write_kgl_container(
        graph,
        writer,
        checkpoint_lsn,
        layout,
        set.sections,
        set.fallbacks,
    )
}

/// The container writer: v9 when `mappable` is non-empty, otherwise the v8
/// bytes this writer has always produced. `layout` also decides which stores
/// are written as shells (their vectors are the `mappable` sections), so it
/// must be the layout `mappable` was collected under.
pub(super) fn write_kgl_container<W: Write>(
    graph: &DirGraph,
    writer: &mut W,
    checkpoint_lsn: Option<u64>,
    layout: SaveLayout,
    mappable: Vec<MappableSection<'_>>,
    mut fallbacks: Vec<crate::graph::dir_graph::MappedFallback>,
) -> io::Result<()> {
    validate_column_keys_registered(graph)?;
    let codec = serde_codec::CodecVersion::PostcardV1;
    let topology = encode_topology(graph, codec)?;
    let columns = column_sections::encode_column_sections(graph, codec, layout)?;
    let (column_meta, column_data) = (columns.meta, columns.compressed);
    let mut mappable = mappable;
    mappable.extend(columns.mappable);
    fallbacks.extend(columns.fallbacks);
    let fallbacks = crate::graph::dir_graph::mapped_report::finalize(fallbacks);
    graph.park_save_notice(&fallbacks);
    let optional = encode_optional_sections(graph, codec, layout)?;
    let core_version = optional.core_version;
    let (entries, _region_len) = plan_data_region(&mappable)?;
    let metadata_json = build_metadata_json(
        graph,
        checkpoint_lsn,
        &topology,
        column_meta,
        &column_data,
        &optional,
        DataRegionRecord {
            entries: entries.clone(),
            fallbacks,
        },
    )?;

    // Header: magic (4B) + codec (1B) + core_data_version (4B) +
    // metadata_length (4B). The codec byte prevents implicit byte sniffing.
    // `metadata_length` counts the JSON plus its 4-byte CRC32 trailer, so
    // `13 + metadata_length` still ends the metadata block.
    let magic = if mappable.is_empty() {
        V8_MAGIC
    } else {
        V9_MAGIC
    };
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&magic);
    header.push(codec.tag());
    header.extend_from_slice(&core_version.to_le_bytes());
    header.extend_from_slice(&((metadata_json.len() + 4) as u32).to_le_bytes());
    writer.write_all(&header)?;
    writer.write_all(&metadata_json)?;
    writer.write_all(&metadata_checksum(&header, &metadata_json).to_le_bytes())?;
    writer.write_all(&topology)?;
    for section_data in &column_data {
        writer.write_all(section_data)?;
    }
    let mut written = (header.len() + metadata_json.len() + 4 + topology.len()) as u64;
    written += column_data.iter().map(|d| d.len() as u64).sum::<u64>();
    for section in optional.in_file_order() {
        write_optional_section(writer, section)?;
        written += section.map_or(0, |s| s.len() as u64);
    }
    if !mappable.is_empty() {
        write_data_region(writer, written, &mappable, &entries)?;
    }
    // Flush the writer's own buffer; the atomic-save wrapper additionally
    // fsyncs the file. A harmless no-op for an in-memory `Vec<u8>` writer.
    writer.flush()?;
    Ok(())
}

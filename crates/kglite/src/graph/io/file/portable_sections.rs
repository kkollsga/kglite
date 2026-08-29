//! Portable-container decode stages split from `file.rs` (source-quality
//! line ceiling): topology decode, the optional trailing sections, and the
//! embedding-norm rebuild they share.

use super::*;

pub(super) fn validate_and_rebuild_embedding_norms(
    embeddings: &mut HashMap<(String, String), EmbeddingStore>,
) -> io::Result<()> {
    for store in embeddings.values_mut() {
        store.validate_shape().map_err(invalid_data)?;
        store.rebuild_norms();
    }
    Ok(())
}

pub(super) fn decode_portable_topology(
    codec: serde_codec::CodecVersion,
    sections: &mut SectionCursor<'_>,
    metadata: FileMetadata,
) -> io::Result<(DirGraph, PortableSectionPlan)> {
    let topology_compressed = sections.take(metadata.topology_compressed_size, TOPOLOGY_SECTION)?;
    let topology_raw = zstd_decompress(topology_compressed)?;
    let mut interner = StringInterner::new();
    let graph: crate::graph::schema::GraphBackend = {
        let _guard = SerdeDeserializeGuard::new(&mut interner);
        codec_deser(codec, &topology_raw, topology_raw.capacity() as u64)?
    };
    let plan = PortableSectionPlan {
        columns: metadata.column_sections.clone(),
        embeddings: metadata.embeddings_compressed_size,
        timeseries: metadata.timeseries_compressed_size,
        secondary_labels: metadata.secondary_labels_compressed_size,
        vector_index: metadata.vector_index_compressed_size,
        text_index: metadata.text_index_compressed_size,
    };
    let mut dir_graph = DirGraph::from_graph(graph);
    dir_graph.interner = interner;
    metadata.apply_to(&mut dir_graph);
    dir_graph.rebuild_type_indices_and_schemas();
    dir_graph.build_connection_types_cache();
    Ok((dir_graph, plan))
}

pub(super) fn load_portable_optional_sections(
    codec: serde_codec::CodecVersion,
    core_version: u32,
    dir_graph: &mut DirGraph,
    sections: &mut SectionCursor<'_>,
    plan: &PortableSectionPlan,
) -> io::Result<()> {
    if plan.embeddings > 0 {
        if core_version < EMBED_PROVENANCE_MIN_VERSION {
            return Err(invalid_data(EMBED_FORMAT_BREAK_MSG));
        }
        let compressed = sections.take(plan.embeddings, EMBEDDINGS_SECTION)?;
        let raw = zstd_decompress(compressed)?;
        let mut embeddings: HashMap<(String, String), EmbeddingStore> =
            codec_deser(codec, &raw, raw.capacity() as u64)?;
        validate_and_rebuild_embedding_norms(&mut embeddings)?;
        dir_graph.embeddings = embeddings;
    }
    if plan.timeseries > 0 {
        let compressed = sections.take(plan.timeseries, TIMESERIES_SECTION)?;
        let raw = zstd_decompress(compressed)?;
        dir_graph.timeseries_store = codec_deser(codec, &raw, raw.capacity() as u64)?;
    }
    if plan.secondary_labels > 0 {
        let compressed = sections.take(plan.secondary_labels, SECONDARY_LABELS_SECTION)?;
        let raw = zstd_decompress(compressed)?;
        decode_secondary_label_index(&raw, dir_graph)?;
    }
    if plan.vector_index > 0 {
        // Framing failures propagate — a section that is truncated or fails
        // its digest means the *file* is damaged, and skipping it would leave
        // the reader with no signal that anything was wrong. The payload
        // itself stays optional: it is self-describing and rebuildable, so an
        // index this build does not recognise is still skipped silently (see
        // `decode_vector_indexes`).
        let compressed = sections.take(plan.vector_index, VECTOR_INDEX_SECTION)?;
        if let Ok(raw) = zstd_decompress(compressed) {
            decode_vector_indexes(&raw, dir_graph);
        }
    }
    if plan.text_index > 0 {
        // Same split as the vector section above: framing failures are file
        // damage and propagate; the payload itself is a rebuildable cache and
        // an unreadable one is skipped silently (see `decode_text_indexes`).
        let compressed = sections.take(plan.text_index, TEXT_INDEX_SECTION)?;
        if let Ok(raw) = zstd_decompress(compressed) {
            decode_text_indexes(&raw, dir_graph);
        }
    }
    Ok(())
}

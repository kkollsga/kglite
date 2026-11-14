//! Portable-container decode stages split from `file.rs` (source-quality
//! line ceiling): topology decode, the optional trailing sections, and the
//! embedding-norm rebuild they share.

use super::*;

pub(super) fn encode_portable_edge_embeddings(
    graph: &DirGraph,
    codec: serde_codec::CodecVersion,
    layout: SaveLayout,
) -> io::Result<(Option<Vec<u8>>, u32)> {
    if !crate::graph::edge_embeddings::has_persisted_edge_embeddings(graph) {
        return Ok((None, NODE_ONLY_CORE_DATA_VERSION));
    }
    for store in graph.edge_embeddings.values() {
        store.numeric().verify_storage()?;
    }
    let ordered = crate::graph::edge_embeddings::persisted_edge_embedding_stores(graph, |store| {
        embedding_sections::store_goes_to_data_region(store, layout)
    });
    let raw = codec_ser(codec, &ordered)?;
    Ok((Some(zstd_compress(&raw)?), CURRENT_CORE_DATA_VERSION))
}

pub(super) fn write_optional_section<W: Write>(
    writer: &mut W,
    section: Option<&[u8]>,
) -> io::Result<()> {
    if let Some(bytes) = section {
        writer.write_all(bytes)?;
    }
    Ok(())
}

/// A section's compressed size for its metadata key; 0 when it is absent.
pub(super) fn compressed_len(section: &Option<Vec<u8>>) -> u64 {
    section.as_ref().map_or(0, |bytes| bytes.len() as u64)
}

pub(super) fn validate_and_rebuild_embedding_norms(
    embeddings: &mut HashMap<(String, String), EmbeddingStore>,
) -> io::Result<()> {
    for store in embeddings.values_mut() {
        store.validate_shape().map_err(invalid_data)?;
        store
            .rebuild_norms_checked()
            .map_err(|error| invalid_data(error.to_string()))?;
    }
    Ok(())
}

pub(super) fn decode_portable_topology(
    codec: serde_codec::CodecVersion,
    sections: &mut SectionCursor<'_>,
    metadata: FileMetadata,
    indexes: Option<&mut index_sections::IndexSections>,
    frozen: Option<topology_sections::TopologySections>,
) -> io::Result<(DirGraph, PortableSectionPlan)> {
    let mut interner = StringInterner::new();
    let graph: crate::graph::schema::GraphBackend = match frozen {
        // The file's raw topology serves the graph in place; the compressed
        // copy of it is stepped over, unread.
        Some(raw) => {
            sections.skip(metadata.topology_compressed_size, TOPOLOGY_SECTION)?;
            crate::graph::schema::GraphBackend::from_frozen(raw.build(
                codec,
                &mut interner,
                sections.ceiling(),
            )?)
        }
        None => {
            let topology_compressed =
                sections.take(metadata.topology_compressed_size, TOPOLOGY_SECTION)?;
            let topology_raw = zstd_decompress_within(topology_compressed, sections.ceiling())?;
            let _guard = SerdeDeserializeGuard::new(&mut interner);
            codec_deser(codec, &topology_raw, topology_raw.capacity() as u64)?
        }
    };
    let is_frozen = graph.is_frozen();
    let plan = PortableSectionPlan {
        columns: metadata.column_sections.clone(),
        embeddings: metadata.embeddings_compressed_size,
        edge_embeddings: metadata.edge_embeddings_compressed_size,
        timeseries: metadata.timeseries_compressed_size,
        secondary_labels: metadata.secondary_labels_compressed_size,
        vector_index: metadata.vector_index_compressed_size,
        text_index: metadata.text_index_compressed_size,
        edge_vector_index: metadata.edge_vector_index_compressed_size,
        edge_text_index: metadata.edge_text_index_compressed_size,
        quantization: metadata.quantization.clone(),
    };
    let mut dir_graph = DirGraph::from_graph(graph);
    dir_graph.interner = interner;
    metadata.apply_to(&mut dir_graph);
    // A mapped load serves the file's member lists and id index when they
    // describe this topology; otherwise (and always into memory storage) they
    // are rebuilt from the nodes.
    let served = match indexes {
        Some(indexes) if indexes.has_types() => {
            dir_graph.rebuild_type_schemas();
            indexes.install_node_indexes(&mut dir_graph)?
        }
        _ => false,
    };
    if !served {
        dir_graph.rebuild_type_indices_and_schemas();
    }
    if is_frozen {
        // The records are read in place, so each must name the row the column
        // loader will bind to it; see `check_rows_against_types`.
        topology_sections::check_rows_against_types(&dir_graph)?;
        if let Some(frozen) = dir_graph.graph.as_frozen() {
            frozen.release_resident();
        }
    }
    dir_graph.build_connection_types_cache();
    Ok((dir_graph, plan))
}

pub(super) fn load_portable_optional_sections(
    codec: serde_codec::CodecVersion,
    core_version: u32,
    dir_graph: &mut DirGraph,
    sections: &mut SectionCursor<'_>,
    plan: &PortableSectionPlan,
    normalization_effects: &legacy_references::NormalizationEffects,
    mappable_table: &mappable::MappableTable,
) -> io::Result<()> {
    if plan.embeddings > 0 {
        if core_version < EMBED_PROVENANCE_MIN_VERSION {
            return Err(invalid_data(EMBED_FORMAT_BREAK_MSG));
        }
        let compressed = sections.take(plan.embeddings, EMBEDDINGS_SECTION)?;
        let raw = zstd_decompress_within(compressed, sections.ceiling())?;
        let mut embeddings: HashMap<(String, String), EmbeddingStore> =
            codec_deser(codec, &raw, raw.capacity() as u64)?;
        // The decompressed payload is as large as the stores it decoded into.
        drop(raw);
        embedding_sections::attach_and_validate_node_stores(
            &mut embeddings,
            mappable_table,
            &plan.quantization,
        )?;
        dir_graph.embeddings = embeddings;
    }
    if plan.edge_embeddings > 0 {
        let compressed = sections.take(plan.edge_embeddings, EDGE_EMBEDDINGS_SECTION)?;
        let raw = zstd_decompress_within(compressed, sections.ceiling())?;
        let decoded: std::collections::BTreeMap<
            crate::graph::edge_embeddings::EdgeEmbeddingKey,
            crate::graph::edge_embeddings::PersistedEdgeEmbeddingStore,
        > = codec_deser(codec, &raw, raw.capacity() as u64)?;
        let mut claimed = std::collections::HashSet::new();
        dir_graph.edge_embeddings =
            crate::graph::edge_embeddings::validate_decoded_edge_embedding_stores_with(
                dir_graph,
                decoded,
                |key, store| {
                    embedding_sections::install_edge_store(
                        mappable_table,
                        &mut claimed,
                        key,
                        store,
                        &plan.quantization,
                    )
                },
            )
            .map_err(|error| invalid_data(format!("edge_embeddings {error}")))?;
        embedding_sections::check_edge_claimed(mappable_table, &claimed)?;
    }
    if plan.timeseries > 0 {
        let compressed = sections.take(plan.timeseries, TIMESERIES_SECTION)?;
        let raw = zstd_decompress_within(compressed, sections.ceiling())?;
        dir_graph.timeseries_store = codec_deser(codec, &raw, raw.capacity() as u64)?;
    }
    if plan.secondary_labels > 0 {
        let compressed = sections.take(plan.secondary_labels, SECONDARY_LABELS_SECTION)?;
        let raw = zstd_decompress_within(compressed, sections.ceiling())?;
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
        if let Ok(raw) = zstd_decompress_within(compressed, sections.ceiling()) {
            decode_vector_indexes(&raw, dir_graph);
        }
    }
    if plan.text_index > 0 {
        // Same split as the vector section above: framing failures are file
        // damage and propagate; the payload itself is a rebuildable cache and
        // an unreadable one is skipped silently (see
        // `decode_text_indexes_after_normalization`).
        let compressed = sections.take(plan.text_index, TEXT_INDEX_SECTION)?;
        if let Ok(raw) = zstd_decompress_within(compressed, sections.ceiling()) {
            decode_text_indexes_after_normalization(&raw, dir_graph, normalization_effects);
        }
    }
    if plan.edge_vector_index > 0 {
        // Same split again; runs after the relationship stores above are
        // installed with their norms, which attaching an index needs.
        let compressed = sections.take(plan.edge_vector_index, EDGE_VECTOR_INDEX_SECTION)?;
        if let Ok(raw) = zstd_decompress_within(compressed, sections.ceiling()) {
            edge_vector_persistence::decode_edge_vector_indexes(&raw, dir_graph);
        }
    }
    if plan.edge_text_index > 0 {
        // The node text section's split: framing failures propagate, an
        // unreadable payload is skipped.
        let compressed = sections.take(plan.edge_text_index, EDGE_TEXT_INDEX_SECTION)?;
        if let Ok(raw) = zstd_decompress_within(compressed, sections.ceiling()) {
            decode_edge_text_indexes(&raw, dir_graph, normalization_effects);
        }
    }
    // Property-sourced vector indexes persist only their declaration (the
    // registry, applied with the metadata); the structure is derived from the
    // finished rows.
    crate::graph::property_vector_indexes::rebuild_declared_property_vectors(dir_graph);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_nonfinite_embedding_is_rejected_before_norm_rebuild() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut store = EmbeddingStore::new(2);
            store.set_embedding(0, &[1.0, bad]);
            let mut stores = HashMap::from([(("Doc".into(), "text_emb".into()), store)]);

            let error = validate_and_rebuild_embedding_norms(&mut stores).unwrap_err();

            assert!(error.to_string().contains("must be finite"));
        }
    }
}

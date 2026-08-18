//! Persistence regression tests extracted from file.rs.

use super::*;

#[cfg(test)]
mod atomic_save_tests {
    use super::*;
    use crate::datatypes::{DataFrame, Value};
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::storage::{GraphRead, GraphWrite};
    use petgraph::graph::NodeIndex;

    /// Add `n` `Doc` nodes to an existing graph, whatever backend it is in.
    fn fill_docs(g: &mut DirGraph, n: i64) {
        let rows: Vec<Vec<Value>> = (1..=n)
            .map(|i| vec![Value::Int64(i), Value::String(format!("t{i}"))])
            .collect();
        let df =
            DataFrame::from_cypher_rows(vec!["id".to_string(), "title".to_string()], rows).unwrap();
        crate::graph::mutation::maintain::add_nodes(
            g,
            df,
            "Doc".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
    }

    /// Stamp + consolidate a filled graph so it is ready for `write_kgl*`.
    fn ready_for_save(g: DirGraph) -> Arc<DirGraph> {
        let mut arc = Arc::new(g);
        prepare_save(&mut arc);
        Arc::make_mut(&mut arc).enable_columnar();
        arc
    }

    /// Build a tiny columnar in-memory graph ready for `write_kgl*`.
    fn tiny_graph(n: i64) -> Arc<DirGraph> {
        let mut g = DirGraph::new();
        fill_docs(&mut g, n);
        ready_for_save(g)
    }

    #[test]
    fn atomic_save_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let g = tiny_graph(5);
        let want = g.graph.node_count();
        write_kgl(&g, path.to_str().unwrap()).unwrap();
        let loaded = load_file(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.graph.node_count(), want);
    }

    #[test]
    fn save_with_fsync_false_still_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let g = tiny_graph(3);
        write_kgl_with(&g, path.to_str().unwrap(), false).unwrap();
        let loaded = load_file(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.graph.node_count(), g.graph.node_count());
    }

    #[test]
    fn to_bytes_roundtrips_via_load_kgl_bytes() {
        let g = tiny_graph(4);
        let mut buf: Vec<u8> = Vec::new();
        write_kgl_to(&g, &mut buf).unwrap();
        assert_eq!(&buf[..4], &V6_MAGIC, "buffer must carry the v6 magic");
        assert_eq!(
            buf[4],
            serde_codec::CodecVersion::PostcardV1.tag(),
            "v6 header must select Postcard explicitly"
        );
        let loaded = load_kgl_bytes(&buf).unwrap();
        assert_eq!(loaded.graph.node_count(), g.graph.node_count());
    }

    /// The v5 container is still decoded. This checks the *dispatch* only —
    /// that a v5 magic reaches the shared reader rather than the
    /// unrecognised-format arm; `tests/test_kgl_v5_compat.py` pins the real
    /// thing against files a published 0.15.14 wheel wrote.
    #[test]
    fn v5_magic_still_reaches_the_shared_reader() {
        let g = tiny_graph(4);
        let mut buf: Vec<u8> = Vec::new();
        write_kgl_to(&g, &mut buf).unwrap();
        buf[3] = V5_MAGIC[3];
        let loaded = load_kgl_bytes(&buf).unwrap();
        assert_eq!(loaded.graph.node_count(), g.graph.node_count());
    }

    #[test]
    fn pre_014_v4_header_is_rejected_with_migration_guidance() {
        let error = load_kgl_bytes(&V4_MAGIC).err().unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("pre-0.14"));
        assert!(error.to_string().contains("0.13.4"));
    }

    #[test]
    fn newer_container_and_invalid_codec_are_rejected_clearly() {
        let newer = [b'R', b'G', b'F', 7];
        let error = load_kgl_bytes(&newer).err().unwrap().to_string();
        assert!(error.contains("version 7") && error.contains("upgrade kglite"));

        // Both readable containers validate the codec byte the same way.
        for version in [5u8, 6u8] {
            let mut invalid = vec![b'R', b'G', b'F', version, 99];
            invalid.extend_from_slice(&CURRENT_CORE_DATA_VERSION.to_le_bytes());
            invalid.extend_from_slice(&0u32.to_le_bytes());
            let error = load_kgl_bytes(&invalid).err().unwrap().to_string();
            assert!(
                error.contains("invalid codec tag"),
                "v{version} must report a bad codec byte, got: {error}"
            );
        }
    }

    /// A v3 (or otherwise unreadable) file is a hard break, but the error must
    /// point the operator at the format-stable export escape hatch so a user
    /// without the original source still has a recovery path (SQLite `.dump`
    /// parity). Guards the recovery hint added to the break messages.
    #[test]
    fn hard_break_errors_point_at_export_recovery() {
        assert!(
            V3_HARD_BREAK_MSG.contains("export_csv")
                && V3_HARD_BREAK_MSG.contains("from_blueprint"),
            "v3 hard-break message must name the export_csv/from_blueprint recovery path"
        );
        // A fabricated v3-magic byte buffer surfaces the hint through load_kgl_bytes.
        let v3_buf = [V3_MAGIC[0], V3_MAGIC[1], V3_MAGIC[2], V3_MAGIC[3], 0, 0];
        let err = load_kgl_bytes(&v3_buf).err().unwrap();
        assert!(err.to_string().contains("export_csv"));
        // An unrecognized *kglite* container carries the hint too — it is a
        // real graph this binary cannot read, so there is something to export.
        let unreadable_container = [V6_MAGIC[0], V6_MAGIC[1], V6_MAGIC[2], 2, 0, 0];
        let err = load_kgl_bytes(&unreadable_container).err().unwrap();
        assert!(err.to_string().contains("from_blueprint"), "{err}");
        // Bytes that are not a kglite container at all get the *opposite*
        // treatment, deliberately (0.16.1): there is no graph to export, so
        // recovery instructions would be advice about a file the user does
        // not have. Rewritten from a pin that asserted the hint here.
        let bad = [0u8, 1, 2, 3, 4, 5];
        let err = load_kgl_bytes(&bad).err().unwrap().to_string();
        assert!(!err.contains("from_blueprint"), "{err}");
        assert!(err.contains("not a kglite graph"), "{err}");
    }

    /// Build a tiny graph carrying one HNSW-indexed embedding store.
    fn tiny_indexed_graph() -> Arc<DirGraph> {
        use crate::graph::algorithms::hnsw::HnswParams;
        use crate::graph::algorithms::vector::DistanceMetric;
        use crate::graph::schema::EmbeddingStore;

        let mut g = tiny_graph(40);
        {
            let dir = Arc::make_mut(&mut g);
            let mut store = EmbeddingStore::with_metric(4, "cosine");
            for i in 0..40usize {
                let v = [i as f32, (i % 3) as f32, 1.0, (i % 7) as f32];
                store.set_embedding(i, &v);
            }
            store
                .build_index(DistanceMetric::Cosine, HnswParams::default(), 7)
                .unwrap();
            dir.embeddings
                .insert(("Doc".to_string(), "vec_emb".to_string()), store);
        }
        g
    }

    #[test]
    fn vector_index_section_roundtrips() {
        let g = tiny_indexed_graph();
        let mut buf: Vec<u8> = Vec::new();
        write_kgl_to(&g, &mut buf).unwrap();
        let loaded = load_kgl_bytes(&buf).unwrap();
        let store = loaded
            .embeddings
            .get(&("Doc".to_string(), "vec_emb".to_string()))
            .expect("embedding store survives round-trip");
        assert!(store.has_index(), "HNSW index must persist in the .kgl");
        assert_eq!(store.index.as_ref().unwrap().len(), 40);
    }

    #[test]
    fn non_default_vector_index_parameters_roundtrip() {
        use crate::graph::algorithms::hnsw::HnswParams;
        use crate::graph::algorithms::vector::DistanceMetric;

        let mut graph = tiny_indexed_graph();
        let key = ("Doc".to_string(), "vec_emb".to_string());
        let params = HnswParams {
            m: 8,
            ef_construction: 80,
            ef_search: 24,
        };
        Arc::make_mut(&mut graph)
            .embeddings
            .get_mut(&key)
            .unwrap()
            .build_index(DistanceMetric::Cosine, params, 91)
            .unwrap();
        let payload = encode_vector_indexes(&graph).unwrap().unwrap();

        let mut destination = tiny_indexed_graph();
        Arc::make_mut(&mut destination)
            .embeddings
            .get_mut(&key)
            .unwrap()
            .index = None;
        decode_vector_indexes(&payload, Arc::make_mut(&mut destination));
        let restored = destination.embeddings[&key]
            .index
            .as_ref()
            .unwrap()
            .params();
        assert_eq!(restored.m, params.m);
        assert_eq!(restored.ef_construction, params.ef_construction);
        assert_eq!(restored.ef_search, params.ef_search);
    }

    #[test]
    fn zero_dimension_embedding_store_roundtrips_without_an_index() {
        use crate::graph::schema::EmbeddingStore;

        let mut graph = tiny_graph(1);
        let mut store = EmbeddingStore::new(0);
        store.set_embedding(0, &[]);
        Arc::make_mut(&mut graph)
            .embeddings
            .insert(("Doc".to_string(), "empty_emb".to_string()), store);

        let mut bytes = Vec::new();
        write_kgl_to(&graph, &mut bytes).unwrap();
        let loaded = load_kgl_bytes(&bytes).unwrap();
        let restored = &loaded.embeddings[&("Doc".to_string(), "empty_emb".to_string())];
        assert_eq!(restored.dimension, 0);
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.get_embedding(0), Some([].as_slice()));
        assert!(!restored.has_index());
    }

    #[test]
    fn corrupt_vector_index_is_skipped_and_exact_search_remains_usable() {
        use crate::graph::algorithms::vector::{
            vector_search, DistanceMetric, VectorSearchOptions,
        };
        use crate::graph::schema::CurrentSelection;
        use petgraph::graph::NodeIndex;

        let mut source = tiny_indexed_graph();
        let key = ("Doc".to_string(), "vec_emb".to_string());
        Arc::make_mut(&mut source)
            .embeddings
            .get_mut(&key)
            .unwrap()
            .index
            .as_mut()
            .unwrap()
            .corrupt_entry_point_for_test();
        let payload = encode_vector_indexes(&source).unwrap().unwrap();

        let mut destination = tiny_indexed_graph();
        Arc::make_mut(&mut destination)
            .embeddings
            .get_mut(&key)
            .unwrap()
            .index = None;
        decode_vector_indexes(&payload, Arc::make_mut(&mut destination));
        let store = destination.embeddings.get(&key).unwrap();
        assert!(
            !store.has_index(),
            "a malformed rebuildable index must not attach to the store"
        );

        let mut selection = CurrentSelection::new();
        selection.get_level_mut(0).unwrap().add_selection(
            None,
            store
                .slot_to_node
                .iter()
                .copied()
                .map(NodeIndex::new)
                .collect(),
        );
        let results = vector_search(
            &destination,
            &selection,
            "vec_emb",
            &[0.0, 0.0, 1.0, 0.0],
            &VectorSearchOptions::default()
                .with_metric(DistanceMetric::Cosine)
                .with_top_k(3)
                .with_exact(true),
        )
        .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].node_idx, NodeIndex::new(0));
    }

    #[test]
    fn pre_014_vector_index_v1_payload_is_skipped() {
        let mut payload = Vec::new();
        payload.extend_from_slice(vector_persistence::VECTOR_INDEX_MAGIC);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&[1, 2, 3]);

        let mut destination = tiny_indexed_graph();
        for store in Arc::make_mut(&mut destination).embeddings.values_mut() {
            store.index = None;
        }
        decode_vector_indexes(&payload, Arc::make_mut(&mut destination));
        assert!(!destination
            .embeddings
            .get(&("Doc".to_string(), "vec_emb".to_string()))
            .unwrap()
            .has_index());
    }

    #[test]
    fn vector_index_decode_skips_unknown_version() {
        // The section is a rebuildable cache: an unknown format version (or a
        // corrupt magic) must be skipped silently, never attached, never panic.
        let g = tiny_indexed_graph();
        let payload = encode_vector_indexes(&g).unwrap().unwrap();

        let mut bumped = payload.clone();
        bumped[8] = bumped[8].wrapping_add(1); // mangle the format-version LSB
        let mut dst = DirGraph::new();
        dst.embeddings.insert(
            ("Doc".to_string(), "vec_emb".to_string()),
            crate::graph::schema::EmbeddingStore::new(4),
        );
        decode_vector_indexes(&bumped, &mut dst);
        assert!(
            !dst.embeddings[&("Doc".to_string(), "vec_emb".to_string())].has_index(),
            "an unknown index format version must be skipped"
        );

        let mut bad_magic = payload.clone();
        bad_magic[0] = b'X';
        decode_vector_indexes(&bad_magic, &mut dst);
        assert!(!dst.embeddings[&("Doc".to_string(), "vec_emb".to_string())].has_index());
    }

    /// Build an equivalent embedding+timeseries graph; `reverse` flips every
    /// map-insertion order that must NOT affect the serialized bytes. Vector
    /// insertion order is kept identical in both builds — slot layout is
    /// legitimately order-dependent; the maps' internal ordering is not.
    fn equivalent_embedding_graph(reverse: bool) -> Arc<DirGraph> {
        use crate::graph::features::timeseries::NodeTimeseries;
        use crate::graph::schema::EmbeddingStore;
        use std::collections::HashMap;

        let mut g = tiny_graph(40);
        let dir = Arc::make_mut(&mut g);

        let mut store_names = vec!["vec_emb", "alt_emb"];
        if reverse {
            store_names.reverse();
        }
        for name in store_names {
            let mut store = EmbeddingStore::with_metric(4, "cosine");
            for i in 0..40usize {
                let v = [i as f32, (i % 3) as f32, 1.0, (i % 7) as f32];
                store.set_embedding(i, &v);
            }
            let mut hash_order: Vec<usize> = (0..40).collect();
            if reverse {
                hash_order.reverse();
            }
            for i in hash_order {
                store.text_hashes.insert(i, (i as u64).wrapping_mul(0x9e37));
            }
            dir.embeddings
                .insert(("Doc".to_string(), name.to_string()), store);
        }

        let mut node_order: Vec<usize> = (0..8).collect();
        if reverse {
            node_order.reverse();
        }
        for n in node_order {
            let mut channels = HashMap::new();
            let mut channel_names = vec!["plays", "skips", "stars"];
            if reverse {
                channel_names.reverse();
            }
            for c in channel_names {
                channels.insert(c.to_string(), vec![n as f64, 2.0]);
            }
            dir.timeseries_store.insert(
                n,
                NodeTimeseries {
                    keys: vec![
                        chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                        chrono::NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(),
                    ],
                    channels,
                },
            );
        }

        // Force the internal Vec<(InternedKey, Value)> into opposite orders.
        // This bypasses HashMap construction so the regression specifically
        // covers EdgeData's map-shaped topology serialization.
        let connection_type = dir.interner.get_or_intern("RELATES_TO");
        let mut edge_properties = vec![
            (
                dir.interner.get_or_intern("confidence"),
                Value::Float64(0.75),
            ),
            (
                dir.interner.get_or_intern("source"),
                Value::String("fixture".to_string()),
            ),
        ];
        if reverse {
            edge_properties.reverse();
        }
        dir.graph.add_edge(
            NodeIndex::new(0),
            NodeIndex::new(1),
            crate::graph::schema::EdgeData::new_interned(connection_type, edge_properties),
        );
        g
    }

    #[test]
    fn kgl_bytes_are_deterministic_across_equivalent_builds() {
        // Regression for sonagram's byte-determinism report (2026-07-20):
        // separately-constructed but equivalent graphs must produce identical
        // `.kgl` bytes. Each HashMap instance carries its own RandomState, so
        // even identical insertion orders iterate differently — serialization
        // must canonicalize (sorted maps) rather than rely on iteration order.
        let mut first = Vec::new();
        write_kgl_to(&equivalent_embedding_graph(false), &mut first).unwrap();
        let mut second = Vec::new();
        write_kgl_to(&equivalent_embedding_graph(true), &mut second).unwrap();
        assert_eq!(
            first, second,
            ".kgl bytes must not depend on HashMap insertion or iteration order"
        );

        let loaded = load_kgl_bytes(&first).unwrap();
        let edge = loaded.graph.edge_weights().next().unwrap();
        assert_eq!(edge.get_property("confidence"), Some(&Value::Float64(0.75)));
        assert_eq!(
            edge.get_property("source"),
            Some(&Value::String("fixture".to_string()))
        );
    }

    #[test]
    fn load_kgl_bytes_rejects_bad_magic() {
        let err = match load_kgl_bytes(b"NOPE and some trailing bytes that are long enough") {
            Ok(_) => panic!("expected an error for a bad-magic buffer"),
            Err(e) => e.to_string().to_lowercase(),
        };
        assert!(
            err.contains("magic") || err.contains("unrecognized"),
            "got: {err}"
        );
    }

    #[test]
    fn load_kgl_bytes_rejects_too_small() {
        assert!(load_kgl_bytes(b"RG").is_err());
        assert!(load_kgl_bytes(&[]).is_err());
    }

    #[test]
    fn load_kgl_bytes_rejects_truncated() {
        let g = tiny_graph(6);
        let mut buf: Vec<u8> = Vec::new();
        write_kgl_to(&g, &mut buf).unwrap();
        // Keep the valid magic+header but cut the body — a torn file.
        let truncated = &buf[..buf.len() / 2];
        assert!(
            load_kgl_bytes(truncated).is_err(),
            "a truncated buffer must be rejected, not silently half-loaded"
        );
    }

    fn rewrite_metadata(buf: &[u8], mutate: impl FnOnce(&mut FileMetadata)) -> Vec<u8> {
        assert_eq!(&buf[..4], &V6_MAGIC);
        let old_len = u32::from_le_bytes(buf[9..13].try_into().unwrap()) as usize;
        let mut metadata: FileMetadata = serde_json::from_slice(&buf[13..13 + old_len]).unwrap();
        mutate(&mut metadata);
        let encoded = serde_json::to_vec(&metadata).unwrap();
        let mut rewritten = Vec::with_capacity(buf.len() - old_len + encoded.len());
        rewritten.extend_from_slice(&buf[..9]);
        rewritten.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        rewritten.extend_from_slice(&encoded);
        rewritten.extend_from_slice(&buf[13 + old_len..]);
        rewritten
    }

    fn assert_invalid_without_panic(bytes: &[u8]) {
        let result = std::panic::catch_unwind(|| load_kgl_bytes(bytes));
        let error = match result.expect("malformed .kgl must return an error, not panic") {
            Ok(_) => panic!("malformed .kgl must not load successfully"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
    }

    #[test]
    fn malformed_section_metadata_is_checked_without_panics() {
        let graph = tiny_graph(2);
        let mut valid = Vec::new();
        write_kgl_to(&graph, &mut valid).unwrap();

        let oversized_topology = rewrite_metadata(&valid, |m| {
            m.topology_compressed_size = u64::MAX;
        });
        assert_invalid_without_panic(&oversized_topology);

        let oversized_column = rewrite_metadata(&valid, |m| {
            m.column_sections[0].compressed_size = u64::MAX;
        });
        assert_invalid_without_panic(&oversized_column);

        let oversized_rows = rewrite_metadata(&valid, |m| {
            m.column_sections[0].row_count = u32::MAX;
        });
        assert_invalid_without_panic(&oversized_rows);

        assert_invalid_without_panic(&valid[..valid.len() - 1]);
    }

    #[test]
    fn serialized_type_names_never_become_temp_paths() {
        let graph = tiny_graph(1);
        let mut valid = Vec::new();
        write_kgl_to(&graph, &mut valid).unwrap();
        for hostile in ["../../outside", "/tmp/kglite-absolute-type"] {
            let mutated = rewrite_metadata(&valid, |m| {
                m.column_sections[0].type_name = hostile.to_string();
            });
            assert_invalid_without_panic(&mutated);
        }
    }

    #[test]
    fn zstd_decompression_respects_expansion_limit() {
        let compressed = zstd_compress(&vec![0u8; 64 * 1024]).unwrap();
        let error = zstd_decompress_limited(&compressed, 1024).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn retained_flat_csr_index_readers_validate_exact_bounds_and_cardinality() {
        let mut interner = crate::graph::storage::interner::StringInterner::new();
        let key = interner.get_or_intern("Person").as_u64();

        let mut type_payload = Vec::new();
        type_payload.extend_from_slice(TYPE_INDICES_MAGIC);
        type_payload.extend_from_slice(&TYPE_INDICES_VERSION.to_le_bytes());
        type_payload.extend_from_slice(&1u32.to_le_bytes());
        type_payload.extend_from_slice(&1u64.to_le_bytes());
        type_payload.extend_from_slice(&key.to_le_bytes());
        type_payload.extend_from_slice(&0u64.to_le_bytes());
        type_payload.extend_from_slice(&1u64.to_le_bytes());
        type_payload.extend_from_slice(&7u32.to_le_bytes());
        assert!(read_type_indices_bin(&type_payload, &interner)
            .unwrap()
            .is_some());
        type_payload.push(0);
        assert_eq!(
            read_type_indices_bin(&type_payload, &interner)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );

        let mut id_payload = Vec::new();
        id_payload.extend_from_slice(ID_INDICES_MAGIC);
        id_payload.extend_from_slice(&ID_INDICES_VERSION.to_le_bytes());
        id_payload.extend_from_slice(&1u32.to_le_bytes());
        id_payload.extend_from_slice(&key.to_le_bytes());
        id_payload.push(0);
        id_payload.extend_from_slice(&[0; 7]);
        id_payload.extend_from_slice(&1u64.to_le_bytes());
        id_payload.extend_from_slice(&7u32.to_le_bytes());
        id_payload.extend_from_slice(&3u32.to_le_bytes());
        assert!(read_id_indices_bin(&id_payload, &interner)
            .unwrap()
            .is_some());
        id_payload.push(0);
        assert_eq!(
            read_id_indices_bin(&id_payload, &interner)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn atomic_save_overwrites_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        let p = path.to_str().unwrap();
        write_kgl(&tiny_graph(2), p).unwrap();
        write_kgl(&tiny_graph(9), p).unwrap();
        let loaded = load_file(p).unwrap();
        assert_eq!(loaded.graph.node_count(), tiny_graph(9).graph.node_count());
    }

    #[test]
    fn successful_save_leaves_no_temp_litter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.kgl");
        write_kgl(&tiny_graph(3), path.to_str().unwrap()).unwrap();
        // Only the destination should remain — no `.tmp.<pid>.<n>` siblings.
        let entries: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["g.kgl".to_string()], "temp file must be gone");
    }

    #[test]
    fn failed_save_to_bad_dir_leaves_dest_untouched() {
        // Write a good file first, then attempt a save into a path whose
        // parent doesn't exist — the temp create fails, and the existing
        // good file must be left intact (no partial overwrite).
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("g.kgl");
        write_kgl(&tiny_graph(4), good.to_str().unwrap()).unwrap();
        let before = std::fs::read(&good).unwrap();

        let bad = dir.path().join("missing_subdir").join("g.kgl");
        assert!(write_kgl(&tiny_graph(7), bad.to_str().unwrap()).is_err());

        // The original file is byte-for-byte unchanged.
        assert_eq!(std::fs::read(&good).unwrap(), before);
    }

    // ── user-schema version stamp ───────────────────────────────────────────
    //
    // The stamp is the caller's data-model revision, not an engine version.
    // Two properties matter and are asserted below: it survives save/load, and
    // it is *completely absent* from the byte stream at the baseline value — so
    // a `.kgl` written before the field existed loads without error, and every
    // pre-existing save stays byte-identical (the golden-digest invariant).

    /// Rewrite a v5 buffer's metadata as raw JSON, so a test can delete a key
    /// outright. `rewrite_metadata` above round-trips through the typed struct
    /// and would re-add any key it knows about; this simulates a file written
    /// by a build whose `FileMetadata` never had the field at all.
    fn rewrite_metadata_json(buf: &[u8], mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        assert_eq!(&buf[..4], &V6_MAGIC);
        let old_len = u32::from_le_bytes(buf[9..13].try_into().unwrap()) as usize;
        let mut raw: serde_json::Value =
            serde_json::from_slice(&buf[13..13 + old_len]).expect("metadata is JSON");
        mutate(&mut raw);
        let encoded = serde_json::to_vec(&raw).unwrap();
        let mut out = Vec::with_capacity(buf.len() - old_len + encoded.len());
        out.extend_from_slice(&buf[..9]);
        out.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        out.extend_from_slice(&encoded);
        out.extend_from_slice(&buf[13 + old_len..]);
        out
    }

    #[test]
    fn user_schema_version_survives_save_and_load() {
        let mut graph = tiny_graph(3);
        Arc::make_mut(&mut graph).user_schema_version = 7;
        let mut bytes = Vec::new();
        write_kgl_to(&graph, &mut bytes).unwrap();

        let loaded = load_kgl_bytes(&bytes).unwrap();
        assert_eq!(
            loaded.user_schema_version, 7,
            "the caller's schema revision must round-trip through .kgl"
        );
    }

    #[test]
    fn unstamped_graph_writes_no_user_schema_version_key() {
        // The baseline value must leave no trace in the metadata JSON: that is
        // what makes the field additive for readers AND byte-neutral for the
        // save-determinism digest.
        let graph = tiny_graph(3);
        assert_eq!(graph.user_schema_version, 0, "fresh graphs are unversioned");
        let mut bytes = Vec::new();
        write_kgl_to(&graph, &mut bytes).unwrap();

        let len = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
        let raw: serde_json::Value = serde_json::from_slice(&bytes[13..13 + len]).unwrap();
        assert!(
            raw.get("user_schema_version").is_none(),
            "an unstamped graph must not emit the key at all, got {raw}"
        );
    }

    #[test]
    fn file_without_user_schema_version_loads_as_unversioned() {
        // Simulates a `.kgl` written by a build predating the field: the key is
        // simply not there. It must load cleanly at the baseline — never an
        // error, and never a value read out of some neighbouring field.
        let mut graph = tiny_graph(3);
        Arc::make_mut(&mut graph).user_schema_version = 11;
        let mut bytes = Vec::new();
        write_kgl_to(&graph, &mut bytes).unwrap();

        let stripped = rewrite_metadata_json(&bytes, |raw| {
            let object = raw.as_object_mut().expect("metadata is a JSON object");
            assert!(
                object.remove("user_schema_version").is_some(),
                "the stamped graph should have written the key"
            );
        });

        let loaded = load_kgl_bytes(&stripped).expect("an older .kgl must still load");
        assert_eq!(
            loaded.user_schema_version, 0,
            "a missing stamp means unversioned, not an error and not garbage"
        );
        assert_eq!(
            loaded.graph.node_count(),
            3,
            "the rest of the graph must be unaffected"
        );
    }

    // ── recorded storage mode ───────────────────────────────────────────────
    //
    // A saved graph records the storage mode that wrote it, so a later open can
    // tell a mapped checkpoint from a memory one. Three properties are asserted
    // below: the mode is written for every portable-capable mode, a file that
    // predates the field still loads (memory, the established fallback), and an
    // unrecognised value is refused **by name** rather than quietly becoming
    // memory — a silent fallback would hand back a graph in a mode nobody asked
    // for, which is indistinguishable from success.

    /// Parse the metadata JSON out of a v5 buffer, so a test can assert on the
    /// bytes actually written rather than on a round-tripped struct.
    fn metadata_json_of(buf: &[u8]) -> serde_json::Value {
        assert_eq!(&buf[..4], &V6_MAGIC);
        let len = u32::from_le_bytes(buf[9..13].try_into().unwrap()) as usize;
        serde_json::from_slice(&buf[13..13 + len]).expect("metadata is JSON")
    }

    /// Build a tiny columnar graph in `mode`, ready for `write_kgl*`.
    fn tiny_graph_in_mode(mode: crate::graph::storage::mode::StorageMode, n: i64) -> Arc<DirGraph> {
        let mut g = crate::graph::storage::mode::new_dir_graph_in_mode(mode, None)
            .expect("portable-capable mode creates without a path");
        fill_docs(&mut g, n);
        ready_for_save(g)
    }

    fn saved_bytes(graph: &Arc<DirGraph>) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_kgl_to(graph, &mut bytes).unwrap();
        bytes
    }

    #[test]
    fn memory_save_omits_the_storage_mode_key() {
        // Memory is the baseline: it writes no key at all, exactly like
        // `user_schema_version` at 0. That is what keeps a memory `.kgl`
        // byte-identical to one written before the field existed (and keeps the
        // save-determinism digest stable), and it is why an absent key means
        // memory on the read side.
        let bytes = saved_bytes(&tiny_graph(3));
        let raw = metadata_json_of(&bytes);
        assert!(
            raw.get("storage_mode").is_none(),
            "a memory graph must not emit the key at all, got {raw}"
        );
        assert_eq!(load_kgl_bytes(&bytes).unwrap().graph.node_count(), 3);
    }

    /// Every `Doc` row, in id order — the value equality a mode switch must
    /// preserve. Read through Cypher so the assertion goes the whole way from
    /// the backend to a materialized row rather than poking at storage.
    fn doc_rows(graph: &DirGraph) -> Vec<Vec<Value>> {
        let params = std::collections::HashMap::new();
        crate::graph::session::execute_read(
            graph,
            "MATCH (n:Doc) RETURN n.id, n.title ORDER BY n.id",
            &crate::graph::session::ExecuteOptions::eager(&params),
        )
        .expect("Doc read")
        .result
        .rows
    }

    #[test]
    fn mapped_saved_file_reopens_mapped_with_identical_rows() {
        // The mode a checkpoint recorded is the mode it comes back in — the
        // whole point of recording it. A mapped-saved graph that reopened as
        // memory is what silently invalidated a mapped-vs-memory comparison.
        let saved = tiny_graph_in_mode(crate::graph::storage::mode::StorageMode::Mapped, 4);
        let loaded = load_kgl_bytes(&saved_bytes(&saved)).unwrap();
        assert!(
            loaded.graph.is_mapped(),
            "a mapped-saved .kgl must reopen mapped, not memory"
        );
        assert_eq!(
            loaded.memory_limit,
            Some(0),
            "the reopened graph must carry mapped mode's spill policy, not just its backend"
        );
        assert_eq!(
            doc_rows(&loaded),
            doc_rows(&saved),
            "rows must be identical"
        );
    }

    #[test]
    fn a_converted_graph_saves_and_reopens_in_its_new_mode() {
        // The conversion has to survive the round trip, or `storage="mapped"`
        // on an existing file would silently revert on the next reopen — the
        // original defect one level down.
        let mut graph = tiny_graph(3);
        let before = doc_rows(&graph);
        crate::graph::storage::mode::convert_dir_graph_to_mode(
            Arc::make_mut(&mut graph),
            crate::graph::storage::mode::StorageMode::Mapped,
        )
        .unwrap();

        let reloaded = load_kgl_bytes(&saved_bytes(&graph)).unwrap();
        assert!(reloaded.graph.is_mapped(), "the new mode must be persisted");
        assert_eq!(doc_rows(&reloaded), before);
    }

    #[test]
    fn memory_saved_and_pre_field_files_still_reopen_as_memory() {
        let memory_saved = tiny_graph(3);
        let loaded = load_kgl_bytes(&saved_bytes(&memory_saved)).unwrap();
        assert!(!loaded.graph.is_mapped() && !loaded.graph.is_disk());
        assert_eq!(loaded.memory_limit, None);
        assert_eq!(doc_rows(&loaded), doc_rows(&memory_saved));

        // A file written before the field existed carries no key at all, and
        // must keep landing in memory rather than inheriting anything.
        let stripped = rewrite_metadata_json(
            &saved_bytes(&tiny_graph_in_mode(
                crate::graph::storage::mode::StorageMode::Mapped,
                3,
            )),
            |raw| {
                raw.as_object_mut().unwrap().remove("storage_mode");
            },
        );
        let old = load_kgl_bytes(&stripped).expect("an older .kgl must still load");
        assert!(!old.graph.is_mapped(), "an unrecorded mode means memory");
        assert_eq!(old.graph.node_count(), 3);
    }

    #[test]
    fn mapped_save_records_the_mapped_mode() {
        let graph = tiny_graph_in_mode(crate::graph::storage::mode::StorageMode::Mapped, 4);
        assert!(
            graph.graph.is_mapped(),
            "mapped is a portable-capable mode and must reach write_kgl in that mode"
        );
        let bytes = saved_bytes(&graph);
        assert_eq!(
            metadata_json_of(&bytes)["storage_mode"],
            serde_json::json!("mapped"),
            "a mapped graph must record the mode that wrote the checkpoint"
        );

        // The recorded key is what the reopen reads back (asserted end-to-end
        // in `mapped_saved_file_reopens_mapped_with_identical_rows`).
        assert_eq!(load_kgl_bytes(&bytes).unwrap().graph.node_count(), 4);
    }

    #[test]
    fn durable_mapped_save_still_records_the_mapped_mode() {
        // `open(path, storage="mapped", durable=True)` is the shape the storage
        // guide recommends, and it saves through a `Recording`-wrapped backend.
        // The wrapper is transparent to the mode, so the checkpoint must still
        // say `mapped` — recording `memory` here would send the graph back as a
        // memory graph on every later reopen.
        let mut g = crate::graph::storage::mode::new_dir_graph_in_mode(
            crate::graph::storage::mode::StorageMode::Mapped,
            None,
        )
        .unwrap();
        // Wrap exactly as `setup_durable` does, so the seam under test is real.
        let inner = std::mem::replace(&mut g.graph, crate::graph::schema::GraphBackend::new());
        g.graph = crate::graph::schema::GraphBackend::Recording(Box::new(
            crate::graph::storage::recording::RecordingGraph::new(inner),
        ));
        fill_docs(&mut g, 2);
        let graph = ready_for_save(g);
        assert!(graph.graph.is_mapped());
        assert_eq!(
            metadata_json_of(&saved_bytes(&graph))["storage_mode"],
            serde_json::json!("mapped"),
            "the durability wrapper must not hide the mode underneath it"
        );
    }

    #[test]
    fn file_without_storage_mode_loads_as_memory() {
        // Simulates a `.kgl` written by a build predating the field: the key is
        // simply not there. It must load cleanly as memory — the established
        // fallback — never an error.
        let bytes = saved_bytes(&tiny_graph_in_mode(
            crate::graph::storage::mode::StorageMode::Mapped,
            4,
        ));
        let stripped = rewrite_metadata_json(&bytes, |raw| {
            let object = raw.as_object_mut().expect("metadata is a JSON object");
            assert!(
                object.remove("storage_mode").is_some(),
                "the mapped graph should have written the key"
            );
        });
        let loaded = load_kgl_bytes(&stripped).expect("an older .kgl must still load");
        assert_eq!(loaded.graph.node_count(), 4);
        assert!(!loaded.graph.is_mapped());

        // An explicitly recorded `memory` is equally legitimate — this build
        // omits it, but the vocabulary is shared with every other binding and a
        // reader must accept the spelled-out form.
        let explicit = rewrite_metadata_json(&bytes, |raw| {
            raw["storage_mode"] = serde_json::json!("memory");
        });
        assert_eq!(load_kgl_bytes(&explicit).unwrap().graph.node_count(), 4);
    }

    #[test]
    fn unrecognised_storage_mode_is_refused_by_name() {
        let bytes = saved_bytes(&tiny_graph(2));
        let corrupt = rewrite_metadata_json(&bytes, |raw| {
            raw["storage_mode"] = serde_json::json!("qubit");
        });
        let error = load_kgl_bytes(&corrupt)
            .err()
            .expect("an unknown storage mode must not silently load as memory");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
        let text = error.to_string();
        assert!(
            text.contains("qubit") && text.contains("storage mode"),
            "the error must name the value it rejected: {text}"
        );
    }

    #[test]
    fn portable_file_claiming_disk_mode_is_refused() {
        // A disk graph is a directory, never a portable file: `save_disk`
        // refuses a non-disk backend and `GraphBackend`'s serializer refuses the
        // disk arm outright, so no writer can produce this. A file that claims
        // it anyway is corrupt and must not be reinterpreted as a portable one.
        let bytes = saved_bytes(&tiny_graph(2));
        let corrupt = rewrite_metadata_json(&bytes, |raw| {
            raw["storage_mode"] = serde_json::json!("disk");
        });
        let error = load_kgl_bytes(&corrupt)
            .err()
            .expect("a portable file claiming disk mode must be refused");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData, "{error}");
        let text = error.to_string();
        assert!(
            text.contains("disk") && text.contains("director"),
            "the error must say a disk graph is a directory: {text}"
        );
    }

    #[test]
    fn disk_directory_records_disk_mode_and_refuses_a_portable_claim() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().to_str().unwrap();
        let mut graph = DirGraph::new();
        fill_docs(&mut graph, 3);
        graph.enable_disk_mode().unwrap();
        graph.save_disk(path).unwrap();

        let snapshot = crate::graph::storage::disk::generation::resolve_snapshot(root.path())
            .unwrap()
            .snapshot_dir;
        let meta_path = snapshot.join("metadata.json");
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        assert_eq!(
            raw["storage_mode"],
            serde_json::json!("disk"),
            "a disk directory must record the mode that wrote it"
        );
        assert_eq!(load_file(path).unwrap().graph.node_count(), 3);

        // The symmetric guard: a disk directory whose metadata claims a portable
        // mode is corrupt — no writer can produce it, and loading it anyway
        // would mean trusting a file that contradicts its own layout.
        let mut mutated = raw.clone();
        mutated["storage_mode"] = serde_json::json!("mapped");
        std::fs::write(&meta_path, serde_json::to_vec(&mutated).unwrap()).unwrap();
        let error = load_file(path)
            .err()
            .expect("a disk directory claiming a portable mode must be refused");
        assert!(
            error.to_string().contains("mapped"),
            "the error must name the value it rejected: {error}"
        );
    }
}

/// Saving **while a lazy view holds the graph** — the `GraphBackend::Forked`
/// arm of `Serialize`, which is the only caller of
/// `ForkedGraph::to_memory_graph`.
///
/// Nothing exercised it before 2026-08-15: every save test in this file owns
/// its graph outright, so the serializer always took the `Memory` arm. The
/// forked arm is a *different* code path — it folds the overlay into a
/// throwaway deep copy of the shared base — and getting it wrong is silent:
/// the save succeeds, the file is well-formed, and it is simply missing (or
/// duplicating) whatever the writer did while the view was outstanding.
#[cfg(test)]
mod save_while_forked_tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::handle::make_dir_graph_mut;
    use crate::graph::session::execute::{execute_mut, ExecuteOptions};
    use crate::graph::storage::GraphRead;
    use std::collections::HashMap;

    fn run(graph: &mut DirGraph, query: &str) {
        let params = HashMap::new();
        let opts = ExecuteOptions::eager(&params);
        execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("query failed: {query}: {e}"));
    }

    /// Every `Item` as `(id, title, qty)`, sorted — read through `node_view`,
    /// so it resolves the column store rather than a bare `NodeData` field.
    fn items(graph: &DirGraph) -> Vec<(Value, Value, Option<Value>)> {
        let mut out: Vec<_> = graph
            .graph
            .node_indices()
            .filter_map(|idx| graph.graph.node_view(idx))
            .map(|node| {
                (
                    node.id().into_owned(),
                    node.title().into_owned(),
                    node.get_property_value("qty"),
                )
            })
            .collect();
        out.sort_by_key(|(id, _, _)| format!("{id:?}"));
        out
    }

    /// A `.kgl` written from a forked backend must carry the **writer's**
    /// content — the overlay's appended nodes and its copy-on-write edits —
    /// and the view must be untouched by the save.
    ///
    /// Both halves matter and neither implies the other. An overlay dropped on
    /// the floor writes the *view's* graph under the writer's name (lost
    /// writes); an overlay folded into the shared base instead of a copy
    /// writes the right file and corrupts the view (`to_memory_graph`'s
    /// `deep_clone` is what separates them, and it is one word away from
    /// `Arc::clone`).
    #[test]
    fn a_save_while_a_view_is_held_writes_the_writers_graph_and_leaves_the_view_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("forked.kgl");
        let path_str = path.to_str().unwrap();

        let mut base = DirGraph::new();
        run(
            &mut base,
            "CREATE (:Item {id: 1, name: 'a', qty: 10}), (:Item {id: 2, name: 'b', qty: 20})",
        );
        let mut writer = Arc::new(base);
        // The lazy view: an `Arc` handle held across the write *and* the save.
        let view = Arc::clone(&writer);
        let view_before = items(&view);
        assert_eq!(view_before.len(), 2, "fixture");

        {
            let graph = make_dir_graph_mut(&mut writer);
            assert!(
                graph.graph.is_forked(),
                "precondition: a held view must fork the writer, or this test saves a \
                 plain backend and proves nothing"
            );
            run(graph, "MATCH (n:Item {id: 1}) SET n.qty = 999");
            run(graph, "CREATE (:Item {id: 3, name: 'c', qty: 30})");
        }

        // `save_inmemory_with` decomposed, so the precondition can be asserted
        // between its two halves: the consolidation pass must leave the backend
        // forked, or `Serialize`'s `Forked` arm never runs.
        prepare_kgl_write(&mut writer);
        assert!(
            writer.graph.is_forked(),
            "precondition: the graph handed to the serializer must still be an overlay"
        );
        let want = items(&writer);
        write_kgl(&writer, path_str).unwrap();

        // Non-vacuity: the writer's content is *different* from the view's in
        // both directions an overlay can differ — an appended node and an
        // overwritten cell.
        assert_eq!(want.len(), 3, "the overlay's appended node must be there");
        assert!(
            want.iter()
                .any(|(id, _, qty)| *id == Value::Int64(1) && *qty == Some(Value::Int64(999))),
            "the overlay's copy-on-write edit must be there: {want:?}"
        );
        assert_ne!(want, view_before);

        let loaded = load_file(path_str).unwrap();
        assert_eq!(
            items(&loaded),
            want,
            "a save taken while a view is held must persist the writer's graph, \
             overlay included"
        );

        assert_eq!(
            items(&writer),
            want,
            "the save must not consume the overlay it serialized — `to_memory_graph` \
             folds into a *copy*, and folding in place (the tempting way to skip that \
             copy) would empty the live writer as a side effect of saving it"
        );
        assert_eq!(
            items(&view),
            view_before,
            "the held view must be byte-identical after the writer saved — a save that \
             folded the overlay into the shared base instead of a copy would show up here"
        );
    }
}

/// A DDL-declared NOT NULL constraint lives in two places: the enforced list
/// (`SchemaDefinition::required_fields`) and the provenance record
/// (`DirGraph::ddl_not_null_constraints`) that says *who* declared it. Only the
/// provenance record distinguishes a constraint the user wrote in Cypher from
/// one an incoming `define_schema` may replace, so it has to survive a save.
#[cfg(test)]
mod ddl_provenance_roundtrip_tests {
    use super::*;
    use crate::datatypes::{DataFrame, Value};
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition, SchemaInstall};

    /// `Person` nodes with an email on every row, so a NOT NULL declaration on
    /// `email` installs cleanly.
    fn person_graph() -> DirGraph {
        let mut graph = DirGraph::new();
        let rows: Vec<Vec<Value>> = (1..=3)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("p{i}")),
                    Value::String(format!("p{i}@example.com")),
                ]
            })
            .collect();
        let df = DataFrame::from_cypher_rows(
            vec!["id".to_string(), "title".to_string(), "email".to_string()],
            rows,
        )
        .unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            df,
            "Person".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        graph
    }

    /// A schema that declares `Person` but says nothing about `email` — the
    /// shape an unrelated `define_schema()` call has.
    fn schema_without_email() -> SchemaDefinition {
        let mut schema = SchemaDefinition::new();
        schema
            .node_schemas
            .insert("Person".to_string(), NodeSchemaDefinition::default());
        schema
    }

    fn save_and_load(graph: DirGraph, dir: &std::path::Path) -> DirGraph {
        let path = dir.join("g.kgl");
        let mut arc = Arc::new(graph);
        prepare_save(&mut arc);
        Arc::make_mut(&mut arc).enable_columnar();
        write_kgl(&arc, path.to_str().unwrap()).unwrap();
        Arc::unwrap_or_clone(load_file(path.to_str().unwrap()).unwrap())
    }

    /// Before the save, the provenance record protects the declaration from an
    /// unrelated schema install. This pins the behaviour the round-trip below
    /// has to preserve — without it, a regression could make the round-trip
    /// test pass by breaking the protection everywhere.
    #[test]
    fn a_declaration_survives_an_unrelated_schema_install_in_memory() {
        let mut graph = person_graph();
        graph.create_not_null_constraint("Person", "email").unwrap();

        graph
            .set_schema(schema_without_email(), SchemaInstall::Replace)
            .unwrap();
        assert!(
            graph.has_not_null_constraint("Person", "email"),
            "an unrelated define_schema must not withdraw a DDL-declared NOT NULL"
        );
    }

    /// The regression: the provenance record is rebuilt from `FileMetadata` on
    /// load, so a field missing from that struct comes back empty and the
    /// declaration silently loses its protection — a `define_schema()` after a
    /// reload un-enforces a constraint the user declared in Cypher, with no
    /// error anywhere.
    #[test]
    fn a_declaration_keeps_its_provenance_across_a_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let mut graph = person_graph();
        graph.create_not_null_constraint("Person", "email").unwrap();

        let mut loaded = save_and_load(graph, dir.path());

        assert!(
            loaded.has_not_null_constraint("Person", "email"),
            "the enforced list must survive the round-trip"
        );
        loaded
            .set_schema(schema_without_email(), SchemaInstall::Replace)
            .unwrap();
        assert!(
            loaded.has_not_null_constraint("Person", "email"),
            "after a reload, an unrelated define_schema silently un-enforced a \
             DDL-declared NOT NULL"
        );
        assert!(
            loaded
                .ddl_not_null_constraints
                .contains(&("Person".to_string(), "email".to_string())),
            "the DDL provenance record must survive the round-trip"
        );
    }

    /// A graph that declares nothing must write the same bytes it wrote before
    /// the provenance field existed, or every `.kgl` in the world shifts format
    /// for a feature almost no graph uses.
    #[test]
    fn an_undeclared_graph_writes_no_provenance_into_the_metadata() {
        let metadata = FileMetadata::from_graph(&person_graph());
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(
            !json.contains("ddl_not_null_constraints"),
            "the empty set must be skipped, or the golden byte digest moves: {json}"
        );

        let mut declared = person_graph();
        declared
            .create_not_null_constraint("Person", "email")
            .unwrap();
        let json = serde_json::to_string(&FileMetadata::from_graph(&declared)).unwrap();
        assert!(
            json.contains("ddl_not_null_constraints"),
            "a declared constraint must be written: {json}"
        );
    }
}

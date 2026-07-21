//! Persistence regression tests extracted from file.rs.

use super::*;

#[cfg(test)]
mod atomic_save_tests {
    use super::*;
    use crate::datatypes::{DataFrame, Value};
    use crate::graph::dir_graph::DirGraph;
    use crate::graph::storage::{GraphRead, GraphWrite};
    use petgraph::graph::NodeIndex;

    /// Build a tiny columnar in-memory graph ready for `write_kgl*`.
    fn tiny_graph(n: i64) -> Arc<DirGraph> {
        let mut g = DirGraph::new();
        let rows: Vec<Vec<Value>> = (1..=n)
            .map(|i| vec![Value::Int64(i), Value::String(format!("t{i}"))])
            .collect();
        let df =
            DataFrame::from_cypher_rows(vec!["id".to_string(), "title".to_string()], rows).unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut g,
            df,
            "Doc".to_string(),
            "id".to_string(),
            Some("title".to_string()),
            None,
        )
        .unwrap();
        let mut arc = Arc::new(g);
        prepare_save(&mut arc);
        Arc::make_mut(&mut arc).enable_columnar();
        arc
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
        assert_eq!(&buf[..4], &V5_MAGIC, "buffer must carry the v5 magic");
        assert_eq!(
            buf[4],
            serde_codec::CodecVersion::PostcardV1.tag(),
            "v5 header must select Postcard explicitly"
        );
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
    fn newer_container_and_invalid_v5_codec_are_rejected_clearly() {
        let newer = [b'R', b'G', b'F', 6];
        let error = load_kgl_bytes(&newer).err().unwrap().to_string();
        assert!(error.contains("version 6") && error.contains("upgrade kglite"));

        let mut invalid = vec![b'R', b'G', b'F', 5, 99];
        invalid.extend_from_slice(&CURRENT_CORE_DATA_VERSION.to_le_bytes());
        invalid.extend_from_slice(&0u32.to_le_bytes());
        let error = load_kgl_bytes(&invalid).err().unwrap().to_string();
        assert!(error.contains("invalid codec tag"));
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
        // An unrecognized buffer carries the hint too.
        let bad = [0u8, 1, 2, 3, 4, 5];
        let err = load_kgl_bytes(&bad).err().unwrap();
        assert!(err.to_string().contains("from_blueprint"));
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
        assert_eq!(&buf[..4], &V5_MAGIC);
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
}

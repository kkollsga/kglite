//! `.kgl` column-section loading — the reader half of the columnar save path.
//!
//! Split out of `io/file.rs` when that file passed its 2500-line ceiling. These
//! three functions are the only place a load installs column stores onto the
//! storage backend, which since D1 Phase 3 is their sole owner: the loader
//! reads a section (or a per-type sidecar), builds a `ColumnStore`, hands it to
//! `DirGraph::install_column_store`, and then points each node of the type at
//! its row.

use super::*;

/// Load `columns/<type>/columns.zst` sidecars onto the storage backend.
/// Skips entries whose type is already loaded (from `columns.bin`'s mmap
/// fast path). Used by both the earlier per-type layout and the additive
/// post-`columns.bin` path that covers types added post-build via
/// `add_nodes`.
pub(super) fn load_column_sidecars(
    dir: &std::path::Path,
    graph: &mut crate::graph::dir_graph::DirGraph,
) -> io::Result<()> {
    use rayon::prelude::*;

    let columns_dir = dir.join("columns");
    if !columns_dir.exists() {
        return Ok(());
    }

    // Collect job descriptors so the heavy work (read + zstd decode +
    // ColumnStore::load_packed) can run in a rayon thread pool. On a
    // 17M-node Wikidata article-author carve with ~4,500 distinct
    // types, the previous sequential loop spent ~70 s in zstd alone;
    // parallelising drops it to a few seconds on a 16-core machine.
    struct Job {
        type_name: String,
        col_file: std::path::PathBuf,
        schema: Arc<crate::graph::schema::TypeSchema>,
        type_meta: std::collections::HashMap<String, String>,
    }

    let mut jobs: Vec<Job> = Vec::new();
    for entry in std::fs::read_dir(&columns_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let type_name = entry.file_name().to_string_lossy().to_string();
        if graph.column_store(&type_name).is_some() {
            // columns.bin mmap path already loaded this type.
            continue;
        }
        let col_file = entry.path().join("columns.zst");
        if !col_file.exists() {
            continue;
        }
        let schema = graph
            .type_schemas
            .get(&type_name)
            .cloned()
            .unwrap_or_else(|| std::sync::Arc::new(crate::graph::schema::TypeSchema::new()));
        let type_meta = graph
            .node_type_metadata
            .get(&type_name)
            .cloned()
            .unwrap_or_default();
        jobs.push(Job {
            type_name,
            col_file,
            schema,
            type_meta,
        });
    }

    // Decompress + load_packed each sidecar in parallel.
    let interner = &graph.interner;
    let results: Vec<io::Result<(String, crate::graph::storage::column_store::ColumnStore)>> = jobs
        .into_par_iter()
        .map(
            |job| -> io::Result<(String, crate::graph::storage::column_store::ColumnStore)> {
                let compressed = std::fs::read(&job.col_file)?;
                let decoded = zstd_decompress(&compressed)?;
                // Current format: `KGLCOLv2` + row_count + Postcard-backed
                // mixed columns. Older sidecars require a pre-0.14 converter.
                if decoded.len() < 12 || &decoded[..8] != b"KGLCOLv2" {
                    return Err(pre_014_bincode_error("KGLCOLv1/raw column sidecar"));
                }
                let packed_slice = &decoded[12..];
                let row_count = u32::from_le_bytes(decoded[8..12].try_into().unwrap());
                let codec = serde_codec::CodecVersion::PostcardV1;
                let store =
                    crate::graph::storage::column_store::ColumnStore::load_packed_with_codec(
                        job.schema,
                        &job.type_meta,
                        interner,
                        packed_slice,
                        row_count,
                        None,
                        codec,
                    )?;
                Ok((job.type_name, store))
            },
        )
        .collect();

    for r in results {
        let (type_name, store) = r?;
        graph.install_column_store(&type_name, Arc::new(store));
    }
    Ok(())
}

pub(super) fn attach_portable_column_stores(dir_graph: &mut DirGraph) {
    // `(type name, has id/title)` snapshot first: the loop below takes a
    // mutable node borrow, and the stores now live on the same backend.
    let types: Vec<(String, bool)> = dir_graph
        .column_stores_by_name()
        .into_iter()
        .map(|(name, store)| (name.to_string(), store.has_id_title_columns()))
        .collect();
    for (type_name, has_id_title) in types {
        let type_name = type_name.as_str();
        let Some(indices) = dir_graph.type_indices.get(type_name) else {
            continue;
        };
        for (row_id, node_idx) in indices.iter().enumerate() {
            let Some(node) = dir_graph.graph.node_weight_mut(node_idx) else {
                continue;
            };
            node.properties = PropertyStorage::Columnar(ColumnarRow::new(row_id as u32));
            if has_id_title {
                node.id = Value::Null;
                node.title = Value::Null;
            }
        }
    }
    dir_graph.rebuild_indices_from_keys();
}
